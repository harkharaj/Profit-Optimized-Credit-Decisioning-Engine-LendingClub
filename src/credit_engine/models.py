"""M2 LightGBM challenger, probability calibration, and the Phase 5 benchmark.

Models compared (higher score = riskier for all of them):
  B0  LendingClub's own decision system: sub-grade (A1=1 ... G5=35), and interest rate
  M1  WoE logistic scorecard (scorecard.py)
  M2  LightGBM on the raw allowlisted features, tuned with Optuna; the production
      version is monotone-constrained if that is (nearly) free on validation

Data use:
  train (2010-12)  fit models
  valid (2013)     tune hyper-parameters, early stopping, fit calibration
  test  (2014-15)  final evaluation only
"""
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

from .evaluation import decile_table, paired_bootstrap_auc_diff, score_metrics
from .features import FeatureBuilder
from .scorecard import WoEScorecard
from .utils import save_csv, save_json

MODEL_NAMES = {"B0": "LC sub-grade (existing system)", "B0_int_rate": "LC interest rate",
               "M1": "WoE scorecard", "M2": "LightGBM"}


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(params: dict, X_train, y_train, X_valid, y_valid, cfg: dict,
                   monotone: dict | None = None) -> lgb.Booster:
    """Fit with early stopping on validation AUC."""
    lc = cfg["lightgbm"]
    params = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": cfg["seed"],
              "deterministic": True, "force_col_wise": True, "bagging_freq": 1, **params}
    if monotone:   # +1: PD may only rise with the feature, -1: only fall, 0: free
        params["monotone_constraints"] = [monotone.get(c, 0) for c in X_train.columns]
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set)
    return lgb.train(params, train_set, num_boost_round=lc["max_boost_rounds"], valid_sets=[valid_set],
                     callbacks=[lgb.early_stopping(lc["early_stopping_rounds"], verbose=False)])


def tune_lightgbm(X_train, y_train, X_valid, y_valid, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Optuna (TPE sampler, fixed seed) search maximizing validation AUC."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        booster = train_lightgbm(params, X_train, y_train, X_valid, y_valid, cfg)
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return booster.best_score["valid_0"]["auc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["lightgbm"]["optuna_trials"])
    trials = study.trials_dataframe()[["number", "value", "user_attrs_best_iteration"]
                                      + [c for c in study.trials_dataframe() if c.startswith("params_")]]
    return study.best_params, trials.rename(columns={"value": "valid_auc"})


def lgb_predict(booster: lgb.Booster, X) -> np.ndarray:
    return booster.predict(X, num_iteration=booster.best_iteration)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """Maps a model's raw probability to a calibrated probability of default.

    Fit on VALIDATION, not train: train-set predictions are over-confident because the
    model has already seen those loans. Candidates:
      isotonic  flexible, monotone step function
      platt     logistic regression on the logit of the raw probability
    The winner has the lower 5-fold cross-validated Brier score *within* validation,
    so the choice isn't judged on the same loans it was fit to.
    """

    def fit(self, raw, y, seed: int) -> "Calibrator":
        raw, y = np.asarray(raw), np.asarray(y)
        cv_brier = {"isotonic": [], "platt": []}
        for fit_idx, eval_idx in KFold(5, shuffle=True, random_state=seed).split(raw):
            for method in cv_brier:
                model = self._fit_method(method, raw[fit_idx], y[fit_idx])
                cv_brier[method].append(brier_score_loss(y[eval_idx], self._apply(method, model, raw[eval_idx])))
        self.cv_brier = {m: float(np.mean(v)) for m, v in cv_brier.items()}
        self.method = min(self.cv_brier, key=self.cv_brier.get)
        self.model = self._fit_method(self.method, raw, y)
        return self

    @staticmethod
    def _fit_method(method, raw, y):
        if method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y)
        return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw), y)

    @staticmethod
    def _apply(method, model, raw):
        return model.predict(raw) if method == "isotonic" else model.predict_proba(_logit(raw))[:, 1]

    def predict(self, raw) -> np.ndarray:
        return self._apply(self.method, self.model, np.asarray(raw))


# ---------------------------------------------------------------------------
# Phase 5: train everything and benchmark against LendingClub's grade
# ---------------------------------------------------------------------------

def subgrade_rank(sub_grade: pd.Series) -> pd.Series:
    """A1 = 1, A2 = 2, ... G5 = 35 (higher = riskier)."""
    return (sub_grade.str[0].map(ord) - ord("A")) * 5 + sub_grade.str[1].astype(int)


def train_and_benchmark(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fit B0 / M1 / M2 + calibration, evaluate on valid & test, save metrics and models.

    Returns one row per loan with every model's raw score and calibrated PD.
    """
    seed, paths = cfg["seed"], cfg["paths"]
    parts = {s: df[df["split"] == s] for s in ["train", "valid", "test"]}
    y = {s: p["target"].to_numpy() for s, p in parts.items()}

    builder = FeatureBuilder().fit(parts["train"], cfg)
    X = {s: builder.transform(p) for s, p in parts.items()}

    # --- B0: LendingClub's grade. Its "raw PD" = train default rate of the loan's sub-grade
    b0_lookup = parts["train"].groupby("sub_grade")["target"].mean()
    b0_raw = {s: p["sub_grade"].map(b0_lookup).fillna(y["train"].mean()).to_numpy() for s, p in parts.items()}

    # --- M1: scorecard
    print("  fitting WoE scorecard...")
    scorecard = WoEScorecard().fit(X["train"], y["train"], X["valid"], y["valid"], cfg)
    m1_raw = {s: scorecard.predict_proba(X[s]) for s in parts}

    # --- M2: LightGBM (+ monotone-constraint experiment)
    print(f"  tuning LightGBM ({cfg['lightgbm']['optuna_trials']} Optuna trials)...")
    best_params, trials = tune_lightgbm(X["train"], y["train"], X["valid"], y["valid"], cfg)
    booster = train_lightgbm(best_params, X["train"], y["train"], X["valid"], y["valid"], cfg)
    mono_booster = train_lightgbm(best_params, X["train"], y["train"], X["valid"], y["valid"], cfg,
                                  monotone=cfg["lightgbm"]["monotone_constraints"])
    free_raw = {s: lgb_predict(booster, X[s]) for s in ["valid", "test"]}
    mono_raw = {s: lgb_predict(mono_booster, X[s]) for s in ["valid", "test"]}

    # production M2 is chosen on VALIDATION: keep the monotone model unless it costs too much AUC
    auc_cost = roc_auc_score(y["valid"], free_raw["valid"]) - roc_auc_score(y["valid"], mono_raw["valid"])
    use_monotone = auc_cost <= cfg["lightgbm"]["max_monotone_auc_cost"]
    booster = mono_booster if use_monotone else booster
    m2_raw = {s: lgb_predict(booster, X[s]) for s in parts}

    # --- calibration, fit on validation for every model
    raw = {"B0": b0_raw, "M1": m1_raw, "M2": m2_raw}
    calibrators = {m: Calibrator().fit(r["valid"], y["valid"], seed) for m, r in raw.items()}
    pd_cal = {m: {s: calibrators[m].predict(r[s]) for s in parts} for m, r in raw.items()}

    # --- rank scores (higher = riskier) used for AUC / KS / capture
    rank_score = {
        "B0": {s: subgrade_rank(p["sub_grade"]).to_numpy() for s, p in parts.items()},
        "B0_int_rate": {s: p["int_rate"].to_numpy() for s, p in parts.items()},
        "M1": m1_raw, "M2": m2_raw,
    }

    metrics = {"model_names": MODEL_NAMES, "models": {}, "decile_tables": {}}
    for m, scores in rank_score.items():
        metrics["models"][m] = {}
        for s in ["valid", "test"]:
            has_pd = m in pd_cal
            metrics["models"][m][s] = score_metrics(
                y[s], scores[s],
                pd_raw=raw[m][s] if has_pd else None,
                pd_calibrated=pd_cal[m][s] if has_pd else None)
    for s in ["valid", "test"]:
        metrics["decile_tables"][s] = {m: decile_table(y[s], pd_cal[m][s]) for m in pd_cal}

    # leakage tripwire: application-time models on this data land ~0.65-0.75
    for m in ["M1", "M2"]:
        if metrics["models"][m]["valid"]["auc"] > 0.85:
            raise RuntimeError(f"{m} validation AUC above 0.85 - investigate leakage before continuing")

    print("  paired bootstrap on test...")
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    metrics["bootstrap_test"] = {
        f"{m}_minus_B0": paired_bootstrap_auc_diff(y["test"], rank_score[m]["test"], rank_score["B0"]["test"],
                                                   n_boot, seed)
        for m in ["M1", "M2"]}

    metrics["monotone_experiment"] = {
        "constraints": cfg["lightgbm"]["monotone_constraints"],
        "valid_auc_unconstrained": roc_auc_score(y["valid"], free_raw["valid"]),
        "valid_auc_monotone": roc_auc_score(y["valid"], mono_raw["valid"]),
        "valid_auc_cost": auc_cost,                     # positive = constraints cost accuracy
        "test_auc_unconstrained": roc_auc_score(y["test"], free_raw["test"]),
        "test_auc_monotone": roc_auc_score(y["test"], mono_raw["test"]),
        "max_allowed_cost": cfg["lightgbm"]["max_monotone_auc_cost"],
        "production_m2": "monotone" if use_monotone else "unconstrained",
    }
    metrics["calibration"] = {m: {"method": c.method, "cv_brier_valid": c.cv_brier} for m, c in calibrators.items()}
    metrics["lightgbm"] = {"best_params": best_params, "best_iteration": booster.best_iteration,
                           "n_trials": len(trials), "best_valid_auc": trials["valid_auc"].max()}
    importance = pd.Series(booster.feature_importance("gain"), index=X["train"].columns)
    metrics["lightgbm"]["feature_importance_gain"] = (importance / importance.sum()).sort_values(ascending=False)
    metrics["scorecard"] = scorecard.summary()
    metrics["features"] = builder.summary()
    save_json(metrics, "model_metrics.json", cfg)
    save_csv(scorecard.points_table, "scorecard.csv", cfg)
    save_csv(scorecard.selection, "scorecard_feature_selection.csv", cfg)
    save_csv(trials, "lightgbm_optuna_trials.csv", cfg)

    # --- persist models
    joblib.dump({"builder": builder, "scorecard": scorecard, "calibrators": calibrators,
                 "b0_lookup": b0_lookup}, paths["models_dir"] / "credit_models.pkl")
    # written via Python: LightGBM's own save_model can't handle non-ASCII paths on Windows
    (paths["models_dir"] / "lightgbm_m2.txt").write_text(booster.model_to_string(), encoding="utf-8")

    # --- one row per loan with every score, for the profit / monitoring phases
    scores = pd.concat([
        pd.DataFrame({"id": parts[s]["id"].to_numpy(), "split": s,
                      "score_b0": rank_score["B0"][s],
                      "raw_m1": m1_raw[s], "raw_m2": m2_raw[s],
                      "pd_b0": pd_cal["B0"][s], "pd_m1": pd_cal["M1"][s], "pd_m2": pd_cal["M2"][s],
                      "points_m1": scorecard.score_points(X[s])})
        for s in parts], ignore_index=True)
    scores.to_parquet(paths["processed_dir"] / "scores.parquet", index=False)
    return scores
