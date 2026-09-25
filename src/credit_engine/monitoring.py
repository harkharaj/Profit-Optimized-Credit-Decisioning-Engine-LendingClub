"""Phase 8a: drift monitoring (PSI / CSI) and weak-spot mining.

PSI (Population Stability Index) compares today's score distribution with the one the
model was built on. Bins = train deciles.  PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)
    < 0.10 stable   |   0.10 - 0.25 monitor   |   > 0.25 action
CSI is the same formula applied to one input feature.

Weak spots: segments of borrowers where the model's predicted default rate is far from
what actually happened, ranked by the dollars at stake, with an early-warning check:
was the problem already visible in the 2013 validation data?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .utils import save_csv, save_json


# ---------------------------------------------------------------------------
# PSI / CSI
# ---------------------------------------------------------------------------

def psi_status(value: float) -> str:
    return "stable" if value < 0.10 else "monitor" if value <= 0.25 else "action"


def psi(expected_share: np.ndarray, actual_share: np.ndarray, eps: float = 1e-4) -> float:
    e, a = np.clip(expected_share, eps, None), np.clip(actual_share, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bin_shares(values: pd.Series, reference: pd.Series, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Shares of `reference` (train) and `values` in bins set from reference deciles.

    Categorical: one bin per level. Missing values always get their own bin, so a
    field that stops (or starts) being missing shows up as drift.
    """
    if isinstance(reference.dtype, pd.CategoricalDtype) or reference.dtype == object:
        levels = list(pd.Series(reference.astype(str).unique()))
        to_bin = lambda s: s.astype(str).map({lvl: i for i, lvl in enumerate(levels)}).fillna(len(levels)).to_numpy()
        n = len(levels) + 1
    else:
        edges = np.unique(np.nanquantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
        to_bin = lambda s: np.where(s.isna(), len(edges) + 1, np.searchsorted(edges, s.fillna(0), side="right"))
        n = len(edges) + 2
    share = lambda s: np.bincount(to_bin(s).astype(int), minlength=n) / len(s)
    return share(reference), share(values)


def drift_table(data: pd.DataFrame, columns: dict, cfg: dict) -> pd.DataFrame:
    """PSI of each column for validation and every test quarter, vs train."""
    n_bins = cfg["monitoring"]["psi_bins"]
    train = data[data["split"] == "train"]
    periods = [("valid 2013", data[data["split"] == "valid"])]
    test = data[data["split"] == "test"]
    periods += list(test.groupby(test["issue_date"].dt.to_period("Q").astype(str)))
    rows = []
    for name, col in columns.items():
        for period, part in periods:
            expected, actual = bin_shares(part[col], train[col], n_bins)
            value = psi(expected, actual)
            rows.append({"variable": name, "period": period, "n_loans": len(part),
                         "psi": value, "status": psi_status(value)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Weak-spot mining
# ---------------------------------------------------------------------------

def segment_labels(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Business-readable segments; quintile edges come from TRAIN only."""
    train = data[data["split"] == "train"]

    def quintiles(col, fmt):
        edges = np.quantile(train[col].dropna(), [0.2, 0.4, 0.6, 0.8])
        bounds = [-np.inf, *edges, np.inf]
        labels = [f"Q{i + 1} ({fmt(lo)}–{fmt(hi)})" if np.isfinite(lo) and np.isfinite(hi)
                  else f"Q1 (< {fmt(hi)})" if not np.isfinite(lo) else f"Q5 (> {fmt(lo)})"
                  for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]))]
        return pd.cut(data[col], bounds, labels=labels).astype(str)

    money = lambda v: f"${v / 1000:,.0f}k"
    fico_lo = (np.floor((data["fico_mid"] - 660) / 20) * 20 + 660).clip(upper=780)
    history_years = data["credit_hist_months"] / 12
    return {
        "purpose": data["purpose"],
        "home_ownership": data["home_ownership"],
        "verification_status": data["verification_status"],
        "emp_length": pd.cut(data["emp_length_years"], [-1, 0, 3, 6, 9, 10],
                             labels=["< 1 yr", "1–3 yrs", "4–6 yrs", "7–9 yrs", "10+ yrs"]).astype(str)
                        .replace("nan", "missing"),
        "fico_band": fico_lo.map(lambda lo: "780+" if lo >= 780 else f"{lo:.0f}–{lo + 19:.0f}"),
        "dti_band": pd.cut(data["dti"], [-np.inf, 10, 15, 20, 25, 30, np.inf],
                           labels=["< 10", "10–15", "15–20", "20–25", "25–30", "30+"]).astype(str),
        "income_quintile": quintiles("annual_inc", money),
        "loan_amount_quintile": quintiles("loan_amnt", money),
        "credit_history": pd.cut(history_years, [-np.inf, 5, 10, 15, 20, np.inf],
                                 labels=["< 5 yrs", "5–10 yrs", "10–15 yrs", "15–20 yrs", "20+ yrs"]).astype(str),
    }


def segment_table(part: pd.DataFrame, labels: dict, cfg: dict) -> pd.DataFrame:
    m = cfg["monitoring"]
    low, high = m["ratio_band"]
    rows = []
    for segment, lab in labels.items():
        for level, g in part.groupby(lab.loc[part.index]):
            if len(g) < m["min_segment_n"]:
                continue
            actual, predicted = g["target"].mean(), g["pd_m2"].mean()
            gap = actual - predicted
            rows.append({
                "segment": segment, "level": level, "n_loans": len(g),
                "actual_default_rate": actual, "predicted_default_rate": predicted,
                "gap_pp": 100 * gap, "actual_to_predicted": actual / predicted,
                "segment_auc": roc_auc_score(g["target"], g["raw_m2"]) if g["target"].nunique() == 2 else np.nan,
                # unexpected (positive) or over-provisioned (negative) loss dollars
                "est_dollar_impact": gap * len(g) * g["loan_amnt"].mean() * g["loss_rate"].mean(),
                "flagged": abs(100 * gap) >= m["gap_flag_pp"] or not (low <= actual / predicted <= high),
            })
    return pd.DataFrame(rows)


def weak_spots(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    labels = segment_labels(data)
    test = segment_table(data[data["split"] == "test"], labels, cfg)
    valid = segment_table(data[data["split"] == "valid"], labels, cfg)
    valid = valid[["segment", "level", "gap_pp", "actual_to_predicted", "flagged"]].rename(columns={
        "gap_pp": "valid_gap_pp", "actual_to_predicted": "valid_actual_to_predicted",
        "flagged": "already_flagged_in_2013"})
    table = test.merge(valid, on=["segment", "level"], how="left")
    table["already_flagged_in_2013"] = table["already_flagged_in_2013"].eq(True)
    # early warning below the flag threshold: was the error already pointing the same way in 2013?
    table["same_direction_in_2013"] = np.sign(table["valid_gap_pp"]) == np.sign(table["gap_pp"])
    table["abs_impact"] = table["est_dollar_impact"].abs()
    table = table.sort_values(["flagged", "abs_impact"], ascending=[False, False]).drop(columns="abs_impact")
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 8a driver
# ---------------------------------------------------------------------------

def run_monitoring(data: pd.DataFrame, X: pd.DataFrame, top_features: list[str], cfg: dict) -> dict:
    """data: policy frame (all splits); X: model feature matrix aligned with data."""
    score_psi = drift_table(data, {"M2 calibrated PD": "pd_m2"}, cfg)
    save_csv(score_psi, "psi.csv", cfg)

    features = X[top_features].assign(split=data["split"].values, issue_date=data["issue_date"].values)
    csi = drift_table(features, {f: f for f in top_features}, cfg)
    save_csv(csi, "csi.csv", cfg)

    spots = weak_spots(data, cfg)
    save_csv(spots, "weak_spots.csv", cfg)

    flagged = spots[spots["flagged"]]
    csi_max = csi[csi["period"] != "valid 2013"].groupby("variable")["psi"].max().sort_values(ascending=False)
    summary = {
        "score_psi_valid": float(score_psi.loc[score_psi["period"] == "valid 2013", "psi"].iloc[0]),
        "score_psi_max_test_quarter": float(score_psi.loc[score_psi["period"] != "valid 2013", "psi"].max()),
        "score_psi_status_by_period": dict(zip(score_psi["period"], score_psi["status"])),
        "csi_max_over_test_quarters": csi_max,
        "features_needing_action": csi_max[csi_max > 0.25].index.tolist(),
        "n_segments_checked": len(spots), "n_segments_flagged": len(flagged),
        "n_flagged_already_visible_in_2013": int(flagged["already_flagged_in_2013"].sum()),
        "n_flagged_with_2013_data": int(flagged["valid_gap_pp"].notna().sum()),
        "n_flagged_same_direction_in_2013": int(flagged["same_direction_in_2013"].sum()),
        "top_flagged_segments": flagged.head(10)[["segment", "level", "n_loans", "actual_default_rate",
                                                  "predicted_default_rate", "gap_pp", "est_dollar_impact",
                                                  "already_flagged_in_2013"]],
    }
    save_json(summary, "monitoring_summary.json", cfg)
    return {"psi": score_psi, "csi": csi, "weak_spots": spots, "summary": summary}
