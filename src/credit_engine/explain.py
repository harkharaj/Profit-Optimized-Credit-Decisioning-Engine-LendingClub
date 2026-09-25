"""Phase 8b: explainability - SHAP for M2 and adverse-action style reason codes.

SHAP splits one applicant's prediction into a contribution from each feature
(in log-odds of default; the calibrated PD is a monotone transform of it, so a
factor that raises the log-odds also raises the PD).

Reason codes mimic US adverse-action notices (ECOA / Reg B): a declined applicant is
told the main factors behind the decision. Here: the 4 features that pushed their
predicted risk up the most.
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from .profit import recommended_approvals
from .utils import save_csv

FRIENDLY_NAMES = {
    "loan_amnt": "Loan amount", "emp_length_years": "Years employed",
    "emp_length_missing": "Employment length not given", "log_annual_inc": "Annual income",
    "dti": "Debt-to-income ratio", "delinq_2yrs": "Delinquencies in last 2 years",
    "inq_last_6mths": "Credit inquiries in last 6 months", "pub_rec": "Public records",
    "pub_rec_bankruptcies": "Bankruptcies on file", "fico_mid": "FICO score",
    "mths_since_last_delinq": "Months since last delinquency", "never_delinquent": "No delinquency on file",
    "mths_since_last_record": "Months since last public record", "no_public_record": "No public record on file",
    "open_acc": "Open credit lines", "total_acc": "Total credit lines", "mort_acc": "Mortgage accounts",
    "revol_bal": "Revolving balance", "revol_util": "Revolving credit utilization",
    "credit_hist_months": "Length of credit history", "loan_to_income": "Loan amount relative to income",
    "revol_bal_to_income": "Revolving balance relative to income", "open_to_total_acc": "Share of credit lines open",
    "bc_util": "Bankcard utilization", "acc_open_past_24mths": "Accounts opened in last 24 months",
    "percent_bc_gt_75": "Share of bankcards above 75% utilization",
    "collections_12_mths_ex_med": "Collections in last 12 months", "home_ownership": "Home ownership",
    "verification_status": "Income verification", "purpose": "Loan purpose",
}


def format_value(feature: str, value) -> str:
    if pd.isna(value):
        return "not reported"
    if isinstance(value, str):
        return value
    if feature == "log_annual_inc":
        return f"${np.expm1(value):,.0f}"
    if feature in ("loan_amnt", "revol_bal"):
        return f"${value:,.0f}"
    if feature in ("dti", "revol_util", "bc_util", "percent_bc_gt_75"):
        return f"{value:.1f}%"
    if feature in ("loan_to_income", "revol_bal_to_income", "open_to_total_acc"):
        return f"{value:.2f}"
    if feature == "credit_hist_months":
        return f"{value / 12:.1f} years"
    if feature in ("mths_since_last_delinq", "mths_since_last_record") and value >= 999:
        return "never"
    if feature in ("emp_length_missing", "never_delinquent", "no_public_record"):
        return "yes" if value == 1 else "no"
    return f"{value:g}"


def load_m2(cfg: dict):
    bundle = joblib.load(cfg["paths"]["models_dir"] / "credit_models.pkl")
    booster = lgb.Booster(model_str=(cfg["paths"]["models_dir"] / "lightgbm_m2.txt").read_text(encoding="utf-8"))
    return bundle["builder"], booster


def reason_codes(shap_values: np.ndarray, X: pd.DataFrame, n: int, direction: int = 1) -> list[list[str]]:
    """For each row, the n features with the largest contribution in `direction` (+1 = raised risk)."""
    columns = X.columns
    out = []
    for i in range(len(X)):
        contrib = direction * shap_values[i]
        top = [j for j in np.argsort(-contrib)[:n] if contrib[j] > 0]
        out.append([f"{FRIENDLY_NAMES[columns[j]]}: {format_value(columns[j], X.iloc[i, j])}" for j in top])
    return out


def run_explain(data: pd.DataFrame, cfg: dict) -> dict:
    """SHAP on a test sample, reason codes for declined applicants, app sample."""
    from . import plots

    e, seed = cfg["explain"], cfg["seed"]
    builder, booster = load_m2(cfg)
    test = data[data["split"] == "test"].reset_index(drop=True)
    test["decision"] = np.where(recommended_approvals(test, cfg), "approve", "decline")

    sample = test.sample(min(e["shap_sample_size"], len(test)), random_state=seed).reset_index(drop=True)
    X = builder.transform(sample)
    values = shap.TreeExplainer(booster).shap_values(X)
    values = values[1] if isinstance(values, list) else values        # older shap returns [neg, pos]

    importance = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    importance["name"] = importance["feature"].map(FRIENDLY_NAMES)
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    save_csv(importance, "shap_importance.csv", cfg)
    plots.shap_summary(values, X, [FRIENDLY_NAMES[c] for c in X.columns], cfg)

    n = e["n_reason_codes"]
    raised, lowered = reason_codes(values, X, n, +1), reason_codes(values, X, 2, -1)
    for k in range(n):
        sample[f"risk_factor_{k + 1}"] = [r[k] if k < len(r) else "" for r in raised]
    for k in range(2):
        sample[f"helping_factor_{k + 1}"] = [r[k] if k < len(r) else "" for r in lowered]

    keep = ["id", "issue_date", "grade", "sub_grade", "int_rate", "loan_amnt", "purpose", "pd_m2",
            "expected_profit", "decision", "target"] + [c for c in sample if c.endswith(("_factor_1", "_factor_2",
                                                                                           "_factor_3", "_factor_4"))]
    explained = sample[keep]
    explained.to_parquet(cfg["paths"]["processed_dir"] / "explain_sample.parquet", index=False)

    declined = explained[explained["decision"] == "decline"]
    examples = declined.sample(e["n_example_declines"], random_state=seed)
    save_csv(examples, "reason_codes_examples.csv", cfg)
    return {"shap_importance": importance, "examples": examples}
