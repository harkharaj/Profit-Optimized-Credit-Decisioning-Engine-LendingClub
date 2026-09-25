"""Feature allowlist, derived features and the leakage guard.

Rule: a model may only see information LendingClub had when the application
arrived. Anything created after the loan was issued (payments, recoveries,
later credit pulls) or LendingClub's own pricing is forbidden.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

# Raw application-time columns (before parsing / deriving)
CORE_RAW_COLUMNS = [
    "loan_amnt", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "pub_rec", "pub_rec_bankruptcies",
    "fico_range_low", "fico_range_high", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "total_acc", "mort_acc", "revol_bal", "revol_util", "earliest_cr_date",
]

# Credit-bureau fields LendingClub started reporting later; kept only if well
# populated in the TRAINING window (see max_missing_rate_train in config)
OPTIONAL_BUREAU_COLUMNS = [
    "tot_cur_bal", "total_rev_hi_lim", "bc_util", "avg_cur_bal", "acc_open_past_24mths",
    "num_tl_op_past_12m", "percent_bc_gt_75", "tot_coll_amt", "collections_12_mths_ex_med",
]

# LendingClub's own risk model output. Allowed for the benchmark and the profit
# math, never as features: using them would copy LC's model instead of competing
# with it (and installment = f(amount, rate, term) leaks int_rate).
PRICING_COLUMNS = ["grade", "sub_grade", "int_rate", "installment"]

# Post-origination information and identifiers: must never be features
DENYLIST = [
    "loan_status", "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_date", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "pymnt_plan", "policy_code", "initial_list_status", "disbursement_method",
    "debt_settlement_flag", "id", "member_id", "url", "desc",
    "target", "issue_date", "split",
]
DENYLIST_PREFIXES = ["hardship_", "settlement_"]

# Excluded for fair lending (ECOA / Reg B): geography can proxy for protected
# characteristics; free-text job titles are noisy and can proxy too
FAIR_LENDING_EXCLUDED = ["zip_code", "addr_state", "emp_title", "title"]

# Model-ready feature names (after 04_features.sql parsing)
NUMERIC_FEATURES = [
    "loan_amnt", "emp_length_years", "emp_length_missing", "log_annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "pub_rec_bankruptcies", "fico_mid",
    "mths_since_last_delinq", "never_delinquent", "mths_since_last_record", "no_public_record",
    "open_acc", "total_acc", "mort_acc", "revol_bal", "revol_util", "credit_hist_months",
    "loan_to_income", "revol_bal_to_income", "open_to_total_acc",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]


class LeakageError(Exception):
    pass


def assert_no_leakage(columns) -> None:
    """Raise if a post-origination, pricing, identifier or fair-lending column is a feature."""
    forbidden = set(DENYLIST) | set(PRICING_COLUMNS) | set(FAIR_LENDING_EXCLUDED)
    bad = [c for c in columns
           if c in forbidden or any(c.startswith(p) for p in DENYLIST_PREFIXES)]
    if bad:
        raise LeakageError(f"Forbidden columns in feature matrix: {bad}")


# ---------------------------------------------------------------------------
# Train-fitted preprocessing
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Learns every preprocessing statistic on TRAIN only, then applies it to any split.

    fit(train)   -> caps, rare-category groups, which bureau fields to keep
    transform(df)-> model feature matrix (numeric + pandas categoricals; NaN kept,
                    because LightGBM and the WoE scorecard both treat missing natively)
    """

    def fit(self, train: pd.DataFrame, cfg: dict) -> "FeatureBuilder":
        f = cfg["features"]
        self.caps = {c: train[c].quantile(f["cap_quantile"]) for c in ["annual_inc", "dti"]}

        missing_rate = train[OPTIONAL_BUREAU_COLUMNS].isna().mean()
        self.bureau_missing_rate = missing_rate.to_dict()
        self.bureau_kept = missing_rate[missing_rate <= f["max_missing_rate_train"]].index.tolist()

        purpose_share = train["purpose"].value_counts(normalize=True)
        self.purpose_levels = sorted(purpose_share[purpose_share >= f["rare_category_share"]].index)
        if "other" not in self.purpose_levels:
            self.purpose_levels.append("other")

        self.categories = {
            "home_ownership": sorted(train["home_ownership"].dropna().unique()),
            "verification_status": sorted(train["verification_status"].dropna().unique()),
            "purpose": self.purpose_levels,
        }
        self.numeric_features = NUMERIC_FEATURES + self.bureau_kept
        self.feature_names = self.numeric_features + CATEGORICAL_FEATURES
        assert_no_leakage(self.feature_names)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for col in self.numeric_features:
            if col == "log_annual_inc":
                X[col] = np.log1p(df["annual_inc"].clip(upper=self.caps["annual_inc"]))
            elif col == "dti":
                X[col] = df["dti"].clip(upper=self.caps["dti"])
            else:
                X[col] = df[col].astype(float)

        purpose = df["purpose"].where(df["purpose"].isin(self.purpose_levels), "other")
        X["purpose"] = pd.Categorical(purpose, categories=self.categories["purpose"])
        for col in ["home_ownership", "verification_status"]:
            X[col] = pd.Categorical(df[col], categories=self.categories[col])

        assert_no_leakage(X.columns)
        return X[self.feature_names]

    def summary(self) -> dict:
        return {
            "n_features": len(self.feature_names),
            "features": self.feature_names,
            "caps_from_train": self.caps,
            "bureau_missing_rate_train": self.bureau_missing_rate,
            "bureau_kept": self.bureau_kept,
            "bureau_dropped": [c for c in OPTIONAL_BUREAU_COLUMNS if c not in self.bureau_kept],
            "purpose_levels": self.purpose_levels,
        }


# ---------------------------------------------------------------------------
# Univariate check (leakage tripwire)
# ---------------------------------------------------------------------------

def univariate_auc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """AUC of each feature on its own. Direction-free: max(AUC, 1 - AUC).

    Numeric: raw value, with missing placed below the minimum so it acts as its own group.
    Categorical: each level replaced by its default rate in this same data.
    """
    rows = []
    for col in X.columns:
        x = X[col]
        if isinstance(x.dtype, pd.CategoricalDtype):
            score = x.map(y.groupby(x, observed=True).mean()).astype(float)
        else:
            score = x.fillna(x.min() - 1)
        auc = roc_auc_score(y, score)
        rows.append({"feature": col, "auc": max(auc, 1 - auc),
                     "direction": "higher = riskier" if auc >= 0.5 else "higher = safer",
                     "missing_rate": x.isna().mean()})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def feature_checks(train: pd.DataFrame, builder: FeatureBuilder, cfg: dict,
                   max_feature_auc: float = 0.80) -> pd.DataFrame:
    """Univariate AUC on train -> univariate_auc.csv, then the leakage tripwire.

    Application-time default models land around 0.65-0.75 AUC in total, so a single
    feature above 0.80 almost certainly carries information from after the loan was issued.
    LendingClub's pricing columns are added for reference only (they are never features).
    """
    from .utils import save_csv

    table = univariate_auc(builder.transform(train), train["target"]).assign(role="model feature")
    reference = univariate_auc(train[["int_rate"]], train["target"]).assign(role="LC pricing (benchmark only)")
    table = pd.concat([table, reference], ignore_index=True).sort_values("auc", ascending=False)
    save_csv(table, "univariate_auc.csv", cfg)

    suspicious = table[(table["role"] == "model feature") & (table["auc"] > max_feature_auc)]
    if len(suspicious):
        raise LeakageError(f"Univariate AUC above {max_feature_auc} - investigate before continuing:\n{suspicious}")
    return table
