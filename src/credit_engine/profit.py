"""Phase 6: the profit decision layer - turn PDs into approve / decline decisions measured in dollars.

Realized profit of a loan (evaluation only, known after the fact):
    profit_i = total_pymnt_i * (1 - servicing_fee) - funded_amnt_i

Expected profit of an applicant (the decision rule, known at application):
    E[profit_i] = (1 - PD_i) * r_good[sub_grade_i] * A_i  -  PD_i * L[grade_i] * A_i  -  c
      r_good  average net return on a loan that is repaid   (estimated on TRAIN)
      L       average net loss rate on a loan that defaults  (estimated on TRAIN)
      A       loan amount,  c  fixed cost per loan

Using LC's sub-grade/grade here is not leakage: the price is a given offer, and our only
decision is approve or decline at that price.

Framing: we only observe loans LendingClub accepted, so "approval rate" = the share of
LendingClub's already-approved book that we would keep. Every policy is a TIGHTENING policy.
"""
import numpy as np
import pandas as pd

from .utils import save_csv, save_json

POLICY_NAMES = {
    "grade": "Grade policy (existing)",
    "m1_pd": "M1 scorecard PD",
    "m2_pd": "M2 LightGBM PD",
    "m2_profit": "M2 expected profit",
    # same profit formula with the grade-based PD: separates "ranking by profit" from "a better PD"
    "b0_profit": "Grade-PD expected profit",
}


# ---------------------------------------------------------------------------
# Profit formulas
# ---------------------------------------------------------------------------

def realized_profit(total_pymnt, funded_amnt, servicing_fee: float):
    return np.asarray(total_pymnt) * (1 - servicing_fee) - np.asarray(funded_amnt)


def expected_profit(pd_, r_good, loss_rate, amount, fixed_cost: float = 0.0):
    pd_, r_good, loss_rate, amount = (np.asarray(v, dtype=float) for v in (pd_, r_good, loss_rate, amount))
    return (1 - pd_) * r_good * amount - pd_ * loss_rate * amount - fixed_cost


def estimate_economics(train: pd.DataFrame, min_n: int = 100) -> dict:
    """r_good by sub-grade and L by grade, from TRAIN loans only.

    r_good naturally includes early prepayment (a repaid loan that prepaid earns less interest).
    Sparse cells fall back to the grade average (r_good) or the overall average (L).
    """
    ret = train["realized_profit"] / train["funded_amnt"]
    good, bad = train["target"] == 0, train["target"] == 1

    r_sub = ret[good].groupby(train.loc[good, "sub_grade"]).agg(["mean", "size"])
    r_grade = ret[good].groupby(train.loc[good, "grade"]).mean()
    r_good = {sg: (row["mean"] if row["size"] >= min_n else r_grade.get(sg[0], ret[good].mean()))
              for sg, row in r_sub.iterrows()}

    loss = (-ret[bad]).groupby(train.loc[bad, "grade"]).agg(["mean", "size"])
    overall_loss = float((-ret[bad]).mean())
    loss_rate = {g: (row["mean"] if row["size"] >= min_n else overall_loss) for g, row in loss.iterrows()}

    return {"r_good_by_subgrade": r_good, "r_good_by_grade": r_grade.to_dict(),
            "overall_r_good": float(ret[good].mean()),
            "loss_rate_by_grade": loss_rate, "overall_loss_rate": overall_loss,
            "loss_rate_n_by_grade": loss["size"].to_dict()}


def attach_economics(df: pd.DataFrame, econ: dict) -> pd.DataFrame:
    """Add each loan's r_good and L (unseen sub-grades/grades fall back to overall averages)."""
    return df.assign(
        r_good=df["sub_grade"].map(econ["r_good_by_subgrade"])
                              .fillna(df["grade"].map(econ["r_good_by_grade"]))
                              .fillna(econ["overall_r_good"]),
        loss_rate=df["grade"].map(econ["loss_rate_by_grade"]).fillna(econ["overall_loss_rate"]),
    )


# ---------------------------------------------------------------------------
# Policies: each one is just an ordering of applicants, best first
# ---------------------------------------------------------------------------

def policy_order(df: pd.DataFrame, policy: str, seed: int) -> np.ndarray:
    """Row positions of df, from most to least preferred applicant."""
    if policy == "grade":
        # best sub-grade first; ties -> lower interest rate; remaining ties -> random (fixed seed)
        tiebreak = np.random.default_rng(seed).random(len(df))
        return np.lexsort((tiebreak, df["int_rate"].to_numpy(), df["score_b0"].to_numpy()))
    if policy == "m1_pd":
        return np.lexsort((df["raw_m1"].to_numpy(), df["pd_m1"].to_numpy()))   # lowest PD first
    if policy == "m2_pd":
        return np.lexsort((df["raw_m2"].to_numpy(), df["pd_m2"].to_numpy()))
    if policy == "m2_profit":
        return np.lexsort((df["raw_m2"].to_numpy(), -df["expected_profit"].to_numpy()))  # highest E[profit] first
    if policy == "b0_profit":
        return np.lexsort((df["score_b0"].to_numpy(), -df["expected_profit_b0"].to_numpy()))
    raise ValueError(policy)


def approve_top(order: np.ndarray, rate: float) -> np.ndarray:
    """Approve the first round(rate * n) applicants of an ordering."""
    approved = np.zeros(len(order), dtype=bool)
    approved[order[: int(round(rate * len(order)))]] = True
    return approved


def book_metrics(approved: np.ndarray, profit: np.ndarray, target: np.ndarray, funded: np.ndarray) -> dict:
    n_approved = int(approved.sum())
    total, dollars = float(profit[approved].sum()), float(funded[approved].sum())
    return {"approval_rate": n_approved / len(approved), "n_approved": n_approved,
            "bad_rate": float(target[approved].mean()), "total_profit": total,
            "profit_per_approved": total / n_approved, "profit_per_applicant": total / len(approved),
            # a policy can make more money simply by lending more dollars; this makes that visible
            "funded_usd": dollars, "return_on_funded": total / dollars}


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def swap_set(df: pd.DataFrame, model_ok: np.ndarray, grade_ok: np.ndarray) -> pd.DataFrame:
    """Loans the two policies disagree on.

    swap-in  = model approves, grade policy declines
    swap-out = grade policy approves, model declines
    If swap-ins have lower bad rates and higher profit than swap-outs, the model is choosing better.
    """
    groups = {"swap-in (model approves, grade declines)": model_ok & ~grade_ok,
              "swap-out (grade approves, model declines)": ~model_ok & grade_ok,
              "both approve": model_ok & grade_ok}
    rows = []
    for name, mask in groups.items():
        part = df[mask]
        rows.append({"group": name, "n_loans": int(mask.sum()), "bad_rate": part["target"].mean(),
                     "total_profit": part["realized_profit"].sum(),
                     "profit_per_loan": part["realized_profit"].mean(),
                     "avg_loan_amnt": part["loan_amnt"].mean(),
                     "return_on_funded": part["realized_profit"].sum() / part["funded_amnt"].sum(),
                     "avg_int_rate": part["int_rate"].mean(), "avg_pd_m2": part["pd_m2"].mean(),
                     "share_grade_A_B": part["grade"].isin(["A", "B"]).mean()})
    return pd.DataFrame(rows)


def _stressed(df: pd.DataFrame, loss_mult: float, cost: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Scenario where losses on defaults are loss_mult x worse and every loan costs `cost`.

    The stress hits both the decision (expected profit) and the outcome (realized profit).
    """
    scenario = df.assign(**{
        col: expected_profit(df[pd_col], df["r_good"], df["loss_rate"] * loss_mult, df["loan_amnt"], cost)
        for col, pd_col in [("expected_profit", "pd_m2"), ("expected_profit_b0", "pd_b0")]})
    real = df["realized_profit"].to_numpy()
    real = np.where((df["target"].to_numpy() == 1) & (real < 0), real * loss_mult, real) - cost
    return scenario, real


def sensitivity(test: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Does the policy ranking survive worse losses and per-loan costs?"""
    p = cfg["profit"]
    rate = p["swap_set_approval_rate"]
    target, funded = test["target"].to_numpy(), test["funded_amnt"].to_numpy()
    rows = []
    for mult in p["loss_rate_multipliers"]:
        for cost in p["fixed_costs"]:
            scenario, real = _stressed(test, mult, cost)
            profits = {pol: book_metrics(approve_top(policy_order(scenario, pol, seed), rate),
                                         real, target, funded)["total_profit"]
                       for pol in POLICY_NAMES}
            # profit-max rule under this scenario, vs the grade policy at the same approval rate
            pm_ok = scenario["expected_profit"].to_numpy() > 0
            grade_same = approve_top(policy_order(scenario, "grade", seed), pm_ok.mean())
            rows.append({"loss_rate_multiplier": mult, "fixed_cost_per_loan": cost,
                         **{f"profit_{pol}_at_{int(rate * 100)}pct": v for pol, v in profits.items()},
                         "best_policy_at_rate": max(profits, key=profits.get),
                         "m2_profit_minus_grade": profits["m2_profit"] - profits["grade"],
                         "profit_max_approval_rate": pm_ok.mean(),
                         "profit_max_profit": real[pm_ok].sum(),
                         "grade_profit_at_same_rate": real[grade_same].sum()})
    return pd.DataFrame(rows)


def bootstrap_profit_diff(profit: np.ndarray, order_a: np.ndarray, order_b: np.ndarray, rate: float,
                          n_resamples: int, seed: int) -> dict:
    """Paired 95% CI for total profit(policy a) - profit(policy b) at a fixed approval rate.

    Each resample re-weights the same loans (w_i = times drawn); each policy approves its
    best applicants until it has approved `rate` of the resampled book.
    """
    def top_profit(order, w):
        w_sorted = w[order]
        keep = np.cumsum(w_sorted) <= rate * w_sorted.sum()
        return float(np.sum((w_sorted * profit[order])[keep]))

    rng, n = np.random.default_rng(seed), len(profit)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        diffs[i] = top_profit(order_a, w) - top_profit(order_b, w)
    ones = np.ones(n)
    return {"profit_diff": top_profit(order_a, ones) - top_profit(order_b, ones),
            "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
            "share_resamples_diff_le_0": float(np.mean(diffs <= 0)), "n_resamples": n_resamples}


# ---------------------------------------------------------------------------
# Phase 6 driver
# ---------------------------------------------------------------------------

def _headline(results: dict, rate: float) -> dict:
    """The key test-set comparisons at the headline approval rate, for the reports."""
    table = results["test_policy_table"]
    at_rate = table[table["target_rate"] == rate].set_index("policy_key")
    grade = at_rate.loc["grade"]
    out = {"approval_rate": rate,
           "approve_all_profit": float(table.loc[table["policy_key"] == "all", "total_profit"].iloc[0])}
    for pol in ["grade", "m1_pd", "m2_pd", "m2_profit", "b0_profit"]:
        row = at_rate.loc[pol]
        out[pol] = {"total_profit": row["total_profit"], "bad_rate": row["bad_rate"],
                    "funded_usd": row["funded_usd"], "return_on_funded": row["return_on_funded"],
                    "profit_vs_grade": row["total_profit"] - grade["total_profit"],
                    "profit_vs_grade_pct": row["total_profit"] / grade["total_profit"] - 1,
                    "funded_vs_grade_pct": row["funded_usd"] / grade["funded_usd"] - 1}
    return out


def run_profit_analysis(df: pd.DataFrame, scores: pd.DataFrame, cfg: dict) -> dict:
    p, seed = cfg["profit"], cfg["seed"]
    data = df.merge(scores.drop(columns="split"), on="id")
    data["realized_profit"] = realized_profit(data["total_pymnt"], data["funded_amnt"], p["servicing_fee"])

    econ = estimate_economics(data[data["split"] == "train"])
    data = attach_economics(data, econ)
    for col, pd_col in [("expected_profit", "pd_m2"), ("expected_profit_b0", "pd_b0")]:
        data[col] = expected_profit(data[pd_col], data["r_good"], data["loss_rate"],
                                    data["loan_amnt"], p["fixed_cost_per_loan"])

    results = {"assumptions": {"servicing_fee": p["servicing_fee"], "fixed_cost_per_loan": p["fixed_cost_per_loan"],
                               "economics_estimated_on": "train (2010-2012)",
                               "approval_rate_meaning": "share of LendingClub's accepted book we would approve"},
               "economics": econ}

    for split in ["valid", "test"]:
        part = data[data["split"] == split].reset_index(drop=True)
        profit, target = part["realized_profit"].to_numpy(), part["target"].to_numpy()
        funded = part["funded_amnt"].to_numpy()
        orders = {pol: policy_order(part, pol, seed) for pol in POLICY_NAMES}

        rows = [{"policy": "Approve all (LC's actual book)", "policy_key": "all", "target_rate": 1.0,
                 **book_metrics(np.ones(len(part), bool), profit, target, funded)}]
        for rate in p["approval_rates"]:
            for pol, order in orders.items():
                rows.append({"policy": POLICY_NAMES[pol], "policy_key": pol, "target_rate": rate,
                             **book_metrics(approve_top(order, rate), profit, target, funded)})
        results[f"{split}_policy_table"] = pd.DataFrame(rows)

        # profit-max rule: approve if E[profit] > 0 (rule and parameters fixed before seeing test)
        pm_ok = part["expected_profit"].to_numpy() > 0
        grade_same = approve_top(orders["grade"], pm_ok.mean())
        results[f"{split}_profit_max"] = {
            "rule": "approve if expected profit > 0 (M2 PD)",
            "profit_max_policy": book_metrics(pm_ok, profit, target, funded),
            "grade_policy_same_approval_rate": book_metrics(grade_same, profit, target, funded),
            "profit_gain_vs_grade": float(profit[pm_ok].sum() - profit[grade_same].sum()),
        }

        if split == "test":
            test, test_orders = part, orders

    # --- swap sets at the headline approval rate (test): where do the policies disagree?
    rate = p["swap_set_approval_rate"]
    grade_ok = approve_top(test_orders["grade"], rate)
    swaps = pd.concat([swap_set(test, approve_top(test_orders[pol], rate), grade_ok)
                       .assign(comparison=f"{POLICY_NAMES[pol]} vs grade policy")
                       for pol in ["m2_profit", "m2_pd"]], ignore_index=True)
    save_csv(swaps, "swap_set.csv", cfg)
    results["swap_set"] = swaps

    # --- sensitivity and uncertainty (test)
    sens = sensitivity(test, cfg, seed)
    save_csv(sens, "profit_sensitivity.csv", cfg)
    results["sensitivity"] = sens
    results["sensitivity_ranking_holds"] = bool((sens["m2_profit_minus_grade"] > 0).all())

    print("  bootstrap CIs for the profit differences...")
    for pol in ["m2_profit", "m2_pd"]:
        results[f"bootstrap_test_{pol}_minus_grade"] = {
            "approval_rate": rate,
            **bootstrap_profit_diff(test["realized_profit"].to_numpy(), test_orders[pol],
                                    test_orders["grade"], rate, cfg["evaluation"]["bootstrap_resamples"], seed)}

    results["headline"] = _headline(results, rate)

    save_json(results, "policy_results.json", cfg)
    return results
