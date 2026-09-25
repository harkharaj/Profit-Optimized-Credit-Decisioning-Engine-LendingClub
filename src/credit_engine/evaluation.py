"""Model metrics used by credit-risk teams.

Convention everywhere: a higher score means a RISKIER applicant.

AUC     probability a random defaulter is scored riskier than a random non-defaulter
Gini    2*AUC - 1 (the scorecard industry's name for the same thing)
KS      max gap between the score distributions of defaulters and non-defaulters
Brier   mean squared error of the predicted probability (lower = better calibrated)
Top-decile capture   share of all defaults found in the riskiest 10% of applicants
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y, score) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def top_decile_capture(y, score) -> float:
    y, score = np.asarray(y), np.asarray(score)
    riskiest = np.argsort(-score, kind="stable")[: int(round(0.10 * len(y)))]
    return float(y[riskiest].sum() / y.sum())


def decile_table(y, pd_hat) -> pd.DataFrame:
    """Calibration by decile of predicted PD: does predicted match actual?"""
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_hat)})
    df["decile"] = pd.qcut(df["pd"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    table = df.groupby("decile").agg(n=("y", "size"), predicted_pd=("pd", "mean"), actual_rate=("y", "mean"))
    return table.reset_index()


def score_metrics(y, score, pd_raw=None, pd_calibrated=None) -> dict:
    auc = roc_auc_score(y, score)
    out = {"auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y, score),
           "top_decile_capture": top_decile_capture(y, score),
           "default_rate": float(np.mean(y)), "n": int(len(y))}
    if pd_raw is not None:
        out["brier_raw"] = brier_score_loss(y, pd_raw)
    if pd_calibrated is not None:
        out["brier_calibrated"] = brier_score_loss(y, pd_calibrated)
        out["mean_predicted_pd"] = float(np.mean(pd_calibrated))
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap for AUC differences
# ---------------------------------------------------------------------------

class _WeightedAUC:
    """AUC for many bootstrap resamples without re-sorting each time.

    A bootstrap resample = the original loans, each repeated w_i times. Sorting by
    score once and reusing the order makes each resample O(n) instead of O(n log n).
    """

    def __init__(self, y, score):
        order = np.argsort(score, kind="stable")
        self.order = order
        self.y = np.asarray(y)[order]
        _, self.tie_group = np.unique(np.asarray(score)[order], return_inverse=True)

    def __call__(self, weights) -> float:
        w = weights[self.order]
        pos = np.bincount(self.tie_group, w * self.y)          # defaulters per distinct score
        neg = np.bincount(self.tie_group, w * (1 - self.y))    # non-defaulters per distinct score
        neg_below = np.cumsum(neg) - neg                        # non-defaulters with a lower score
        wins = np.sum(pos * (neg_below + 0.5 * neg))            # ties count half
        return wins / (pos.sum() * neg.sum())


def paired_bootstrap_auc_diff(y, score_a, score_b, n_resamples: int, seed: int) -> dict:
    """95% CI for AUC(a) - AUC(b), resampling the SAME loans for both models."""
    rng = np.random.default_rng(seed)
    auc_a, auc_b = _WeightedAUC(y, score_a), _WeightedAUC(y, score_b)
    n = len(y)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        diffs[i] = auc_a(weights) - auc_b(weights)
    point = auc_a(np.ones(n)) - auc_b(np.ones(n))
    return {"auc_diff": point, "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
            "share_resamples_diff_le_0": float(np.mean(diffs <= 0)), "n_resamples": n_resamples}
