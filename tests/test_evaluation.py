"""Metric helpers agree with scikit-learn and with hand-computed values."""
import numpy as np
from sklearn.metrics import roc_auc_score

from credit_engine.evaluation import _WeightedAUC, ks_statistic, top_decile_capture


def test_fast_bootstrap_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    score = np.round(y + rng.normal(0, 2, 3000))          # rounding creates many ties
    counts = np.bincount(rng.integers(0, 3000, 3000), minlength=3000)
    resample = np.repeat(np.arange(3000), counts)
    assert np.isclose(_WeightedAUC(y, score)(np.ones(3000)), roc_auc_score(y, score))
    assert np.isclose(_WeightedAUC(y, score)(counts.astype(float)), roc_auc_score(y[resample], score[resample]))


def test_ks_perfect_and_useless():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([1, 2, 3, 4])) == 1.0
    assert ks_statistic(y, np.array([1, 1, 1, 1])) == 0.0


def test_top_decile_capture():
    y = np.array([1] + [0] * 9 + [1] + [0] * 9)            # 20 loans, 2 defaults
    score = np.arange(20)[::-1].astype(float)               # loan 0 is the riskiest
    assert top_decile_capture(y, score) == 0.5              # top 2 loans catch 1 of 2 defaults
