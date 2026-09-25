"""PSI is 0 for identical distributions and grows as the distribution shifts."""
import numpy as np
import pandas as pd

from credit_engine.monitoring import bin_shares, psi, psi_status


def test_psi_zero_for_identical_distributions():
    x = pd.Series(np.random.default_rng(0).normal(size=10_000))
    assert psi(*bin_shares(x, x, 10)) == 0.0


def test_psi_positive_and_growing_with_shift():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=20_000))
    small = psi(*bin_shares(pd.Series(rng.normal(0.2, 1, 20_000)), reference, 10))
    large = psi(*bin_shares(pd.Series(rng.normal(1.0, 1, 20_000)), reference, 10))
    assert 0 < small < large
    assert psi_status(large) == "action"


def test_missing_values_count_as_drift():
    reference = pd.Series([1.0, 2.0, 3.0, np.nan] * 1000)
    now = pd.Series([1.0, 2.0, 3.0, 3.0] * 1000)           # the field stopped being missing
    assert psi(*bin_shares(now, reference, 10)) > 0.10


def test_categorical_psi():
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi(*bin_shares(reference, reference, 10)) == 0.0
    assert psi(*bin_shares(pd.Series(["a"] * 900 + ["b"] * 100), reference, 10)) > 0.25
