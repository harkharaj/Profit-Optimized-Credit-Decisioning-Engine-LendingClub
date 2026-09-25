"""No post-origination, pricing, identifier or fair-lending column can reach a model."""
import pandas as pd
import pytest

from credit_engine.data import load_model_base
from credit_engine.features import (CATEGORICAL_FEATURES, DENYLIST, FAIR_LENDING_EXCLUDED, NUMERIC_FEATURES,
                                    OPTIONAL_BUREAU_COLUMNS, PRICING_COLUMNS, FeatureBuilder, LeakageError,
                                    assert_no_leakage)

FORBIDDEN = set(DENYLIST) | set(PRICING_COLUMNS) | set(FAIR_LENDING_EXCLUDED)


def test_candidate_features_are_all_allowed():
    candidates = set(NUMERIC_FEATURES) | set(OPTIONAL_BUREAU_COLUMNS) | set(CATEGORICAL_FEATURES)
    assert candidates & FORBIDDEN == set()


@pytest.mark.parametrize("column", ["int_rate", "grade", "sub_grade", "installment", "total_pymnt",
                                    "recoveries", "last_fico_range_high", "hardship_flag",
                                    "settlement_amount", "addr_state", "zip_code", "id"])
def test_guard_raises_on_forbidden_column(column):
    with pytest.raises(LeakageError):
        assert_no_leakage(["fico_mid", column])


def test_feature_matrix_has_no_forbidden_columns(cfg, needs_db):
    df = load_model_base(cfg)
    train = df[df["split"] == "train"]
    builder = FeatureBuilder().fit(train, cfg)
    X = builder.transform(df.sample(5000, random_state=0))
    assert set(X.columns) & FORBIDDEN == set()
    assert not any(c.startswith(("hardship_", "settlement_")) for c in X.columns)
