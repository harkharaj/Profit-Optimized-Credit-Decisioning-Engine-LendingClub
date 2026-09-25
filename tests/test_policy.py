"""Approval policies approve the right number of loans in the right order."""
import numpy as np
import pandas as pd
import pytest

from credit_engine.models import subgrade_rank
from credit_engine.profit import approve_top, policy_order


@pytest.mark.parametrize("rate", [0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("n", [10, 997, 1000])
def test_approved_count_is_round_k_times_n(rate, n):
    order = np.random.default_rng(0).permutation(n)
    assert approve_top(order, rate).sum() == round(rate * n)


def test_subgrade_rank():
    assert subgrade_rank(pd.Series(["A1", "A5", "B1", "G5"])).tolist() == [1, 5, 6, 35]


def test_grade_policy_orders_subgrades_then_rate():
    df = pd.DataFrame({"sub_grade": ["C2", "A1", "B3", "A1", "A2"],
                       "int_rate": [14.0, 6.0, 11.0, 5.5, 6.5]})
    df["score_b0"] = subgrade_rank(df["sub_grade"])
    ordered = df.iloc[policy_order(df, "grade", seed=42)]
    assert ordered["sub_grade"].tolist() == ["A1", "A1", "A2", "B3", "C2"]
    assert ordered["int_rate"].tolist()[:2] == [5.5, 6.0]     # tie on sub-grade -> lower rate first


def test_grade_policy_approves_best_subgrades():
    df = pd.DataFrame({"sub_grade": ["D1", "A1", "C1", "B1"], "int_rate": [17.0, 6.0, 13.0, 9.0]})
    df["score_b0"] = subgrade_rank(df["sub_grade"])
    approved = approve_top(policy_order(df, "grade", seed=42), 0.5)
    assert set(df.loc[approved, "sub_grade"]) == {"A1", "B1"}
