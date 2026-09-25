"""Profit formulas against hand-computed examples."""
import numpy as np
import pandas as pd
import pytest

from credit_engine.profit import estimate_economics, expected_profit, realized_profit


def test_realized_profit_hand_example():
    # $10,000 loan, $11,500 paid back, 1% servicing fee -> 11,500 * 0.99 - 10,000 = 1,385
    assert realized_profit(11_500, 10_000, 0.01) == pytest.approx(1_385)
    # charged off after paying back $4,000 -> 4,000 * 0.99 - 10,000 = -6,040
    assert realized_profit(4_000, 10_000, 0.01) == pytest.approx(-6_040)


def test_expected_profit_hand_example():
    # PD 10%, good-loan return 15%, loss rate 40%, $10,000, $50 cost
    # 0.9 * 0.15 * 10,000 - 0.1 * 0.40 * 10,000 - 50 = 1,350 - 400 - 50 = 900
    assert expected_profit(0.10, 0.15, 0.40, 10_000, 50) == pytest.approx(900)


def test_expected_profit_breakeven_pd():
    # break-even PD = r / (r + L): 0.15 / 0.55
    assert expected_profit(0.15 / 0.55, 0.15, 0.40, 10_000) == pytest.approx(0, abs=1e-9)


def test_estimate_economics_uses_goods_for_return_and_bads_for_loss():
    train = pd.DataFrame({
        "sub_grade": ["A1"] * 200 + ["A1"] * 100,
        "grade": ["A"] * 300,
        "target": [0] * 200 + [1] * 100,
        "funded_amnt": [1000.0] * 300,
        "realized_profit": [100.0] * 200 + [-500.0] * 100,
    })
    econ = estimate_economics(train)
    assert econ["r_good_by_subgrade"]["A1"] == pytest.approx(0.10)
    assert econ["loss_rate_by_grade"]["A"] == pytest.approx(0.50)
