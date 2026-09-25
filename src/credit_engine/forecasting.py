"""Phase 7: expected-loss forecasting vs actuals, by issue quarter of the test period.

For each quarter's loans:
    forecast defaults      = sum of PD_i
    actual defaults        = sum of target_i
    forecast net loss $    = sum of PD_i * L[grade_i] * loan_amount_i
    actual net loss $      = sum of -realized_profit_i over charged-off loans

Forecasters:
    M1, M2       calibrated PDs (calibration fit on 2013)
    Benchmark    the naive method: TRAIN default rate of each grade, applied to the quarter's grade mix
"""
import numpy as np
import pandas as pd

from .profit import prepare_policy_frame, recommended_approvals
from .utils import save_json

FORECASTERS = {"M1": "pd_m1", "M2": "pd_m2", "Benchmark": "pd_grade_hist"}


def quarterly_forecast(book: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per quarter x forecaster."""
    book = book.assign(quarter=book["issue_date"].dt.to_period("Q").astype(str),
                       actual_loss=np.where(book["target"] == 1, -book["realized_profit"], 0.0))
    rows = []
    for quarter, q in book.groupby("quarter"):
        actual_defaults, actual_loss = q["target"].sum(), q["actual_loss"].sum()
        for name, col in FORECASTERS.items():
            forecast_defaults = q[col].sum()
            forecast_loss = (q[col] * q["loss_rate"] * q["loan_amnt"]).sum()
            rows.append({"quarter": quarter, "forecaster": name, "n_loans": len(q),
                         "actual_defaults": actual_defaults, "forecast_defaults": forecast_defaults,
                         "actual_default_rate": actual_defaults / len(q), "forecast_default_rate": forecast_defaults / len(q),
                         "default_error_pct": 100 * (forecast_defaults - actual_defaults) / actual_defaults,
                         "actual_loss_usd": actual_loss, "forecast_loss_usd": forecast_loss,
                         "loss_error_pct": 100 * (forecast_loss - actual_loss) / actual_loss})
    return pd.DataFrame(rows)


def mape(table: pd.DataFrame) -> dict:
    """Mean absolute % error across quarters, per forecaster (and the average signed bias)."""
    out = {}
    for name, g in table.groupby("forecaster"):
        out[name] = {"defaults_mape_pct": g["default_error_pct"].abs().mean(),
                     "loss_mape_pct": g["loss_error_pct"].abs().mean(),
                     "defaults_mean_error_pct": g["default_error_pct"].mean(),
                     "loss_mean_error_pct": g["loss_error_pct"].mean(),
                     "total_forecast_loss_usd": g["forecast_loss_usd"].sum(),
                     "total_actual_loss_usd": g["actual_loss_usd"].sum()}
    return out


def run_forecast(df: pd.DataFrame, scores: pd.DataFrame, cfg: dict) -> dict:
    data, _ = prepare_policy_frame(df, scores, cfg)
    train_rate = data[data["split"] == "train"].groupby("grade")["target"].mean()
    data["pd_grade_hist"] = data["grade"].map(train_rate)

    test = data[data["split"] == "test"].reset_index(drop=True)
    books = {"whole_book": test, "approved_book": test[recommended_approvals(test, cfg)]}
    results = {"recommended_policy": cfg["profit"]["recommended_policy"],
               "approval_rate": cfg["profit"]["swap_set_approval_rate"],
               "benchmark_train_default_rate_by_grade": train_rate}
    for name, book in books.items():
        table = quarterly_forecast(book)
        results[name] = {"by_quarter": table, "summary": mape(table)}
    save_json(results, "forecast.json", cfg)
    return results
