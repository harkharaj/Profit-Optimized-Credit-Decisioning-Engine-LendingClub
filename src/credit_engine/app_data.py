"""Phase 9 support: copy small, precomputed artifacts into app/data/ for the Streamlit app.

The app never trains or reads the raw data; it only loads what is written here.
"""
import shutil

import pandas as pd

from .profit import prepare_policy_frame

BOOK_COLUMNS = ["id", "issue_date", "grade", "sub_grade", "int_rate", "loan_amnt", "funded_amnt", "purpose",
                "target", "realized_profit", "pd_b0", "pd_m1", "pd_m2", "raw_m1", "raw_m2", "score_b0",
                "r_good", "loss_rate"]
METRIC_FILES = ["model_metrics.json", "policy_results.json", "forecast.json", "monitoring_summary.json",
                "data_summary.json", "splits.json", "vintage_by_grade.csv", "psi.csv", "csi.csv",
                "weak_spots.csv", "shap_importance.csv", "profit_sensitivity.csv", "swap_set.csv"]
FIGURES = ["roc.png", "ks.png", "calibration.png", "shap_summary.png", "forecast_vs_actual.png",
           "vintage_default_by_grade.png", "profit_curve.png", "swap_set.png", "psi_by_quarter.png"]


def build_app_data(df: pd.DataFrame, scores: pd.DataFrame, cfg: dict) -> dict:
    out = cfg["paths"]["app_data_dir"]
    (out / "figures").mkdir(parents=True, exist_ok=True)

    data, _ = prepare_policy_frame(df, scores, cfg)
    book = data.loc[data["split"] == "test", BOOK_COLUMNS]
    if len(book) > cfg["app"]["max_rows"]:
        book = book.sample(cfg["app"]["max_rows"], random_state=cfg["seed"])
    book.to_parquet(out / "test_book.parquet", index=False)

    shutil.copy(cfg["paths"]["processed_dir"] / "explain_sample.parquet", out / "explain_sample.parquet")
    for name in METRIC_FILES:
        shutil.copy(cfg["paths"]["metrics_dir"] / name, out / name)
    for name in FIGURES:
        shutil.copy(cfg["paths"]["figures_dir"] / name, out / "figures" / name)

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    return {"n_book_rows": len(book), "app_data_mb": size_mb}
