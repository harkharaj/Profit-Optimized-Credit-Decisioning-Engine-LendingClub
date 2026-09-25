"""One entry point, raw file -> every metric, figure and report:

    python -m credit_engine.run_all

Each phase writes its results to reports/metrics/ (numbers) and reports/figures/ (charts).
"""
import time
import warnings

import matplotlib

matplotlib.use("Agg")   # render charts to files, no window

import numpy as np
import pandas as pd

from . import app_data, data, explain, forecasting, models, monitoring, plots, profit, report
from .config import load_config
from .features import FeatureBuilder, feature_checks
from .splits import summarize_splits
from .utils import load_json

warnings.filterwarnings("ignore", category=UserWarning)


def phase_1_load_and_clean(cfg):
    data.build_clean_tables(cfg)
    summary = data.data_summary(cfg)
    print(f"  {summary['n_loans']:,} loans, default rate {summary['overall_default_rate']:.1%}")


def phase_2_vintages(cfg):
    tables = data.vintage_analysis(cfg)
    plots.vintage_default_by_grade(tables["by_grade"], cfg)
    plots.cum_default_curves(tables["cum_default"], cfg)
    plots.pricing_vs_risk(tables["pricing"], cfg)


def phase_3_4_features_and_splits(cfg) -> pd.DataFrame:
    data.build_model_base(cfg)
    df = data.load_model_base(cfg)
    summarize_splits(df, cfg)
    train = df[df["split"] == "train"]
    table = feature_checks(train, FeatureBuilder().fit(train, cfg), cfg)
    top = table[table["role"] == "model feature"].iloc[0]
    print(f"  leakage tripwire passed: strongest single feature {top['feature']} (AUC {top['auc']:.3f})")
    return df


def phase_5_models(cfg, df) -> pd.DataFrame:
    scores = models.train_and_benchmark(df, cfg)
    metrics = load_json("model_metrics.json", cfg)

    test = scores[scores["split"] == "test"].merge(df[["id", "target"]], on="id")
    rank_scores = {"B0": test["score_b0"], "M1": test["raw_m1"], "M2": test["raw_m2"]}
    test_metrics = {m: metrics["models"][m]["test"] for m in rank_scores}
    plots.roc_curves(test["target"], rank_scores, {m: v["auc"] for m, v in test_metrics.items()}, cfg)
    plots.ks_panels(test["target"], rank_scores, {m: v["ks"] for m, v in test_metrics.items()}, cfg)
    plots.calibration_panels({s: {m: pd.DataFrame(t) for m, t in metrics["decile_tables"][s].items()}
                              for s in ["valid", "test"]}, cfg)
    for m, v in test_metrics.items():
        print(f"  test AUC {m}: {v['auc']:.3f}")
    return scores


def phase_6_profit(cfg, df, scores) -> dict:
    results = profit.run_profit_analysis(df, scores, cfg)
    plots.profit_curves(results["test_policy_table"], results["test_profit_max"], cfg)
    plots.swap_set_bars(results["swap_set"], cfg)
    h = results["headline"]
    for pol in ["m2_pd", "m2_profit"]:
        print(f"  {pol} vs grade at {h['approval_rate']:.0%} approval: "
              f"{h[pol]['profit_vs_grade'] / 1e6:+.1f}M ({h[pol]['profit_vs_grade_pct']:+.1%})")
    return results


def phase_7_forecast(cfg, df, scores):
    results = forecasting.run_forecast(df, scores, cfg)
    plots.forecast_vs_actual(results["whole_book"]["by_quarter"], cfg)
    for name, s in results["whole_book"]["summary"].items():
        print(f"  {name:9s} MAPE defaults {s['defaults_mape_pct']:.1f}%, net loss {s['loss_mape_pct']:.1f}%")


def phase_8_monitoring(cfg, df, scores):
    policy_frame, _ = profit.prepare_policy_frame(df, scores, cfg)
    builder, _ = explain.load_m2(cfg)
    importance = load_json("model_metrics.json", cfg)["lightgbm"]["feature_importance_gain"]
    top = list(importance)[: cfg["monitoring"]["csi_top_features"]]
    result = monitoring.run_monitoring(policy_frame, builder.transform(policy_frame), top, cfg)
    plots.psi_by_quarter(result["psi"], result["csi"], cfg)
    explain.run_explain(policy_frame, cfg)
    s = result["summary"]
    print(f"  max score PSI in test {s['score_psi_max_test_quarter']:.3f}; "
          f"{s['n_segments_flagged']} of {s['n_segments_checked']} segments flagged")


def phase_9_app_data(cfg, df, scores):
    info = app_data.build_app_data(df, scores, cfg)
    print(f"  app/data: {info['n_book_rows']:,} test loans, {info['app_data_mb']:.1f} MB")


def phase_10_reports(cfg):
    for path in report.run_reports(cfg):
        print(f"  wrote {path.relative_to(path.parents[1]) if path.parent.name == 'reports' else path.name}")


def main():
    cfg = load_config()
    np.random.seed(cfg["seed"])
    start = time.time()

    def run(name, fn, *args):
        t = time.time()
        print(f"\n== {name}")
        out = fn(*args)
        print(f"  ({time.time() - t:.0f}s)")
        return out

    run("Phase 1: load, clean, population", phase_1_load_and_clean, cfg)
    run("Phase 2: vintage analysis", phase_2_vintages, cfg)
    df = run("Phases 3-4: features, leakage checks, splits", phase_3_4_features_and_splits, cfg)
    scores = run("Phase 5: models and benchmark", phase_5_models, cfg, df)
    run("Phase 6: profit decision layer", phase_6_profit, cfg, df, scores)
    run("Phase 7: loss forecasting", phase_7_forecast, cfg, df, scores)
    run("Phase 8: monitoring, weak spots, explainability", phase_8_monitoring, cfg, df, scores)
    run("Phase 9: app data", phase_9_app_data, cfg, df, scores)
    run("Phase 10: reports", phase_10_reports, cfg)

    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Metrics in reports/metrics/, charts in reports/figures/.")


if __name__ == "__main__":
    main()
