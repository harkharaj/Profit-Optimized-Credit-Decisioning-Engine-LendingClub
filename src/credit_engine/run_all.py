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

from . import data, models, plots, profit
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

    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Metrics in reports/metrics/, charts in reports/figures/.")


if __name__ == "__main__":
    main()
