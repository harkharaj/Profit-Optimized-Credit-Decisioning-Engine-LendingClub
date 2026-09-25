"""Out-of-time splits: no overlap, and every loan lands in exactly one split."""
import pandas as pd

from credit_engine.data import load_model_base
from credit_engine.splits import SPLIT_NAMES, assign_split


def test_split_windows_do_not_overlap(cfg):
    windows = sorted((pd.Timestamp(cfg["splits"][s][0]), pd.Timestamp(cfg["splits"][s][1])) for s in SPLIT_NAMES)
    for (_, end), (next_start, _) in zip(windows, windows[1:]):
        assert end < next_start


def test_split_boundaries(cfg):
    dates = pd.Series(pd.to_datetime(["2010-01-01", "2012-12-01", "2013-01-01", "2013-12-01",
                                      "2014-01-01", "2015-12-01"]))
    assert assign_split(dates, cfg).tolist() == ["train", "train", "valid", "valid", "test", "test"]


def test_every_loan_in_exactly_one_split(cfg, needs_db):
    df = load_model_base(cfg)
    assert df["split"].notna().all()
    assert set(df["split"]) == set(SPLIT_NAMES)
    # train strictly before valid strictly before test
    last = df.groupby("split")["issue_date"].agg(["min", "max"])
    assert last.loc["train", "max"] < last.loc["valid", "min"]
    assert last.loc["valid", "max"] < last.loc["test", "min"]
