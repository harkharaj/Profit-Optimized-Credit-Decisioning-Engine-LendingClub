"""Out-of-time splits by issue date.

Train 2010-2012 | Validation 2013 | Test 2014-2015 (dates in config.yaml).

Why out-of-time and not random? A lender builds a model on past loans and
uses it on future applicants. A random split would leak future economic
conditions into training and overstate performance.
"""
import pandas as pd

from .utils import save_json

SPLIT_NAMES = ["train", "valid", "test"]


def assign_split(issue_date: pd.Series, cfg: dict) -> pd.Series:
    """Label each loan 'train', 'valid' or 'test' from its issue date."""
    dates = pd.to_datetime(issue_date)
    split = pd.Series(pd.NA, index=issue_date.index, dtype="object")
    for name in SPLIT_NAMES:
        start, end = (pd.Timestamp(d) for d in cfg["splits"][name])
        split[(dates >= start) & (dates <= end)] = name
    return split


def split_case_sql(cfg: dict, column: str = "issue_date") -> str:
    """The same split rule as a SQL CASE expression, for DuckDB queries."""
    whens = [
        f"WHEN {column} BETWEEN DATE '{cfg['splits'][name][0]}' AND DATE '{cfg['splits'][name][1]}' THEN '{name}'"
        for name in SPLIT_NAMES
    ]
    return "CASE " + " ".join(whens) + " END"


def summarize_splits(df: pd.DataFrame, cfg: dict) -> dict:
    """Size, default rate and date range of each split -> splits.json."""
    summary = {}
    for name in SPLIT_NAMES:
        part = df[df["split"] == name]
        summary[name] = {
            "n_loans": len(part),
            "default_rate": part["target"].mean(),
            "first_issue": part["issue_date"].min(),
            "last_issue": part["issue_date"].max(),
        }
    save_json(summary, "splits.json", cfg)
    return summary
