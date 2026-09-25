"""Small helpers for reading and writing the metrics files.

Every number quoted anywhere in this project is read from reports/metrics/,
so these two functions are the only way results leave the pipeline.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _to_jsonable(obj):
    """Convert numpy / pandas / date objects into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _to_jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return obj.strftime("%Y-%m-%d")
    return obj


def save_json(obj, name: str, cfg: dict) -> Path:
    path = cfg["paths"]["metrics_dir"] / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    return path


def load_json(name: str, cfg: dict):
    with open(cfg["paths"]["metrics_dir"] / name, encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, name: str, cfg: dict) -> Path:
    path = cfg["paths"]["metrics_dir"] / name
    df.to_csv(path, index=False)
    return path
