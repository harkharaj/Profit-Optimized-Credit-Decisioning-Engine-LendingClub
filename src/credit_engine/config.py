"""Load config.yaml and turn every path into an absolute path.

Absolute paths let the same code run from the command line, a notebook,
the tests or the Streamlit app without caring about the working directory.
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# directories the pipeline writes into; created on first use
_OUTPUT_DIRS = ["processed_dir", "metrics_dir", "figures_dir", "models_dir", "app_data_dir"]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg["paths"].items():
        if key != "raw_glob":
            cfg["paths"][key] = PROJECT_ROOT / value

    for key in _OUTPUT_DIRS:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)

    return cfg
