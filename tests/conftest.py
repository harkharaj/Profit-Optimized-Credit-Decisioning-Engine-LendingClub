import pytest

from credit_engine.config import load_config
from credit_engine.data import table_exists


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def needs_db(cfg):
    """Skip data-dependent tests until the pipeline has built the DuckDB tables."""
    if not table_exists(cfg, "loans_clean"):
        pytest.skip("DuckDB tables not built yet: run `python -m credit_engine.run_all` first")
