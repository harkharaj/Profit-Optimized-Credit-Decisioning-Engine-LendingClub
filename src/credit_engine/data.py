"""Run the SQL pipeline in DuckDB and hand clean tables to Python.

Heavy lifting (reading 2.2M rows, filtering, aggregating) happens in SQL.
Only the filtered modeling population (~0.5M rows) is ever pulled into pandas.
"""
from pathlib import Path
from string import Template

import duckdb
import pandas as pd

from .config import PROJECT_ROOT
from .features import CORE_RAW_COLUMNS, OPTIONAL_BUREAU_COLUMNS
from .splits import split_case_sql
from .utils import save_json

SQL_DIR = PROJECT_ROOT / "sql"
TERMINAL_STATUSES = ("Fully Paid", "Charged Off", "Default")


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def find_raw_file(cfg: dict) -> Path:
    matches = sorted(cfg["paths"]["raw_dir"].glob(cfg["paths"]["raw_glob"]))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{cfg['paths']['raw_glob']}' in {cfg['paths']['raw_dir']}.\n"
            "Download it with: kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip"
        )
    return matches[0]


def connect(cfg: dict, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(cfg["paths"]["duckdb"]), read_only=read_only)
    con.execute("SET enable_progress_bar = false")
    return con


def run_sql_file(con: duckdb.DuckDBPyConnection, filename: str, **params) -> None:
    """Execute a file from sql/, filling its $placeholders from params."""
    sql = Template((SQL_DIR / filename).read_text(encoding="utf-8")).substitute(**params)
    con.execute(sql)


def query(cfg: dict, sql: str) -> pd.DataFrame:
    with connect(cfg, read_only=True) as con:
        return con.execute(sql).df()


def table_exists(cfg: dict, table: str) -> bool:
    if not cfg["paths"]["duckdb"].exists():
        return False
    with connect(cfg, read_only=True) as con:
        return table in set(con.execute("SHOW TABLES").df()["name"])


# ---------------------------------------------------------------------------
# Phase 1: load + clean
# ---------------------------------------------------------------------------

def build_clean_tables(cfg: dict, force_reload: bool = False) -> None:
    """01_load.sql (slow, cached unless force_reload) then 02_clean.sql (always re-run)."""
    pop = cfg["population"]
    with connect(cfg) as con:
        if force_reload or not _has_table(con, "loans_raw"):
            print("  loading raw CSV into DuckDB (takes a few minutes the first time)...")
            run_sql_file(con, "01_load.sql", raw_path=find_raw_file(cfg).as_posix())
        run_sql_file(
            con, "02_clean.sql",
            term=f"{pop['term_months']} months",
            application_type=pop["application_type"],
            issue_start=pop["issue_start"],
            issue_end=pop["issue_end"],
        )


def _has_table(con, table: str) -> bool:
    return table in set(con.execute("SHOW TABLES").df()["name"])


def load_table(cfg: dict, table: str) -> pd.DataFrame:
    df = query(cfg, f"SELECT * FROM {table}")
    for col in df.columns:
        if col.endswith("_date"):
            df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------------------
# Phase 1: data summary -> reports/metrics/data_summary.json
# ---------------------------------------------------------------------------

def population_waterfall(cfg: dict) -> pd.DataFrame:
    """How many loans survive each filter, from raw file to modeling population."""
    pop = cfg["population"]
    filters = [
        ("all rows in file", "TRUE"),
        ("valid id and loan amount", "id IS NOT NULL AND loan_amnt IS NOT NULL"),
        (f"{pop['term_months']}-month term", f"term = '{pop['term_months']} months'"),
        ("individual application", f"application_type = '{pop['application_type']}'"),
        (f"issued {pop['issue_start']} to {pop['issue_end']}",
         f"issue_date BETWEEN DATE '{pop['issue_start']}' AND DATE '{pop['issue_end']}'"),
    ]
    rows, condition = [], "TRUE"
    for step, new_condition in filters:
        condition = f"{condition} AND {new_condition}"
        n = query(cfg, f"SELECT COUNT(*) AS n FROM loans_raw WHERE {condition}")["n"][0]
        rows.append({"step": step, "n_loans": n})
    for step, table in [("de-duplicated on id", "loans_population"),
                        ("final outcome known (modeling population)", "loans_clean")]:
        rows.append({"step": step, "n_loans": query(cfg, f"SELECT COUNT(*) AS n FROM {table}")["n"][0]})
    return pd.DataFrame(rows)


def exclusions_by_quarter(cfg: dict) -> pd.DataFrame:
    """Loans dropped because their outcome isn't final yet, per issue quarter.

    If many recent loans are still 'Current', dropping them would keep only the
    fast defaulters and fast payers, biasing recent vintages. Flag > 2%.
    """
    df = query(cfg, """
        SELECT date_trunc('quarter', issue_date)                                         AS issue_quarter,
               COUNT(*)                                                                  AS n_population,
               COUNT(*) FILTER (WHERE loan_status IN ('Fully Paid','Charged Off','Default')) AS n_kept,
               COUNT(*) FILTER (WHERE loan_status = 'Current'
                                   OR loan_status = 'In Grace Period'
                                   OR loan_status LIKE 'Late%')                          AS n_not_final,
               COUNT(*) FILTER (WHERE loan_status LIKE 'Does not meet%')                 AS n_legacy
        FROM loans_population
        GROUP BY 1 ORDER BY 1
    """)
    df["pct_not_final"] = 100 * df["n_not_final"] / df["n_population"]
    df["flag_over_2pct"] = df["pct_not_final"] > 2.0
    return df


def missingness_table(cfg: dict) -> pd.DataFrame:
    """Share of missing values per candidate feature, overall and in each split window."""
    columns = CORE_RAW_COLUMNS + OPTIONAL_BUREAU_COLUMNS
    null_rates = ",\n".join(f"AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0 END) AS {c}" for c in columns)
    by_split = query(cfg, f"""
        SELECT {split_case_sql(cfg)} AS split, {null_rates}
        FROM loans_clean GROUP BY 1
    """).set_index("split").T
    overall = query(cfg, f"SELECT {null_rates} FROM loans_clean").T[0]
    table = pd.DataFrame({"feature": columns, "missing_overall": overall.values})
    for split in ["train", "valid", "test"]:
        table[f"missing_{split}"] = by_split[split].values
    return table.sort_values("missing_train", ascending=False)


def payment_identity_check(cfg: dict) -> dict:
    """Check total_pymnt = principal + interest + late fees + recoveries."""
    return query(cfg, """
        WITH d AS (
            SELECT target,
                   abs(total_pymnt - (total_rec_prncp + total_rec_int
                                      + total_rec_late_fee + recoveries)) AS gap
            FROM loans_clean
        )
        SELECT COUNT(*)                          AS n_loans,
               AVG(CASE WHEN gap < 1 THEN 1.0 ELSE 0 END) AS share_within_1_usd,
               AVG(gap)                          AS mean_abs_gap_usd,
               quantile_cont(gap, 0.99)          AS p99_abs_gap_usd,
               MAX(gap)                          AS max_abs_gap_usd
        FROM d
    """).iloc[0].to_dict()


def data_summary(cfg: dict) -> dict:
    summary = {
        "population_waterfall": population_waterfall(cfg),
        "status_counts_in_population": query(cfg, """
            SELECT loan_status, COUNT(*) AS n_loans FROM loans_population
            GROUP BY 1 ORDER BY 2 DESC"""),
        "exclusions_by_quarter": exclusions_by_quarter(cfg),
        "rows_and_default_rate_by_year": query(cfg, """
            SELECT year(issue_date) AS issue_year, COUNT(*) AS n_loans,
                   AVG(target) AS default_rate, SUM(funded_amnt) AS funded_usd
            FROM loans_clean GROUP BY 1 ORDER BY 1"""),
        "default_rate_by_year_and_grade": query(cfg, """
            SELECT year(issue_date) AS issue_year, grade, COUNT(*) AS n_loans, AVG(target) AS default_rate
            FROM loans_clean GROUP BY 1, 2 ORDER BY 1, 2"""),
        "missingness": missingness_table(cfg),
        "payment_identity": payment_identity_check(cfg),
    }
    overall = query(cfg, "SELECT COUNT(*) AS n, AVG(target) AS dr FROM loans_clean").iloc[0]
    summary["n_loans"] = int(overall["n"])
    summary["overall_default_rate"] = float(overall["dr"])
    summary["quarters_flagged_over_2pct_not_final"] = [
        q.strftime("%Y-Q") + str((q.month - 1) // 3 + 1)
        for q in summary["exclusions_by_quarter"].query("flag_over_2pct")["issue_quarter"]
    ]
    save_json(summary, "data_summary.json", cfg)
    return summary
