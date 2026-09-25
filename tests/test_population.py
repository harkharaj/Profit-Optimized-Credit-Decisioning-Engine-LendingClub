"""The modeling population is exactly what the spec says it is."""
from credit_engine.data import query


def test_only_36_month_individual_loans(cfg, needs_db):
    df = query(cfg, "SELECT DISTINCT term, application_type FROM loans_clean")
    assert df["term"].tolist() == ["36 months"]
    assert df["application_type"].tolist() == ["INDIVIDUAL"]


def test_issue_dates_inside_window(cfg, needs_db):
    df = query(cfg, "SELECT MIN(issue_date) AS lo, MAX(issue_date) AS hi FROM loans_clean")
    assert str(df["lo"][0])[:10] >= str(cfg["population"]["issue_start"])
    assert str(df["hi"][0])[:10] <= str(cfg["population"]["issue_end"])


def test_only_terminal_statuses(cfg, needs_db):
    statuses = set(query(cfg, "SELECT DISTINCT loan_status FROM loans_clean")["loan_status"])
    assert statuses <= {"Fully Paid", "Charged Off", "Default"}


def test_target_mapping(cfg, needs_db):
    df = query(cfg, "SELECT loan_status, target, COUNT(*) AS n FROM loans_clean GROUP BY 1, 2")
    assert set(df["target"]) == {0, 1}
    for _, row in df.iterrows():
        expected = 0 if row["loan_status"] == "Fully Paid" else 1
        assert row["target"] == expected


def test_ids_unique(cfg, needs_db):
    df = query(cfg, "SELECT COUNT(*) AS n, COUNT(DISTINCT id) AS n_ids FROM loans_clean")
    assert df["n"][0] == df["n_ids"][0]
