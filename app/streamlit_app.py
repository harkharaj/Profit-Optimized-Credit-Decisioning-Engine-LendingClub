"""Credit policy simulator for the LendingClub decisioning engine.

    streamlit run app/streamlit_app.py

Loads only precomputed files from app/data/ (written by `python -m credit_engine.run_all`).
It never trains anything. The policy math is imported from the pipeline itself, so the
simulator reproduces the numbers in the reports exactly.
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data"
sys.path.insert(0, str(APP_DIR.parent / "src"))
from credit_engine.profit import POLICY_NAMES, approve_top, book_metrics, policy_order, stress_scenario  # noqa: E402

SEED = 42
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b"
STATUS_COLORS = {"stable": "#0ca30c", "monitor": "#fab219", "action": "#d03b3b"}

st.set_page_config(page_title="Credit Decisioning Engine", layout="wide")


# ---------------------------------------------------------------------------
# Data (cached: loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def load_book() -> pd.DataFrame:
    return pd.read_parquet(DATA / "test_book.parquet")


@st.cache_data
def load_explain_sample() -> pd.DataFrame:
    return pd.read_parquet(DATA / "explain_sample.parquet")


@st.cache_data
def scenario_orders(loss_mult: float, cost: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Apply a stress scenario and rank applicants under every policy (cached per scenario)."""
    scenario, realized = stress_scenario(load_book(), loss_mult, cost)
    orders = {pol: policy_order(scenario, pol, SEED) for pol in POLICY_NAMES}
    return scenario, realized, orders


def money(v: float, signed: bool = False) -> str:
    sign = "-" if v < 0 else "+" if signed else ""
    return f"{sign}${abs(v) / 1e6:,.1f}M"


def dollars(v: float, signed: bool = False) -> str:
    sign = "-" if v < 0 else "+" if signed else ""
    return f"{sign}${abs(v):,.0f}"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    summary, policy, models = load_json("data_summary.json"), load_json("policy_results.json"), load_json("model_metrics.json")
    h = policy["headline"]
    st.title("Profit-Optimized Credit Decisioning Engine")
    st.caption("LendingClub 36-month loans, 2010–2015 · SQL + Python · scorecard, LightGBM and a profit decision layer")
    st.markdown(
        "**Business question.** If we ran credit policy, which applicants should we approve to maximize profit, "
        "and how much better is a data-driven policy than LendingClub's own grade-based one?")

    c = st.columns(4)
    c[0].metric("Loans analysed", f"{summary['n_loans']:,}", help="36-month individual loans with a final outcome")
    c[1].metric("Test AUC: M2 LightGBM", f"{models['models']['M2']['test']['auc']:.3f}",
                f"{models['models']['M2']['test']['auc'] - models['models']['B0']['test']['auc']:+.3f} vs LC grade")
    c[2].metric(f"M2 PD policy at {h['approval_rate']:.0%} approval", money(h["m2_pd"]["total_profit"]),
                f"{money(h['m2_pd']['profit_vs_grade'], True)} vs grade ({h['m2_pd']['profit_vs_grade_pct']:+.1%})")
    c[3].metric(f"M2 expected-profit policy at {h['approval_rate']:.0%}", money(h["m2_profit"]["total_profit"]),
                f"{money(h['m2_profit']['profit_vs_grade'], True)} vs grade ({h['m2_profit']['profit_vs_grade_pct']:+.1%})")

    st.subheader("How it works")
    st.markdown(
        "1. **SQL (DuckDB):** clean 2.26M raw loans into a 611,816-loan population with a final outcome; vintage analysis.\n"
        "2. **Models:** WoE logistic scorecard (M1) and monotone LightGBM (M2), benchmarked against LendingClub's sub-grade; "
        "out-of-time test on 2014–15 loans.\n"
        "3. **Profit layer:** expected profit = (1 − PD) × return × amount − PD × loss × amount; policies compared on realized profit.\n"
        "4. **Validation:** bootstrap confidence intervals, loss forecast vs actual, PSI drift, weak-spot mining, SHAP reason codes.")
    st.info("Only loans LendingClub accepted are observed, so **approval rate = share of LendingClub's accepted book "
            "we would keep**. Every policy here tightens LendingClub's book.")


def page_vintages():
    st.header("Vintage explorer")
    df = load_csv("vintage_by_grade.csv")
    df["vintage"] = pd.to_datetime(df["vintage"])
    grades = st.multiselect("Grades", sorted(df["grade"].unique()), default=["A", "B", "C", "D", "E"])
    min_n = st.slider("Hide cells with fewer loans than", 0, 500, 100, step=50)
    view = df[df["grade"].isin(grades) & (df["n_loans"] >= min_n)]
    chart = alt.Chart(view).mark_line(strokeWidth=2).encode(
        x=alt.X("vintage:T", title="Issue quarter"),
        y=alt.Y("default_rate:Q", title="Lifetime default rate", axis=alt.Axis(format="%")),
        color=alt.Color("grade:N", title="Grade", scale=alt.Scale(scheme="blues")),
        tooltip=["grade", alt.Tooltip("vintage:T", format="%Y-Q%q"), alt.Tooltip("default_rate:Q", format=".1%"), "n_loans"])
    st.altair_chart(chart, width="stretch")
    st.caption("Default rates rise with grade and drift up for 2014–15 vintages, even within the same grade.")


def page_benchmark():
    st.header("Model benchmark")
    m = load_json("model_metrics.json")
    rows = []
    for key, name in m["model_names"].items():
        for split in ["valid", "test"]:
            v = m["models"][key][split]
            rows.append({"Model": name, "Split": "Validation 2013" if split == "valid" else "Test 2014–15",
                         "AUC": v["auc"], "Gini": v["gini"], "KS": v["ks"], "Top-decile capture": v["top_decile_capture"],
                         "Brier (calibrated)": v.get("brier_calibrated")})
    st.dataframe(pd.DataFrame(rows).style.format({"AUC": "{:.3f}", "Gini": "{:.3f}", "KS": "{:.3f}",
                                                  "Top-decile capture": "{:.1%}", "Brier (calibrated)": "{:.4f}"},
                                                 na_rep="–"), hide_index=True, width="stretch")
    b = m["bootstrap_test"]
    st.markdown(
        f"**Paired bootstrap on test ({b['M2_minus_B0']['n_resamples']:,} resamples):** "
        f"AUC(M2) − AUC(LC grade) = {b['M2_minus_B0']['auc_diff']:+.4f}, 95% CI "
        f"[{b['M2_minus_B0']['ci_low']:+.4f}, {b['M2_minus_B0']['ci_high']:+.4f}] → statistically tied. "
        f"AUC(M1) − AUC(LC grade) = {b['M1_minus_B0']['auc_diff']:+.4f}, 95% CI "
        f"[{b['M1_minus_B0']['ci_low']:+.4f}, {b['M1_minus_B0']['ci_high']:+.4f}].")
    tabs = st.tabs(["ROC", "KS", "Calibration", "SHAP"])
    for tab, fig in zip(tabs, ["roc.png", "ks.png", "calibration.png", "shap_summary.png"]):
        tab.image(str(DATA / "figures" / fig), width="stretch")


def page_simulator():
    st.header("Policy simulator")
    st.caption("Move the cutoff and the assumptions; results are realized profit on the 445k-loan 2014–15 test book.")
    c = st.columns(4)
    rate = c[0].slider("Approval rate", 30, 100, 70, step=1, format="%d%%") / 100
    labels = {k: v for k, v in POLICY_NAMES.items() if k != "grade"}
    policy = c[1].selectbox("Policy", list(labels), index=list(labels).index("m2_pd"), format_func=labels.get)
    loss_mult = c[2].slider("Loss-rate multiplier", 0.6, 1.6, 1.0, step=0.1,
                            help="Stress: losses on defaulted loans are this many times the historical level")
    cost = c[3].slider("Fixed cost per loan ($)", 0, 200, 0, step=10)

    scenario, realized, orders = scenario_orders(loss_mult, float(cost))
    target, funded = scenario["target"].to_numpy(), scenario["funded_amnt"].to_numpy()
    ours = book_metrics(approve_top(orders[policy], rate), realized, target, funded)
    grade = book_metrics(approve_top(orders["grade"], rate), realized, target, funded)

    k = st.columns(5)
    k[0].metric("Approved loans", f"{ours['n_approved']:,}", f"{ours['approval_rate']:.0%} of book", delta_color="off")
    k[1].metric("Bad rate", f"{ours['bad_rate']:.2%}", f"{100 * (ours['bad_rate'] - grade['bad_rate']):+.2f} pp vs grade",
                delta_color="inverse")
    k[2].metric("Total profit", money(ours["total_profit"]), f"{money(ours['total_profit'] - grade['total_profit'], True)} vs grade")
    k[3].metric("Profit per approved loan", dollars(ours["profit_per_approved"]),
                f"{dollars(ours['profit_per_approved'] - grade['profit_per_approved'], True)} vs grade")
    k[4].metric("Dollars lent", money(ours["funded_usd"]), f"{ours['funded_usd'] / grade['funded_usd'] - 1:+.1%} vs grade",
                delta_color="off")

    # profit curve under this scenario: cumulative profit along each ranking
    grid = np.arange(0.30, 1.0001, 0.02)
    curve = []
    for key, name in [(policy, labels[policy]), ("grade", POLICY_NAMES["grade"])]:
        cum = np.cumsum(realized[orders[key]])
        for r in grid:
            curve.append({"Approval rate": r, "Total profit ($M)": cum[int(round(r * len(cum))) - 1] / 1e6, "Policy": name})
    chart = alt.Chart(pd.DataFrame(curve)).mark_line(strokeWidth=2.2).encode(
        x=alt.X("Approval rate:Q", axis=alt.Axis(format="%")),
        y=alt.Y("Total profit ($M):Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Policy:N", scale=alt.Scale(domain=[labels[policy], POLICY_NAMES["grade"]], range=[ORANGE, GRAY])),
        tooltip=["Policy", alt.Tooltip("Approval rate:Q", format=".0%"), alt.Tooltip("Total profit ($M):Q", format=",.1f")])
    rule = alt.Chart(pd.DataFrame({"x": [rate]})).mark_rule(color=INK, strokeWidth=1).encode(x="x:Q")
    st.altair_chart(chart + rule, width="stretch")
    st.caption("Approval rate = share of LendingClub's accepted book kept. Losses and costs apply to both the "
               "expected-profit ranking and the realized outcome.")


def page_monitoring():
    st.header("Drift & weak spots")
    psi, csi, spots = load_csv("psi.csv"), load_csv("csi.csv"), load_csv("weak_spots.csv")
    summary = load_json("monitoring_summary.json")
    left, right = st.columns([1.1, 1])
    chart = alt.Chart(psi).mark_bar().encode(
        x=alt.X("period:N", sort=None, title=None), y=alt.Y("psi:Q", title="PSI of M2 PD vs train"),
        color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values()))),
        tooltip=["period", alt.Tooltip("psi:Q", format=".3f"), "status"])
    rules = alt.Chart(pd.DataFrame({"y": [0.10, 0.25]})).mark_rule(color=GRAY).encode(y="y:Q")
    left.subheader("Score drift (PSI)")
    left.altair_chart(chart + rules, width="stretch")
    worst = (csi[csi["period"] != "valid 2013"].groupby("variable")["psi"].max().sort_values(ascending=False)
             .rename("worst CSI 2014–15").reset_index())
    right.subheader("Feature drift (CSI)")
    right.dataframe(worst.style.format({"worst CSI 2014–15": "{:.3f}"}), hide_index=True, width="stretch")

    st.subheader(f"Weak spots: {summary['n_segments_flagged']} of {summary['n_segments_checked']} segments flagged")
    st.markdown(
        f"None were flagged on 2013 data, but **{summary['n_flagged_same_direction_in_2013']} of the "
        f"{summary['n_flagged_with_2013_data']}** flagged segments were already under-predicted in the same direction "
        "in 2013, below the flag threshold. A trend alert would have caught them earlier.")
    flagged = spots[spots["flagged"]][["segment", "level", "n_loans", "actual_default_rate", "predicted_default_rate",
                                       "gap_pp", "segment_auc", "est_dollar_impact", "valid_gap_pp"]]
    st.dataframe(flagged.style.format({"n_loans": "{:,}", "actual_default_rate": "{:.1%}", "predicted_default_rate": "{:.1%}",
                                       "gap_pp": "{:+.1f}", "segment_auc": "{:.3f}", "est_dollar_impact": "${:,.0f}",
                                       "valid_gap_pp": "{:+.1f}"}, na_rep="–"),
                 hide_index=True, width="stretch")


def page_explain():
    st.header("Explain a decision")
    sample = load_explain_sample()
    decision = st.radio("Show", ["decline", "approve"], horizontal=True,
                        format_func=lambda d: "Declined applicants" if d == "decline" else "Approved applicants")
    pool = sample[sample["decision"] == decision].head(500)
    loan_id = st.selectbox("Loan id", pool["id"].tolist())
    row = pool[pool["id"] == loan_id].iloc[0]

    c = st.columns(4)
    c[0].metric("Decision (M2 PD policy)", row["decision"].title())
    c[1].metric("Predicted PD", f"{row['pd_m2']:.1%}")
    c[2].metric("LC sub-grade / rate", f"{row['sub_grade']} · {row['int_rate']:.2f}%")
    c[3].metric("Actual outcome", "Defaulted" if row["target"] == 1 else "Repaid")

    left, right = st.columns(2)
    left.subheader("Factors that raised predicted risk")
    for k in range(1, 5):
        if row[f"risk_factor_{k}"]:
            left.markdown(f"{k}. {row[f'risk_factor_{k}']}")
    right.subheader("Factors that lowered it")
    for k in range(1, 3):
        if row[f"helping_factor_{k}"]:
            right.markdown(f"- {row[f'helping_factor_{k}']}")
    st.caption("Reason codes = the features with the largest SHAP contributions toward default, "
               "the way US adverse-action notices list the main reasons for a decline.")


PAGES = {"Overview": page_overview, "Vintage explorer": page_vintages, "Model benchmark": page_benchmark,
         "Policy simulator": page_simulator, "Drift & weak spots": page_monitoring, "Explain a decision": page_explain}
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
