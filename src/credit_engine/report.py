"""Phase 10: write every report from reports/metrics/ - no number is typed by hand.

    reports/results.md          tables only
    README.md                   project overview with headline results
    reports/business_memo.md    one-page memo to the Head of Credit Policy
    reports/resume_bullets.md   two resume bullets + a 60-second pitch

Each document is a template filled from the metrics files, so re-running the
pipeline updates every number everywhere.
"""
import pandas as pd

from .config import PROJECT_ROOT
from .utils import load_json


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def pp(x: float, d: int = 1) -> str:
    return f"{x:+.{d}f} pp"


def usd_m(x: float, signed: bool = False) -> str:
    sign = "-" if x < 0 else "+" if signed else ""
    return f"{sign}${abs(x) / 1e6:,.1f}M"


def usd_b(x: float) -> str:
    return f"${x / 1e9:,.2f}B"


def usd(x: float, signed: bool = False) -> str:
    sign = "-" if x < 0 else "+" if signed else ""
    return f"{sign}${abs(x):,.0f}"


def num(x: float) -> str:
    return f"{x:,.0f}"


def md_table(df: pd.DataFrame, formats: dict | None = None) -> str:
    """DataFrame -> Markdown table; `formats` maps column -> formatting function."""
    formats = formats or {}
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(formats.get(c, str)(v) if pd.notna(v) else "–" for c, v in row.items()) + " |"
            for _, row in df.iterrows()]
    return "\n".join([header, rule, *rows])


# ---------------------------------------------------------------------------
# Load every metrics file once
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self, cfg: dict):
        d = cfg["paths"]["metrics_dir"]
        self.data = load_json("data_summary.json", cfg)
        self.splits = load_json("splits.json", cfg)
        self.vintage = load_json("vintage_summary.json", cfg)
        self.models = load_json("model_metrics.json", cfg)
        self.policy = load_json("policy_results.json", cfg)
        self.forecast = load_json("forecast.json", cfg)
        self.monitor = load_json("monitoring_summary.json", cfg)
        self.univariate = pd.read_csv(d / "univariate_auc.csv")
        self.weak_spots = pd.read_csv(d / "weak_spots.csv")
        self.reasons = pd.read_csv(d / "reason_codes_examples.csv")
        self.shap = pd.read_csv(d / "shap_importance.csv")
        self.cfg = cfg

        # frequently used numbers
        self.h = self.policy["headline"]
        self.rate = self.h["approval_rate"]
        self.test = {k: v["test"] for k, v in self.models["models"].items()}
        self.valid = {k: v["valid"] for k, v in self.models["models"].items()}
        self.boot_pd = self.policy["bootstrap_test_m2_pd_minus_grade"]
        self.boot_profit = self.policy["bootstrap_test_m2_profit_minus_grade"]
        self.boot_auc = self.models["bootstrap_test"]
        sens = pd.DataFrame(self.policy["sensitivity"])
        rate_tag = f"{int(self.rate * 100)}pct"
        self.sens_pd_gain = sens[f"profit_m2_pd_at_{rate_tag}"] - sens[f"profit_grade_at_{rate_tag}"]
        self.sens_profit_gain = sens["m2_profit_minus_grade"]
        self.n_scenarios = len(sens)
        self.fc = self.forecast["whole_book"]["summary"]
        self.fc_approved = self.forecast["approved_book"]["summary"]
        self.swaps = pd.DataFrame(self.policy["swap_set"])
        self.pm = self.policy["test_profit_max"]
        self.raw_rows = self.data["population_waterfall"][0]["n_loans"]
        self.top_feature = self.univariate[self.univariate["role"] == "model feature"].iloc[0]

    def swap(self, comparison_prefix: str, group_prefix: str) -> dict:
        s = self.swaps
        return s[s["comparison"].str.startswith(comparison_prefix) & s["group"].str.startswith(group_prefix)].iloc[0]


# ---------------------------------------------------------------------------
# reports/results.md  (tables only)
# ---------------------------------------------------------------------------

def build_results(m: Metrics) -> str:
    waterfall = pd.DataFrame(m.data["population_waterfall"])
    splits = pd.DataFrame([{"split": k, **v} for k, v in m.splits.items()])

    model_rows = []
    for key, name in m.models["model_names"].items():
        for split in ["valid", "test"]:
            v = m.models["models"][key][split]
            model_rows.append({"model": name, "split": split, "AUC": v["auc"], "Gini": v["gini"], "KS": v["ks"],
                               "top-decile capture": v["top_decile_capture"],
                               "Brier (calibrated)": v.get("brier_calibrated"),
                               "mean predicted PD": v.get("mean_predicted_pd"), "actual default rate": v["default_rate"]})
    f3 = lambda x: f"{x:.3f}"
    f4 = lambda x: f"{x:.4f}"

    boot = pd.DataFrame([{"comparison": k.replace("_minus_", " − "), **v} for k, v in m.boot_auc.items()])
    policy = pd.DataFrame(m.policy["test_policy_table"])
    policy = policy[["policy", "target_rate", "bad_rate", "total_profit", "profit_per_approved",
                     "funded_usd", "return_on_funded"]]
    profit_boot = pd.DataFrame([{"comparison": "M2 PD − grade", **m.boot_pd},
                                {"comparison": "M2 expected profit − grade", **m.boot_profit}])
    swaps = m.swaps[["comparison", "group", "n_loans", "bad_rate", "avg_int_rate", "profit_per_loan",
                     "avg_loan_amnt", "return_on_funded"]]
    sens = pd.DataFrame(m.policy["sensitivity"])
    sens = sens.assign(m2_pd_minus_grade=m.sens_pd_gain)[
        ["loss_rate_multiplier", "fixed_cost_per_loan", "best_policy_at_rate", "m2_profit_minus_grade",
         "m2_pd_minus_grade", "profit_max_approval_rate"]]
    forecast = pd.DataFrame([{"book": book.replace("_", " "), "forecaster": name, **s}
                             for book in ["whole_book", "approved_book"]
                             for name, s in m.forecast[book]["summary"].items()])
    psi = pd.read_csv(m.cfg["paths"]["metrics_dir"] / "psi.csv")
    csi = pd.Series(m.monitor["csi_max_over_test_quarters"]).rename("worst CSI 2014–15").reset_index()
    csi.columns = ["feature", "worst CSI 2014–15"]
    spots = m.weak_spots[m.weak_spots["flagged"]][["segment", "level", "n_loans", "actual_default_rate",
                                                   "predicted_default_rate", "gap_pp", "est_dollar_impact",
                                                   "valid_gap_pp", "same_direction_in_2013"]]
    reasons = m.reasons[["id", "sub_grade", "loan_amnt", "pd_m2", "target",
                         "risk_factor_1", "risk_factor_2", "risk_factor_3", "risk_factor_4"]]

    return f"""# Results

_Auto-generated by `src/credit_engine/report.py` from `reports/metrics/`. Tables only._

## 1. Population

{md_table(waterfall, {"n_loans": num})}

Overall default rate: {pct(m.data["overall_default_rate"])}.

## 2. Out-of-time splits

{md_table(splits, {"n_loans": num, "default_rate": pct})}

## 3. Model benchmark

{md_table(pd.DataFrame(model_rows), {"AUC": f3, "Gini": f3, "KS": f3, "top-decile capture": pct,
                                     "Brier (calibrated)": f4, "mean predicted PD": pct, "actual default rate": pct})}

Paired bootstrap on test ({m.boot_auc["M2_minus_B0"]["n_resamples"]:,} resamples):

{md_table(boot[["comparison", "auc_diff", "ci_low", "ci_high"]], {"auc_diff": f4, "ci_low": f4, "ci_high": f4})}

Monotone-constraint experiment (validation AUC): unconstrained {m.models["monotone_experiment"]["valid_auc_unconstrained"]:.4f},
monotone {m.models["monotone_experiment"]["valid_auc_monotone"]:.4f} → production M2 = {m.models["monotone_experiment"]["production_m2"]}.

## 4. Policy comparison (test 2014–15)

{md_table(policy, {"target_rate": pct, "bad_rate": pct, "total_profit": usd_m, "profit_per_approved": usd,
                   "funded_usd": usd_b, "return_on_funded": lambda x: pct(x, 2)})}

Profit difference at {pct(m.rate, 0)} approval, paired bootstrap:

{md_table(profit_boot[["comparison", "profit_diff", "ci_low", "ci_high"]],
          {"profit_diff": usd_m, "ci_low": usd_m, "ci_high": usd_m})}

Profit-max rule (approve if E[profit] > 0): approval rate {pct(m.pm["profit_max_policy"]["approval_rate"])},
profit {usd_m(m.pm["profit_max_policy"]["total_profit"])} vs grade policy at the same rate
{usd_m(m.pm["grade_policy_same_approval_rate"]["total_profit"])}.

### Swap sets at {pct(m.rate, 0)} approval

{md_table(swaps, {"n_loans": num, "bad_rate": pct, "avg_int_rate": lambda x: f"{x:.1f}%", "profit_per_loan": usd,
                  "avg_loan_amnt": usd, "return_on_funded": lambda x: pct(x, 2)})}

### Sensitivity (profit differences at {pct(m.rate, 0)} approval)

{md_table(sens, {"loss_rate_multiplier": lambda x: f"{x:.1f}×", "fixed_cost_per_loan": usd,
                 "m2_profit_minus_grade": lambda x: usd_m(x, True), "m2_pd_minus_grade": lambda x: usd_m(x, True),
                 "profit_max_approval_rate": pct})}

## 5. Loss forecast vs actual (test quarters)

{md_table(forecast[["book", "forecaster", "defaults_mape_pct", "loss_mape_pct", "defaults_mean_error_pct",
                    "total_forecast_loss_usd", "total_actual_loss_usd"]],
          {"defaults_mape_pct": lambda x: f"{x:.1f}%", "loss_mape_pct": lambda x: f"{x:.1f}%",
           "defaults_mean_error_pct": lambda x: f"{x:+.1f}%", "total_forecast_loss_usd": usd_m,
           "total_actual_loss_usd": usd_m})}

## 6. Monitoring

### Score PSI (M2 calibrated PD vs train)

{md_table(psi[["period", "n_loans", "psi", "status"]], {"n_loans": num, "psi": f3})}

### Feature CSI (worst quarter)

{md_table(csi, {"worst CSI 2014–15": f3})}

### Flagged weak spots (test), ranked by estimated dollar impact

{md_table(spots, {"n_loans": num, "actual_default_rate": pct, "predicted_default_rate": pct, "gap_pp": pp,
                  "est_dollar_impact": lambda x: usd_m(x, True), "valid_gap_pp": pp})}

## 7. Reason-code examples (declined under the recommended policy)

{md_table(reasons, {"loan_amnt": usd, "pd_m2": pct})}
"""


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------

def build_readme(m: Metrics) -> str:
    t, h = m.test, m.h
    wf = {row["step"]: row["n_loans"] for row in m.data["population_waterfall"]}
    swap_in, swap_out = m.swap("M2 LightGBM PD", "swap-in"), m.swap("M2 LightGBM PD", "swap-out")
    return f"""<!-- Generated by src/credit_engine/report.py from reports/metrics/. Edit the template there, not this file. -->

# Profit-Optimized Credit Decisioning Engine — LendingClub

An end-to-end credit decisioning system built on {num(m.raw_rows)} historical LendingClub loans:
**SQL vintage analysis → default-probability models → a profit-maximizing approve/decline policy → bank-style validation and monitoring → an interactive policy simulator.**

## Business question

> If we were the credit policy team, which applicants should we approve to maximize portfolio profit,
> and how much better is a data-driven policy than LendingClub's existing grade-based one?

## Headline results (out-of-time test: {num(m.splits["test"]["n_loans"])} loans issued 2014–2015)

| | Result |
|---|---|
| **Profit, same volume and capital** | Ranking applicants by the LightGBM PD instead of LendingClub's grade earns **{usd_m(h["m2_pd"]["profit_vs_grade"], True)} ({pct(h["m2_pd"]["profit_vs_grade_pct"])})** at {pct(m.rate, 0)} approval, 95% CI [{usd_m(m.boot_pd["ci_low"])}, {usd_m(m.boot_pd["ci_high"])}], with the same bad rate ({pct(h["m2_pd"]["bad_rate"])} vs {pct(h["grade"]["bad_rate"])}) and the same dollars lent ({pct(h["m2_pd"]["funded_vs_grade_pct"])}) |
| **Profit, expected-profit ranking** | Ranking by expected profit earns **{usd_m(h["m2_profit"]["profit_vs_grade"], True)} ({pct(h["m2_profit"]["profit_vs_grade_pct"])})**, 95% CI [{usd_m(m.boot_profit["ci_low"])}, {usd_m(m.boot_profit["ci_high"])}], but lends {pct(h["m2_profit"]["funded_vs_grade_pct"])} more dollars |
| **Robustness** | Both model policies beat the grade policy in all {m.n_scenarios} loss-severity × cost scenarios (M2 PD: {usd_m(m.sens_pd_gain.min(), True)} to {usd_m(m.sens_pd_gain.max(), True)}) |
| **Model benchmark** | LightGBM test AUC **{t["M2"]["auc"]:.3f}** vs LendingClub sub-grade {t["B0"]["auc"]:.3f}: statistically tied (Δ {m.boot_auc["M2_minus_B0"]["auc_diff"]:+.4f}, 95% CI [{m.boot_auc["M2_minus_B0"]["ci_low"]:+.4f}, {m.boot_auc["M2_minus_B0"]["ci_high"]:+.4f}]), with the best calibration (Brier {t["M2"]["brier_calibrated"]:.4f}) and top-decile capture ({pct(t["M2"]["top_decile_capture"])}) |
| **Loss forecasting** | M2 forecasts quarterly net losses with **{m.fc["M2"]["loss_mape_pct"]:.1f}% MAPE** (grade benchmark {m.fc["Benchmark"]["loss_mape_pct"]:.1f}%) |
| **Monitoring** | Score PSI enters "monitor" in late 2014 (max {m.monitor["score_psi_max_test_quarter"]:.3f}); {m.monitor["n_segments_flagged"]} of {m.monitor["n_segments_checked"]} segments flagged as weak spots, and {m.monitor["n_flagged_same_direction_in_2013"]} of {m.monitor["n_flagged_with_2013_data"]} were already drifting the same way in 2013 |

**Why the model makes money even though AUC is tied:** where the M2 PD policy and the grade policy disagree,
both sets of loans default at the same rate ({pct(swap_in["bad_rate"])} vs {pct(swap_out["bad_rate"])}), but the loans M2 picks
pay {swap_in["avg_int_rate"]:.1f}% interest instead of {swap_out["avg_int_rate"]:.1f}%, earning {usd(swap_in["profit_per_loan"])} vs {usd(swap_out["profit_per_loan"])} per loan.
M2 finds loans that LendingClub priced as riskier than they turned out to be.

![Profit by approval rate](reports/figures/profit_curve.png)

## Data and population

- Source: Kaggle [`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club), accepted loans 2007–2018.
- Population: 36-month, individual applications issued {m.cfg["population"]["issue_start"]} to {m.cfg["population"]["issue_end"]} with a final outcome → **{num(m.data["n_loans"])} loans**, default rate {pct(m.data["overall_default_rate"])}. Every loan in the window had time to mature before the data cutoff.
- Target: 1 = Charged Off / Default, 0 = Fully Paid.
- **Framing:** only loans LendingClub accepted are observed, so *approval rate* means the share of LendingClub's accepted book we would keep. Every policy here is a **tightening** policy.

| Split | Issued | Loans | Default rate |
|---|---|---|---|
| Train | 2010–2012 | {num(m.splits["train"]["n_loans"])} | {pct(m.splits["train"]["default_rate"])} |
| Validation | 2013 | {num(m.splits["valid"]["n_loans"])} | {pct(m.splits["valid"]["default_rate"])} |
| Test (touched once) | 2014–2015 | {num(m.splits["test"]["n_loans"])} | {pct(m.splits["test"]["default_rate"])} |

## Approach

```mermaid
flowchart LR
    A["Raw CSV<br/>{num(m.raw_rows)} loans"] -->|DuckDB SQL| B["Population<br/>{num(m.data["n_loans"])} loans"]
    B --> V["Vintage analysis<br/>(window functions)"]
    B --> F["Allowlisted features<br/>+ leakage guard"]
    F --> S["Out-of-time split<br/>train / valid / test"]
    S --> M1["M1 WoE scorecard"]
    S --> M2["M2 monotone LightGBM"]
    M1 --> C["Calibration<br/>(fit on 2013)"]
    M2 --> C
    C --> P["Profit layer<br/>E[profit] per applicant"]
    P --> R["Policies vs grade<br/>swap sets, bootstrap CIs"]
    C --> Q["Loss forecast · PSI/CSI<br/>weak spots · SHAP"]
    R --> APP["Streamlit simulator<br/>+ reports"]
    Q --> APP
```

1. **SQL (DuckDB)** loads the 2.2M-row gzipped CSV without pandas, cleans it, defines the population and runs vintage analysis (default rate by quarter × grade, risk-based pricing check, cumulative default curves with window functions).
2. **Features** use an allowlist of application-time fields only. LendingClub's grade, sub-grade, interest rate and installment are *never* features (they are LC's own model), and `zip_code`/`addr_state` are excluded for fair lending (ECOA / Reg B). A code guard and a test enforce it; the strongest single feature (`{m.top_feature["feature"]}`) has univariate AUC {m.top_feature["auc"]:.3f}, far below the 0.80 leakage tripwire.
3. **Models:** B0 = LendingClub sub-grade (benchmark); M1 = WoE logistic scorecard ({m.models["scorecard"]["n_selected_features"]} features, 600 points at 50:1 odds, PDO 20); M2 = LightGBM tuned with Optuna. Monotone constraints (FICO −, DTI +, inquiries +, utilization +) cost nothing on validation, so production M2 is monotone. PDs are calibrated on 2013.
4. **Profit layer:** `E[profit] = (1 − PD)·r_good[sub-grade]·amount − PD·L[grade]·amount − cost`, with returns and loss rates estimated on 2010–12 only; realized profit = payments × (1 − 1% servicing fee) − funded amount.
5. **Validation:** paired bootstrap CIs, loss forecast vs actual by quarter, PSI/CSI drift, segment weak-spot mining with an early-warning check, SHAP reason codes.

## Model benchmark (test 2014–15)

| Model | AUC | Gini | KS | Top-decile capture | Brier |
|---|---|---|---|---|---|
| LendingClub sub-grade (existing) | {t["B0"]["auc"]:.3f} | {t["B0"]["gini"]:.3f} | {t["B0"]["ks"]:.3f} | {pct(t["B0"]["top_decile_capture"])} | {t["B0"]["brier_calibrated"]:.4f} |
| M1 WoE scorecard | {t["M1"]["auc"]:.3f} | {t["M1"]["gini"]:.3f} | {t["M1"]["ks"]:.3f} | {pct(t["M1"]["top_decile_capture"])} | {t["M1"]["brier_calibrated"]:.4f} |
| M2 LightGBM (monotone) | {t["M2"]["auc"]:.3f} | {t["M2"]["gini"]:.3f} | {t["M2"]["ks"]:.3f} | {pct(t["M2"]["top_decile_capture"])} | {t["M2"]["brier_calibrated"]:.4f} |

On 2013 validation both models beat the grade (M2 {m.valid["M2"]["auc"]:.3f}, M1 {m.valid["M1"]["auc"]:.3f} vs {m.valid["B0"]["auc"]:.3f}); by 2014–15 LendingClub's grade had caught up.

## Key charts

| | |
|---|---|
| ![Vintages](reports/figures/vintage_default_by_grade.png) | ![ROC](reports/figures/roc.png) |
| ![Swap sets](reports/figures/swap_set.png) | ![Forecast](reports/figures/forecast_vs_actual.png) |
| ![Drift](reports/figures/psi_by_quarter.png) | ![SHAP](reports/figures/shap_summary.png) |

## How to run

```bash
pip install -r requirements.txt
pip install -e .
# data: kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
#   (or place accepted_2007_to_2018Q4.csv.gz in data/raw/ manually)
python -m credit_engine.run_all        # raw file -> every metric, chart and report (~10 min)
python -m pytest                       # tests
streamlit run app/streamlit_app.py     # policy simulator
```

`make setup`, `make all`, `make test` and `make app` wrap the same commands. The notebook
[`notebooks/credit_decisioning_walkthrough.ipynb`](notebooks/credit_decisioning_walkthrough.ipynb) walks through every result (`pip install jupyter` to run it).

## Interactive app

`streamlit run app/streamlit_app.py`: overview, vintage explorer, model benchmark, **policy simulator** (approval-rate slider, policy, loss-rate multiplier and cost per loan, with deltas vs the grade policy), drift and weak spots, and "explain a decision" with reason codes.
The app only reads `app/data/` (precomputed, {num(m.splits["test"]["n_loans"])} test loans), so it deploys directly to Streamlit Community Cloud (entry point `app/streamlit_app.py`).

## Limitations

- **Accepted loans only.** No reject inference is possible (rejected applications have no outcomes), so policies can only tighten LendingClub's book.
- **Simplified profit.** No time value of money or funding cost; the 1% servicing fee and cost per loan are assumptions covered by the sensitivity analysis. With a funding cost, low-rate loans look less attractive and the profit-max rule (which approves {pct(m.pm["profit_max_policy"]["approval_rate"])} of the book) would decline more.
- **Different economics.** LendingClub is a peer-to-peer installment lender, not a bank card issuer: the method transfers, the exact numbers do not.
- **One benign credit period** (2010–2015), with no recession in the test window.
- **Coarse loss rates**, estimated by grade only.
- **Calibration drift:** PDs calibrated on 2013 under-predict 2015 defaults ({pct(t["M2"]["mean_predicted_pd"])} predicted vs {pct(t["M2"]["default_rate"])} actual across the test period); the monitoring section shows how it would have been caught.
- **The expected-profit ranking** earns the most but lends more capital into riskier, larger loans; most of its gain comes from the decision rule rather than the model (the same rule with the grade-based PD earns {usd_m(h["b0_profit"]["profit_vs_grade"], True)}).

## Repository map

```
config.yaml                 every date, threshold and parameter
sql/                        01_load, 02_clean, 03_vintage, 04_features (DuckDB)
src/credit_engine/          data, features, splits, scorecard, models, evaluation, profit,
                            forecasting, monitoring, explain, plots, report, run_all
app/streamlit_app.py        policy simulator (reads app/data/ only)
notebooks/                  walkthrough notebook
reports/metrics/            every number (single source of truth)
reports/figures/            every chart
reports/results.md          all result tables (generated)
reports/business_memo.md    one-page recommendation (generated)
tests/                      leakage, population, splits, profit, policy, PSI, metrics
PROGRESS.md                 decision log
```
"""


# ---------------------------------------------------------------------------
# reports/business_memo.md
# ---------------------------------------------------------------------------

def build_memo(m: Metrics) -> str:
    h, rate = m.h, pct(m.rate, 0)
    swap_in, swap_out = m.swap("M2 LightGBM PD", "swap-in"), m.swap("M2 LightGBM PD", "swap-out")
    top_spot = m.weak_spots[m.weak_spots["flagged"]].iloc[0]
    csi = pd.Series(m.monitor["csi_max_over_test_quarters"])
    structural = " and ".join(f"`{f}`" for f in csi[csi > 1.0].index)          # fields LC started reporting in 2012
    shifted = " and ".join(f"`{f}`" for f in csi[(csi > 0.25) & (csi <= 1.0)].index)
    return f"""# Memo: replace grade-based approvals with a model-based, profit-aware policy

**To:** Head of Credit Policy  **From:** Credit Decisioning Analytics  **Re:** Approval policy for 36-month personal loans

_All figures are realized results on {num(m.splits["test"]["n_loans"])} loans issued 2014–2015 that no model or rule was tuned on._

## Recommendation

**Rank applicants by the new LightGBM default model (M2) instead of by LendingClub grade.**
If volume must be held at {rate} of today's approvals, this earns **{usd_m(h["m2_pd"]["total_profit"])} vs {usd_m(h["grade"]["total_profit"])}**,
an extra **{usd_m(h["m2_pd"]["profit_vs_grade"], True)} ({pct(h["m2_pd"]["profit_vs_grade_pct"])})**, 95% CI {usd_m(m.boot_pd["ci_low"])}–{usd_m(m.boot_pd["ci_high"])},
with the **same number of loans, the same bad rate ({pct(h["m2_pd"]["bad_rate"])}) and the same dollars lent** ({pct(h["m2_pd"]["funded_vs_grade_pct"])}).

## Why it works

Where the two policies disagree, the loans M2 approves and the grade policy declines default at the same rate
({pct(swap_in["bad_rate"])} vs {pct(swap_out["bad_rate"])}) but pay {swap_in["avg_int_rate"]:.1f}% instead of {swap_out["avg_int_rate"]:.1f}%.
M2 identifies borrowers whose grade overstates their risk, and it drops grade A/B loans that behave like riskier grades.

## Options considered

| Option at {rate} approval | Profit | vs grade | Bad rate | Dollars lent |
|---|---|---|---|---|
| Grade policy (today) | {usd_m(h["grade"]["total_profit"])} | – | {pct(h["grade"]["bad_rate"])} | {usd_b(h["grade"]["funded_usd"])} |
| **M2 PD ranking (recommended)** | {usd_m(h["m2_pd"]["total_profit"])} | {usd_m(h["m2_pd"]["profit_vs_grade"], True)} | {pct(h["m2_pd"]["bad_rate"])} | {usd_b(h["m2_pd"]["funded_usd"])} |
| M2 expected-profit ranking | {usd_m(h["m2_profit"]["total_profit"])} | {usd_m(h["m2_profit"]["profit_vs_grade"], True)} | {pct(h["m2_profit"]["bad_rate"])} | {usd_b(h["m2_profit"]["funded_usd"])} |

The expected-profit ranking earns more, but it lends {pct(h["m2_profit"]["funded_vs_grade_pct"])} more capital into larger, riskier loans,
its advantage shrinks from {usd_m(m.sens_profit_gain.max(), True)} to {usd_m(m.sens_profit_gain.min(), True)} as loss severity rises, and it declines small loans for size rather than risk,
which is harder to explain to customers. Pilot it only where capital is available, with a loss-severity limit.
Without a volume constraint, {pct(m.pm["profit_max_policy"]["approval_rate"])} of today's approvals already have positive expected profit, so there is no case for broad tightening.

## Risks

- **Selection bias:** only approved loans have outcomes, so the model cannot judge applicants we decline today. This recommendation only re-orders the current book.
- **Drift:** PDs calibrated on 2013 under-predicted 2015 defaults ({pct(m.test["M2"]["mean_predicted_pd"])} vs {pct(m.test["M2"]["default_rate"])}). Score PSI reached {m.monitor["score_psi_max_test_quarter"]:.2f} in late 2014. {structural} changed distribution completely once LendingClub began reporting them in 2012{f", and {shifted} also crossed the action level" if shifted else ""}. Recalibrate at least annually.
- **Weak spots:** {m.monitor["n_segments_flagged"]} of {m.monitor["n_segments_checked"]} segments are under-predicted by ≥ 2 pp; the largest is {top_spot["segment"].replace("_", " ")} = {top_spot["level"]} ({pp(top_spot["gap_pp"])}, ≈{usd_m(top_spot["est_dollar_impact"])} of unexpected loss).
- **Fair lending:** geography and free-text fields are excluded, every decline gets SHAP-based reason codes, and constraints keep the model's direction intuitive (a higher FICO never raises predicted risk). A disparate-impact review is still required before launch.
- **Loss severity:** loss rates are estimated by grade from 2010–12; the policy ranking held in all {m.n_scenarios} stress scenarios (losses ±20%, $0–100 cost per loan).

## Monitoring plan

| What | How often | Trigger |
|---|---|---|
| Score PSI (M2 PD vs development sample) | Monthly | ≥ 0.10 investigate, > 0.25 recalibrate |
| Feature CSI (top 15 features) | Monthly | > 0.25 root-cause (data or population change) |
| Segment calibration (purpose, FICO, DTI, income, tenure …) | Quarterly | gap ≥ 2 pp or actual/predicted outside 0.8–1.25; also alert on the same-direction gap two quarters running |
| Forecast vs actual losses by vintage | Quarterly | MAPE > 10% or bias > 5% for two quarters |
| Champion vs grade profit (holdout of approvals) | Quarterly | gain CI includes 0 |
"""


# ---------------------------------------------------------------------------
# reports/resume_bullets.md
# ---------------------------------------------------------------------------

def build_resume(m: Metrics) -> str:
    h, t = m.h, m.test
    bullet_1 = (f"Built end-to-end credit risk engine in SQL (DuckDB) and Python on {num(m.data['n_loans'])} LendingClub loans; "
                f"benchmarked statistical modeling (WoE scorecard) and machine learning (LightGBM) against LendingClub grades "
                f"with out-of-time model validation.")
    bullet_2 = (f"Designed profit optimization policy from calibrated PDs, lifting realized profit {usd_m(h['m2_pd']['profit_vs_grade'], True)} "
                f"({pct(h['m2_pd']['profit_vs_grade_pct'])}) vs grade-based approvals at equal volume; added loss forecasting "
                f"({m.fc['M2']['loss_mape_pct']:.1f}% MAPE) and PSI drift monitoring.")
    words = lambda s: len(s.split())
    return f"""# Resume bullets

_Generated from `reports/metrics/` by `report.py`; every number is computed._

- {bullet_1}
- {bullet_2}

<sub>Word counts: {words(bullet_1)} and {words(bullet_2)}. Keywords covered: SQL, Python, statistical modeling, machine learning,
benchmarking, forecasting, credit risk, profit optimization, model validation.</sub>

### Alternative second bullet (expected-profit framing)

- Built profit-maximizing approve/decline rule from calibrated default probabilities, earning {usd_m(h['m2_profit']['profit_vs_grade'], True)} ({pct(h['m2_profit']['profit_vs_grade_pct'])}) over grade-based policy at {pct(m.rate, 0)} approval, 95% CI {usd_m(m.boot_profit['ci_low'])}–{usd_m(m.boot_profit['ci_high'])}.

## 60-second pitch

"LendingClub prices every loan with its own grade. I asked: if I ran credit policy, could a model plus a profit rule
do better than approving the best grades first? I cleaned {num(m.raw_rows)} loans in DuckDB SQL, built a WoE scorecard and a
LightGBM model using only application-time data, with a leakage guard, and tested them out-of-time on {num(m.splits['test']['n_loans'])} loans from 2014–15.
On ranking alone, LightGBM tied LendingClub's grade (AUC {t['M2']['auc']:.3f} vs {t['B0']['auc']:.3f}), which I confirmed with a bootstrap.
But it had the best-calibrated probabilities, and when I turned those into approval decisions it earned {usd_m(h['m2_pd']['profit_vs_grade'], True)}
more than the grade policy at the same volume, same capital and same bad rate. It picked loans that paid higher rates
for the same risk. I stress-tested that across {m.n_scenarios} loss and cost scenarios, forecast losses by quarter with
{m.fc['M2']['loss_mape_pct']:.0f}% average error, and built PSI and segment monitoring that shows the drift appearing in late 2014. It all runs from one command
and there's a Streamlit simulator where you can move the cutoff and watch profit change."
"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_reports(cfg: dict) -> list:
    m = Metrics(cfg)
    outputs = {
        PROJECT_ROOT / "reports" / "results.md": build_results(m),
        PROJECT_ROOT / "README.md": build_readme(m),
        PROJECT_ROOT / "reports" / "business_memo.md": build_memo(m),
        PROJECT_ROOT / "reports" / "resume_bullets.md": build_resume(m),
    }
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
    return list(outputs)
