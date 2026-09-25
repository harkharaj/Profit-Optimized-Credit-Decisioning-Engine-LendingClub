# PROGRESS

Running log of what was done, the key numbers (copied from `reports/metrics/`), decisions and open issues.

---

## Phase 0 — Setup ✅

- Repo structure, `config.yaml`, `requirements.txt`, `pyproject.toml` (editable install so `python -m credit_engine.run_all` works from anywhere), `Makefile`, `.gitignore`.
- **Data download:** no Kaggle credentials on this machine, so the single file `accepted_2007_to_2018Q4.csv.gz` (374 MB) was fetched with `kagglehub` (public datasets need no login) and moved to `data/raw/`. Only the accepted-loans file was downloaded, not the full ~5 GB bundle (disk space).
- **Windows note:** `make` is not installed here. Every Makefile target is a single Python command, so it can be run directly (e.g. `python -m credit_engine.run_all`).

## Phase 1 — Load, clean, population and target ✅

Source: `reports/metrics/data_summary.json`.

- DuckDB reads the gzipped CSV directly (`read_csv_auto(..., all_varchar=true, ignore_errors=true)`), keeps 52 of 151 columns and casts types with `TRY_CAST`. Takes ~20 s; the raw table is cached in `data/processed/lending_club.duckdb`.
- **Population waterfall:** 2,260,701 rows in file → 2,260,668 valid → 1,609,754 36-month → 1,539,983 individual → 612,653 issued 2010–2015 (no duplicate ids) → **611,816 with a final outcome**.
- **Target:** 1 = Charged Off / Default, 0 = Fully Paid. Overall default rate **13.9%**.
- **Exclusions:** 690 legacy "Does not meet the credit policy" loans (all in 2010) and 147 loans without a final outcome (Current / Late / Grace), almost all in 2015-Q4 (0.16% of that quarter). **No quarter exceeds the 2% flag**, so dropping non-terminal loans doesn't bias recent vintages.
- **Payment identity** `total_pymnt = principal + interest + late fees + recoveries` holds for 100% of loans (max gap $0.01), so `total_pymnt` can be trusted for realized profit.
- **Default rate rises over time:** 9.9% (2010) → 14.9% (2015), and it rises *within* grades too (e.g. grade C: 12.7% → 19.4%). The test period (2014–15) is riskier than train (2010–12). Expect calibrated PDs from 2013 to under-forecast 2015 → a real-world drift story for Phases 7–8.
- **Missingness surprise:** several bureau fields (`tot_cur_bal`, `total_rev_hi_lim`, `avg_cur_bal`, `tot_coll_amt`, `num_tl_op_past_12m`) are 68% missing in the train window but ~0% in 2013–15: LendingClub only started reporting them in late 2012. They fail the 50% rule and are dropped. `bc_util`, `percent_bc_gt_75`, `acc_open_past_24mths` and `mort_acc` are ~43% missing in train and pass. This missingness is *structural* (depends on issue date), not borrower behavior.

### Decisions
- Cleaning is split in two: `01_load.sql` only parses/casts (cached, slow), `02_clean.sql` applies the population rules (fast, always re-run so config edits take effect).
- `application_type` and `home_ownership` are upper-cased and `purpose` lower-cased at load time so later comparisons are simple.

### Resolved
- Repo uses a local git identity (personal account), not the global work one; pushed to GitHub.

## Phase 2 — Vintage analysis ✅

Sources: `vintage_*.csv`, `vintage_summary.json`. Figures: `vintage_default_by_grade.png`, `cum_default_curves.png`, `pricing_vs_risk.png`.

- **Risk-based pricing works:** across the 35 sub-grades, average interest rate and realized default rate have Spearman correlation **0.986**. Price rises from ~5.7% (A1, 2.8% default) to ~25% (F4/F5, 35%+ default).
- **Later vintages are worse at every grade** (e.g. grade D ~20% in 2013 → ~26–27% by late 2015): LendingClub's grades did not fully re-price the deterioration.
- **Timing:** most charge-offs land in months 8–30 on book; curves flatten after ~36 months, confirming that every 36-month loan in the population has a final outcome.
- Chart note: grade G and cells with < 100 loans are hidden (too noisy); sub-grades with < 500 loans are hidden on the pricing chart. The cumulative curves are aggregated to annual vintages for readability (the CSV keeps quarters). Months-on-book is approximated as months from issue to last payment and labelled as such.

## Phases 3–4 — Features, leakage control, splits ✅

Sources: `univariate_auc.csv`, `splits.json`, `model_metrics.json → features`.

- **Split:** train 2010–12 = 66,037 loans (12.5% default), validation 2013 = 100,422 (12.3%), test 2014–15 = 445,357 (14.5%). The test period is riskier than anything the models trained on.
- **30 features** after the train-fitted builder: 23 core/derived numerics + 4 bureau fields that pass the 50% missing rule (`bc_util`, `acc_open_past_24mths`, `percent_bc_gt_75`, `collections_12_mths_ex_med`) + 3 categoricals. 5 bureau fields dropped (68% missing in train).
- **Leakage tripwire passed:** strongest single feature is `fico_mid` (univariate AUC **0.620**); nothing near 0.80. For reference, LC's own `int_rate` scores 0.631 on train (not a feature).
- **Leakage guard:** `assert_no_leakage` runs inside `FeatureBuilder.fit` and `.transform`; `tests/test_no_leakage.py` checks denylist, pricing, fair-lending and `hardship_*`/`settlement_*` columns.
- **Design:** stateless parsing in SQL (`04_features.sql`); everything learned from data (99.5% caps, rare `purpose` levels < 1% → `other`, bureau keep/drop) is fit on train only in `FeatureBuilder`.
- **Quirk:** `mort_acc` looks "riskier when higher" in train only because its missing values are all 2010–12 loans (which defaulted less). Structural missingness, not borrower behavior.

## Phase 5 — Models and benchmark ✅ (checkpoint)

Sources: `model_metrics.json`, `scorecard.csv`, `scorecard_feature_selection.csv`, `lightgbm_optuna_trials.csv`. Figures: `roc.png`, `ks.png`, `calibration.png`.

| Test 2014–15 | AUC | Gini | KS | Top-decile capture | Brier (calibrated) |
|---|---|---|---|---|---|
| B0 LC sub-grade | 0.671 | 0.342 | 0.252 | 19.5% | 0.1193 |
| B0 LC int_rate | 0.668 | 0.335 | 0.246 | 19.4% | – |
| M1 WoE scorecard | 0.653 | 0.306 | 0.221 | 19.1% | 0.1196 |
| M2 LightGBM (monotone) | 0.670 | 0.340 | 0.245 | 20.0% | 0.1183 |

- **Validation (2013):** M2 0.669 and M1 0.652 both beat LC's sub-grade (0.649).
- **Test:** LC's grade improves to 0.671 while our models stay flat, so M2 vs B0 is a **statistical tie**: paired bootstrap AUC(M2) − AUC(B0) = −0.0013, 95% CI [−0.0034, +0.0006] (1,000 resamples). M1 is clearly behind: −0.018, CI [−0.020, −0.016].
  - Likely reasons: LC re-trained its grading on far more data over 2013–15 (its grade embeds full-bureau information we don't have), while our models only saw 66k loans from 2010–12, before most bureau fields existed.
- **M2 has the best calibrated probabilities** (lowest Brier on test) and captures the most defaults in the riskiest decile, which matters for the profit layer: it uses PDs, not just ranks.
- **Calibration drift:** every model under-predicts on test (actual 14.5% vs predicted mean 13.3% M2, 12.5% M1, 12.0% B0). All were calibrated on 2013, a more benign year. M2's PDs move the most with the riskier 2014–15 applicant mix. This feeds the loss-forecasting phase.
- **Monotone-constraint experiment:** constraining `fico_mid` (−), `dti` (+), `inq_last_6mths` (+), `revol_util` (+) *gained* 0.0005 validation AUC (0.6684 → 0.6689). Rule set in config: keep the constrained model unless it costs > 0.002 validation AUC → **production M2 is monotone**. Decision taken on validation only.
- **Scorecard:** 30 candidates → 12 pass IV ≥ 0.02 → `percent_bc_gt_75` dropped (|corr| 0.84 with `bc_util`) → `mort_acc` and `dti` dropped for **wrong-sign coefficients** (their effect flips once `loan_to_income`/`revol_util` and the shared-missingness bureau fields are in). **9 features** in the final scorecard; validation AUC is flat across C (best C = 10). Points: 600 at 50:1, PDO 20.
- **Calibration choice** (5-fold CV Brier within validation): isotonic for B0 and M2, Platt for M1. The two methods differ by < 0.0001 Brier, so the choice barely matters.
- **LightGBM:** Optuna picked a very regularized model (8 leaves, 522 min child samples, lr 0.018, 1,100 trees), consistent with a weak, noisy signal. Top gain features: `fico_mid`, `log_annual_inc`, `loan_to_income`, `purpose`, `acc_open_past_24mths`.

### Engineering notes
- LightGBM's C++ `save_model` can't write to this folder on Windows (non-ASCII "—" in the path); the model text is written through Python instead.
- The paired bootstrap re-weights one sorted order instead of re-sorting 445k scores 1,000 times (`evaluation._WeightedAUC`); a test checks it against scikit-learn, including ties.

## Phase 6 — Profit decision layer ✅ (checkpoint)

Sources: `policy_results.json` (incl. `headline`), `swap_set.csv`, `profit_sensitivity.csv`. Figures: `profit_curve.png`, `swap_set.png`.

**Economics (train only):** net return on a repaid loan `r_good` rises from 8.8% (grade A) to 31.5% (G); net loss on a defaulted loan `L` is ~39–44% of the funded amount for every grade (F and G fall back to the overall mean: < 100 defaults). Break-even PD = r / (r + L), so ~17% for grade A and ~32% for grade C.

**Test 2014–15 at 70% approval** (keep 70% of LC's accepted book):

| Policy | Profit | vs grade | Dollars lent | Bad rate | Profit per $ lent |
|---|---|---|---|---|---|
| Grade policy (existing) | $270.4M | – | $4.07B | 10.5% | 6.65% |
| M1 scorecard PD | $293.1M | +8.4% | $4.20B | 11.0% | 6.97% |
| M2 LightGBM PD | $292.9M | **+8.3%** | $4.06B | 10.5% | **7.22%** |
| M2 expected profit | $315.4M | **+16.6%** | $4.71B | 14.4% | 6.70% |
| Grade-PD expected profit (decomposition) | $314.5M | +16.3% | $4.96B | 15.1% | 6.34% |
| Approve all (LC's whole book) | $360.2M | | $5.67B | 14.5% | 6.36% |

- **Two different sources of gain:**
  1. **Better risk ranking (M2 PD policy): +$22.5M, 95% CI [$20.1M, $24.7M]**, with the same number of loans, the same dollars lent (−0.3%) and the same bad rate. This is the clean "our model adds value" result.
  2. **Ranking by expected dollar profit: +$44.9M, 95% CI [$41.7M, $47.9M]** (spec's headline policy). But it lends 16% more dollars (bigger, higher-rate loans), and the same formula with the *grade-based* PD gets +$44.0M. So most of this gain comes from the profit-aware decision rule, not from M2. M2's PD adds only +$0.9M on top at the base case.
- **Swap sets (70%):**
  - *M2 PD vs grade:* 56,187 loans swapped each way. **Both groups default at the same ~18.4%**, but the loans M2 takes pay 15.7% interest vs 11.4% → $961 vs $562 profit per loan. The swap-outs are 58% grade A/B: loans LC priced as safe that defaulted like riskier ones, which M2 flagged (mean PD 21%).
  - *M2 expected profit vs grade:* 101,393 swapped each way. Swap-ins are riskier (23.4% vs 11.4% default) but pay 16.3% vs 9.3%, and larger → $828 vs $385 profit per loan. Lowest risk ≠ highest profit.
- **Model-based policies beat the grade policy on total profit at every approval rate** (50–90%). On profit per dollar lent, the M2 PD policy is best everywhere (7.4% at 50% approval vs 6.6% for grade).
- **Profit-max rule (approve if E[profit] > 0):** approves **98.7%** of the test book (99.6% of validation). Its gain over the grade policy at the same approval rate is tiny (+$0.2M). Under this simplified profit (no funding cost or time value), nearly every loan LC accepted was worth making at its price, so a pure profit-maximizer barely tightens. Tightening only pays when volume or capital is constrained, which is where the ranking results above matter.
- **Sensitivity (loss × 0.8/1.0/1.2, cost $0/50/100):**
  - M2 expected profit beats the grade policy in **all 9 scenarios** (+$26.1M to +$66.5M).
  - The M2 PD gain is **stable at +$22.1–22.8M** in every scenario.
  - Fixed cost doesn't change differences at a fixed approval rate (every policy approves the same count).
  - When losses are 20% *lower*, the grade-PD profit ranking edges out M2 (−$0.9M); when losses are 20% *higher*, M2's better PDs matter more (+$5.2M over grade-PD ranking).
- **Framing:** only LC-accepted loans are observed, so every policy is a tightening of LC's book; "approval rate" = share of LC's accepted book.

### Decisions
- Added two things beyond the spec, because an interviewer would ask about them: dollars lent / profit per dollar (a policy can "win" by lending more), and the grade-PD expected-profit policy (to separate the value of the formula from the value of the model).
- Bootstrap = paired resampling of test loans; each policy re-approves its top 70% of the resampled book.

## Recommended policy changed: M2 PD ranking (decided while building reason codes)

- The expected-profit ranking declines mostly **small, low-risk loans** ($2–5k), because dollar profit scales with loan size. Those declines are not risk-based, so SHAP "risk factor" reason codes would misdescribe them. Adverse-action notices must state the real reasons.
- `config.yaml → profit.recommended_policy` is now **`m2_pd`**: +$22.5M at 70% approval with the same volume, capital and bad rate, the most robust gain (+$22.1M to +$22.8M in every stress scenario), and every decline is risk-based. The expected-profit ranking stays in all reports as the higher-profit, higher-capital option.
- Used for: the approved-book forecast, reason codes, the app's "Explain a decision" page and the memo.

## Phase 7 — Loss forecasting ✅

Source: `forecast.json`. Figure: `forecast_vs_actual.png`.

| Whole test book, 8 quarters | Defaults MAPE | Net-loss MAPE |
|---|---|---|
| M2 LightGBM | **6.9%** | **9.0%** |
| Benchmark (2010–12 default rate by grade × quarter's grade mix) | 7.6% | 10.5% |
| M1 scorecard | 11.3% | 10.5% |

- M2's two-year net-loss total is within ~0.2% of actual ($295.4M vs $294.7M), but the quarterly errors trend. Defaults go from ≈ on target (2014-Q1) to −12% (2015-Q4); dollar losses from +20% (2014-Q1) to −10% (2015-Q4). Drift again, visible only quarter by quarter.
- **Approved book (M2 PD policy, 70%):** M2 net-loss MAPE 9.5%, defaults under-forecast by 8.6% on average. The grade benchmark *over*-forecasts losses by 32%, because M2 picks the safer loans within each grade. Selecting on a model concentrates its optimism (a selection effect), so monitor the approved book specifically.

## Phase 8 — Monitoring, weak spots, explainability ✅

Sources: `psi.csv`, `csi.csv`, `weak_spots.csv`, `monitoring_summary.json`, `shap_importance.csv`, `reason_codes_examples.csv`. Figures: `psi_by_quarter.png`, `shap_summary.png`.

- **Score PSI** (M2 calibrated PD, train-decile bins; isotonic output has 100 distinct values, so edges are de-duplicated): 0.036 in 2013 → **monitor in 2014-Q4 to 2015-Q2 (max 0.121)** → back under 0.10 in late 2015. It coincides with the forecast under-prediction and needs no outcomes.
- **CSI:** `acc_open_past_24mths` 3.9 and `bc_util` 1.9 (structural: ~43% missing in train, ~0% later), `fico_mid` 0.25 in 2014-Q4 (action), `purpose` 0.23 and `credit_hist_months` 0.15 (monitor). Missing values get their own bin, so a field that starts being reported counts as drift.
- **Weak spots:** 13 of 51 segments flagged (|gap| ≥ 2 pp or actual/predicted outside 0.8–1.25, n ≥ 500), all under-predicted. The largest by dollar impact is RENT (+2.5 pp, ≈$21.9M), then Verified income (+2.5 pp, ≈$17.8M) and 5–10 yrs credit history (+2.8 pp).
- **Early warning:** 0 flagged on 2013 data, but **12 of 12** flagged segments with 2013 data were already under-predicted in the same direction (below the threshold). That motivates the memo's "same-direction two quarters running" trend alert.
- **SHAP** (TreeExplainer on the monotone LightGBM, 20k test loans; values identical to LightGBM's native `pred_contrib`): top drivers FICO, accounts opened in 24 months, income, loan-to-income, inquiries.
- **Reason codes:** the top-4 risk-raising SHAP features in plain language with the applicant's value, e.g. "FICO score: 662 · Accounts opened in last 24 months: 7 · Credit inquiries in last 6 months: 2". Five declined examples are in `reason_codes_examples.csv`.

## Phase 9 — Streamlit app ✅

- `app/streamlit_app.py`, six pages (overview, vintage explorer, benchmark, **policy simulator**, drift & weak spots, explain a decision). It reads only `app/data/` (23.3 MB: the whole 445k-loan test book plus metrics and figures), built by `app_data.py`.
- The app imports `credit_engine.profit`, so the simulator uses the pipeline's exact policy code. Verified with Streamlit's `AppTest`: every page runs without exceptions, and the default setting reproduces the report ($292.9M, +$22.5M vs grade).

## Phase 10 — Reporting ✅

- `report.py` generates `README.md`, `reports/results.md`, `reports/business_memo.md` and `reports/resume_bullets.md` from templates filled with `reports/metrics/`, so no number is hand-typed.
- `notebooks/credit_decisioning_walkthrough.ipynb`: executed walkthrough of every phase (reads saved results; live SQL cells only when the DuckDB file exists).
- **Reproducibility:** deleted every generated artifact (DuckDB, models, metrics, figures, app data) and rebuilt from the raw CSV with `python -m credit_engine.run_all` in **5.9 min**. Every JSON was identical; one CSV differed by 1e-12 (parallel-sum order in DuckDB), so CSVs are now rounded to 10 decimals. SHAP beeswarm jitter is now seeded, so reruns are byte-identical.
- **Not done / needs the user:** deploying to Streamlit Community Cloud needs the user's account (entry point `app/streamlit_app.py`, repo already on GitHub). `make` isn't installed on this Windows machine, so the Makefile targets are untested here; each is a single command that was run directly.

## Definition of done (§7)

- [x] One command runs raw file → reports (`python -m credit_engine.run_all`; `make all` wraps it)
- [x] All tests pass (51)
- [x] `model_metrics.json`, `policy_results.json`, `forecast.json`, `psi.csv`, `weak_spots.csv` exist
- [x] All figures exist in `reports/figures/`
- [x] README, results.md, business memo and resume bullets are generated from `reports/metrics/`
- [x] Streamlit app runs locally (`streamlit run app/streamlit_app.py`)
- [x] PROGRESS.md documents decisions, assumptions and surprises
