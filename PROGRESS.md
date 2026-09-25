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

### Open issues
- Git identity is the global work account; the user may want a personal email for this CV repo.

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
| B0 LC sub-grade | 0.671 | 0.343 | 0.252 | 19.5% | 0.1193 |
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
- **LightGBM:** Optuna picked a very regularized model (8 leaves, 522 min child samples, lr 0.018, ~1,100 trees), consistent with a weak, noisy signal. Top gain features: `fico_mid`, `log_annual_inc`, `loan_to_income`, `purpose`, `revol_util`.

### Engineering notes
- LightGBM's C++ `save_model` can't write to this folder on Windows (non-ASCII "—" in the path); the model text is written through Python instead.
- The paired bootstrap re-weights one sorted order instead of re-sorting 445k scores 1,000 times (`evaluation._WeightedAUC`); a test checks it against scikit-learn, including ties.
