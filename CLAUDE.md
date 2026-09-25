# CLAUDE.md — Profit-Optimized Credit Decisioning Engine

This file is the full specification for this repository. Read it completely before writing any code.

---

## 0. How to work on this project (read first)

**Context.** This is a portfolio project for an M.Tech student at IISc applying for Capital One's *Associate, Business Analysis* role in Bangalore. The audience is analytics and credit-risk reviewers, so the business framing (profit, policy, risk) matters as much as the ML. Every result should answer a business question, not only report a model metric.

**Working style**

- Work through the phases in §5 in order. After each phase: run the tests, update `PROGRESS.md` (what was done, key numbers, open issues), and make a small git commit.
- **Checkpoints — stop and show the user a short summary, then wait for a go-ahead:**
  - after Phase 1 (data summary)
  - after Phase 5 (model benchmark)
  - after Phase 6 (profit results)
- Explain decisions briefly in code comments and in `PROGRESS.md`. The user must be able to defend every choice in an interview.

**Hard rules**

1. **Never invent or hand-type a metric.** Every number in the README, reports, memo or resume bullets must be read from files in `reports/metrics/` that the code produced.
2. **The test set (loans issued 2014–2015) is used once, for final evaluation.** No tuning, feature selection, binning, calibration or cutoff choice may use it.
3. **Leakage tripwire.** If any model's validation AUC is above 0.85, or any single feature's univariate AUC is above 0.80, stop and investigate before continuing. Application-time default models on this data usually land somewhere around 0.65–0.75 AUC, so a much higher score almost certainly means leakage.
4. **Never load the full raw CSV into pandas.** It is about 2.2M rows × 151 columns. Use DuckDB to select only the needed columns and rows.
5. Ask the user before adding any dependency not listed in §3.
6. Keep everything reproducible:
   - fixed seeds (`seed: 42` in config)
   - all dates and parameters in `config.yaml`
   - one entry point: `python -m credit_engine.run_all` (also expose `make all`)

---

## 1. Project summary

Build an end-to-end credit decisioning system on historical LendingClub loans:

- **SQL vintage analysis** of default rates by issue quarter and grade.
- **Default-probability (PD) models** — a logistic-regression scorecard and LightGBM — benchmarked against LendingClub's own grade-based decision system.
- **A profit-maximizing approve/decline policy** built from calibrated PDs, compared with the grade policy at matched approval rates, including swap-set analysis.
- **Bank-style validation**: out-of-time testing, expected-loss forecasting versus actuals, drift monitoring (PSI) and segment-level weak-spot detection.
- **A Streamlit policy simulator** where a user moves the approval cutoff and sees approval rate, bad rate and profit update live.

**Business question.** If we were the credit policy team, which applicants should we approve to maximize portfolio profit? How much better is a data-driven policy than the existing grade-based one?

### How this maps to the job description

| JD requirement | Where this project covers it |
|---|---|
| Proficiency in SQL and Python | DuckDB SQL pipeline (`sql/`), Python package (`src/credit_engine/`) |
| Data analysis, statistical modeling & forecasting | Vintage analysis, WoE logistic scorecard, expected-loss forecast vs actuals |
| Machine learning solutions development | LightGBM PD model, calibration, deployed Streamlit app |
| Benchmark existing solutions against state-of-the-art decision systems | LC grade vs scorecard vs LightGBM; profit policy vs grade policy |
| Data mining & pattern recognition | Segment weak-spot mining, swap-set analysis, SHAP |
| Identify weak spots in model predictions earlier (AI & risk management) | PSI/CSI drift monitoring, segment calibration gaps, early-warning check |
| Execution: hypotheses, testing, monitoring plan | Business memo and monitoring section |

---

## 2. Data

**Source.** Kaggle dataset `wordsforthewise/lending-club`: all LendingClub accepted loans, 2007–2018.

- Use the accepted-loans file, typically `accepted_2007_to_2018Q4.csv.gz`. The name can vary by version, so glob for `accepted_*.csv*`.
- Ignore the rejected-loans file. It has no outcomes; see the limitations in §8.

**Download**

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY)
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
```

If the user has no Kaggle credentials, ask them to download the file manually into `data/raw/`. `data/raw/` must be git-ignored.

**Known quirks to handle and document**

- **Messy text columns:**
  - `term` has a leading space (" 36 months").
  - `int_rate` and `revol_util` may arrive as strings with `%` in some versions.
  - `emp_length` is text ("10+ years", "< 1 year", "n/a").
- **Date columns** (`issue_d`, `earliest_cr_line`, `last_pymnt_d`) use the format `Mon-YYYY`, e.g. `Dec-2015`. Parse them with `strptime(col, '%b-%Y')`.
- **Junk rows:** some rows may be summary or footer lines. Drop rows where `id` is not numeric or `loan_amnt` is null, and deduplicate on `id`.
- **`application_type` casing** can differ ("Individual" / "INDIVIDUAL"). Compare with `upper(trim(...))`.
- **Legacy statuses:** statuses of the form "Does not meet the credit policy. Status: ..." are legacy loans; exclude them.
- **Payment identity:** verify on the data that `total_pymnt ≈ total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries`, and document the result.

---

## 3. Tech stack and repo structure

**Python 3.11.** Dependencies (`requirements.txt`, loosely pinned):

```
duckdb, pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm, optuna, shap,
matplotlib, streamlit, pyyaml, pytest
```

```
credit-decisioning-engine/
├── CLAUDE.md                  # this spec
├── README.md
├── PROGRESS.md                # running log, updated after every phase
├── config.yaml                # dates, splits, params, seed
├── requirements.txt
├── Makefile
├── data/
│   ├── raw/                   # git-ignored
│   └── processed/             # git-ignored (duckdb file, parquet)
├── sql/
│   ├── 01_load.sql
│   ├── 02_clean.sql
│   ├── 03_vintage.sql
│   └── 04_features.sql
├── src/credit_engine/
│   ├── config.py
│   ├── data.py                # runs SQL, returns clean frames
│   ├── features.py            # allowlist, derived features, leakage guard
│   ├── splits.py
│   ├── scorecard.py           # WoE binning, IV, logistic scorecard, points
│   ├── models.py              # LightGBM, calibration
│   ├── evaluation.py          # AUC, Gini, KS, Brier, bootstrap
│   ├── profit.py              # realized profit, expected profit, policies, swap sets
│   ├── forecasting.py         # expected loss vs actual by vintage
│   ├── monitoring.py          # PSI/CSI, weak-spot mining
│   ├── explain.py             # SHAP, reason codes
│   ├── plots.py
│   ├── report.py              # builds results.md from metrics JSON
│   └── run_all.py
├── app/
│   ├── streamlit_app.py
│   └── data/                  # small precomputed artifacts only (<50 MB)
├── models/
├── reports/
│   ├── metrics/               # all JSON/CSV outputs (single source of truth)
│   ├── figures/
│   ├── results.md             # auto-generated
│   ├── business_memo.md
│   └── resume_bullets.md
└── tests/
```

---

## 4. Config (`config.yaml`, starting values)

```yaml
seed: 42
population:
  term_months: 36
  issue_start: 2010-01-01
  issue_end: 2015-12-01      # 36-month loans issued by Dec-2015 reach maturity by the data cutoff
  application_type: INDIVIDUAL
splits:
  train: [2010-01-01, 2012-12-01]
  valid: [2013-01-01, 2013-12-01]
  test:  [2014-01-01, 2015-12-01]   # out-of-time, touched once
features:
  max_missing_rate_train: 0.50
profit:
  servicing_fee: 0.01        # assumption: fee on payments received; document it
  fixed_cost_per_loan: 0     # sensitivity: [0, 50, 100]
  loss_rate_multipliers: [0.8, 1.0, 1.2]
  approval_rates: [0.5, 0.6, 0.7, 0.8, 0.9]
  swap_set_approval_rate: 0.7
monitoring:
  psi_bins: 10
  min_segment_n: 500
  gap_flag_pp: 2.0
```

---

## 5. Phases

### Phase 0 — Setup

- Create the repo structure, `requirements.txt`, `config.yaml`, `Makefile`, `.gitignore` and an empty `PROGRESS.md`.
- Download the data, or ask the user to place it in `data/raw/`.

**Done when:** `make setup` works and the raw file is found.

### Phase 1 — Load, clean, define population and target (SQL)

**Load (`01_load.sql`).** Read with DuckDB: `read_csv_auto(path, all_varchar=true, ignore_errors=true)`. Select only the columns needed (allowlist in Phase 3, plus the ID, date, status, pricing and payment columns) into table `loans_raw`. Cast types in SQL.

**Clean (`02_clean.sql`) → table `loans_clean`**

- Keep rows where:
  - `trim(term) = '36 months'`
  - `upper(trim(application_type)) = 'INDIVIDUAL'`
  - `issue_date` is within the population window
- **Target:** `target = 1` if `loan_status IN ('Charged Off', 'Default')`, `0` if `'Fully Paid'`. Exclude every other status: Current, Late, In Grace Period, and the legacy "Does not meet the credit policy" statuses.
- For each issue quarter, log how many loans were excluded for having a non-terminal status. If more than 2% of any quarter is excluded, flag it in `PROGRESS.md`, because it would bias recent vintages.

**Output:** `reports/metrics/data_summary.json`, containing:
- rows per issue year
- default rate by year and grade
- the missingness table
- exclusion counts

**Checkpoint:** show the user the data summary.

### Phase 2 — Vintage analysis (SQL)

**`03_vintage.sql`** should produce these tables:

- Default rate, loan count and funded dollars by issue quarter × grade.
- Average interest rate against realized default rate by sub-grade. This is a risk-based pricing check: is higher risk actually priced higher?
- A cumulative default curve by vintage, using window functions. The approximation to use: for charged-off loans, months-on-book ≈ months from `issue_date` to `last_pymnt_date`, or 0 if no payment was ever made. Label this clearly as an approximation.

Reference queries (adapt as needed):

```sql
-- default rate by vintage and grade
SELECT date_trunc('quarter', issue_date) AS vintage,
       grade,
       COUNT(*)          AS n_loans,
       AVG(target)       AS default_rate,
       SUM(funded_amnt)  AS funded_usd
FROM loans_clean
GROUP BY 1, 2
ORDER BY 1, 2;

-- cumulative default curve by vintage (months-on-book approximated via last payment date)
WITH base AS (
  SELECT date_trunc('quarter', issue_date) AS vintage,
         COALESCE(date_diff('month', issue_date, last_pymnt_date), 0) AS mob,
         target
  FROM loans_clean
),
new_defaults AS (
  SELECT vintage, mob, SUM(target) AS d
  FROM base WHERE target = 1
  GROUP BY 1, 2
),
sizes AS (SELECT vintage, COUNT(*) AS n FROM base GROUP BY 1)
SELECT nd.vintage, nd.mob,
       SUM(nd.d) OVER (PARTITION BY nd.vintage ORDER BY nd.mob) * 1.0 / s.n AS cum_default_rate
FROM new_defaults nd JOIN sizes s USING (vintage)
ORDER BY 1, 2;
```

**Figures:**
- `vintage_default_by_grade.png`
- `cum_default_curves.png`
- `pricing_vs_risk.png`

**Output:** `reports/metrics/vintage_*.csv`.

### Phase 3 — Features and leakage control

Use an **allowlist**: only information available at application time.

**Core allowlist**

| Column | Treatment |
|---|---|
| `loan_amnt` | as-is |
| `emp_length` | parse to 0–10, plus a missing flag |
| `home_ownership` | collapse ANY/NONE/OTHER → OTHER |
| `annual_inc` | log; cap at the train 99.5th percentile |
| `verification_status` | categorical |
| `purpose` | categorical; group levels under 1% of train into OTHER |
| `dti` | cap at the train 99.5th percentile; handle null/negative |
| `delinq_2yrs`, `inq_last_6mths`, `pub_rec`, `pub_rec_bankruptcies` | as-is |
| `fico_range_low`, `fico_range_high` | → `fico_mid` |
| `mths_since_last_delinq`, `mths_since_last_record` | null means "never": fill with a large value and add a flag |
| `open_acc`, `total_acc`, `mort_acc`, `revol_bal` | as-is |
| `revol_util` | parse `%` |
| `earliest_cr_line` | → `credit_hist_months` at `issue_date` |

**Derived features**
- `loan_to_income = loan_amnt / annual_inc`
- `revol_bal_to_income`
- `open_to_total_acc`

**Optional bureau fields.** Include each only if its missing rate in the *training window* is at or below `max_missing_rate_train`:

`tot_cur_bal`, `total_rev_hi_lim`, `bc_util`, `avg_cur_bal`, `acc_open_past_24mths`, `num_tl_op_past_12m`, `percent_bc_gt_75`, `tot_coll_amt`, `collections_12_mths_ex_med`.

**Pricing columns — allowed for the benchmark and the profit calculation, never as model features:**

`grade`, `sub_grade`, `int_rate`, `installment`.

- These are the *outputs* of LendingClub's own risk model. Using them as features would copy LC's model instead of competing with it.
- `installment` is a function of amount, rate and term, so it leaks `int_rate`.

**Denylist — must never be features.** These are post-origination leakage or identifiers:

```
loan_status, funded_amnt, funded_amnt_inv, out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv,
total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries, collection_recovery_fee,
last_pymnt_d, last_pymnt_amnt, next_pymnt_d, last_credit_pull_d, last_fico_range_high,
last_fico_range_low, pymnt_plan, policy_code, initial_list_status, disbursement_method,
debt_settlement_flag, hardship_*, settlement_*, id, member_id, url, desc
```

**Fair-lending exclusions:**
- `zip_code` and `addr_state` are excluded because they can proxy for protected characteristics; US lenders are subject to fair-lending rules (ECOA / Reg B).
- `emp_title` and `title` (free text) are excluded as noisy and potential proxies.
- Document all of these in the README.

**Leakage guard.** `features.py` must raise an error if any denylisted or pricing column reaches a model feature matrix. `tests/test_no_leakage.py` asserts this.

**Univariate check.** Compute each feature's univariate AUC on train and save it to `reports/metrics/univariate_auc.csv`. Apply the tripwire rule from §0.

### Phase 4 — Time-based splits

- Split by `issue_date` using the config windows: train 2010–2012, validation 2013, test 2014–2015.
- There is no random splitting and no refit on the test period.
- All preprocessing statistics (caps, bins, category groupings, imputation values) are fit on **train only**.
- Save split sizes and default rates to `reports/metrics/splits.json`.

### Phase 5 — Models and benchmark

**B0 — Existing decision system (benchmark)**
- Score each loan with LC's `sub_grade` as an ordinal (A1=1 … G5=35; higher = riskier), and separately with `int_rate`.
- Evaluate both directly as risk scores.

**M1 — WoE logistic-regression scorecard (industry-standard baseline)**
- **Numeric features:** up to 10 quantile bins on train, each with at least 5% of the population. Merge adjacent bins to enforce a monotonic bad rate where it makes business sense (`fico_mid`, `dti`, `revol_util`, `inq_last_6mths`, `annual_inc`).
- **Categorical features:** group rare levels.
- **Feature selection:**
  - compute WoE and Information Value per feature; keep IV ≥ 0.02
  - for any feature pair with |corr of WoE| > 0.7, drop the one with lower IV
- **Model:** L2 logistic regression with `C` tuned on validation.
- **Points scaling:** 600 points at 50:1 good:bad odds, PDO = 20.
- **Save** the scorecard table (feature, bin, WoE, points) to `reports/metrics/scorecard.csv`.

**M2 — LightGBM challenger**
- Uses raw allowlisted features with native missing-value and categorical handling.
- Tune with Optuna (30–50 trials) to maximize validation AUC, with early stopping.
- **Experiment:** add monotone constraints (`fico_mid` −, `dti` +, `inq_last_6mths` +, `revol_util` +) and report how much AUC this costs, if any. This is an interview talking point about interpretability versus accuracy.

**Calibration**
- Fit isotonic calibration (or Platt, whichever has the lower validation Brier score) on validation, for both M1 and M2.
- Calibrated PDs are required because they feed the profit math.

**Metrics** (validation and test):
- AUC
- Gini (= 2·AUC − 1)
- KS
- Brier score (before and after calibration)
- top-decile capture rate
- decile calibration table

On test, add a paired bootstrap (1,000 resamples) giving 95% confidence intervals for AUC(M1) − AUC(B0) and AUC(M2) − AUC(B0).

**Outputs**
- `models/`
- `reports/metrics/model_metrics.json`
- `roc.png`, `ks.png`, `calibration.png`

**Checkpoint:** show the user the benchmark table.

### Phase 6 — Profit decision layer (the core business result)

**Realized profit per loan** (used for evaluation only):

```
realized_profit_i = total_pymnt_i × (1 − servicing_fee) − funded_amnt_i
```

**Economic parameters.** Estimate these on TRAIN only:

- `r_good[sub_grade]`: mean of `realized_profit / funded_amnt` over fully paid loans. This is the net return on a good loan, and it naturally includes the effect of early prepayment.
- `L[grade]`: mean of `−realized_profit / funded_amnt` over charged-off loans. This is the net loss rate on a bad loan. Fall back to the overall mean for sparse grades.

**Expected profit per applicant** (used for decisions):

```
E[π_i] = (1 − PD_i) · r_good[sub_grade_i] · A_i  −  PD_i · L[grade_i] · A_i  −  c
```

Here `A_i` = `loan_amnt` and `c` = `fixed_cost_per_loan`.

Using `sub_grade` and `grade` here is legitimate, not leakage. LC's price is a given offer; our decision is only approve or decline at that price.

**Policies to compare on test**, at each approval rate in `approval_rates`:

1. **Grade policy (existing system):** approve the best sub-grades first. Break ties by lower `int_rate`, then randomly with the fixed seed.
2. **M1 policy:** approve the lowest PDs first.
3. **M2 policy:** approve the lowest PDs first.
4. **M2 expected-profit policy:** approve the highest `E[π]` first.

For each policy and approval rate, report:
- approval rate
- bad rate among approved loans
- total realized profit
- profit per approved loan
- profit per applicant

**Profit-max policy.** Approve if `E[π] > 0`, using the rule and parameters fixed on validation. Report its test approval rate and realized profit, compared with the grade policy at the same approval rate.

**Swap-set analysis** at `swap_set_approval_rate`, comparing M2-profit with the grade policy:
- **swap-ins:** loans the model approves and the grade policy rejects
- **swap-outs:** loans the grade policy approves and the model rejects

For each group report count, bad rate and realized profit. Also make a bar chart.

**Sensitivity analysis:** repeat the comparison across `loss_rate_multipliers` × `c ∈ {0, 50, 100}`. Report whether the policy ranking holds.

**Uncertainty:** paired bootstrap 95% CI for the profit difference (M2-profit minus grade) at 70% approval.

**Framing note (put this in the README).** Only loans LC accepted are observed. Here, "approval rate" means the share of LC's already-approved book we would approve, so it is a *tightening* policy.

**Outputs**
- `reports/metrics/policy_results.json`
- `swap_set.csv`
- `profit_curve.png` (x = approval rate, y = total realized profit; one line per policy)
- `swap_set.png`

**Checkpoint:** show the user the policy comparison.

### Phase 7 — Loss forecasting

For each test issue quarter, compute:

| Measure | Formula |
|---|---|
| Forecast defaults | Σ PD_i |
| Actual defaults | Σ target_i |
| Forecast net loss $ | Σ PD_i · L[grade_i] · A_i |
| Actual net loss $ | Σ −realized_profit_i over charged-off loans |

- Do this for the whole book and for the book approved under the chosen policy.
- **Benchmark forecast:** apply train-period default rates by grade to each test quarter's grade mix.
- **Report:**
  - error % per quarter
  - MAPE for M1, M2 and the benchmark forecast
- **Chart:** `forecast_vs_actual.png`
- **Output:** `reports/metrics/forecast.json`

### Phase 8 — Monitoring, weak spots, explainability

**Drift**
- **PSI** of calibrated M2 PD, using bins set from train deciles: compute it for validation and for each test quarter.
- **CSI** for the top 15 features.
- **Thresholds:** below 0.10 stable, 0.10–0.25 monitor, above 0.25 action.
- **Outputs:** `psi.csv`, `psi_by_quarter.png`

**Weak-spot mining**
- **Segments:** `purpose`, `home_ownership`, `verification_status`, `emp_length` bucket, 20-point FICO bands, DTI bands, income quintiles, loan-amount quintiles, credit-history bands.
- For each segment with n ≥ `min_segment_n` in test, compute:
  - actual vs predicted default rate
  - calibration gap in percentage points
  - actual/predicted ratio
  - segment AUC
- **Flag** a segment if |gap| ≥ `gap_flag_pp` or the ratio falls outside [0.8, 1.25].
- **Rank** flagged segments by estimated dollar impact: gap × count × average amount × L.
- **Early-warning check:** was each flagged segment already visible in validation (2013)? This answers "could we have caught it earlier?", which the JD's risk-management bullet asks for.
- **Output:** `weak_spots.csv`

**Explainability**
- SHAP `TreeExplainer` on M2, on a 20k-row test sample. Produce `shap_summary.png`.
- **Reason codes:** for declined applicants, list the top 4 features that pushed PD up. This mimics US adverse-action notices.
- Save 5 example declined applicants with their reason codes to `reason_codes_examples.csv`.

### Phase 9 — Streamlit policy simulator

**Data rule.** The app loads only precomputed artifacts from `app/data/`:
- a parquet file of test-set scores plus key columns, sampled if needed to stay under 50 MB
- the metrics JSON files

The app never trains anything.

**Pages**

1. **Overview:** the business question and headline numbers.
2. **Vintage explorer:** default rate by quarter, filtered by grade.
3. **Model benchmark:** ROC, KS, calibration and the AUC/Gini table.
4. **Policy simulator:**
   - inputs: approval-rate slider (30–100%), policy selector, loss-rate multiplier, cost per loan
   - outputs: approval rate, bad rate, total profit and profit per loan, each shown with its delta against the grade policy
5. **Drift & weak spots:** PSI chart and the flagged-segments table.
6. **Explain a decision:** pick a loan to see its PD, the decision and its reason codes.

**Deployment.** Use relative paths and a `requirements.txt` so the app deploys to Streamlit Community Cloud.

### Phase 10 — Reporting

**`reports/results.md`** is generated by `report.py` from the metrics files. It contains tables only; nothing is hand-typed.

**`README.md`** contains:
- problem and business question
- data and population
- approach diagram (Mermaid is fine)
- headline results, pulled from the metrics files
- key charts
- how to run the project
- the app link
- limitations (§8)
- a repo map

**`reports/business_memo.md`** is a one-page memo to a "Head of Credit Policy", in plain business language:
- the recommended policy and approval rate
- expected impact against the grade policy
- risks: selection bias, drift, fair lending
- a monitoring plan (PSI thresholds, segment review cadence)

**`reports/resume_bullets.md`** contains:
- **Two resume bullets,** each at most about 30 words, filled only with computed numbers. Make them ATS-friendly by naturally including: SQL, Python, statistical modeling, machine learning, benchmarking, forecasting, credit risk, profit optimization, model validation.
- **A 60-second verbal pitch** of the project for interviews.

---

## 6. Tests (`pytest`, run after every phase)

| Test file | What it checks |
|---|---|
| `test_no_leakage.py` | Feature-matrix columns ∩ (denylist ∪ pricing columns) = ∅ |
| `test_population.py` | Only 36-month, individual loans with terminal statuses; target ∈ {0, 1} with the correct mapping |
| `test_splits.py` | Split date ranges don't overlap; every row is in exactly one split |
| `test_profit.py` | Expected-profit and realized-profit functions against hand-computed examples |
| `test_policy.py` | At approval rate k, the approved count is round(k·n); the grade policy orders sub-grades correctly |
| `test_psi.py` | PSI is 0 for identical distributions and positive for shifted ones |

---

## 7. Definition of done

- [ ] `make all` runs end-to-end from the raw file to reports without manual steps
- [ ] All tests pass
- [ ] `model_metrics.json`, `policy_results.json`, `forecast.json`, `psi.csv` and `weak_spots.csv` exist
- [ ] All figures listed above exist in `reports/figures/`
- [ ] README, results.md, business memo and resume bullets exist, with every number traceable to `reports/metrics/`
- [ ] The Streamlit app runs locally with `streamlit run app/streamlit_app.py`
- [ ] `PROGRESS.md` documents decisions, assumptions and anything surprising

---

## 8. Limitations to state honestly (README and memo)

- **Accepted loans only.** There is no reject inference, because rejected applications have no outcomes. Policies can only tighten LC's existing book.
- **Simplified profit.** It ignores the time value of money and funding costs. The servicing fee and cost per loan are assumptions, covered by the sensitivity analysis.
- **Different economics.** LendingClub is a peer-to-peer installment-loan platform, so its economics differ from a bank's credit-card business. The method transfers; the exact numbers do not.
- **A single, fairly benign credit period** (2010–2015). There is no recession in the test window.
- **Coarse loss rates.** They are estimated by grade only.

---

## 9. Priority if time is short

| Priority | Scope |
|---|---|
| **Must have** | Phases 0–6 and a basic README with real numbers |
| **Should have** | Phase 7, PSI and weak spots from Phase 8, bootstrap CIs, Phase 9 app |
| **Nice to have** | Optuna tuning, monotone-constraint experiment, SHAP reason codes, business memo polish |
