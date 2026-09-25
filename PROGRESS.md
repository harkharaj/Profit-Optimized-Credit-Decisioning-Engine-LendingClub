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
