-- 04_features.sql
-- Row-level feature parsing: transformations that need no statistics from the data
-- (anything learned from data, like caps or rare-category groups, is fit on TRAIN in features.py).
--
-- model_base = keys + outcome + pricing (never features) + application-time features.

CREATE OR REPLACE TABLE model_base AS
SELECT
    -- keys, outcome, pricing and cash flows: NOT features (evaluation / benchmark / profit only)
    id, issue_date, target, grade, sub_grade, int_rate, installment,
    funded_amnt, total_pymnt, last_pymnt_date,

    -- application-time features ------------------------------------------------------------
    loan_amnt,

    CASE WHEN emp_length = '< 1 year'  THEN 0
         WHEN emp_length = '10+ years' THEN 10
         ELSE TRY_CAST(regexp_extract(emp_length, '(\d+)', 1) AS DOUBLE)      -- 'n/a' / NULL -> NULL
    END                                                              AS emp_length_years,
    CASE WHEN emp_length IS NULL OR emp_length = 'n/a' THEN 1 ELSE 0 END AS emp_length_missing,

    CASE WHEN home_ownership IN ('ANY', 'NONE', 'OTHER') THEN 'OTHER'
         ELSE home_ownership END                                     AS home_ownership,
    annual_inc,
    verification_status,
    purpose,
    CASE WHEN dti < 0 THEN NULL ELSE dti END                         AS dti,   -- negative DTI is a data error
    delinq_2yrs,
    inq_last_6mths,
    pub_rec,
    pub_rec_bankruptcies,
    (fico_range_low + fico_range_high) / 2                           AS fico_mid,

    -- NULL "months since" means the event never happened: fill with a large value + flag
    COALESCE(mths_since_last_delinq, $never)                         AS mths_since_last_delinq,
    CASE WHEN mths_since_last_delinq IS NULL THEN 1 ELSE 0 END       AS never_delinquent,
    COALESCE(mths_since_last_record, $never)                         AS mths_since_last_record,
    CASE WHEN mths_since_last_record IS NULL THEN 1 ELSE 0 END       AS no_public_record,

    open_acc,
    total_acc,
    mort_acc,
    revol_bal,
    revol_util,
    date_diff('month', earliest_cr_date, issue_date)                 AS credit_hist_months,

    -- derived ratios
    loan_amnt / NULLIF(annual_inc, 0)                                AS loan_to_income,
    revol_bal / NULLIF(annual_inc, 0)                                AS revol_bal_to_income,
    open_acc  / NULLIF(total_acc, 0)                                 AS open_to_total_acc,

    -- optional bureau fields (features.py keeps only those populated enough in train)
    tot_cur_bal, total_rev_hi_lim, bc_util, avg_cur_bal, acc_open_past_24mths,
    num_tl_op_past_12m, percent_bc_gt_75, tot_coll_amt, collections_12_mths_ex_med

FROM loans_clean;
