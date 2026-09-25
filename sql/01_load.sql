-- 01_load.sql
-- Read the raw LendingClub CSV with DuckDB (never pandas: the file is ~2.2M rows x 151 columns),
-- keep only the columns this project needs, and cast them to proper types.
--
-- Every column is first read as text (all_varchar) so a single malformed value can't break the load.
-- TRY_CAST / try_strptime turn anything unparseable (e.g. footer lines) into NULL; 02_clean.sql drops those rows.
-- Dates arrive as 'Mon-YYYY' (e.g. 'Dec-2015'); some dataset versions store rates as '13.99%'.

CREATE OR REPLACE TABLE loans_raw AS
SELECT
    -- identifiers, status and dates
    TRY_CAST(id AS BIGINT)                                  AS id,
    trim(loan_status)                                       AS loan_status,
    trim(term)                                              AS term,              -- raw value is ' 36 months'
    upper(trim(application_type))                           AS application_type,  -- 'Individual' vs 'INDIVIDUAL'
    CAST(try_strptime(issue_d, '%b-%Y') AS DATE)            AS issue_date,
    CAST(try_strptime(earliest_cr_line, '%b-%Y') AS DATE)   AS earliest_cr_date,
    CAST(try_strptime(last_pymnt_d, '%b-%Y') AS DATE)       AS last_pymnt_date,

    -- LendingClub's own risk model output = the price of the loan.
    -- Used for the benchmark and the profit math, NEVER as model features.
    trim(grade)                                             AS grade,
    trim(sub_grade)                                         AS sub_grade,
    TRY_CAST(replace(int_rate, '%', '') AS DOUBLE)          AS int_rate,
    TRY_CAST(installment AS DOUBLE)                         AS installment,

    -- amounts and realized cash flows (known only after origination: evaluation only)
    TRY_CAST(loan_amnt AS DOUBLE)                           AS loan_amnt,
    TRY_CAST(funded_amnt AS DOUBLE)                         AS funded_amnt,
    TRY_CAST(total_pymnt AS DOUBLE)                         AS total_pymnt,
    TRY_CAST(total_rec_prncp AS DOUBLE)                     AS total_rec_prncp,
    TRY_CAST(total_rec_int AS DOUBLE)                       AS total_rec_int,
    TRY_CAST(total_rec_late_fee AS DOUBLE)                  AS total_rec_late_fee,
    TRY_CAST(recoveries AS DOUBLE)                          AS recoveries,
    TRY_CAST(collection_recovery_fee AS DOUBLE)             AS collection_recovery_fee,

    -- application-time information (candidate model features)
    trim(emp_length)                                        AS emp_length,
    upper(trim(home_ownership))                             AS home_ownership,
    TRY_CAST(annual_inc AS DOUBLE)                          AS annual_inc,
    trim(verification_status)                               AS verification_status,
    lower(trim(purpose))                                    AS purpose,
    TRY_CAST(dti AS DOUBLE)                                 AS dti,
    TRY_CAST(delinq_2yrs AS DOUBLE)                         AS delinq_2yrs,
    TRY_CAST(inq_last_6mths AS DOUBLE)                      AS inq_last_6mths,
    TRY_CAST(pub_rec AS DOUBLE)                             AS pub_rec,
    TRY_CAST(pub_rec_bankruptcies AS DOUBLE)                AS pub_rec_bankruptcies,
    TRY_CAST(fico_range_low AS DOUBLE)                      AS fico_range_low,
    TRY_CAST(fico_range_high AS DOUBLE)                     AS fico_range_high,
    TRY_CAST(mths_since_last_delinq AS DOUBLE)              AS mths_since_last_delinq,
    TRY_CAST(mths_since_last_record AS DOUBLE)              AS mths_since_last_record,
    TRY_CAST(open_acc AS DOUBLE)                            AS open_acc,
    TRY_CAST(total_acc AS DOUBLE)                           AS total_acc,
    TRY_CAST(mort_acc AS DOUBLE)                            AS mort_acc,
    TRY_CAST(revol_bal AS DOUBLE)                           AS revol_bal,
    TRY_CAST(replace(revol_util, '%', '') AS DOUBLE)        AS revol_util,

    -- optional credit-bureau fields (kept only if well populated in the training window)
    TRY_CAST(tot_cur_bal AS DOUBLE)                         AS tot_cur_bal,
    TRY_CAST(total_rev_hi_lim AS DOUBLE)                    AS total_rev_hi_lim,
    TRY_CAST(bc_util AS DOUBLE)                             AS bc_util,
    TRY_CAST(avg_cur_bal AS DOUBLE)                         AS avg_cur_bal,
    TRY_CAST(acc_open_past_24mths AS DOUBLE)                AS acc_open_past_24mths,
    TRY_CAST(num_tl_op_past_12m AS DOUBLE)                  AS num_tl_op_past_12m,
    TRY_CAST(percent_bc_gt_75 AS DOUBLE)                    AS percent_bc_gt_75,
    TRY_CAST(tot_coll_amt AS DOUBLE)                        AS tot_coll_amt,
    TRY_CAST(collections_12_mths_ex_med AS DOUBLE)          AS collections_12_mths_ex_med

FROM read_csv_auto('$raw_path', all_varchar = true, ignore_errors = true);
