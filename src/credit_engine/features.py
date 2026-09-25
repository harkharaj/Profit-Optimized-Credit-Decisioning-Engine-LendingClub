"""Feature allowlist, derived features and the leakage guard.

Rule: a model may only see information LendingClub had when the application
arrived. Anything created after the loan was issued (payments, recoveries,
later credit pulls) or LendingClub's own pricing is forbidden.
"""

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

# Raw application-time columns (before parsing / deriving)
CORE_RAW_COLUMNS = [
    "loan_amnt", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "pub_rec", "pub_rec_bankruptcies",
    "fico_range_low", "fico_range_high", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "total_acc", "mort_acc", "revol_bal", "revol_util", "earliest_cr_date",
]

# Credit-bureau fields LendingClub started reporting later; kept only if well
# populated in the TRAINING window (see max_missing_rate_train in config)
OPTIONAL_BUREAU_COLUMNS = [
    "tot_cur_bal", "total_rev_hi_lim", "bc_util", "avg_cur_bal", "acc_open_past_24mths",
    "num_tl_op_past_12m", "percent_bc_gt_75", "tot_coll_amt", "collections_12_mths_ex_med",
]

# LendingClub's own risk model output. Allowed for the benchmark and the profit
# math, never as features: using them would copy LC's model instead of competing
# with it (and installment = f(amount, rate, term) leaks int_rate).
PRICING_COLUMNS = ["grade", "sub_grade", "int_rate", "installment"]

# Post-origination information and identifiers: must never be features
DENYLIST = [
    "loan_status", "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_date", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "pymnt_plan", "policy_code", "initial_list_status", "disbursement_method",
    "debt_settlement_flag", "id", "member_id", "url", "desc",
    "target", "issue_date", "split",
]
DENYLIST_PREFIXES = ["hardship_", "settlement_"]

# Excluded for fair lending (ECOA / Reg B): geography can proxy for protected
# characteristics; free-text job titles are noisy and can proxy too
FAIR_LENDING_EXCLUDED = ["zip_code", "addr_state", "emp_title", "title"]
