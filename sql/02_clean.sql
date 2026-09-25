-- 02_clean.sql
-- Define the modeling population and the default target.
--
-- Step 1 -> loans_population: valid, de-duplicated, 36-month, individual loans issued in the window.
-- Step 2 -> loans_clean: only loans with a FINAL outcome, plus the target.
--     target = 1  'Charged Off' or 'Default'
--     target = 0  'Fully Paid'
--     excluded    'Current', 'Late', 'In Grace Period' (no final outcome yet) and the legacy
--                 'Does not meet the credit policy. Status: ...' loans.

CREATE OR REPLACE TABLE loans_population AS
SELECT *
FROM loans_raw
WHERE id IS NOT NULL                -- footer/summary lines have a non-numeric id -> NULL after TRY_CAST
  AND loan_amnt IS NOT NULL
  AND term = '$term'
  AND application_type = '$application_type'
  AND issue_date BETWEEN DATE '$issue_start' AND DATE '$issue_end'
QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY issue_date) = 1;   -- de-duplicate on id


CREATE OR REPLACE TABLE loans_clean AS
SELECT
    *,
    CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END AS target
FROM loans_population
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default');
