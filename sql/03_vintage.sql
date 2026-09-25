-- 03_vintage.sql
-- Vintage analysis: how did each issue cohort ("vintage") of loans perform?


-- 1. Default rate, loan count and funded dollars by issue quarter x grade
CREATE OR REPLACE TABLE vintage_by_grade AS
SELECT date_trunc('quarter', issue_date) AS vintage,
       grade,
       COUNT(*)                          AS n_loans,
       AVG(target)                       AS default_rate,
       SUM(funded_amnt)                  AS funded_usd
FROM loans_clean
GROUP BY 1, 2
ORDER BY 1, 2;


-- 2. Risk-based pricing check: is higher realized risk actually charged a higher rate?
CREATE OR REPLACE TABLE pricing_by_subgrade AS
SELECT sub_grade,
       grade,
       COUNT(*)          AS n_loans,
       AVG(int_rate::DECIMAL(6, 2)) AS avg_int_rate,   -- exact decimal sum: same result on every run
       AVG(target)       AS default_rate
FROM loans_clean
GROUP BY 1, 2
ORDER BY 1;


-- 3. Cumulative default curve by vintage (window function).
-- APPROXIMATION: the data has no default date, so for a charged-off loan
-- months-on-book at default ~ months from issue to its last payment (0 if it never paid).
-- A full vintage x month grid is built so months with no new defaults still appear.
CREATE OR REPLACE TABLE vintage_cum_default AS
WITH base AS (
    SELECT date_trunc('quarter', issue_date)                                       AS vintage,
           LEAST(COALESCE(date_diff('month', issue_date, last_pymnt_date), 0), 60) AS mob,
           target
    FROM loans_clean
),
sizes AS (
    SELECT vintage, COUNT(*) AS n_loans FROM base GROUP BY 1
),
new_defaults AS (
    SELECT vintage, mob, SUM(target) AS new_defaults
    FROM base WHERE target = 1
    GROUP BY 1, 2
),
grid AS (
    SELECT s.vintage, m.mob, s.n_loans
    FROM sizes s CROSS JOIN range(0, 61) AS m(mob)
)
SELECT g.vintage,
       g.mob,
       g.n_loans,
       SUM(COALESCE(nd.new_defaults, 0)) OVER (PARTITION BY g.vintage ORDER BY g.mob)              AS cum_defaults,
       SUM(COALESCE(nd.new_defaults, 0)) OVER (PARTITION BY g.vintage ORDER BY g.mob) / g.n_loans  AS cum_default_rate
FROM grid g
LEFT JOIN new_defaults nd USING (vintage, mob)
ORDER BY 1, 2;
