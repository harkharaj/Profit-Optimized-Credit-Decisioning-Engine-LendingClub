# Resume bullets

_Generated from `reports/metrics/` by `report.py`; every number is computed._

- Built end-to-end credit risk engine in SQL (DuckDB) and Python on 611,816 LendingClub loans; benchmarked statistical modeling (WoE scorecard) and machine learning (LightGBM) against LendingClub grades with out-of-time model validation.
- Designed profit optimization policy from calibrated PDs, lifting realized profit +$22.5M (8.3%) vs grade-based approvals at equal volume; added loss forecasting (9.0% MAPE) and PSI drift monitoring.

<sub>Word counts: 30 and 27. Keywords covered: SQL, Python, statistical modeling, machine learning,
benchmarking, forecasting, credit risk, profit optimization, model validation.</sub>

### Alternative second bullet (expected-profit framing)

- Built profit-maximizing approve/decline rule from calibrated default probabilities, earning +$44.9M (16.6%) over grade-based policy at 70% approval, 95% CI $41.7M–$47.9M.

## 60-second pitch

"LendingClub prices every loan with its own grade. I asked: if I ran credit policy, could a model plus a profit rule
do better than approving the best grades first? I cleaned 2,260,701 loans in DuckDB SQL, built a WoE scorecard and a
LightGBM model using only application-time data, with a leakage guard, and tested them out-of-time on 445,357 loans from 2014–15.
On ranking alone, LightGBM tied LendingClub's grade (AUC 0.670 vs 0.671), which I confirmed with a bootstrap.
But it had the best-calibrated probabilities, and when I turned those into approval decisions it earned +$22.5M
more than the grade policy at the same volume, same capital and same bad rate. It picked loans that paid higher rates
for the same risk. I stress-tested that across 9 loss and cost scenarios, forecast losses by quarter with
9% average error, and built PSI and segment monitoring that shows the drift appearing in late 2014. It all runs from one command
and there's a Streamlit simulator where you can move the cutoff and watch profit change."
