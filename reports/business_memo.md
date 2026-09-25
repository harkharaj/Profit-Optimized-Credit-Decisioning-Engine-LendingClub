# Memo: replace grade-based approvals with a model-based, profit-aware policy

**To:** Head of Credit Policy  **From:** Credit Decisioning Analytics  **Re:** Approval policy for 36-month personal loans

_All figures are realized results on 445,357 loans issued 2014–2015 that no model or rule was tuned on._

## Recommendation

**Rank applicants by the new LightGBM default model (M2) instead of by LendingClub grade.**
If volume must be held at 70% of today's approvals, this earns **$292.9M vs $270.4M**,
an extra **+$22.5M (8.3%)**, 95% CI $20.1M–$24.7M,
with the **same number of loans, the same bad rate (10.5%) and the same dollars lent** (-0.3%).

## Why it works

Where the two policies disagree, the loans M2 approves and the grade policy declines default at the same rate
(18.3% vs 18.4%) but pay 15.7% instead of 11.4%.
M2 identifies borrowers whose grade overstates their risk, and it drops grade A/B loans that behave like riskier grades.

## Options considered

| Option at 70% approval | Profit | vs grade | Bad rate | Dollars lent |
|---|---|---|---|---|
| Grade policy (today) | $270.4M | – | 10.5% | $4.07B |
| **M2 PD ranking (recommended)** | $292.9M | +$22.5M | 10.5% | $4.06B |
| M2 expected-profit ranking | $315.4M | +$44.9M | 14.4% | $4.71B |

The expected-profit ranking earns more, but it lends 15.8% more capital into larger, riskier loans,
its advantage shrinks from +$66.5M to +$26.1M as loss severity rises, and it declines small loans for size rather than risk,
which is harder to explain to customers. Pilot it only where capital is available, with a loss-severity limit.
Without a volume constraint, 98.7% of today's approvals already have positive expected profit, so there is no case for broad tightening.

## Risks

- **Selection bias:** only approved loans have outcomes, so the model cannot judge applicants we decline today. This recommendation only re-orders the current book.
- **Drift:** PDs calibrated on 2013 under-predicted 2015 defaults (13.3% vs 14.5%). Score PSI reached 0.12 in late 2014. `acc_open_past_24mths` and `bc_util` changed distribution completely once LendingClub began reporting them in 2012, and `fico_mid` also crossed the action level. Recalibrate at least annually.
- **Weak spots:** 13 of 51 segments are under-predicted by ≥ 2 pp; the largest is home ownership = RENT (+2.5 pp, ≈$21.9M of unexpected loss).
- **Fair lending:** geography and free-text fields are excluded, every decline gets SHAP-based reason codes, and constraints keep the model's direction intuitive (a higher FICO never raises predicted risk). A disparate-impact review is still required before launch.
- **Loss severity:** loss rates are estimated by grade from 2010–12; the policy ranking held in all 9 stress scenarios (losses ±20%, $0–100 cost per loan).

## Monitoring plan

| What | How often | Trigger |
|---|---|---|
| Score PSI (M2 PD vs development sample) | Monthly | ≥ 0.10 investigate, > 0.25 recalibrate |
| Feature CSI (top 15 features) | Monthly | > 0.25 root-cause (data or population change) |
| Segment calibration (purpose, FICO, DTI, income, tenure …) | Quarterly | gap ≥ 2 pp or actual/predicted outside 0.8–1.25; also alert on the same-direction gap two quarters running |
| Forecast vs actual losses by vintage | Quarterly | MAPE > 10% or bias > 5% for two quarters |
| Champion vs grade profit (holdout of approvals) | Quarterly | gain CI includes 0 |
