-- Explicit descriptive cohorts, not automatically causal subgroups.
SELECT treat,
       CASE WHEN re74 = 0 AND re75 = 0 THEN 'no prior earnings'
            ELSE 'some prior earnings' END AS earnings_cohort,
       COUNT(*) AS n, AVG(age) AS mean_age, AVG(educ) AS mean_education,
       AVG(re78) AS mean_1978_earnings
FROM observations GROUP BY treat, earnings_cohort ORDER BY earnings_cohort, treat;
-- Standardized mean differences use pooled sample variance, not a significance
-- test. A small difference is no proof of exchangeability.
WITH moments AS (
  SELECT treat, AVG(age) AS mean_age, VAR_SAMP(age) AS var_age
  FROM observations GROUP BY treat
)
SELECT (a.mean_age-b.mean_age)/SQRT((a.var_age+b.var_age)/2) AS age_smd
FROM moments a CROSS JOIN moments b WHERE a.treat=1 AND b.treat=0;
