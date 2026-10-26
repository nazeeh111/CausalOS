-- Mirrors the executed, safely quoted queries in causalos.profiling.
-- observations is a registered immutable pandas relation. No transformations
-- are silently applied. Substitute identifiers only with SQL double quoting.
SELECT COUNT(*) AS rows FROM observations;
SELECT COUNT(*) AS unique_rows FROM (SELECT DISTINCT * FROM observations);
-- Example for the packaged NSW dataset:
SELECT treat, COUNT(*) AS n, AVG(re78) AS mean_earnings_1978,
       STDDEV_SAMP(re78) / SQRT(COUNT(*)) AS standard_error
FROM observations GROUP BY treat ORDER BY treat;
SELECT COUNT(*) FILTER (WHERE re78 IS NULL) AS missing_outcome,
       COUNT(*) FILTER (WHERE re74 = 0 AND re75 = 0) AS no_prior_earnings,
       AVG(re78) FILTER (WHERE treat = 1) -
       AVG(re78) FILTER (WHERE treat = 0) AS unadjusted_difference
FROM observations;
