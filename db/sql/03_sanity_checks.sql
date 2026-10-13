
-- 1. Count records
SELECT 'customers' AS dataset, COUNT(*) AS total
FROM customers
UNION ALL
SELECT 'unique_usage_customers', COUNT(DISTINCT customer_id)
FROM usage_monthly
UNION ALL
SELECT 'churn_labels', COUNT(*)
FROM churn_labels
UNION ALL
SELECT 'model_source', COUNT(*)
FROM model_source;

-- 2. Churn rate
SELECT
    churn,
    COUNT(*) AS n,
    ROUND(
        100.0 * COUNT(*) /
        (SELECT COUNT(*) FROM churn_labels),
        2
    ) AS pct
FROM churn_labels
GROUP BY churn;

-- 3. Churn by contract type
SELECT
    c.contract_type,
    l.churn,
    COUNT(*) AS n
FROM customers c
JOIN churn_labels l USING (customer_id)
GROUP BY c.contract_type, l.churn
ORDER BY c.contract_type, l.churn;

-- 4. Average complaints by churn
SELECT
    l.churn,
    ROUND(AVG(u.complaints), 2) AS avg_complaints
FROM usage_monthly u
JOIN churn_labels l USING (customer_id)
GROUP BY l.churn;
