DROP VIEW IF EXISTS customer_usage_features;
CREATE VIEW customer_usage_features AS
SELECT
customer_id,
COUNT(*) AS months_observed,
AVG(call_minutes) AS avg_call_minutes,
AVG(data_usage_gb) AS avg_data_usage_gb,
AVG(sms_count) AS avg_sms_count,
SUM(COALESCE(complaints,0)) AS total_complaints,
AVG(COALESCE(complaints,0)) AS avg_complaints
FROM usage_monthly
GROUP BY customer_id;
DROP VIEW IF EXISTS model_source;
CREATE VIEW model_source AS
SELECT
c.customer_id,
c.age, c.gender, c.region, c.signup_date,
c.contract_type, c.monthly_charges,
u.months_observed, u.avg_call_minutes, u.avg_data_usage_gb,
u.avg_sms_count, u.total_complaints, u.avg_complaints,
l.churn
FROM customers c
JOIN customer_usage_features u USING (customer_id)
JOIN churn_labels l USING (customer_id);