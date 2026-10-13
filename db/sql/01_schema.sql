DROP TABLE IF EXISTS churn_labels;
DROP TABLE IF EXISTS usage_monthly;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    region TEXT,
    signup_date TEXT,
    contract_type TEXT,
    monthly_charges REAL
);

CREATE TABLE usage_monthly (
    customer_id TEXT NOT NULL,
    month TEXT NOT NULL,
    call_minutes REAL,
    data_usage_gb REAL,
    sms_count INTEGER,
    complaints INTEGER,
    PRIMARY KEY (customer_id, month),
    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);

CREATE TABLE churn_labels (
    customer_id TEXT PRIMARY KEY,
    churn INTEGER NOT NULL CHECK(churn IN (0,1)),
    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);