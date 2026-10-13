# Telecom Customer Churn — End-to-End MLOps Pipeline

## Project Overview

This is a hands-on MLOps learning and portfolio project focused on building an end-to-end machine learning pipeline for predicting telecom customer churn.

The project is part of my transition into an MLOps engineering career. My goal is to gain practical experience implementing machine learning workflows, managing datasets and experiments, building reproducible pipelines, deploying models, and automating ML systems.

Rather than focusing only on training an accurate model, this project emphasizes the engineering practices required to make machine learning systems reproducible, maintainable, testable, and deployable.

**Project status:** In active development.

## Business Problem

Customer churn occurs when a customer stops using a company's services.

For telecom companies, identifying customers who are likely to churn can help teams understand retention risks and potentially take proactive measures.

**Objective:** Build a machine learning system that predicts customer churn using customer demographics, contract information, monthly usage patterns, and complaint history.

The target variable is:

- `0` — Customer retained
- `1` — Customer churned

The project also explores how to turn an experimental ML model into a more reliable, production-oriented system.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python, SQL |
| Data processing | Pandas, NumPy |
| Database | SQLite |
| Machine learning | scikit-learn |
| Environment management | uv |
| Version control | Git, GitHub |
| Data versioning | DVC |
| Cloud storage | AWS S3 |
| Experiment tracking | MLflow (planned) |
| Model serving | FastAPI (planned) |
| Containerization | Docker (planned) |
| Orchestration | Kubernetes (planned) |
| CI/CD | GitHub Actions (planned) |
| Monitoring | To be implemented |

## MLOps Architecture

The intended pipeline is:

```text
Raw Telecom Dataset
        |
        v
Data Validation & Cleaning
        |
        v
SQLite Database
        |
        v
SQL Analytics Views
        |
        v
Python Feature Engineering
        |
        v
Train / Validation / Test Split
        |
        v
scikit-learn Preprocessing Pipeline
        |
        v
Model Training & Evaluation
        |
        v
MLflow Experiment Tracking
        |
        v
Model Packaging & Registry
        |
        v
FastAPI Model Serving
        |
        v
Docker & Kubernetes
        |
        v
CI/CD and Monitoring
```

DVC and AWS S3 provide data-versioning and storage capabilities alongside the pipeline. Git tracks source code and configuration.

## Dataset

The project uses three telecom customer datasets:

| Dataset | Description |
|---|---|
| `customer_info.csv` | Customer demographics, signup dates, contracts, and monthly charges |
| `usage_data.csv` | Monthly call minutes, data usage, SMS activity, and complaints |
| `churn_labels.csv` | Customer churn outcomes |

The dataset contains:

- 10,000 unique customers after cleaning.
- 120,000 monthly usage records.
- 10,000 churn labels.
- Usage data covering January through December 2023.

Raw datasets are excluded from regular Git tracking. Data versioning is handled separately through DVC.

## Implementation Progress

### Phase 1 — Data Ingestion and Validation

**Implemented**

- Created a Python project environment.
- Established a reproducible project directory structure.
- Created SQLite database tables.
- Built a Python data-ingestion script.
- Validated primary keys, duplicate records, churn labels, and foreign-key relationships.
- Implemented duplicate analysis and cleaning logic.

### Phase 2 — SQL Analytics

**Implemented**

Created SQL views for customer-level analytics, including:

- Average monthly call minutes.
- Average monthly data usage.
- Average SMS activity.
- Total customer complaints.
- Average complaint frequency.
- Months of usage history.

The `model_source` SQL view combines customer information, usage aggregates, and churn labels.

### Phase 3 — Feature Engineering

**Implemented**

Created `src/features/build_features.py` to produce a reusable tabular feature dataset.

Engineered features include:

- Customer tenure in months.
- Complaint rate.
- Monthly charges relative to tenure.

The feature builder supports a configurable historical cutoff date.

**Output:** `data/processed/features.csv` — 10,000 customer records.

### Phase 4 — Model Training

**In progress**

- Train/validation/test split.
- Numerical imputation and scaling.
- Categorical imputation and one-hot encoding.
- Logistic Regression baseline.
- Accuracy, precision, recall, F1, and ROC AUC evaluation.
- Model serialization using Joblib.

The preprocessing and estimator are combined using a scikit-learn `Pipeline` to prevent preprocessing statistics from being learned from validation or test data.

### Phase 5 — Experiment Tracking and Deployment

**Planned**

- MLflow experiment tracking and model comparison.
- DVC pipeline orchestration.
- Automated testing and data-quality checks.
- FastAPI prediction endpoint.
- Docker containerization.
- Kubernetes deployment.
- GitHub Actions CI/CD.
- Monitoring and model lifecycle management.

## Key Data Quality Challenge

During ingestion, the customer dataset contained 10,200 rows representing 10,000 unique customers.

The duplicate analysis identified:

| Issue | Result |
|---|---:|
| Records associated with repeated customer IDs | 400 |
| Exact duplicate rows | 195 |
| Customer IDs with conflicting signup dates | 5 |

This highlighted the importance of validating datasets before training machine learning models.

The project uses documented cleaning rules and preserves the original input files so data preparation can be reproduced.

## Project Structure

```text
telecom-churn-mlops/
├── artifacts/
│   └── feature_schema.json
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── db/
│   └── sql/
│       ├── 01_schema.sql
│       └── 02_features.sql
├── src/
│   ├── data/
│   │   └── build_database.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── create_feature_schema.py
│   └── models/
│       └── train.py
├── models/
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

The structure will evolve as additional MLOps components are implemented.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd telecom-churn-mlops
```

### 2. Set up the Python environment

```bash
uv sync
```

### 3. Prepare the dataset

Restore or place the required dataset files in `data/raw/`, using the configured DVC remote where appropriate.

### 4. Build the SQLite database

```bash
uv run python src/data/build_database.py
```

### 5. Create SQL feature views

```bash
sqlite3 db/telecom.db < db/sql/02_features.sql
```

### 6. Build the feature dataset

```bash
uv run python src/features/build_features.py \
  --as-of-date 2023-12-31
```

The cutoff is currently used for this learning exercise; the churn-label timing still needs confirmation before making claims about leakage-free predictive performance.

### 7. Train the baseline model

```bash
uv run python src/models/train.py
```

Training and evaluation commands will be updated as the experiment-tracking pipeline develops.

## Engineering Lessons

This project focuses on learning important MLOps engineering practices:

- Why dataset versioning matters.
- Why validation should happen before model training.
- How deterministic transformations improve reproducibility.
- Why preprocessing must be learned from training data only.
- How SQL analytics and model preprocessing serve different purposes.
- Why a model needs more than accuracy to evaluate its performance.
- How infrastructure permissions, data access, and deployment configuration affect real ML workflows.

## Current Limitations

The project is a learning implementation and is not yet a production-ready system.

In particular:

- Historical churn-label timing requires further verification.
- Current feature aggregation uses available 2023 usage history.
- Temporal model evaluation is not yet implemented.
- Model monitoring and automated retraining are future work.
- Cloud deployment and CI/CD require additional validation.

## Project Goal

By the end of this project, I aim to demonstrate practical proficiency across the MLOps lifecycle, from raw data and reproducible model training to deployment, automation, monitoring, and troubleshooting.

The final outcome will be a documented MLOps pipeline that can serve as a technical portfolio project and a reference for future engineering work.
