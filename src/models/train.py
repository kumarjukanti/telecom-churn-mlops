
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# =====================================================
# 1. PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/features.csv"
MODEL_DIR = ROOT / "models"


# =====================================================
# 2. FEATURE DEFINITIONS
# =====================================================

NUMERIC = [
    "age",
    "monthly_charges",
    "months_observed",
    "avg_call_minutes",
    "avg_data_usage_gb",
    "avg_sms_count",
    "total_complaints",
    "avg_complaints",
    "tenure_months",
    "complaint_rate",
    "charges_per_tenure_month",
]

CATEGORICAL = [
    "gender",
    "region",
    "contract_type",
]


# =====================================================
# 3. MAIN TRAINING PIPELINE
# =====================================================

def main():

    # -------------------------------------------------
    # STEP 1: LOAD PROCESSED FEATURES
    # -------------------------------------------------

    print("\nSTEP 1: Loading features...")

    df = pd.read_csv(DATA)

    print(f"Rows loaded: {len(df)}")

    required = NUMERIC + CATEGORICAL + ["churn"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["churn"].isna().any():
        raise ValueError("Missing churn labels found")

    X = df[NUMERIC + CATEGORICAL]
    y = df["churn"].astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError("Churn labels must be 0 or 1")

    # -------------------------------------------------
    # STEP 2: CREATE TRAIN / VALIDATION / TEST SETS
    # -------------------------------------------------

    print("\nSTEP 2: Splitting dataset...")

    # Reserve 20% for final testing
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Reserve 20% of remaining data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        stratify=y_trainval,
        random_state=42,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Testing records: {len(X_test)}")

    # -------------------------------------------------
    # STEP 3: NUMERICAL PREPROCESSING
    # -------------------------------------------------

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # -------------------------------------------------
    # STEP 4: CATEGORICAL PREPROCESSING
    # -------------------------------------------------

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    # -------------------------------------------------
    # STEP 5: COMBINE PREPROCESSING PIPELINES
    # -------------------------------------------------

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL),
    ])

    # -------------------------------------------------
    # STEP 6: BUILD COMPLETE ML PIPELINE
    # -------------------------------------------------

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # -------------------------------------------------
    # STEP 7: TRAIN MODEL
    # -------------------------------------------------

    print("\nSTEP 3: Training Logistic Regression...")

    model.fit(X_train, y_train)

    # -------------------------------------------------
    # STEP 8: EVALUATE ON VALIDATION DATA
    # -------------------------------------------------

    print("\nSTEP 4: Evaluating validation dataset...")

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if y_val.nunique() == 2:
        auc = roc_auc_score(y_val, probabilities)
        print(f"ROC AUC:   {auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    # -------------------------------------------------
    # STEP 9: SAVE TRAINED PIPELINE
    # -------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "churn_pipeline.joblib"

    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")
    print("\nSUCCESS: Baseline model training completed!")

    print(
        "NOTE: Test dataset remains reserved "
        "for final model evaluation."
    )


# =====================================================
# 4. ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
