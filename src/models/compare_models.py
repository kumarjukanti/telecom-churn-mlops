
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data/processed/train.parquet"
VALIDATION_FILE = ROOT / "data/processed/validation.parquet"

MODEL_FILE = ROOT / "models/selected_churn_model.joblib"
REPORT_FILE = ROOT / "artifacts/model_selection.json"

RANDOM_STATE = 42
MIN_PRECISION = 0.50

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

FEATURES = NUMERIC + CATEGORICAL


def make_pipeline(classifier):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def validate_dataset(df, name):
    required = FEATURES + ["customer_id", "churn"]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )

    if df.empty:
        raise ValueError(f"{name}: dataset is empty")

    if df["customer_id"].isna().any():
        raise ValueError(f"{name}: missing customer IDs")

    if df["customer_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate customer IDs")

    if df["churn"].isna().any():
        raise ValueError(f"{name}: missing churn labels")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError(f"{name}: invalid churn labels")

    if df["churn"].nunique() != 2:
        raise ValueError(
            f"{name}: both churn classes are required"
        )


def choose_threshold(y_true, probabilities):
    precision, recall, thresholds = (
        precision_recall_curve(y_true, probabilities)
    )

    valid_indices = np.flatnonzero(
        precision[:-1] >= MIN_PRECISION
    )

    if len(valid_indices):
        best_index = valid_indices[
            np.argmax(recall[valid_indices])
        ]

        return float(thresholds[best_index]), True

    # Diagnostic fallback only.
    # Never interpret this as a passed acceptance gate.
    best_index = int(np.argmax(precision[:-1]))

    return float(thresholds[best_index]), False


def evaluate_at_threshold(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predicted, zero_division=0)
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    print("STEP 1: Loading DVC-managed datasets...")

    train_df = pd.read_parquet(TRAIN_FILE)
    val_df = pd.read_parquet(VALIDATION_FILE)

    validate_dataset(train_df, "train")
    validate_dataset(val_df, "validation")

    if set(train_df["customer_id"]) & set(val_df["customer_id"]):
        raise ValueError(
            "Training and validation customer IDs overlap"
        )

    X_train = train_df[FEATURES]
    y_train = train_df["churn"].astype(int)

    X_val = val_df[FEATURES]
    y_val = val_df["churn"].astype(int)

    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_val)}")

    class_prevalence = float(y_val.mean())

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    trained_models = {}

    print("\nSTEP 2: Training and comparing models...")

    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_val)[:, 1]

        results[name] = {
            "roc_auc": float(
                roc_auc_score(y_val, probabilities)
            ),
            "pr_auc": float(
                average_precision_score(y_val, probabilities)
            ),
        }

        trained_models[name] = pipeline

        print(
            f"{name}: "
            f"ROC-AUC={results[name]['roc_auc']:.4f}, "
            f"Average Precision={results[name]['pr_auc']:.4f}"
        )

    print("\nSTEP 3: Selecting best candidate...")

    best_name = max(
        results,
        key=lambda name: results[name]["pr_auc"],
    )

    best_model = trained_models[best_name]

    probabilities = best_model.predict_proba(X_val)[:, 1]

    threshold, threshold_feasible = choose_threshold(
        y_val, probabilities
    )

    validation_metrics = evaluate_at_threshold(
        y_val, probabilities, threshold
    )

    precision_requirement_met = bool(
        threshold_feasible
        and validation_metrics["precision"] >= MIN_PRECISION
    )

    business_criteria_met = precision_requirement_met

    # Model selection is not production authorization.
    deployment_approved = False

    report = {
        "selected_model": best_name,
        "selection_metric": "average_precision",
        "minimum_precision": MIN_PRECISION,
        "class_prevalence": class_prevalence,
        "pr_auc_baseline": class_prevalence,
        "precision_requirement_met": precision_requirement_met,
        "business_criteria_met": business_criteria_met,
        "deployment_approved": deployment_approved,
        "threshold_selection_feasible": bool(
            threshold_feasible
        ),
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "model_comparison": results,
        "random_state": RANDOM_STATE,
        "split": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": 2000,
        },
        "test_evaluated": False,
    }

    print(f"Selected candidate: {best_name}")
    print(f"Threshold: {threshold:.6f}")
    print(
        "Validation precision:",
        f"{validation_metrics['precision']:.4f}"
    )
    print(
        "Validation recall:",
        f"{validation_metrics['recall']:.4f}"
    )
    print("Business criteria met:", business_criteria_met)
    print("Deployment approved:", deployment_approved)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_FILE)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved report: {REPORT_FILE}")
    print("SUCCESS: Training stage completed.")
    print("Test dataset was not accessed.")


if __name__ == "__main__":
    main()
