
import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
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

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data/processed/train.parquet"
VALIDATION_FILE = ROOT / "data/processed/validation.parquet"

MODEL_FILE = ROOT / "models/selected_churn_model.joblib"
REPORT_FILE = ROOT / "artifacts/model_selection.json"

EXPERIMENT_NAME = "telecom-churn"
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


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def file_sha256(path):
    """Record exact dataset file identity."""
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


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
        raise ValueError(f"{name}: empty dataset")

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
            f"{name}: both churn classes required"
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

    # Diagnostic fallback; NOT production approval.
    best_index = int(np.argmax(precision[:-1]))

    return float(thresholds[best_index]), False


def evaluate_at_threshold(
    y_true, probabilities, threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
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


# ==================================================
# MAIN WORKFLOW
# ==================================================

def main():

    print("STEP 1: Connecting to MLflow...")

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is missing. "
            "Export the tracking URI before running."
        )

    mlflow.set_tracking_uri(tracking_uri)

    # Fail early rather than completing training
    # without a functioning tracking server.
    mlflow.MlflowClient().search_experiments()
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", EXPERIMENT_NAME)

    print("\nSTEP 2: Loading DVC datasets...")

    train_df = pd.read_parquet(TRAIN_FILE)
    val_df = pd.read_parquet(VALIDATION_FILE)

    validate_dataset(train_df, "train")
    validate_dataset(val_df, "validation")

    if set(train_df["customer_id"]) & set(
        val_df["customer_id"]
    ):
        raise ValueError(
            "Training and validation customer IDs overlap"
        )

    X_train = train_df[FEATURES]
    y_train = train_df["churn"].astype(int)

    X_val = val_df[FEATURES]
    y_val = val_df["churn"].astype(int)

    prevalence = float(y_val.mean())

    train_hash = file_sha256(TRAIN_FILE)
    val_hash = file_sha256(VALIDATION_FILE)

    print("Training customers:", len(train_df))
    print("Validation customers:", len(val_df))

    print("\nSTEP 3: Training and logging models...")

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
    run_ids = {}

    for name, classifier in candidates.items():

        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name) as run:

            run_ids[name] = run.info.run_id

            mlflow.set_tags({
                "project": "telecom-churn-mlops",
                "dataset": "telecom-churn",
                "evaluation_split": "validation",
                "model_candidate": name,
                "production_approved": "false",
            })

            params = {
                "model_type": type(
                    classifier
                ).__name__,
                "random_state": RANDOM_STATE,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "train_sha256": train_hash,
                "validation_sha256": val_hash,
            }

            # Include model-specific hyperparameters.
            for key, value in (
                classifier.get_params(deep=False).items()
            ):
                params[f"model__{key}"] = str(value)

            mlflow.log_params(params)

            pipeline = make_pipeline(classifier)

            # Preprocessing fits only on training data.
            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(
                X_val
            )[:, 1]

            scores = {
                "roc_auc": float(
                    roc_auc_score(y_val, probabilities)
                ),
                "pr_auc": float(
                    average_precision_score(
                        y_val, probabilities
                    )
                ),
                "class_prevalence": prevalence,
            }

            mlflow.log_metrics(scores)

            # Log the complete fitted preprocessing
            # and classification pipeline.
            # Do not register it for production.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="pipeline",
                serialization_format="cloudpickle",
            )

            results[name] = {
                "roc_auc": scores["roc_auc"],
                "pr_auc": scores["pr_auc"],
            }

            trained_models[name] = pipeline

            print(
                f"ROC-AUC: {scores['roc_auc']:.4f}"
            )
            print(
                f"Average Precision: "
                f"{scores['pr_auc']:.4f}"
            )
            print("MLflow run ID:", run.info.run_id)

    print("\nSTEP 4: Selecting best candidate...")

    best_name = max(
        results,
        key=lambda name: results[name]["pr_auc"],
    )

    best_model = trained_models[best_name]

    best_probabilities = best_model.predict_proba(
        X_val
    )[:, 1]

    threshold, feasible = choose_threshold(
        y_val, best_probabilities
    )

    validation_metrics = evaluate_at_threshold(
        y_val,
        best_probabilities,
        threshold,
    )

    precision_met = bool(
        feasible
        and validation_metrics["precision"]
        >= MIN_PRECISION
    )

    business_criteria_met = precision_met

    # Training never authorizes deployment.
    deployment_approved = False

    print("Selected model:", best_name)
    print("Threshold:", threshold)
    print(
        "Business criteria met:",
        business_criteria_met
    )

    print("\nSTEP 5: Saving DVC outputs...")

    MODEL_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    REPORT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(best_model, MODEL_FILE)

    report = {
        "selected_model": best_name,
        "selection_metric": "average_precision",
        "minimum_precision": MIN_PRECISION,
        "class_prevalence": prevalence,
        "pr_auc_baseline": prevalence,
        "precision_requirement_met": precision_met,
        "business_criteria_met": business_criteria_met,
        "deployment_approved": deployment_approved,
        "threshold_selection_feasible": bool(feasible),
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

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSTEP 6: Logging selection results...")

    # Reopen the selected model's MLflow run
    # to attach threshold and acceptance metrics.
    with mlflow.start_run(
        run_id=run_ids[best_name]
    ):

        mlflow.set_tags({
            "selected_candidate": "true",
            "business_criteria_met": str(
                business_criteria_met
            ).lower(),
            "deployment_approved": "false",
        })

        mlflow.log_metrics({
            "threshold": threshold,
            "precision": validation_metrics["precision"],
            "recall": validation_metrics["recall"],
            "f1": validation_metrics["f1"],
            "accuracy": validation_metrics["accuracy"],
            "minimum_precision": MIN_PRECISION,
        })

        mlflow.log_artifact(str(REPORT_FILE))

    print("\nSUCCESS: Three MLflow runs recorded.")
    print("Selected model:", best_name)
    print("Saved model:", MODEL_FILE)
    print("Saved report:", REPORT_FILE)
    print("Test dataset was not accessed.")


if __name__ == "__main__":
    main()
