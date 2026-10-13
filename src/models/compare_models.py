
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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/features.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"

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


# =====================================================
# 2. PREPROCESSING PIPELINE
# =====================================================

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


# =====================================================
# 3. MODEL RANKING METRICS
# =====================================================

def evaluate_model(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
    }


# =====================================================
# 4. THRESHOLD SELECTION
# =====================================================

def choose_threshold(y_true, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(y_true, probabilities)
    )

    valid = precision[:-1] >= MIN_PRECISION

    if valid.any():
        candidates = np.flatnonzero(valid)

        best_index = candidates[
            np.argmax(recall[candidates])
        ]

        threshold = float(thresholds[best_index])

        return threshold, True

    # Diagnostic fallback only.
    # This does not approve deployment.
    best_index = int(np.argmax(precision[:-1]))

    threshold = float(thresholds[best_index])

    print(
        "WARNING: No threshold meets minimum precision."
    )

    return threshold, False


# =====================================================
# 5. EVALUATION AND CONFUSION MATRIX
# =====================================================

def threshold_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

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
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


# =====================================================
# 6. MAIN WORKFLOW
# =====================================================

def main():

    print("\nSTEP 1: Loading features...")

    df = pd.read_csv(DATA)

    required = NUMERIC + CATEGORICAL + ["churn"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["churn"].isna().any():
        raise ValueError("Missing churn labels")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must be 0 or 1")

    X = df[NUMERIC + CATEGORICAL]
    y = df["churn"].astype(int)

    print(f"Total customers: {len(df)}")

    # -------------------------------------------------
    # STEP 2: TRAIN / VALIDATION / TEST SPLIT
    # -------------------------------------------------

    print("\nSTEP 2: Splitting data...")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    print(f"Training: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test reserved: {len(X_test)}")

    # -------------------------------------------------
    # STEP 3: CLASS PREVALENCE BASELINE
    # -------------------------------------------------

    print("\nSTEP 3: Calculating churn baseline...")

    class_prevalence = float(y_val.mean())

    print(
        f"Validation churn prevalence: "
        f"{class_prevalence:.4f}"
    )

    print(
        f"PR-AUC baseline: {class_prevalence:.4f}"
    )

    # -------------------------------------------------
    # STEP 4: TRAIN THREE MODELS
    # -------------------------------------------------

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

    print("\nSTEP 4: Training and comparing models...")

    for name, classifier in candidates.items():

        print(f"\nTraining {name}...")

        pipeline = make_pipeline(classifier)

        pipeline.fit(X_train, y_train)

        scores = evaluate_model(
            pipeline, X_val, y_val
        )

        results[name] = scores
        trained_models[name] = pipeline

        print(f"ROC-AUC: {scores['roc_auc']:.4f}")
        print(f"PR-AUC:  {scores['pr_auc']:.4f}")

    # -------------------------------------------------
    # STEP 5: SELECT BEST CANDIDATE
    # -------------------------------------------------

    print("\nSTEP 5: Selecting best candidate...")

    best_name = max(
        results,
        key=lambda name: results[name]["pr_auc"],
    )

    best_model = trained_models[best_name]

    print(f"Selected candidate: {best_name}")

    # -------------------------------------------------
    # STEP 6: SELECT THRESHOLD
    # -------------------------------------------------

    probabilities = best_model.predict_proba(
        X_val
    )[:, 1]

    threshold, threshold_feasible = choose_threshold(
        y_val, probabilities
    )

    metrics = threshold_metrics(
        y_val, probabilities, threshold
    )

    # -------------------------------------------------
    # STEP 7: BUSINESS ACCEPTANCE CHECK
    # -------------------------------------------------

    business_criteria_met = bool(
        threshold_feasible
        and metrics["precision"] >= MIN_PRECISION
    )

    # Conservative default:
    # The model must not be deployed merely because
    # it was the best model in the comparison.
    deployment_approved = False

    print("\nSTEP 6: Validation results...")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(
        "Business criteria met:",
        business_criteria_met
    )

    print(
        "Deployment approved:",
        deployment_approved
    )

    if not business_criteria_met:
        print(
            "\nWARNING: Business acceptance criteria "
            "were NOT met."
        )

    # -------------------------------------------------
    # STEP 8: SAVE MODEL AND RESULTS
    # -------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True, exist_ok=True
    )

    ARTIFACT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    model_path = (
        MODEL_DIR / "selected_churn_model.joblib"
    )

    report_path = (
        ARTIFACT_DIR / "model_selection.json"
    )

    joblib.dump(best_model, model_path)

    report = {
        "selected_model": best_name,
        "selection_metric": "average_precision",
        "minimum_precision": MIN_PRECISION,
        "class_prevalence": class_prevalence,
        "pr_auc_baseline": class_prevalence,
        "precision_requirement_met": bool(
            metrics["precision"] >= MIN_PRECISION
        ),
        "business_criteria_met": business_criteria_met,
        "deployment_approved": deployment_approved,
        "threshold_selection_feasible": threshold_feasible,
        "threshold": threshold,
        "validation_metrics": metrics,
        "model_comparison": results,
        "random_state": RANDOM_STATE,
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "test_evaluated": False,
    }

    with open(
        report_path, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved report: {report_path}")

    print("\nSUCCESS: Evaluation report generated.")
    print("Test dataset remains untouched.")


if __name__ == "__main__":
    main()
