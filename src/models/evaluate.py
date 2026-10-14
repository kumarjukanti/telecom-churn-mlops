
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = ROOT / "data/processed/test.parquet"
MODEL_FILE = ROOT / "models/selected_churn_model.joblib"
SELECTION_FILE = ROOT / "artifacts/model_selection.json"

REPORT_DIR = ROOT / "reports"
METRICS_FILE = REPORT_DIR / "metrics.json"
CONFUSION_PLOT = REPORT_DIR / "confusion_matrix.png"
ROC_PLOT = REPORT_DIR / "roc_curve.png"

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


# =====================================================
# 2. EVALUATION
# =====================================================

def main():

    print("STEP 1: Loading frozen model and metadata...")

    for path in [MODEL_FILE, SELECTION_FILE, TEST_FILE]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    # Only load model artifacts created by trusted code.
    model = joblib.load(MODEL_FILE)

    with SELECTION_FILE.open("r", encoding="utf-8") as f:
        selection = json.load(f)

    threshold = float(selection["threshold"])

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid stored threshold")

    selected_model = selection["selected_model"]

    print(f"Selected model: {selected_model}")
    print(f"Frozen threshold: {threshold:.6f}")

    print("\nSTEP 2: Checking deployment eligibility...")

    business_criteria_met = bool(
        selection["business_criteria_met"]
    )

    validation_approval = bool(
        selection["deployment_approved"]
    )

    # This stage must never grant deployment approval.
    deployment_approved = False

    if not business_criteria_met:
        print(
            "WARNING: Model failed validation "
            "business criteria."
        )

    if validation_approval:
        print(
            "NOTE: Evaluation alone cannot authorize "
            "production deployment."
        )

    print("\nSTEP 3: Loading reserved test dataset...")

    test_df = pd.read_parquet(TEST_FILE)

    required_columns = FEATURES + ["customer_id", "churn"]

    missing = sorted(
        set(required_columns) - set(test_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing test columns: {missing}"
        )

    if test_df.empty:
        raise ValueError("Test dataset is empty")

    if test_df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs in test data")

    if test_df["churn"].isna().any():
        raise ValueError("Missing test labels")

    if not test_df["churn"].isin([0, 1]).all():
        raise ValueError("Test labels must be 0 or 1")

    X_test = test_df[FEATURES]
    y_test = test_df["churn"].astype(int)

    if y_test.nunique() != 2:
        raise ValueError(
            "Test dataset must contain both classes"
        )

    print(f"Test customers: {len(test_df)}")
    print(f"Actual churners: {int(y_test.sum())}")

    print("\nSTEP 4: Generating predictions...")

    # Do not call fit() here.
    # This is evaluation of the frozen trained pipeline.
    probabilities = model.predict_proba(X_test)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities returned")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities outside [0, 1]")

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("\nSTEP 5: Calculating metrics...")

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    test_metrics = {
        "customers": len(y_test),
        "actual_churners": int(y_test.sum()),
        "class_prevalence": float(y_test.mean()),
        "threshold": threshold,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    for name in [
        "accuracy", "precision", "recall",
        "f1", "roc_auc", "pr_auc"
    ]:
        print(f"{name}: {test_metrics[name]:.4f}")

    print(
        f"Confusion matrix: "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )

    print("\nSTEP 6: Generating evaluation charts...")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Retained", "Churn"]
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Telecom Churn — Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=selected_model,
        ax=ax,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier"
    )

    ax.set_title("Telecom Churn — Test ROC Curve")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(ROC_PLOT, dpi=160)
    plt.close(fig)

    print("\nSTEP 7: Saving evaluation report...")

    report = {
        "selected_model": selected_model,
        "evaluation_dataset": "test",
        "test_evaluated": True,
        "threshold_frozen_from_validation": True,
        "minimum_precision": float(
            selection["minimum_precision"]
        ),
        "validation_business_criteria_met": (
            business_criteria_met
        ),
        "business_criteria_met": False,
        "deployment_approved": deployment_approved,
        "evaluation_purpose": "experimental_benchmark",
        "test_metrics": test_metrics,
        "limitations": [
        (
        "Validation precision requirement not met"
        if not business_criteria_met
        else "Production authorization not established"
        ),
        (
        "Churn label timing relative to feature "
        "cutoff has not been verified"
        ),
        (
        "Test metrics must not be used for "
        "further model or threshold tuning"
        ),
        ],
    }

    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Metrics: {METRICS_FILE}")
    print(f"Confusion matrix: {CONFUSION_PLOT}")
    print(f"ROC curve: {ROC_PLOT}")

    print("\nSUCCESS: Final test evaluation completed.")
    print("Deployment approved: False")


if __name__ == "__main__":
    main()
