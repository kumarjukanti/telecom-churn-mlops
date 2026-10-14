
"""FastAPI application for telecom churn model demonstrations."""

import json
from pathlib import Path

from fastapi import FastAPI

from api.schemas.request import ChurnRequest
from api.services.model_service import predict

ROOT = Path(__file__).resolve().parents[1]
LINEAGE_PATH = ROOT / "artifacts/model_lineage.json"

app = FastAPI(
    title="Telecom Churn API",
    version="1.0.0",
    description=(
        "Development API for telecom customer churn prediction. "
        "The current model is not approved for production deployment."
    ),
)


@app.get("/health")
def health():
    """Report that the API process is responding."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Return non-sensitive lineage and model status metadata."""
    with LINEAGE_PATH.open(encoding="utf-8") as file:
        lineage = json.load(file)

    return {
        "model_name": lineage["registered_model_name"],
        "model_version": lineage["registered_model_version"],
        "model_type": lineage["model_type"],
        "lifecycle_status": lineage["lifecycle_status"],
        "deployment_approved": lineage["deployment_approved"],
        "feature_schema_version": lineage["feature_schema_version"],
        "training_timestamp": lineage["training_started_at_utc"],
        "threshold": lineage["decision_threshold"],
        "test_pr_auc": lineage["test_metrics"]["pr_auc"],
        "test_roc_auc": lineage["test_metrics"]["roc_auc"],
    }


@app.post("/predict")
def churn(req: ChurnRequest):
    """Predict churn using the validated request and frozen model."""
    return predict(req)
