
"""Churn model inference service for local development."""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.schemas.request import ChurnRequest

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/selected_churn_model.joblib"
SELECTION_PATH = ROOT / "artifacts/model_selection.json"
LINEAGE_PATH = ROOT / "artifacts/model_lineage.json"
SCHEMA_PATH = ROOT / "artifacts/feature_schema.json"


@lru_cache(maxsize=1)
def load_model_bundle():
    """Load the trusted local model and its frozen metadata."""

    with SELECTION_PATH.open(encoding="utf-8") as file:
        selection = json.load(file)

    with LINEAGE_PATH.open(encoding="utf-8") as file:
        lineage = json.load(file)

    with SCHEMA_PATH.open(encoding="utf-8") as file:
        schema = json.load(file)

    if selection["selected_model"] != lineage["model_type"]:
        raise RuntimeError("Model selection and lineage disagree")

    threshold = float(selection["threshold"])

    if threshold != float(lineage["decision_threshold"]):
        raise RuntimeError("Frozen thresholds do not match")

    if selection["deployment_approved"] or lineage["deployment_approved"]:
        raise RuntimeError(
            "Unexpected deployment approval: review promotion policy"
        )

    model = joblib.load(MODEL_PATH)

    return model, threshold, list(schema["features"])


def predict(payload: ChurnRequest) -> dict:
    """Return a churn probability using the frozen model pipeline."""

    if not isinstance(payload, ChurnRequest):
        raise TypeError("Expected a validated ChurnRequest")

    model, threshold, features = load_model_bundle()

    data = payload.model_dump()
    frame = pd.DataFrame([data], columns=features)

    # The fitted numeric imputer handles missing age.
    frame["age"] = pd.to_numeric(
        frame["age"], errors="coerce"
    )

    probabilities = model.predict_proba(frame)
    probability = float(probabilities[0, 1])

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Model returned an invalid probability")

    return {
        "churn_probability": probability,
        "churn_prediction": int(probability >= threshold),
        "threshold": threshold,
    }
