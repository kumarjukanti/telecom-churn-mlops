
"""Create and optionally publish selected-model lineage metadata."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from mlflow import MlflowClient

from src.models.compare_models import FEATURES

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "9bc25a76a0174fd295c194db8481229b"
LOGGED_MODEL_ID = "m-5751ae2016c34628bd8a0309eb1f6bfc"
MODEL_NAME = "telecom-churn-model"
MODEL_VERSION = "1"

OUTPUT = ROOT / "artifacts/model_lineage.json"


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def md5_file(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_hash(stage, group, path):
    for item in stage[group]:
        if item["path"] == path:
            return item["md5"]
    raise ValueError(f"Missing DVC entry: {path}")


def main(sync_mlflow=False):
    selection = load_json(ROOT / "artifacts/model_selection.json")
    evaluation = load_json(ROOT / "reports/metrics.json")
    schema = load_json(ROOT / "artifacts/feature_schema.json")

    with (ROOT / "dvc.lock").open(encoding="utf-8") as file:
        lock = yaml.safe_load(file)["stages"]

    client = MlflowClient()
    run = client.get_run(RUN_ID)
    version = client.get_model_version(MODEL_NAME, MODEL_VERSION)

    if run.info.status != "FINISHED":
        raise ValueError("Selected MLflow run is not finished")

    if version.run_id and version.run_id != RUN_ID:
        raise ValueError("Registered version belongs to another run")

    if not version.run_id and LOGGED_MODEL_ID not in version.source:
        raise ValueError("Cannot verify registered model source")

    if set(schema["features"]) != set(FEATURES):
        raise ValueError("Feature schema does not match training features")

    if selection["selected_model"] != "HistGradientBoosting":
        raise ValueError("Unexpected selected model")

    if evaluation["selected_model"] != selection["selected_model"]:
        raise ValueError("Evaluation model does not match selection")

    threshold = selection["threshold"]
    test_metrics = evaluation["test_metrics"]

    if threshold != test_metrics["threshold"]:
        raise ValueError("Validation and test thresholds differ")

    if threshold != run.data.metrics["threshold"]:
        raise ValueError("MLflow threshold does not match selection")

    if selection["deployment_approved"] or evaluation["deployment_approved"]:
        raise ValueError("Unexpected production approval")

    paths = {
        "train": ("features", "outs", "data/processed/train.parquet"),
        "validation": (
            "features",
            "outs",
            "data/processed/validation.parquet",
        ),
        "test": ("features", "outs", "data/processed/test.parquet"),
        "model": (
            "train",
            "outs",
            "models/selected_churn_model.joblib",
        ),
        "selection": (
            "train",
            "outs",
            "artifacts/model_selection.json",
        ),
        "test_metrics": (
            "evaluate",
            "outs",
            "reports/metrics.json",
        ),
    }

    hashes = {}

    for name, (stage, group, path) in paths.items():
        expected = find_hash(lock[stage], group, path)
        actual = md5_file(ROOT / path)

        if actual != expected:
            raise ValueError(f"DVC hash mismatch: {path}")

        hashes[name] = expected

    training_sha = run.data.tags.get("mlflow.source.git.commit")

    if not training_sha:
        raise ValueError("Training Git SHA missing from MLflow")

    aliases = list(version.aliases or [])

    if "champion" in aliases:
        raise ValueError("Unapproved model has champion alias")

    started_at = datetime.fromtimestamp(
        run.info.start_time / 1000,
        tz=UTC,
    ).isoformat()

    lineage = {
        "registered_model_name": MODEL_NAME,
        "registered_model_version": MODEL_VERSION,
        "registered_model_aliases": aliases,
        "model_type": selection["selected_model"],
        "lifecycle_status": "development_candidate",
        "deployment_approved": False,
        "training_git_sha": training_sha,
        "mlflow_run_id": RUN_ID,
        "mlflow_logged_model_id": LOGGED_MODEL_ID,
        "training_started_at_utc": started_at,
        "dvc_hashes": hashes,
        "feature_schema_version": schema["schema_version"],
        "feature_build_version": schema["feature_build_version"],
        "feature_schema_md5": md5_file(
            ROOT / "artifacts/feature_schema.json"
        ),
        "decision_threshold": threshold,
        "validation_pr_auc": run.data.metrics["pr_auc"],
        "test_metrics": test_metrics,
        "limitations": evaluation["limitations"],
        "provenance_notes": [
    (
        "Feature schema created after this training run; "
        "feature names match, historical schema approval unverified."
    ),
    (
        "DVC lock records historical pipeline dependencies; "
        "source files were subsequently reformatted."
    ),
    (
        "Registered model and local joblib artifact equivalence "
        "has not been independently verified."
    ),
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(lineage, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Lineage written to {OUTPUT}")
    print(json.dumps(lineage, indent=2))

    if sync_mlflow:
        tags = {
            "lineage.training_git_sha": training_sha,
            "lineage.mlflow_run_id": RUN_ID,
            "lineage.logged_model_id": LOGGED_MODEL_ID,
            "lineage.feature_schema_version": schema["schema_version"],
            "lineage.feature_build_version": schema["feature_build_version"],
            "lineage.training_timestamp": started_at,
            "lineage.threshold": str(threshold),
            "lineage.dvc_train_md5": hashes["train"],
            "lineage.dvc_validation_md5": hashes["validation"],
            "lineage.dvc_test_md5": hashes["test"],
            "lineage.dvc_model_md5": hashes["model"],
            "lineage.test_pr_auc": str(test_metrics["pr_auc"]),
            "lineage.test_roc_auc": str(test_metrics["roc_auc"]),
            "lineage.test_precision": str(test_metrics["precision"]),
            "lineage.test_recall": str(test_metrics["recall"]),
            "lineage.deployment_approved": "false",
            "lineage.status": "development_candidate",
        }

        for key, value in tags.items():
            client.set_tag(RUN_ID, key, value)
            client.set_model_version_tag(
                MODEL_NAME, MODEL_VERSION, key, value
            )

        client.log_artifact(
            RUN_ID,
            str(OUTPUT),
            artifact_path="lineage",
        )

        print("MLflow run and model-version tags updated.")
        print("Lineage JSON uploaded as a run artifact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync-mlflow",
        action="store_true",
        help="Publish verified lineage to MLflow",
    )
    args = parser.parse_args()
    main(sync_mlflow=args.sync_mlflow)
