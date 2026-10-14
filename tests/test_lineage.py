
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_lineage_matches_existing_artifacts():
    with (ROOT / "artifacts/model_lineage.json").open() as f:
        lineage = json.load(f)

    with (ROOT / "artifacts/model_selection.json").open() as f:
        selection = json.load(f)

    with (ROOT / "reports/metrics.json").open() as f:
        evaluation = json.load(f)

    with (ROOT / "artifacts/feature_schema.json").open() as f:
        schema = json.load(f)

    with (ROOT / "dvc.lock").open() as f:
        stages = yaml.safe_load(f)["stages"]

    assert lineage["model_type"] == selection["selected_model"]
    assert lineage["feature_schema_version"] == schema["schema_version"]

    assert lineage["decision_threshold"] == selection["threshold"]
    assert lineage["decision_threshold"] == (
        evaluation["test_metrics"]["threshold"]
    )

    assert lineage["test_metrics"] == evaluation["test_metrics"]

    assert lineage["deployment_approved"] is False
    assert "champion" not in lineage["registered_model_aliases"]

    train = next(
        item
        for item in stages["features"]["outs"]
        if item["path"] == "data/processed/train.parquet"
    )

    assert lineage["dvc_hashes"]["train"] == train["md5"]
