
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTION_FILE = ROOT / "artifacts/model_selection.json"

MIN_PRECISION = 0.50
MIN_RECALL = 0.20


def check_promotion(
    report,
    tests_passed=False,
    schema_compatible=False,
    leakage_checks_passed=False,
    champion_pr_auc=None,
):
    """
    Evaluate the candidate's promotion eligibility.

    This function does not register a model,
    assign an MLflow alias, or deploy anything.
    """

    metrics = report["validation_metrics"]
    candidate_pr_auc = report["model_comparison"][
        report["selected_model"]
    ]["pr_auc"]

    checks = {
        "precision": (
            metrics["precision"] >= MIN_PRECISION
        ),
        "recall": (
            metrics["recall"] >= MIN_RECALL
        ),
        "business_criteria": bool(
            report["business_criteria_met"]
        ),
        "unit_and_api_tests": bool(tests_passed),
        "schema_compatible": bool(schema_compatible),
        "leakage_checks": bool(leakage_checks_passed),
        "champion_comparison": (
            True
            if champion_pr_auc is None
            else candidate_pr_auc >= champion_pr_auc
        ),
    }

    return {
        "selected_model": report["selected_model"],
        "candidate_pr_auc": candidate_pr_auc,
        "minimum_precision": MIN_PRECISION,
        "minimum_recall": MIN_RECALL,
        "champion_pr_auc": champion_pr_auc,
        "checks": checks,
        "eligible_for_promotion": all(checks.values()),
    }


def main():
    with SELECTION_FILE.open("r", encoding="utf-8") as f:
        report = json.load(f)

    # Conservative defaults:
    # Nothing is considered verified until
    # the corresponding process confirms it.
    result = check_promotion(report)

    print(json.dumps(result, indent=2))

    if not result["eligible_for_promotion"]:
        print("\nPROMOTION BLOCKED")
        print("No MLflow champion alias was modified.")


if __name__ == "__main__":
    main()
