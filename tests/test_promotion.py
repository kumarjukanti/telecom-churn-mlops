
from src.models.promotion_policy import check_promotion


def make_report(precision, recall):
    return {
        "selected_model": "ExampleModel",
        "business_criteria_met": precision >= 0.50,
        "validation_metrics": {
            "precision": precision,
            "recall": recall,
        },
        "model_comparison": {
            "ExampleModel": {"pr_auc": 0.15}
        },
    }


def test_low_precision_blocks_promotion():
    result = check_promotion(
        make_report(precision=0.08, recall=0.40),
        tests_passed=True,
        schema_compatible=True,
        leakage_checks_passed=True,
    )

    assert result["eligible_for_promotion"] is False
    assert result["checks"]["precision"] is False


def test_low_recall_blocks_promotion():
    result = check_promotion(
        make_report(precision=0.60, recall=0.05),
        tests_passed=True,
        schema_compatible=True,
        leakage_checks_passed=True,
    )

    assert result["eligible_for_promotion"] is False
    assert result["checks"]["recall"] is False


def test_missing_checks_block_promotion():
    result = check_promotion(
        make_report(precision=0.60, recall=0.40)
    )

    assert result["eligible_for_promotion"] is False
    assert result["checks"]["unit_and_api_tests"] is False


def test_worse_than_champion_blocks_promotion():
    result = check_promotion(
        make_report(precision=0.60, recall=0.40),
        tests_passed=True,
        schema_compatible=True,
        leakage_checks_passed=True,
        champion_pr_auc=0.20,
    )

    assert result["eligible_for_promotion"] is False
    assert result["checks"]["champion_comparison"] is False
