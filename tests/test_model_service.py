
import pytest

from api.schemas.request import ChurnRequest
from api.services.model_service import predict


@pytest.fixture
def customer():
    return ChurnRequest(
        age=45,
        gender="Male",
        region="Urban",
        contract_type="Monthly",
        monthly_charges=75.0,
        months_observed=12,
        avg_call_minutes=350.0,
        avg_data_usage_gb=15.0,
        avg_sms_count=100.0,
        total_complaints=2.0,
        avg_complaints=0.17,
        tenure_months=36.0,
        complaint_rate=0.17,
        charges_per_tenure_month=2.08,
    )


def test_predict_returns_expected_fields(customer):
    result = predict(customer)

    assert set(result) == {
        "churn_probability",
        "churn_prediction",
        "threshold",
    }


def test_probability_is_valid(customer):
    result = predict(customer)

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in (0, 1)


def test_uses_frozen_threshold(customer):
    result = predict(customer)

    assert result["threshold"] == 0.26234641106951156
    assert result["churn_prediction"] == int(
        result["churn_probability"] >= result["threshold"]
    )


def test_missing_age_supported(customer):
    payload = customer.model_dump()
    payload["age"] = None

    result = predict(ChurnRequest(**payload))

    assert 0 <= result["churn_probability"] <= 1


def test_rejects_unvalidated_dictionary(customer):
    with pytest.raises(TypeError):
        predict(customer.model_dump())
