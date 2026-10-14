
import pytest
from pydantic import ValidationError

from api.schemas.request import ChurnRequest


@pytest.fixture
def valid_customer():
    return {
        "age": 45,
        "gender": "Male",
        "region": "Urban",
        "contract_type": "Monthly",
        "monthly_charges": 75.0,
        "months_observed": 12,
        "avg_call_minutes": 350.0,
        "avg_data_usage_gb": 15.0,
        "avg_sms_count": 100.0,
        "total_complaints": 2.0,
        "avg_complaints": 0.17,
        "tenure_months": 36.0,
        "complaint_rate": 0.17,
        "charges_per_tenure_month": 2.08,
    }


def test_valid_customer(valid_customer):
    request = ChurnRequest(**valid_customer)
    assert request.age == 45
    assert len(request.model_dump()) == 14


def test_missing_age_allowed(valid_customer):
    valid_customer.pop("age")
    request = ChurnRequest(**valid_customer)
    assert request.age is None


def test_invalid_age_rejected(valid_customer):
    valid_customer["age"] = 15

    with pytest.raises(ValidationError):
        ChurnRequest(**valid_customer)


def test_negative_charges_rejected(valid_customer):
    valid_customer["monthly_charges"] = -10

    with pytest.raises(ValidationError):
        ChurnRequest(**valid_customer)


def test_invalid_contract_rejected(valid_customer):
    valid_customer["contract_type"] = "Unknown"

    with pytest.raises(ValidationError):
        ChurnRequest(**valid_customer)


def test_extra_field_rejected(valid_customer):
    valid_customer["unexpected_field"] = "test"

    with pytest.raises(ValidationError):
        ChurnRequest(**valid_customer)
