
"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "telecom-churn-model"
    assert data["model_version"] == "1"
    assert data["feature_schema_version"] == "1.0"
    assert data["deployment_approved"] is False
    assert data["threshold"] == 0.26234641106951156


def test_predict(valid_customer):
    response = client.post(
        "/predict",
        json=valid_customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in (0, 1)
    assert data["threshold"] == 0.26234641106951156


def test_predict_missing_age(valid_customer):
    valid_customer.pop("age")

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1


def test_predict_invalid_age(valid_customer):
    valid_customer["age"] = 15

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_predict_invalid_category(valid_customer):
    valid_customer["contract_type"] = "Unknown"

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_predict_missing_required_field(valid_customer):
    valid_customer.pop("monthly_charges")

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422
