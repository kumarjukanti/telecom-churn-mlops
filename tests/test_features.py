
import numpy as np
import pandas as pd

from src.features.build_features import build_features
from src.models.compare_models import FEATURES, NUMERIC
from src.models.promotion_policy import ROOT


def test_feature_engineering_is_deterministic():
    source = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "signup_date": ["2022-01-01", "2023-01-01"],
        "monthly_charges": [50.0, 70.0],
        "total_complaints": [4, 2],
        "months_observed": [12, 12],
        "churn": [0, 1],
    })

    first = build_features(source, "2023-12-31")
    second = build_features(source, "2023-12-31")

    pd.testing.assert_frame_equal(first, second)


def test_numeric_features_have_no_infinity():
    df = pd.read_parquet(
        ROOT / "data/processed/train.parquet"
    )

    numeric = df[NUMERIC].to_numpy(dtype=float)

    # Missing age values (NaN) are allowed because
    # the model uses median imputation.
    # Positive and negative infinity are not allowed.
    assert not np.isinf(numeric).any()


def test_preprocessed_features_are_finite():
    import joblib

    model = joblib.load(
        ROOT / "models/selected_churn_model.joblib"
    )

    df = pd.read_parquet(
        ROOT / "data/processed/train.parquet"
    )

    transformed = model.named_steps[
        "preprocessor"
    ].transform(df[FEATURES].head(100))

    assert np.isfinite(
        np.asarray(transformed, dtype=float)
    ).all()


def test_target_not_in_model_inputs():
    assert "churn" not in FEATURES
    assert "customer_id" not in FEATURES
    assert "signup_date" not in FEATURES
