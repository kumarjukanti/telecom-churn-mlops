
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_loads_and_predicts():
    model_path = (
        ROOT / "models/selected_churn_model.joblib"
    )

    assert model_path.exists()

    # Load only our locally generated trusted artifact.
    model = joblib.load(model_path)

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    df = pd.read_parquet(
        ROOT / "data/processed/validation.parquet"
    )

    from src.models.compare_models import FEATURES

    probabilities = model.predict_proba(
        df[FEATURES].head(5)
    )

    assert probabilities.shape == (5, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (
        probabilities <= 1
    )).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(5),
    )
