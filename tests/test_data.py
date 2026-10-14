
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"

EXPECTED_COLUMNS = {
    "customer_id",
    "churn",
    "age",
    "gender",
    "region",
    "contract_type",
    "tenure_months",
    "complaint_rate",
    "charges_per_tenure_month",
}


@pytest.mark.parametrize(
    "name,expected_rows",
    [
        ("train", 6400),
        ("validation", 1600),
        ("test", 2000),
    ],
)
def test_dataset_schema(name, expected_rows):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    assert len(df) == expected_rows
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert df["customer_id"].is_unique
    assert df["customer_id"].notna().all()
    assert df["churn"].isin([0, 1]).all()


def test_no_customer_overlap():
    ids = {}

    for name in ["train", "validation", "test"]:
        df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
        ids[name] = set(df["customer_id"])

    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
