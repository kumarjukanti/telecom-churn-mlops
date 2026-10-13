
from pathlib import Path
import argparse
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "db/telecom.db"
OUTPUT = ROOT / "data/processed/features.csv"


def build_features(df, as_of_date):
    x = df.copy()

    x["signup_date"] = pd.to_datetime(
        x["signup_date"], errors="coerce"
    )

    if x["signup_date"].isna().any():
        raise ValueError("Invalid signup dates found")

    cutoff = pd.Timestamp(as_of_date)

    if (x["signup_date"] > cutoff).any():
        raise ValueError(
            "Some customers signed up after the cutoff date"
        )

    x["tenure_days"] = (
        cutoff - x["signup_date"]
    ).dt.days

    x["tenure_months"] = x["tenure_days"] / 30.44

    x["complaint_rate"] = (
        x["total_complaints"] /
        x["months_observed"].clip(lower=1)
    )

    x["charges_per_tenure_month"] = (
        x["monthly_charges"] /
        x["tenure_months"].clip(lower=1)
    )

    x = x.drop(columns=["signup_date", "tenure_days"])

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of-date",
        required=True
    )
    args = parser.parse_args()

    print("STEP 1: Reading SQLite database...")

    with sqlite3.connect(DB) as con:
        df = pd.read_sql_query(
            "SELECT * FROM model_source", con
        )

    print(f"Rows loaded: {len(df)}")

    if df.empty:
        raise ValueError("No data found in model_source")

    print("STEP 2: Building features...")

    features = build_features(df, args.as_of_date)

    print("STEP 3: Saving features.csv...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(OUTPUT, index=False)

    print(f"Rows exported: {len(features)}")
    print(f"File saved: {OUTPUT}")
    print("SUCCESS: Feature engineering completed!")


if __name__ == "__main__":
    main()
