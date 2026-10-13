
from pathlib import Path
import argparse
import sqlite3
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "db/telecom.db"
OUTPUT = ROOT / "data/processed/features.csv"


def build_features(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """Create business-readable customer churn features."""

    x = df.copy()

    # Convert signup date to datetime
    x["signup_date"] = pd.to_datetime(
        x["signup_date"], errors="coerce"
    )

    if x["signup_date"].isna().any():
        raise ValueError("Invalid or missing signup dates found")

    cutoff = pd.Timestamp(as_of_date)

    if (x["signup_date"] > cutoff).any():
        raise ValueError(
            "Some customers signed up after the cutoff date"
        )

    # Calculate customer tenure
    x["tenure_days"] = (
        cutoff - x["signup_date"]
    ).dt.days

    x["tenure_months"] = x["tenure_days"] / 30.44

    # Calculate complaints per observed month
    x["complaint_rate"] = (
        x["total_complaints"]
        / x["months_observed"].clip(lower=1)
    )

    # Calculate monthly charges relative to tenure
    x["charges_per_tenure_month"] = (
        x["monthly_charges"]
        / x["tenure_months"].clip(lower=1)
    )

    # Remove intermediate columns
    x = x.drop(columns=["signup_date", "tenure_days"])

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Feature cutoff date, e.g. 2023-12-31"
    )
    args = parser.parse_args()

    print("STEP 1: Reading SQLite model_source...")

    if not DB.exists():
        raise FileNotFoundError(f"Database not found: {DB}")

    with sqlite3.connect(DB) as con:
        df = pd.read_sql_query(
            "SELECT * FROM model_source",
            con
        )

    print(f"Rows loaded: {len(df)}")

    if df.empty:
        raise ValueError("model_source contains no records")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found")

    print("STEP 2: Building features...")

    features = build_features(
        df, as_of_date=args.as_of_date
    )

    print("STEP 3: Saving processed dataset...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, index=False)

    print(f"Rows exported: {len(features)}")
    print(f"Columns exported: {len(features.columns)}")
    print(f"Saved to: {OUTPUT}")

    print("\nFeature preview:")
    print(features.head().to_string(index=False))

    print("\nSUCCESS: Feature engineering completed!")


if __name__ == "__main__":
    main()
