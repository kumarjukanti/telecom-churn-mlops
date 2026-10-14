
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/interim/model_source.parquet"
OUTPUT_DIR = ROOT / "data/processed"

RANDOM_STATE = 42


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
        x["total_complaints"]
        / x["months_observed"].clip(lower=1)
    )

    x["charges_per_tenure_month"] = (
        x["monthly_charges"]
        / x["tenure_months"].clip(lower=1)
    )

    x = x.drop(columns=["signup_date", "tenure_days"])

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of-date",
        default="2023-12-31",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
    )
    args = parser.parse_args()

    print("STEP 1: Reading model source Parquet...")

    if not INPUT.is_file():
        raise FileNotFoundError(
            f"Missing model source: {INPUT}"
        )

    df = pd.read_parquet(INPUT)

    if df.empty:
        raise ValueError("Model source is empty")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found")

    if df["churn"].isna().any():
        raise ValueError("Missing churn labels")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn labels must be 0 or 1")

    print(f"Records loaded: {len(df)}")

    print("STEP 2: Engineering features...")

    features = build_features(
        df, args.as_of_date
    )

    features["churn"] = features["churn"].astype(int)

    print("STEP 3: Creating stratified splits...")

    # 80% train+validation, 20% test
    trainval, test = train_test_split(
        features,
        test_size=0.20,
        stratify=features["churn"],
        random_state=args.random_state,
    )

    # 80% of trainval for training,
    # 20% of trainval for validation.
    # Final proportions: 64% / 16% / 20%.
    train, validation = train_test_split(
        trainval,
        test_size=0.20,
        stratify=trainval["churn"],
        random_state=args.random_state,
    )

    # Ensure no customers appear in multiple splits.
    train_ids = set(train["customer_id"])
    val_ids = set(validation["customer_id"])
    test_ids = set(test["customer_id"])

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):
        raise ValueError(
            "Customer IDs overlap between datasets"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, frame in outputs.items():
        path = OUTPUT_DIR / f"{name}.parquet"

        frame.to_parquet(path, index=False)

        print(
            f"{name}: {len(frame)} rows, "
            f"churn rate: {frame['churn'].mean():.4f}"
        )

        print(f"Saved: {path}")

    if len(train) + len(validation) + len(test) != len(df):
        raise ValueError("Split sizes do not add up")

    print("\nSUCCESS: Features and splits created!")
    print("Test dataset is reserved for final evaluation.")


if __name__ == "__main__":
    main()
