
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DB = ROOT / "db" / "telecom.db"
SCHEMA = ROOT / "db" / "sql" / "01_schema.sql"


def normalize_columns(df):
    """Standardize common CSV column-name variations."""
    df = df.copy()

    aliases = {
        "customerid": "customer_id",
        "customer_id": "customer_id",
        "signupdate": "signup_date",
        "signup_date": "signup_date",
        "contracttype": "contract_type",
        "contract_type": "contract_type",
        "monthlycharges": "monthly_charges",
        "monthly_charges": "monthly_charges",
        "callminutes": "call_minutes",
        "call_minutes": "call_minutes",
        "datausagegb": "data_usage_gb",
        "data_usage_gb": "data_usage_gb",
        "smscount": "sms_count",
        "sms_count": "sms_count",
        "churnlabel": "churn",
        "churned": "churn",
        "churn": "churn",
    }

    def clean(name):
        name = str(name).strip().lower()
        name = name.replace(" ", "_").replace("-", "_")
        return aliases.get(name, name)

    df.columns = [clean(c) for c in df.columns]

    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns after normalization")

    return df


def check_columns(df, expected, name):
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def prepare_customers(df):
    columns = [
        "customer_id", "age", "gender", "region",
        "signup_date", "contract_type", "monthly_charges"
    ]
    check_columns(df, columns, "customer_info.csv")

    df = df[columns].copy()
    df = df.drop_duplicates()

    df["customer_id"] = df["customer_id"].astype("string").str.strip()

    if df["customer_id"].isna().any() or (
        df["customer_id"] == ""
    ).any():
        raise ValueError("Missing customer IDs")

    df["signup_date"] = pd.to_datetime(
        df["signup_date"], errors="coerce"
    )

    if df["signup_date"].isna().any():
        raise ValueError("Invalid signup dates")

    df["age"] = pd.to_numeric(df["age"], errors="raise")
    df["monthly_charges"] = pd.to_numeric(
        df["monthly_charges"], errors="raise"
    )

    # Where only the signup date differs for the same
    # customer, retain the earliest recorded date.
    # Fail if other customer attributes disagree.
    other_columns = [
        "age", "gender", "region",
        "contract_type", "monthly_charges"
    ]

    conflicts = (
        df.groupby("customer_id", dropna=False)[other_columns]
        .nunique(dropna=False)
    )

    bad_ids = conflicts.index[
        conflicts.gt(1).any(axis=1)
    ].tolist()

    if bad_ids:
        raise ValueError(
            f"Conflicting customer attributes: {bad_ids[:10]}"
        )

    df = (
        df.sort_values(["customer_id", "signup_date"])
        .drop_duplicates("customer_id", keep="first")
    )

    df["signup_date"] = df["signup_date"].dt.strftime(
        "%Y-%m-%d"
    )

    return df


def prepare_usage(df):
    columns = [
        "customer_id", "month", "call_minutes",
        "data_usage_gb", "sms_count", "complaints"
    ]
    check_columns(df, columns, "usage_data.csv")

    df = df[columns].copy().drop_duplicates()

    df["customer_id"] = df["customer_id"].astype("string").str.strip()

    if df["customer_id"].isna().any() or (
        df["customer_id"] == ""
    ).any():
        raise ValueError("Missing usage customer IDs")

    parsed_month = pd.to_datetime(df["month"], errors="coerce")

    if parsed_month.isna().any():
        raise ValueError("Invalid usage dates")

    df["month"] = parsed_month.dt.strftime("%Y-%m-%d")

    numeric = [
        "call_minutes", "data_usage_gb",
        "sms_count", "complaints"
    ]

    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Same customer and month must not contain
    # contradictory usage records.
    keys = ["customer_id", "month"]

    if df.duplicated(keys).any():
        raise ValueError(
            "Conflicting usage records for the same "
            "customer and month"
        )

    return df


def prepare_labels(df):
    columns = ["customer_id", "churn"]
    check_columns(df, columns, "churn_labels.csv")

    df = df[columns].copy().drop_duplicates()

    df["customer_id"] = df["customer_id"].astype("string").str.strip()

    if df["customer_id"].isna().any() or (
        df["customer_id"] == ""
    ).any():
        raise ValueError("Missing label customer IDs")

    df["churn"] = pd.to_numeric(df["churn"], errors="raise")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn labels must contain only 0 or 1")

    if df["customer_id"].duplicated().any():
        raise ValueError("Conflicting churn labels")

    df["churn"] = df["churn"].astype(int)

    return df


def main():
    print("STEP 1: Reading raw CSV files...")

    paths = {
        "customers": RAW / "customer_info.csv",
        "usage": RAW / "usage_data.csv",
        "labels": RAW / "churn_labels.csv",
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} CSV: {path}"
            )

    customers = prepare_customers(
        normalize_columns(pd.read_csv(paths["customers"]))
    )
    usage = prepare_usage(
        normalize_columns(pd.read_csv(paths["usage"]))
    )
    labels = prepare_labels(
        normalize_columns(pd.read_csv(paths["labels"]))
    )

    print(f"Customers: {len(customers)}")
    print(f"Usage records: {len(usage)}")
    print(f"Churn labels: {len(labels)}")

    print("STEP 2: Validating customer references...")

    known_ids = set(customers["customer_id"])

    for name, frame in [
        ("usage", usage),
        ("labels", labels),
    ]:
        unknown = set(frame["customer_id"]) - known_ids

        if unknown:
            raise ValueError(
                f"{name} contains unknown customer IDs: "
                f"{sorted(unknown)[:10]}"
            )

    if not SCHEMA.is_file():
        raise FileNotFoundError(
            f"Missing SQL schema: {SCHEMA}"
        )

    print("STEP 3: Building SQLite database...")

    DB.parent.mkdir(parents=True, exist_ok=True)

    # Build in a temporary file to avoid leaving a
    # partially constructed database after an error.
    fd, temporary_path = tempfile.mkstemp(
        prefix="telecom_build_",
        suffix=".db",
        dir=DB.parent,
    )
    os.close(fd)

    try:
        with sqlite3.connect(temporary_path) as con:
            con.execute("PRAGMA foreign_keys = ON")

            con.executescript(
                SCHEMA.read_text(encoding="utf-8")
            )

            customers.to_sql(
                "customers", con,
                if_exists="append", index=False
            )

            usage.to_sql(
                "usage_monthly", con,
                if_exists="append", index=False
            )

            labels.to_sql(
                "churn_labels", con,
                if_exists="append", index=False
            )

            con.commit()

            for table, expected in [
                ("customers", len(customers)),
                ("usage_monthly", len(usage)),
                ("churn_labels", len(labels)),
            ]:
                actual = con.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]

                if actual != expected:
                    raise ValueError(
                        f"{table}: expected {expected}, "
                        f"got {actual}"
                    )

            errors = con.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

            if errors:
                raise ValueError(
                    f"Foreign key validation failed: {errors[:5]}"
                )

        os.replace(temporary_path, DB)

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(f"Database created: {DB}")
    print("SUCCESS: SQLite database built!")


if __name__ == "__main__":
    main()
