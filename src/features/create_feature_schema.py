
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/features.csv"
OUTPUT = ROOT / "artifacts/feature_schema.json"

FEATURE_BUILD_VERSION = "1.0.0"

NUMERIC = [
    "age",
    "monthly_charges",
    "months_observed",
    "avg_call_minutes",
    "avg_data_usage_gb",
    "avg_sms_count",
    "total_complaints",
    "avg_complaints",
    "tenure_months",
    "complaint_rate",
    "charges_per_tenure_month",
]

CATEGORICAL = [
    "gender",
    "region",
    "contract_type",
]

TARGET = "churn"
IDENTIFIER = "customer_id"


def main():
    print("Loading processed features...")

    df = pd.read_csv(INPUT)

    required = NUMERIC + CATEGORICAL + [
        TARGET, IDENTIFIER
    ]

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    schema = {
        "schema_version": "1.0",
        "feature_build_version": FEATURE_BUILD_VERSION,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": "data/processed/features.csv",
        "identifier": IDENTIFIER,
        "target": {
            "name": TARGET,
            "type": "integer",
            "allowed_values": [0, 1],
            "definition": {
                "0": "Customer retained",
                "1": "Customer churned"
            }
        },
        "features": {}
    }

    # Numeric features
    for column in NUMERIC:
        values = pd.to_numeric(
            df[column], errors="coerce"
        )

        invalid = df[column].notna() & values.isna()

        if invalid.any():
            raise ValueError(
                f"Non-numeric values found in {column}"
            )

        schema["features"][column] = {
            "type": "number",
            "nullable": bool(values.isna().any())
        }

    # Categorical features
    for column in CATEGORICAL:
        categories = sorted(
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        schema["features"][column] = {
            "type": "string",
            "nullable": bool(df[column].isna().any()),
            "allowed_categories": categories
        }

    # Save contract
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2)

    print("\nFeature contract created successfully!")
    print(f"Saved to: {OUTPUT}")
    print(f"Number of model features: {len(schema['features'])}")

    print("\nAllowed categories:")
    for column in CATEGORICAL:
        print(
            column,
            schema["features"][column]["allowed_categories"]
        )


if __name__ == "__main__":
    main()
