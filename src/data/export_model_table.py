
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DB = ROOT / "db/telecom.db"
OUTPUT = ROOT / "data/interim/model_source.parquet"


def main():
    print("Exporting model source from SQLite...")

    if not DB.exists():
        raise FileNotFoundError(
            f"Database not found: {DB}"
        )

    with sqlite3.connect(DB) as con:
        df = pd.read_sql_query(
            "SELECT * FROM model_source",
            con
        )

    if df.empty:
        raise ValueError("model_source is empty")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found")

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT,
        index=False
    )

    print(f"Rows exported: {len(df)}")
    print(f"File saved: {OUTPUT}")
    print("Export completed successfully!")


if __name__ == "__main__":
    main()
