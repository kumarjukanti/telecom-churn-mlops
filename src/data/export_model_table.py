import sqlite3
import pandas as pd
from pathlib import Path
OUT = Path("data/interim/model_source.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)
feature_sql = Path("db/sql/02_features.sql").read_text()
with sqlite3.connect("db/analytics.db") as con:
    con.executescript(feature_sql)
df = pd.read_sql_query("SELECT * FROM model_source", con)
df.to_parquet(OUT, index=False)
print(df.shape)