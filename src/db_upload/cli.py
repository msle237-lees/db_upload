import json
import os
import pathlib

import dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import TIMESTAMP as PG_TIMESTAMP


def load_schema(schema_path: str) -> dict:
    """Load column type schema from a JSON file."""
    if not os.path.exists(schema_path):
        print(f"No schema file found at {schema_path}, skipping type enforcement.")
        return {}
    with open(schema_path) as f:
        return json.load(f)


def localize_series(series: pd.Series, timezone: str) -> pd.Series:
    """Localize a datetime series one timestamp at a time to safely handle multiple DST transitions."""
    def localize_single(ts):
        if pd.isna(ts):
            return ts
        try:
            return ts.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            return pd.NaT

    return series.apply(localize_single)


def apply_schema(
    df: pd.DataFrame, table_name: str, schema: dict, timezone: str
) -> tuple[pd.DataFrame, dict]:
    """
    Cast columns to their schema-defined types.
    Returns the updated DataFrame and a sqlalchemy dtype map for to_sql().
    """
    sql_dtypes = {}
    table_schema = schema.get(table_name, {})

    for col, col_type in table_schema.items():
        if col not in df.columns:
            print(f"  Warning: schema column '{col}' not found in {table_name}, skipping.")
            continue

        if col_type == "timestamptz":
            df[col] = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
            df[col] = localize_series(df[col], timezone)
            sql_dtypes[col] = PG_TIMESTAMP(timezone=True)
            print(f"  Cast '{col}' to timestamptz ({timezone})")

    return df, sql_dtypes


def load_data(
    folder_path: str, timezone: str, schema: dict
) -> list[tuple[str, pd.DataFrame, dict]]:
    """Load CSV / JSON / Excel files from the specified folder."""
    print(f"Loading data from folder: {folder_path}")

    dir = os.listdir(folder_path)
    print(f"Files found: {dir}")

    dataframes = []
    for file in dir:
        file_path = os.path.join(folder_path, file)
        if file.endswith(".csv"):
            df = pd.read_csv(file_path)
            print(f"Loaded CSV: {file}")
        elif file.endswith(".json"):
            df = pd.read_json(file_path)
            print(f"Loaded JSON: {file}")
        elif file.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
            print(f"Loaded Excel: {file}")
        else:
            print(f"Unsupported file type (skipping): {file}")
            continue

        table_name = os.path.splitext(file)[0]
        df, sql_dtypes = apply_schema(df, table_name, schema, timezone)
        dataframes.append((file, df, sql_dtypes))
        # print(df.head())

    return dataframes


def app():
    dotenv.load_dotenv()

    timezone = os.getenv("TIMEZONE", "UTC")
    print(f"Using timezone: {timezone}")

    data_folder = os.getenv(
        "DATA_UPLOAD_PATH", pathlib.Path(__file__).parent.parent.parent / "data"
    )
    schema_path = os.getenv(
        "SCHEMA_PATH", pathlib.Path(__file__).parent.parent.parent / "schema.json"
    )

    schema = load_schema(schema_path)
    dataframes = load_data(data_folder, timezone, schema)

    engine = create_engine(
        f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    with engine.connect() as conn:
        for file_name, df, sql_dtypes in dataframes:
            table_name = os.path.splitext(file_name)[0]
            print(f"Uploading {file_name} to table {table_name}...")
            df.to_sql(
                table_name,
                conn,
                schema=os.getenv("DB_SCHEMA", "public"),
                if_exists=os.getenv("DB_IF_EXISTS", "replace"),
                index=False,
                dtype=sql_dtypes,  # enforces TIMESTAMPTZ in Postgres
            )
            print(f"Uploaded {file_name} to table {table_name} successfully.")
