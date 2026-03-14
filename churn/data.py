"""
churn/data.py

1) Downloads the Telco churn CSV from a public URL
2) Cleans it into a modeling-ready dataset
3) Saves to data/processed/telco_processed.csv

Run:
  python -m churn.data --download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from churn.config import (
    CATEGORICAL_COLS,
    ID_COL,
    NUMERIC_COLS,
    PATHS,
    TARGET_COL,
    TELCO_CSV_URL,
)
from churn.utils import ensure_dirs


def download_telco_csv(dest: Path) -> None:
    print(f"Downloading dataset to: {dest}")
    r = requests.get(TELCO_CSV_URL, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print("Download complete.")


def load_raw() -> pd.DataFrame:
    raw_path = PATHS.raw_dir / "telco.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Raw dataset not found. Run: python -m churn.data --download")
    return pd.read_csv(raw_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert TotalCharges safely (sometimes blank strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop ID column (not predictive; doesn't generalize)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Target: Yes/No -> 1/0
    df[TARGET_COL] = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Validate expected columns exist (guards against dataset changes)
    expected = set(NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    return df


def save_processed(df: pd.DataFrame) -> Path:
    out_path = PATHS.processed_dir / "telco_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download raw dataset")
    args = parser.parse_args()

    ensure_dirs()

    raw_path = PATHS.raw_dir / "telco.csv"
    if args.download:
        download_telco_csv(raw_path)

    df_raw = load_raw()
    df = clean(df_raw)
    save_processed(df)


if __name__ == "__main__":
    main()
