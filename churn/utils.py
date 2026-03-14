"""
churn/utils.py

Small shared helpers:
- directory creation
- safe load paths
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from churn.config import PATHS


def ensure_dirs() -> None:
    """Create all project directories used by scripts."""
    PATHS.raw_dir.mkdir(parents=True, exist_ok=True)
    PATHS.processed_dir.mkdir(parents=True, exist_ok=True)
    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    PATHS.tableau_dir.mkdir(parents=True, exist_ok=True)

    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    PATHS.monitoring_dir.mkdir(parents=True, exist_ok=True)
    PATHS.metrics_dir.mkdir(parents=True, exist_ok=True)


def load_processed_csv(filename: str = "telco_processed.csv") -> pd.DataFrame:
    path = PATHS.processed_dir / filename
    if not path.exists():
        raise FileNotFoundError("Processed dataset not found. Run: python -m churn.data --download")
    return pd.read_csv(path)


def must_exist(path: Path, help_msg: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(help_msg)
    return path
