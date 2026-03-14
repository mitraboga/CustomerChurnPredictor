"""
churn/config.py

Single source of truth for paths, dataset URL, columns, and business assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    logs_dir: Path = data_dir / "logs"
    tableau_dir: Path = data_dir / "tableau"

    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    figures_dir: Path = reports_dir / "figures"
    monitoring_dir: Path = reports_dir / "monitoring"
    metrics_dir: Path = reports_dir / "metrics"


PATHS = Paths()

# Public dataset source (no Kaggle keys needed).
TELCO_CSV_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TEST_SIZE = 0.2
RANDOM_SEED = 42

# --- Business assumptions for ROI simulation ---
DEFAULT_OFFER_COST = 12.0          # $ cost per targeted customer
DEFAULT_SAVE_RATE = 0.35           # chance we save a true churner
DEFAULT_MARGIN = 0.30              # profit margin (not revenue)
DEFAULT_MONTHS_SAVED = 12          # months of retained value if saved
