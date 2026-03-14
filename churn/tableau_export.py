"""
churn/tableau_export.py

Creates Tableau-ready datasets so Tableau is the PRIMARY visualization layer.

Exports to data/tableau/:
1) telco_cleaned.csv
2) telco_scored.csv
3) roi_thresholds.csv
4) feature_importance.csv
5) model_metrics.csv + threshold_scan.csv (copied for convenience)

Run:
  python -m churn.tableau_export
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from churn.config import (
    DEFAULT_MARGIN,
    DEFAULT_MONTHS_SAVED,
    DEFAULT_OFFER_COST,
    DEFAULT_SAVE_RATE,
    PATHS,
    RANDOM_SEED,
    TARGET_COL,
)
from churn.utils import ensure_dirs, load_processed_csv, must_exist


def load_model():
    path = must_exist(PATHS.models_dir / "model.joblib", "Model not found. Run: python -m churn.train")
    return joblib.load(path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Business-friendly feature: average spend proxy
    df["AvgChargePerMonth"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Buckets (great for Tableau filters + segmentation)
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 36, 48, 60, 120],
        labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61+"],
    )

    df["MonthlyChargesBucket"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-1, 25, 50, 75, 100, 200],
        labels=["0-25", "26-50", "51-75", "76-100", "100+"],
    )

    return df


def score_dataset(model, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df.drop(columns=[TARGET_COL])
    proba = model.predict_proba(X)[:, 1]
    df["churn_probability"] = proba

    df["risk_bucket"] = pd.cut(
        df["churn_probability"],
        bins=[-0.000001, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    )

    ranks = df["churn_probability"].rank(method="first")
    df["risk_decile"] = pd.qcut(ranks, 10, labels=list(range(1, 11)))

    df["recommended_action"] = np.where(
        df["risk_bucket"].astype(str) == "High",
        "Target retention offer",
        np.where(df["risk_bucket"].astype(str) == "Medium", "Watchlist / proactive support", "No action"),
    )

    return df


def profit_if_saved(monthly_charges: np.ndarray, months_saved: int, margin: float) -> np.ndarray:
    return monthly_charges * months_saved * margin


def build_roi_table(df_scored: pd.DataFrame) -> pd.DataFrame:
    monthly = df_scored["MonthlyCharges"].to_numpy()
    profit_saved = profit_if_saved(monthly, DEFAULT_MONTHS_SAVED, DEFAULT_MARGIN)
    proba = df_scored["churn_probability"].to_numpy()

    rows = []
    for t in np.linspace(0.1, 0.9, 17):
        targeted = proba >= t
        if targeted.sum() == 0:
            continue

        expected_per_customer = (proba[targeted] * DEFAULT_SAVE_RATE * profit_saved[targeted]) - DEFAULT_OFFER_COST

        rows.append(
            {
                "threshold": float(t),
                "targeted_customers": int(targeted.sum()),
                "avg_expected_value_per_targeted": float(expected_per_customer.mean()),
                "total_expected_value": float(expected_per_customer.sum()),
                "offer_cost": float(DEFAULT_OFFER_COST),
                "save_rate": float(DEFAULT_SAVE_RATE),
                "margin": float(DEFAULT_MARGIN),
                "months_saved": int(DEFAULT_MONTHS_SAVED),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold")


def build_feature_importance(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).to_numpy()

    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def copy_if_exists(src, dst):
    src = PATHS.metrics_dir / src
    if src.exists():
        (PATHS.tableau_dir / dst).write_bytes(src.read_bytes())


def main() -> None:
    ensure_dirs()
    df = load_processed_csv()
    df = add_engineered_features(df)

    # 1) Cleaned
    cleaned_path = PATHS.tableau_dir / "telco_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved: {cleaned_path}")

    # 2) Scored
    model = load_model()
    df_scored = score_dataset(model, df)
    scored_path = PATHS.tableau_dir / "telco_scored.csv"
    df_scored.to_csv(scored_path, index=False)
    print(f"Saved: {scored_path}")

    # 3) ROI table
    roi = build_roi_table(df_scored)
    roi_path = PATHS.tableau_dir / "roi_thresholds.csv"
    roi.to_csv(roi_path, index=False)
    print(f"Saved: {roi_path}")

    # 4) Feature importance
    fi = build_feature_importance(model, df)
    fi_path = PATHS.tableau_dir / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    print(f"Saved: {fi_path}")

    # Convenience: copy model metrics + threshold scan (if they exist)
    copy_if_exists("model_metrics.csv", "model_metrics.csv")
    copy_if_exists("threshold_scan.csv", "threshold_scan.csv")

    print("\n✅ Tableau exports ready: data/tableau/")
    print("Open Tableau -> Connect -> Text file -> pick the CSVs.")


if __name__ == "__main__":
    main()
