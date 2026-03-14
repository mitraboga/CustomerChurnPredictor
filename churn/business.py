"""
churn/business.py

Business simulation: decide WHO to target based on churn probability.

We compute expected NET BENEFIT per targeted customer:
  expected_value = (p_churn * save_rate * profit_if_saved) - offer_cost

Then we evaluate thresholds and pick the best one (max total expected value).

Outputs:
- reports/figures/business_simulation_thresholds.csv
- reports/figures/business_total_value_vs_threshold.png
- reports/metrics/best_threshold.json

Run:
  python -m churn.business
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn.config import (
    DEFAULT_MARGIN,
    DEFAULT_MONTHS_SAVED,
    DEFAULT_OFFER_COST,
    DEFAULT_SAVE_RATE,
    PATHS,
    TARGET_COL,
)
from churn.utils import ensure_dirs, load_processed_csv, must_exist


def load_model():
    path = must_exist(PATHS.models_dir / "model.joblib", "Model not found. Run: python -m churn.train")
    return joblib.load(path)


def profit_if_saved(monthly_charges: np.ndarray, months_saved: int, margin: float) -> np.ndarray:
    return monthly_charges * months_saved * margin


def simulate_thresholds(
    df: pd.DataFrame,
    proba: np.ndarray,
    offer_cost: float,
    save_rate: float,
    margin: float,
    months_saved: int,
) -> pd.DataFrame:
    monthly = df["MonthlyCharges"].to_numpy()
    profit_saved = profit_if_saved(monthly, months_saved, margin)

    rows = []
    for t in np.linspace(0.1, 0.9, 17):
        targeted = proba >= t
        if targeted.sum() == 0:
            continue

        expected_per_customer = (proba[targeted] * save_rate * profit_saved[targeted]) - offer_cost
        rows.append(
            {
                "threshold": float(t),
                "targeted_customers": int(targeted.sum()),
                "avg_expected_value_per_targeted": float(expected_per_customer.mean()),
                "total_expected_value": float(expected_per_customer.sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold")


def main() -> None:
    ensure_dirs()

    df = load_processed_csv()
    X = df.drop(columns=[TARGET_COL])

    model = load_model()
    proba = model.predict_proba(X)[:, 1]

    table = simulate_thresholds(
        df=df,
        proba=proba,
        offer_cost=DEFAULT_OFFER_COST,
        save_rate=DEFAULT_SAVE_RATE,
        margin=DEFAULT_MARGIN,
        months_saved=DEFAULT_MONTHS_SAVED,
    )

    out_csv = PATHS.figures_dir / "business_simulation_thresholds.csv"
    table.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Best threshold by total expected value
    best_row = table.sort_values("total_expected_value", ascending=False).iloc[0].to_dict()
    best_path = PATHS.metrics_dir / "best_threshold.json"
    best_path.write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    print(f"Saved: {best_path}")

    # Plot total expected value vs threshold
    plt.figure()
    plt.plot(table["threshold"], table["total_expected_value"], marker="o")
    plt.title("Total Expected Value vs Threshold")
    plt.xlabel("Churn Probability Threshold")
    plt.ylabel("Total Expected Value ($)")
    plt.tight_layout()
    out_png = PATHS.figures_dir / "business_total_value_vs_threshold.png"
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")

    print("\nTop 5 strategies (by total expected value):")
    print(table.sort_values("total_expected_value", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
