"""
churn/evaluate.py

Loads the saved model and prints:
- ROC-AUC, PR-AUC
- Confusion matrix @ threshold
- Threshold scan (precision/recall tradeoffs)

Run:
  python -m churn.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn.config import PATHS, RANDOM_SEED, TARGET_COL, TEST_SIZE
from churn.utils import load_processed_csv, must_exist


def load_model():
    path = must_exist(PATHS.models_dir / "model.joblib", "Model not found. Run: python -m churn.train")
    return joblib.load(path)


def scan_thresholds(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "predicted_churn_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_processed_csv()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).to_numpy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    model = load_model()
    proba = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr:.4f}")

    t = 0.5
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, pred)

    print(f"\nConfusion Matrix @ threshold={t}:")
    print(cm)

    table = scan_thresholds(y_test, proba)
    out_path = PATHS.metrics_dir / "threshold_scan.csv"
    table.to_csv(out_path, index=False)

    print("\nThreshold scan saved to:", out_path)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
