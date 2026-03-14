"""
churn/train.py

Trains multiple models and saves the best (by ROC-AUC) as models/model.joblib.

Also saves:
- models/metadata.json
- reports/metrics/model_metrics.csv
- data/processed/reference_sample.csv  (for monitoring)

Run:
  python -m churn.train
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.config import PATHS, RANDOM_SEED, TARGET_COL, TEST_SIZE
from churn.modeling import build_preprocess, candidate_models
from churn.utils import ensure_dirs, load_processed_csv


def score_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }


def main() -> None:
    ensure_dirs()
    df = load_processed_csv()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    preprocess = build_preprocess()

    results: list[dict[str, object]] = []
    best_name: str | None = None
    best_pipe: Pipeline | None = None
    best_roc = -1.0

    models = candidate_models()
    print(f"Training candidates: {list(models.keys())}")

    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)

        metrics = score_model(pipe, X_test, y_test)
        print(f"[{name}] ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f}")
        results.append({"model": name, **metrics})

        if metrics["roc_auc"] > best_roc:
            best_roc = metrics["roc_auc"]
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None and best_name is not None

    # Save model + metadata
    model_path = PATHS.models_dir / "model.joblib"
    joblib.dump(best_pipe, model_path)

    meta = {
        "best_model": best_name,
        "ranking_metric": "roc_auc",
        "results": results,
        "target_col": TARGET_COL,
    }
    (PATHS.models_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save metrics table (for README + Tableau if needed)
    metrics_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    metrics_csv = PATHS.metrics_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Save monitoring reference sample (includes target column)
    ref_path = PATHS.processed_dir / "reference_sample.csv"
    X_test.copy().assign(**{TARGET_COL: y_test}).sample(
        n=min(800, len(X_test)), random_state=RANDOM_SEED
    ).to_csv(ref_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {PATHS.models_dir / 'metadata.json'}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved monitoring reference: {ref_path}")


if __name__ == "__main__":
    main()
