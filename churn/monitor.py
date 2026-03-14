"""
churn/monitor.py

Monitoring with Evidently:
- Compares reference sample (saved during training) vs current data (from API logs)
- Generates an HTML drift report

Output:
- reports/monitoring/data_drift_report.html

Run:
  python -m churn.monitor
"""

from __future__ import annotations

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from churn.config import PATHS, TARGET_COL
from churn.utils import ensure_dirs, must_exist


def load_reference() -> pd.DataFrame:
    ensure_dirs()
    path = must_exist(PATHS.processed_dir / "reference_sample.csv", "Reference sample missing. Run: python -m churn.train")
    return pd.read_csv(path)


def load_current() -> pd.DataFrame:
    ensure_dirs()
    log_path = PATHS.logs_dir / "predictions_log.csv"

    if log_path.exists():
        df = pd.read_csv(log_path)
        cols = [c for c in df.columns if c not in ["timestamp_utc", "churn_probability"]]
        current = df[cols].copy()
        current[TARGET_COL] = 0  # placeholder target for schema alignment
        return current.sample(n=min(800, len(current)), random_state=42)

    # fallback to reference if no logs yet
    ref = load_reference()
    return ref.sample(n=min(800, len(ref)), random_state=42)


def main() -> None:
    ensure_dirs()
    reference = load_reference()
    current = load_current()

    report = Report(metrics=[DataDriftPreset()])
    eval_result = report.run(reference_data=reference, current_data=current)

    out = PATHS.monitoring_dir / "data_drift_report.html"
    eval_result.save_html(str(out))
    print(f"Saved drift report: {out}")


if __name__ == "__main__":
    main()
