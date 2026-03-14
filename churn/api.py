"""
churn/api.py

FastAPI service:
- GET /health
- POST /predict (single record)
- POST /predict_batch (list of records)

Logs every prediction to: data/logs/predictions_log.csv

Run:
  python -m churn.api
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS, PATHS
from churn.utils import ensure_dirs, must_exist

APP = FastAPI(title="Customer Churn Predictor", version="1.0")


def load_model():
    path = must_exist(PATHS.models_dir / "model.joblib", "Model not found. Run: python -m churn.train")
    return joblib.load(path)


MODEL = None


def ensure_log_file() -> Path:
    ensure_dirs()
    log_path = PATHS.logs_dir / "predictions_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["timestamp_utc", *NUMERIC_COLS, *CATEGORICAL_COLS, "churn_probability"]
            writer.writerow(header)
    return log_path


LOG_PATH = ensure_log_file()


class ChurnRequest(BaseModel):
    tenure: float = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


class ChurnResponse(BaseModel):
    churn_probability: float


@APP.on_event("startup")
def _startup() -> None:
    global MODEL
    MODEL = load_model()


@APP.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _predict_proba_one(payload: dict[str, Any]) -> float:
    assert MODEL is not None
    df = pd.DataFrame([payload])
    return float(MODEL.predict_proba(df)[:, 1][0])


def _append_log(payload: dict[str, Any], proba: float) -> None:
    row = [
        datetime.utcnow().isoformat(),
        *[payload[c] for c in NUMERIC_COLS],
        *[payload[c] for c in CATEGORICAL_COLS],
        proba,
    ]
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


@APP.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest) -> ChurnResponse:
    payload = req.model_dump()
    proba = _predict_proba_one(payload)
    _append_log(payload, proba)
    return ChurnResponse(churn_probability=proba)


@APP.post("/predict_batch")
def predict_batch(requests: list[ChurnRequest]) -> dict[str, Any]:
    results = []
    for r in requests:
        payload = r.model_dump()
        proba = _predict_proba_one(payload)
        _append_log(payload, proba)
        results.append({"churn_probability": proba})
    return {"count": len(results), "results": results}


def main() -> None:
    uvicorn.run("churn.api:APP", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
