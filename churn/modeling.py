"""
churn/modeling.py

Builds preprocess pipeline + candidate models.

We always include:
- Logistic Regression (interpretable, strong baseline)
- RandomForest (non-linear, robust)

Optionally includes:
- XGBoost (if installed) for a strong "resume model"
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_SEED


def build_preprocess() -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logreg": LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    # Optional XGBoost (only if installed)
    try:
        from xgboost import XGBClassifier  # type: ignore
        models["xgb"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1,
        )
    except Exception:
        # Not installed. That's fine.
        pass

    return models
