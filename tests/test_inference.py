import pandas as pd
import joblib

from churn.config import PATHS, TARGET_COL


def test_model_artifact_exists():
    assert (PATHS.models_dir / "model.joblib").exists(), "Model artifact missing. Run training first."


def test_model_predict_proba_shape():
    model = joblib.load(PATHS.models_dir / "model.joblib")

    ref = pd.read_csv(PATHS.processed_dir / "reference_sample.csv")
    X = ref.drop(columns=[TARGET_COL])

    proba = model.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert proba.shape[1] == 2
