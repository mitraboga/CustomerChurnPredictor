"""
churn/explain.py

Explainability outputs:
1) Permutation importance on RAW INPUT columns (business-friendly)
2) SHAP global + local plots (if SHAP works in your environment)

Outputs saved to: reports/figures/

Run:
  python -m churn.explain
"""

from __future__ import annotations

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from churn.config import PATHS, RANDOM_SEED, TARGET_COL
from churn.utils import ensure_dirs, load_processed_csv, must_exist


def load_model():
    path = must_exist(PATHS.models_dir / "model.joblib", "Model not found. Run: python -m churn.train")
    return joblib.load(path)


def save_perm_importance(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).to_numpy()

    print("Computing permutation importance (raw input columns)...")
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
    )

    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    csv_path = PATHS.figures_dir / "permutation_importance.csv"
    importances.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot top 15
    top = importances.head(15).iloc[::-1]
    plt.figure()
    plt.barh(top["feature"], top["importance_mean"])
    plt.title("Permutation Importance (Top 15)")
    plt.tight_layout()
    png_path = PATHS.figures_dir / "permutation_importance_top15.png"
    plt.savefig(png_path, dpi=160)
    print(f"Saved: {png_path}")

    return importances


def save_shap_plots(model, df: pd.DataFrame) -> None:
    try:
        import shap  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"SHAP not available ({e}). Skipping SHAP.")
        return

    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["model"]

    X = df.drop(columns=[TARGET_COL])
    X_trans = preprocess.transform(X)

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = None

    warnings.filterwarnings("ignore")

    print("Computing SHAP values (may take a bit)...")
    background = X_trans[:500] if X_trans.shape[0] > 500 else X_trans
    explainer = shap.Explainer(clf, background, feature_names=feature_names)
    shap_values = explainer(X_trans[:800] if X_trans.shape[0] > 800 else X_trans)

    # -------------------------
    # Export SHAP data for UI
    # -------------------------
    try:
        values = np.asarray(shap_values.values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if getattr(shap_values, "feature_names", None) is not None:
            names = list(shap_values.feature_names)
        elif feature_names is not None:
            names = list(feature_names)
        else:
            names = [f"feature_{i}" for i in range(values.shape[1])]

        mean_abs = np.mean(np.abs(values), axis=0)
        global_df = (
            pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        global_csv = PATHS.figures_dir / "shap_global_importance.csv"
        global_df.to_csv(global_csv, index=False)
        print(f"Saved: {global_csv}")

        idx = 0
        local_vals = values[idx, :]
        local_data = getattr(shap_values, "data", None)
        if local_data is not None:
            local_feat_vals = np.asarray(local_data)[idx, :]
        else:
            local_feat_vals = np.array([np.nan] * len(local_vals))

        local_df = (
            pd.DataFrame(
                {
                    "feature": names,
                    "shap_value": local_vals,
                    "feature_value": local_feat_vals,
                    "abs_shap_value": np.abs(local_vals),
                }
            )
            .sort_values("abs_shap_value", ascending=False)
            .reset_index(drop=True)
        )
        local_csv = PATHS.figures_dir / "shap_local_example.csv"
        local_df.to_csv(local_csv, index=False)
        print(f"Saved: {local_csv}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not export SHAP CSVs ({e}). Continuing with plots.")

    plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    out1 = PATHS.figures_dir / "shap_global_importance_top15.png"
    plt.savefig(out1, dpi=160)
    print(f"Saved: {out1}")

    # Local example
    idx = 0
    plt.figure()
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    plt.tight_layout()
    out2 = PATHS.figures_dir / "shap_local_waterfall_example.png"
    plt.savefig(out2, dpi=160)
    print(f"Saved: {out2}")


def main() -> None:
    ensure_dirs()
    df = load_processed_csv()
    model = load_model()

    save_perm_importance(model, df)
    save_shap_plots(model, df)


if __name__ == "__main__":
    main()
