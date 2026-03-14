r"""
CustomerChurnPredictor — Streamlit App (Modern SaaS Dashboard)

Run:
1) Terminal A (API):
   .\.venv\Scripts\Activate.ps1
   python -m churn.api

2) Terminal B (UI):
   .\.venv\Scripts\Activate.ps1
   python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

# Altair gives clean SaaS-style charts (Streamlit usually installs it).
try:
    import altair as alt
except Exception:  # noqa: BLE001
    alt = None


# ---------------------------
# Paths / Artifacts
# ---------------------------

PATH_BEST_THRESHOLD = Path("reports/metrics/best_threshold.json")
PATH_MODEL_METRICS = Path("reports/metrics/model_metrics.csv")

PATH_PERM_IMPORTANCE_PNG = Path("reports/figures/permutation_importance_top15.png")
PATH_SHAP_GLOBAL_PNG = Path("reports/figures/shap_global_importance_top15.png")
PATH_SHAP_LOCAL_PNG = Path("reports/figures/shap_local_waterfall_example.png")
PATH_BUSINESS_PNG = Path("reports/figures/business_total_value_vs_threshold.png")
PATH_SHAP_GLOBAL_CSV = Path("reports/figures/shap_global_importance.csv")
PATH_SHAP_LOCAL_CSV = Path("reports/figures/shap_local_example.csv")

PATH_LOGS = Path("data/logs/predictions_log.csv")

# Tableau exports (reused for Streamlit analytics)
PATH_TELCO_CLEANED = Path("data/tableau/telco_cleaned.csv")
PATH_TELCO_SCORED = Path("data/tableau/telco_scored.csv")
PATH_ROI_THRESHOLDS = Path("data/tableau/roi_thresholds.csv")
PATH_FEATURE_IMPORTANCE = Path("data/tableau/feature_importance.csv")
PATH_THRESHOLD_SCAN = Path("data/tableau/threshold_scan.csv")  # exported by tableau_export.py
PATH_THRESHOLD_SCAN_FALLBACK = Path("reports/metrics/threshold_scan.csv")  # fallback


# ---------------------------
# Safe config resolution
# ---------------------------

def resolve_api_url() -> str:
    """
    Resolve API URL without crashing if Streamlit secrets are not configured.

    Priority:
    1) env var CHURN_API_URL or API_URL
    2) Streamlit secrets (if available)
    3) default http://localhost:8000
    """
    env_url = os.getenv("CHURN_API_URL") or os.getenv("API_URL")
    if env_url:
        return env_url.strip()

    try:
        url = st.secrets.get("API_URL", None)  # may throw if no secrets.toml exists
        if url:
            return str(url).strip()
    except Exception:
        pass

    return "http://localhost:8000"


# ---------------------------
# Page setup
# ---------------------------

st.set_page_config(
    page_title="CustomerChurnPredictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Styling (Modern SaaS CSS)
# ---------------------------

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }
small { opacity: 0.85; }

.hero {
  background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(59,130,246,0.10));
  border: 1px solid rgba(148,163,184,0.40);
  padding: 18px 18px;
  border-radius: 18px;
}

.kpi {
  border: 1px solid rgba(148,163,184,0.40);
  background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(59,130,246,0.10));
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 20px rgba(15,23,42,0.06);
}

.card {
  border: 1px solid rgba(148,163,184,0.40);
  background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(59,130,246,0.10));
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 20px rgba(15,23,42,0.06);
}

.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.45);
  font-size: 12px;
  margin-right: 8px;
  opacity: 0.95;
  background: rgba(255,255,255,0.7);
}

.metric-card-title {
  font-size: 12px;
  color: rgba(255,255,255,0.88);
  font-weight: 600;
  margin-bottom: 6px;
}
.metric-card-value {
  font-size: 22px;
  font-weight: 800;
  margin: 0;
  line-height: 1.1;
  color: rgba(255,255,255,0.98);
}
.metric-card-sub {
  font-size: 12px;
  color: rgba(255,255,255,0.78);
  margin-top: 6px;
}
.mini-bar {
  height: 10px;
  width: 100%;
  border-radius: 999px;
  background: rgba(255,255,255,0.22);
  overflow: hidden;
  margin-top: 10px;
}
.mini-bar > div {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(255,255,255,0.92), rgba(255,255,255,0.55));
}

.stButton>button {
  border-radius: 12px;
  padding: 0.6rem 1rem;
  font-weight: 700;
}
hr { display: none; }

/* Make Streamlit bordered containers match cards (keeps headings inside) */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid rgba(148,163,184,0.40) !important;
  background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(59,130,246,0.10)) !important;
  border-radius: 14px !important;
  padding: 14px 16px !important;
  box-shadow: 0 8px 20px rgba(15,23,42,0.06) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(show_spinner=False)
def api_health(api_url: str) -> Tuple[bool, str]:
    try:
        r = requests.get(f"{api_url}/health", timeout=4)
        if r.status_code == 200:
            return True, "Online"
        return False, f"Error ({r.status_code})"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


def risk_bucket(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def bucket_icon(bucket: str) -> str:
    return {"High": "🚨", "Medium": "⚠️", "Low": "✅"}.get(bucket, "ℹ️")


def expected_value_per_target(
    p_churn: float,
    monthly_charges: float,
    margin: float,
    months_saved: int,
    save_rate: float,
    offer_cost: float,
) -> float:
    """
    Per-customer EV:
    EV = P(churn) * save_rate * (margin * months_saved * monthly_charges) - offer_cost
    """
    retained_profit = margin * months_saved * monthly_charges
    return (p_churn * save_rate * retained_profit) - offer_cost


def make_payload_from_state() -> Dict[str, Any]:
    return {
        "tenure": float(st.session_state["tenure"]),
        "MonthlyCharges": float(st.session_state["MonthlyCharges"]),
        "TotalCharges": float(st.session_state["TotalCharges"]),
        "gender": st.session_state["gender"],
        "SeniorCitizen": st.session_state["SeniorCitizen"],
        "Partner": st.session_state["Partner"],
        "Dependents": st.session_state["Dependents"],
        "PhoneService": st.session_state["PhoneService"],
        "MultipleLines": st.session_state["MultipleLines"],
        "InternetService": st.session_state["InternetService"],
        "OnlineSecurity": st.session_state["OnlineSecurity"],
        "OnlineBackup": st.session_state["OnlineBackup"],
        "DeviceProtection": st.session_state["DeviceProtection"],
        "TechSupport": st.session_state["TechSupport"],
        "StreamingTV": st.session_state["StreamingTV"],
        "StreamingMovies": st.session_state["StreamingMovies"],
        "Contract": st.session_state["Contract"],
        "PaperlessBilling": st.session_state["PaperlessBilling"],
        "PaymentMethod": st.session_state["PaymentMethod"],
    }


def set_state_from_preset(preset: Dict[str, Any]) -> None:
    for k, v in preset.items():
        st.session_state[k] = v


# ---------------------------
# Presets (smooth demos)
# ---------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "Custom (default)": {
        "tenure": 12.0,
        "MonthlyCharges": 70.0,
        "TotalCharges": 800.0,
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    },
    "High Risk (demo)": {
        "tenure": 3.0,
        "MonthlyCharges": 95.0,
        "TotalCharges": 250.0,
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    },
    "Low Risk (demo)": {
        "tenure": 48.0,
        "MonthlyCharges": 55.0,
        "TotalCharges": 2600.0,
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
    },
}

if "initialized" not in st.session_state:
    set_state_from_preset(PRESETS["Custom (default)"])
    st.session_state["initialized"] = True


# ---------------------------
# Load artifacts
# ---------------------------

best_thr_obj = read_json_if_exists(PATH_BEST_THRESHOLD)
default_threshold = float(best_thr_obj.get("threshold", 0.5)) if best_thr_obj else 0.5

model_metrics_df = read_csv_if_exists(PATH_MODEL_METRICS)
roi_df = read_csv_if_exists(PATH_ROI_THRESHOLDS)
feat_df = read_csv_if_exists(PATH_FEATURE_IMPORTANCE)

telco_clean = read_csv_if_exists(PATH_TELCO_CLEANED)
telco_scored = read_csv_if_exists(PATH_TELCO_SCORED)

thr_df = read_csv_if_exists(PATH_THRESHOLD_SCAN)
if thr_df.empty:
    thr_df = read_csv_if_exists(PATH_THRESHOLD_SCAN_FALLBACK)


# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    API_URL = st.text_input("API URL", value=resolve_api_url())

    api_ok, api_msg = api_health(API_URL)
    if api_ok:
        st.success(f"API: {api_msg}")
    else:
        st.error("API: Offline")
        st.caption(f"Reason: {api_msg}")
        st.code("python -m churn.api", language="bash")

    st.write("")

    st.markdown("### 🎯 Decision Threshold")
    threshold = st.slider(
        "Intervene if churn probability ≥",
        0.05,
        0.95,
        float(default_threshold),
        0.01,
        help="Business decision point (not a model metric).",
    )
    if best_thr_obj:
        st.caption(f"Saved best threshold (ROI): **{float(default_threshold):.2f}**")

    st.write("")

    st.markdown("### 💰 Business Assumptions")
    offer_cost = st.number_input("Retention offer cost ($)", min_value=0.0, value=12.0, step=1.0)
    save_rate = st.slider("Save rate if targeted", 0.0, 1.0, 0.35, 0.01)
    margin = st.slider("Profit margin", 0.0, 1.0, 0.30, 0.01)
    months_saved = st.slider("Months saved if retained", 1, 36, 12, 1)

    st.write("")

    st.markdown("### 🧪 Demo Profiles")
    preset_name = st.selectbox("Load a preset customer", list(PRESETS.keys()))
    if st.button("Load preset into form", use_container_width=True):
        set_state_from_preset(PRESETS[preset_name])
        st.rerun()

    st.write("")
    st.markdown("### 📦 Data Availability")
    if telco_scored.empty:
        st.warning("No `telco_scored.csv` found.")
        st.caption("Generate it with:")
        st.code("python -m churn.tableau_export", language="bash")
    else:
        st.success("Analytics dataset loaded ✅")


# ---------------------------
# Hero Header
# ---------------------------

st.markdown(
    """
<div class="hero">
  <div style="display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;">
    <div>
      <h1 style="margin:0;">📉 CustomerChurnPredictor</h1>
      <p style="margin:6px 0 0 0; font-size: 15px; opacity:0.92;">
        Predict churn risk → choose a threshold → target retention offers → quantify ROI.
      </p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# ---------------------------
# KPI Row
# ---------------------------

k1, k2, k3, k4 = st.columns(4, gap="large")

best_model = "logreg"
roc_auc = None
pr_auc = None

if not model_metrics_df.empty and "roc_auc" in model_metrics_df.columns:
    best_row = model_metrics_df.sort_values("roc_auc", ascending=False).iloc[0]
    best_model = str(best_row.get("model", "logreg"))
    roc_auc = float(best_row.get("roc_auc", 0.0))
    pr_auc = float(best_row.get("pr_auc", 0.0))

with k1:
    st.markdown(
        f"""
<div class="kpi">
  <div class="metric-card-title">Best Model</div>
  <div class="metric-card-value">{best_model}</div>
  <div class="metric-card-sub">Top ROC-AUC on holdout</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k2:
    roc_txt = f"{roc_auc:.4f}" if roc_auc is not None else "—"
    st.markdown(
        f"""
<div class="kpi">
  <div class="metric-card-title">ROC-AUC</div>
  <div class="metric-card-value">{roc_txt}</div>
  <div class="metric-card-sub">Ranking quality</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k3:
    pr_txt = f"{pr_auc:.4f}" if pr_auc is not None else "—"
    st.markdown(
        f"""
<div class="kpi">
  <div class="metric-card-title">PR-AUC</div>
  <div class="metric-card-value">{pr_txt}</div>
  <div class="metric-card-sub">Performance on positives</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
<div class="kpi">
  <div class="metric-card-title">Decision Threshold</div>
  <div class="metric-card-value">{threshold:.2f}</div>
  <div class="metric-card-sub">Intervene if p ≥ threshold</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")


# ---------------------------
# Tabs
# ---------------------------

tab_predict, tab_analytics, tab_explain, tab_business, tab_logs = st.tabs(
    ["🧾 Predict", "📊 Analytics Dashboard", "🔎 Explainability", "📈 Business", "🧾 Logs & Batch"]
)

# ============================================================
# TAB: Predict
# ============================================================

with tab_predict:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("### Customer Input Form")
        st.caption("Fill the form → click Predict → get a decision + business value estimate.")

        with st.form("predict_form", clear_on_submit=False):
            with st.container(border=True):
                st.markdown("#### Core")
                n1, n2, n3 = st.columns(3)
                with n1:
                    st.number_input("Tenure (months)", min_value=0.0, step=1.0, key="tenure")
                with n2:
                    st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0, key="MonthlyCharges")
                with n3:
                    st.number_input("Total Charges ($)", min_value=0.0, step=10.0, key="TotalCharges")

                st.markdown("#### Customer")
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.selectbox("Gender", ["Male", "Female"], key="gender")
                with a2:
                    st.selectbox("Senior Citizen", ["0", "1"], key="SeniorCitizen")
                with a3:
                    st.selectbox("Partner", ["Yes", "No"], key="Partner")
                with a4:
                    st.selectbox("Dependents", ["Yes", "No"], key="Dependents")

                st.markdown("#### Services")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.selectbox("Phone Service", ["Yes", "No"], key="PhoneService")
                    st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="MultipleLines")
                with s2:
                    st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="InternetService")
                    st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="OnlineSecurity")
                    st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="TechSupport")
                with s3:
                    st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="OnlineBackup")
                    st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="DeviceProtection")
                    st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="StreamingTV")
                    st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="StreamingMovies")

                st.markdown("#### Contract & Billing")
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
                with b2:
                    st.selectbox("Paperless Billing", ["Yes", "No"], key="PaperlessBilling")
                with b3:
                    st.selectbox(
                        "Payment Method",
                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                        key="PaymentMethod",
                    )
            submitted = st.form_submit_button("🚀 Predict Churn Probability", use_container_width=True)

    if submitted:
        if not api_ok:
            st.error("API is offline. Start it first: `python -m churn.api`")
        else:
            payload = make_payload_from_state()
            with st.spinner("Scoring customer..."):
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=12)
                    r.raise_for_status()
                    p = float(r.json()["churn_probability"])
                    st.session_state["last_prediction"] = p
                    st.success("Prediction complete.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Prediction failed: {e}")

    with right:
        st.markdown("### Decision Panel")

        p = float(st.session_state.get("last_prediction", -1.0))
        if p < 0:
            st.info("Run a prediction to see the decision + expected value.")
        else:
            bucket = risk_bucket(p)
            icon = bucket_icon(bucket)

            ev = expected_value_per_target(
                p_churn=p,
                monthly_charges=float(st.session_state["MonthlyCharges"]),
                margin=float(margin),
                months_saved=int(months_saved),
                save_rate=float(save_rate),
                offer_cost=float(offer_cost),
            )

            bar_w = int(min(max(p, 0.0), 1.0) * 100)
            st.markdown(
                f"""
<div class="card">
  <div class="metric-card-title">Churn Probability</div>
  <div class="metric-card-value">{p:.2%}</div>
  <div class="mini-bar"><div style="width:{bar_w}%;"></div></div>
  <div class="metric-card-sub">Model score for this customer</div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown(
                f"""
<div class="card">
  <div class="metric-card-title">Risk Bucket</div>
  <div class="metric-card-value">{icon} {bucket}</div>
  <div class="metric-card-sub">Based on probability bands</div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.write("")
            decision_txt = "INTERVENE" if p >= threshold else "DO NOT TARGET"
            decision_sub = f"p ≥ {threshold:.2f}" if p >= threshold else f"p < {threshold:.2f}"
            st.markdown(
                f"""
<div class="card">
  <div class="metric-card-title">Decision</div>
  <div class="metric-card-value">{decision_txt}</div>
  <div class="metric-card-sub">{decision_sub}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.write("")
            ev_sub = "Positive EV under assumptions" if ev >= 0 else "Negative EV — adjust threshold/cost"
            st.markdown(
                f"""
<div class="card">
  <div class="metric-card-title">Expected Value (per customer)</div>
  <div class="metric-card-value">${ev:,.2f}</div>
  <div class="metric-card-sub">{ev_sub}</div>
</div>
""",
                unsafe_allow_html=True,
            )


# ============================================================
# TAB: Analytics Dashboard
# ============================================================

with tab_analytics:
    st.markdown("### 📊 Analytics Dashboard")
    st.caption("All charts are based on your real scored dataset + evaluation exports.")

    if telco_scored.empty:
        st.warning("No analytics data found yet. Generate Tableau exports first:")
        st.code("python -m churn.tableau_export", language="bash")
    else:
        df = telco_scored.copy()

        if "churn_probability" not in df.columns:
            for alt_col in ["probability", "proba", "churn_proba"]:
                if alt_col in df.columns:
                    df["churn_probability"] = df[alt_col]
                    break

        if "churn_probability" not in df.columns:
            st.error("Could not find churn_probability column in data/tableau/telco_scored.csv")
        else:
            df["decision"] = (df["churn_probability"] >= float(threshold)).map({True: "Intervene", False: "No Action"})
            df["risk_bucket"] = df["churn_probability"].map(risk_bucket)

            # KPI row
            a1, a2, a3, a4 = st.columns(4, gap="large")
            with a1:
                st.markdown(
                    f"""
<div class="kpi">
  <div class="metric-card-title">Customers Scored</div>
  <div class="metric-card-value">{len(df):,}</div>
  <div class="metric-card-sub">Dataset rows</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with a2:
                st.markdown(
                    f"""
<div class="kpi">
  <div class="metric-card-title">Avg Churn Probability</div>
  <div class="metric-card-value">{df['churn_probability'].mean():.2%}</div>
  <div class="metric-card-sub">Overall risk level</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with a3:
                st.markdown(
                    f"""
<div class="kpi">
  <div class="metric-card-title">Targeted @ Threshold</div>
  <div class="metric-card-value">{(df['decision'] == 'Intervene').mean():.2%}</div>
  <div class="metric-card-sub">Action rate</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with a4:
                st.markdown(
                    f"""
<div class="kpi">
  <div class="metric-card-title">High Risk Share</div>
  <div class="metric-card-value">{(df['risk_bucket'] == 'High').mean():.2%}</div>
  <div class="metric-card-sub">Bucketed at p ≥ 0.70</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.write("")

            c_left, c_right = st.columns([1.25, 0.75], gap="large")

            # Chart 1: stacked bar distribution (like your screenshot)
            with c_left:
                st.markdown("#### Churn Probability Distribution (stacked by decision)")

                bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                df["prob_bin"] = pd.cut(df["churn_probability"], bins=bins, include_lowest=True)

                bin_df = (
                    df.groupby(["prob_bin", "decision"], observed=True)
                    .size()
                    .reset_index(name="customers")
                )

                if alt is not None:
                    chart = (
                        alt.Chart(bin_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("prob_bin:N", title="Churn probability (binned)", sort=None),
                            y=alt.Y("customers:Q", title="Customers"),
                            color=alt.Color("decision:N", title="Decision"),
                            tooltip=["prob_bin:N", "decision:N", "customers:Q"],
                        )
                        .properties(height=340)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    pivot = bin_df.pivot(index="prob_bin", columns="decision", values="customers").fillna(0)
                    st.bar_chart(pivot)

                st.caption("Business view: how many customers get targeted at the current threshold.")

            # Chart 2: contract insight
            with c_right:
                st.markdown("#### Avg Churn Probability by Contract")
                if "Contract" in df.columns:
                    contract_df = (
                        df.groupby("Contract", dropna=False)["churn_probability"]
                        .mean()
                        .reset_index()
                        .sort_values("churn_probability", ascending=False)
                    )

                    if alt is not None:
                        chart2 = (
                            alt.Chart(contract_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Contract:N", title="Contract"),
                                y=alt.Y("churn_probability:Q", title="Avg churn probability", axis=alt.Axis(format="%")),
                                tooltip=["Contract:N", alt.Tooltip("churn_probability:Q", format=".2%")],
                            )
                            .properties(height=340)
                        )
                        st.altair_chart(chart2, use_container_width=True)
                    else:
                        st.dataframe(contract_df, use_container_width=True, hide_index=True)

                    st.caption("Story: month-to-month contracts typically carry higher churn risk.")
                else:
                    st.info("`Contract` column not found (still okay).")

            st.write("")

            # NEW: two extra charts (you asked for BOTH)
            x1, x2 = st.columns([1.0, 1.0], gap="large")

            # Chart 3: Feature importance (Top Drivers)
            with x1:
                st.markdown("#### Top Churn Drivers (Feature Importance)")
                if feat_df.empty or "feature" not in feat_df.columns:
                    st.info("feature_importance.csv not found. Run: `python -m churn.tableau_export`")
                else:
                    top = feat_df.sort_values("importance_mean", ascending=False).head(15).copy()
                    # reverse for nicer horizontal bars (largest at top)
                    top = top.iloc[::-1]

                    if alt is not None:
                        fi_chart = (
                            alt.Chart(top)
                            .mark_bar()
                            .encode(
                                x=alt.X("importance_mean:Q", title="Importance (mean)"),
                                y=alt.Y("feature:N", sort=None, title="Feature"),
                                tooltip=["feature:N", "importance_mean:Q"],
                            )
                            .properties(height=340)
                        )
                        st.altair_chart(fi_chart, use_container_width=True)
                    else:
                        st.dataframe(top, use_container_width=True, hide_index=True)

                    st.caption("Great for presentation: explains which factors influence churn risk most.")

            # Chart 4: Precision/Recall vs Threshold
            with x2:
                st.markdown("#### Precision vs Recall vs Threshold")
                if thr_df.empty or not {"threshold", "precision", "recall"}.issubset(set(thr_df.columns)):
                    st.info("threshold_scan.csv not found. Run: `python -m churn.evaluate`")
                else:
                    tdf = thr_df.copy()
                    # Make sure numeric
                    for col in ["threshold", "precision", "recall", "predicted_churn_rate"]:
                        if col in tdf.columns:
                            tdf[col] = pd.to_numeric(tdf[col], errors="coerce")

                    long_df = tdf.melt(
                        id_vars=["threshold"],
                        value_vars=[c for c in ["precision", "recall"] if c in tdf.columns],
                        var_name="metric",
                        value_name="value",
                    )

                    if alt is not None:
                        base = (
                            alt.Chart(long_df)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("threshold:Q", title="Threshold"),
                                y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                                color=alt.Color("metric:N", title="Metric"),
                                tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "metric:N", alt.Tooltip("value:Q", format=".3f")],
                            )
                            .properties(height=300)
                        )

                        # Vertical rule for current UI threshold
                        rule = (
                            alt.Chart(pd.DataFrame({"threshold": [float(threshold)]}))
                            .mark_rule()
                            .encode(x="threshold:Q")
                        )

                        st.altair_chart(base + rule, use_container_width=True)
                    else:
                        st.dataframe(tdf, use_container_width=True, hide_index=True)

                    st.caption("This proves you didn’t pick threshold=0.5 blindly — it’s a tradeoff.")

            st.write("")

            # Table preview (like screenshot)
            st.markdown("#### Top High-Risk Customers (Preview)")
            show_cols = [c for c in ["Contract", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges", "churn_probability", "risk_bucket", "decision"] if c in df.columns]
            top_df = df.sort_values("churn_probability", ascending=False).head(25)[show_cols].copy()
            st.dataframe(top_df, use_container_width=True, hide_index=True)
            st.caption("Use this table during your demo: it shows exactly who gets targeted and why.")


# ============================================================
# TAB: Explainability
# ============================================================

with tab_explain:
    st.markdown("### 🔎 Explainability")
    st.caption("Use these visuals in your presentation to explain *why* the model behaves the way it does.")

    e1, e2 = st.columns(2, gap="large")

    with e1:
        st.markdown("#### Permutation Importance (Global)")
        if not feat_df.empty and {"feature", "importance_mean"}.issubset(set(feat_df.columns)):
            top_pi = feat_df.sort_values("importance_mean", ascending=False).head(15).copy().iloc[::-1]
            if alt is not None:
                pi_chart = (
                    alt.Chart(top_pi)
                    .mark_bar()
                    .encode(
                        x=alt.X("importance_mean:Q", title="Importance (mean)"),
                        y=alt.Y("feature:N", sort=None, title="Feature"),
                        tooltip=["feature:N", "importance_mean:Q"],
                    )
                    .properties(height=340)
                )
                st.altair_chart(pi_chart, use_container_width=True)
            else:
                st.dataframe(top_pi.iloc[::-1], use_container_width=True, hide_index=True)
            pass
        elif PATH_PERM_IMPORTANCE_PNG.exists():
            st.image(str(PATH_PERM_IMPORTANCE_PNG), use_container_width=True)
            pass
        else:
            st.warning("No feature importance export found. Run: `python -m churn.tableau_export`")

    with e2:
        st.markdown("#### SHAP Global Importance")
        shap_global_df = read_csv_if_exists(PATH_SHAP_GLOBAL_CSV)
        if not shap_global_df.empty and {"feature", "mean_abs_shap"}.issubset(set(shap_global_df.columns)):
            top_sg = shap_global_df.sort_values("mean_abs_shap", ascending=False).head(15).copy().iloc[::-1]
            if alt is not None:
                sg_chart = (
                    alt.Chart(top_sg)
                    .mark_bar()
                    .encode(
                        x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
                        y=alt.Y("feature:N", sort=None, title="Feature"),
                        tooltip=["feature:N", alt.Tooltip("mean_abs_shap:Q", format=".4f")],
                    )
                    .properties(height=340)
                )
                st.altair_chart(sg_chart, use_container_width=True)
            else:
                st.dataframe(top_sg.iloc[::-1], use_container_width=True, hide_index=True)
            pass
        elif PATH_SHAP_GLOBAL_PNG.exists():
            st.image(str(PATH_SHAP_GLOBAL_PNG), use_container_width=True)
            pass
        else:
            st.info("Missing SHAP artifacts. Run: `python -m churn.explain`")

    st.write("")

    st.markdown("#### SHAP Local Example (Waterfall)")
    shap_local_df = read_csv_if_exists(PATH_SHAP_LOCAL_CSV)
    if not shap_local_df.empty and {"feature", "shap_value"}.issubset(set(shap_local_df.columns)):
        local_top = shap_local_df.copy()
        if "abs_shap_value" in local_top.columns:
            local_top = local_top.sort_values("abs_shap_value", ascending=False)
        else:
            local_top["abs_shap_value"] = pd.to_numeric(local_top["shap_value"], errors="coerce").abs()
            local_top = local_top.sort_values("abs_shap_value", ascending=False)
        local_top = local_top.head(15).copy().iloc[::-1]
        local_top["direction"] = (pd.to_numeric(local_top["shap_value"], errors="coerce") >= 0).map(
            {True: "Increases churn risk", False: "Decreases churn risk"}
        )

        if alt is not None:
            sl_chart = (
                alt.Chart(local_top)
                .mark_bar()
                .encode(
                    x=alt.X("shap_value:Q", title="SHAP value (signed)"),
                    y=alt.Y("feature:N", sort=None, title="Feature"),
                    color=alt.Color("direction:N", title="Effect", scale=alt.Scale(domain=["Decreases churn risk", "Increases churn risk"])),
                    tooltip=[
                        "feature:N",
                        alt.Tooltip("shap_value:Q", format=".4f"),
                        alt.Tooltip("feature_value:Q", format=".4f") if "feature_value" in local_top.columns else alt.Tooltip("direction:N"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(sl_chart, use_container_width=True)
        else:
            st.dataframe(local_top.iloc[::-1], use_container_width=True, hide_index=True)
        pass
    elif PATH_SHAP_LOCAL_PNG.exists():
        st.image(str(PATH_SHAP_LOCAL_PNG), use_container_width=True)
        pass
    else:
        st.info("Missing local SHAP artifacts. Run: `python -m churn.explain`")


# ============================================================
# TAB: Business
# ============================================================

with tab_business:
    st.markdown("### 📈 Business Layer: Threshold → ROI")
    st.caption("This is what makes the project feel real: model outputs power a profitable decision.")

    b1, b2 = st.columns([1.2, 0.8], gap="large")

    with b1:
        st.markdown("#### ROI vs Threshold")
        if not roi_df.empty and {"threshold", "total_expected_value"}.issubset(set(roi_df.columns)):
            roi_plot = roi_df.copy()
            roi_plot["threshold"] = pd.to_numeric(roi_plot["threshold"], errors="coerce")
            roi_plot["total_expected_value"] = pd.to_numeric(roi_plot["total_expected_value"], errors="coerce")
            roi_plot = roi_plot.dropna(subset=["threshold", "total_expected_value"]).sort_values("threshold")

            if alt is not None:
                base = (
                    alt.Chart(roi_plot)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("threshold:Q", title="Threshold"),
                        y=alt.Y("total_expected_value:Q", title="Total expected value ($)"),
                        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("total_expected_value:Q", format=",.0f")],
                    )
                    .properties(height=340)
                )
                rule = (
                    alt.Chart(pd.DataFrame({"threshold": [float(threshold)]}))
                    .mark_rule()
                    .encode(x="threshold:Q")
                )
                st.altair_chart(base + rule, use_container_width=True)
            else:
                st.line_chart(roi_plot.set_index("threshold")["total_expected_value"])

            pass
        elif PATH_BUSINESS_PNG.exists():
            st.image(str(PATH_BUSINESS_PNG), use_container_width=True)
            pass
        else:
            st.warning("ROI export not found. Run: `python -m churn.tableau_export`")

    with b2:
        st.markdown("#### Quick ROI Table")
        if not roi_df.empty and {"threshold", "total_expected_value"}.issubset(set(roi_df.columns)):
            roi_show = roi_df.sort_values("total_expected_value", ascending=False).head(10).copy()
            st.dataframe(roi_show, use_container_width=True, hide_index=True)
        else:
            st.info("ROI table not found. Run: `python -m churn.tableau_export`")

        st.write("")
        st.markdown("#### Download ROI Export")
        if PATH_ROI_THRESHOLDS.exists():
            st.download_button(
                "⬇️ Download roi_thresholds.csv",
                data=PATH_ROI_THRESHOLDS.read_bytes(),
                file_name="roi_thresholds.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# TAB: Logs & Batch
# ============================================================

with tab_logs:
    st.markdown("### 🧾 Logs + Batch Scoring")
    st.caption("API logs predictions. You can also score a CSV in bulk.")

    l1, l2 = st.columns([1.25, 0.75], gap="large")

    with l1:
        st.markdown("#### Recent Predictions Log")
        logs_df = read_csv_if_exists(PATH_LOGS)
        if logs_df.empty:
            st.info("No logs found yet. Make predictions in the Predict tab (API logs them).")
        else:
            show_n = st.slider("Show last N rows", 10, 500, 50, 10)
            st.dataframe(logs_df.tail(show_n), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download predictions_log.csv",
                data=PATH_LOGS.read_bytes(),
                file_name="predictions_log.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with l2:
        st.markdown("#### Batch Score a CSV")
        st.write("Upload a CSV with the same input columns as the form (one row per customer).")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                st.write("Preview:")
                st.dataframe(df_in.head(10), use_container_width=True)

                if st.button("Score File via API", use_container_width=True):
                    if not api_ok:
                        st.error("API offline. Start: `python -m churn.api`")
                    else:
                        rows: List[Dict[str, Any]] = df_in.to_dict(orient="records")
                        with st.spinner("Scoring batch..."):
                            r = requests.post(f"{API_URL}/predict_batch", json=rows, timeout=45)
                            r.raise_for_status()
                            out = r.json()
                            probs = [float(x["churn_probability"]) for x in out["results"]]
                            df_out = df_in.copy()
                            df_out["churn_probability"] = probs
                            df_out["risk_bucket"] = df_out["churn_probability"].map(risk_bucket)
                            st.success(f"Scored {len(df_out)} rows.")
                            st.dataframe(df_out.head(20), use_container_width=True)

                            st.download_button(
                                "⬇️ Download scored CSV",
                                data=df_out.to_csv(index=False).encode("utf-8"),
                                file_name="scored_customers.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read/score CSV: {e}")