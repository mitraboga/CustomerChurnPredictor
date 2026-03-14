<h1 align="center">📊 Customer Churn Predictor</h1>
<h3 align="center">Machine Learning Risk Intelligence + Executive Business Dashboard</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Model-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/FastAPI-Model%20API-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-SaaS%20Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Evidently-Drift%20Monitoring-6B46C1" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-111827" />
  <img src="https://img.shields.io/badge/Tableau-Data%20Visualization-E97627?logo=tableau" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Churn%20Prediction-brightgreen" />
  <img src="https://img.shields.io/badge/Analytics-Business%20Intelligence-blueviolet" />
  <img src="https://img.shields.io/badge/Architecture-End--to--End%20ML%20System-0EA5E9" />
</p>

<p align="center">
  <!-- Replace the src below with your centered demo GIF path, e.g. assets/demo.gif -->
  <img src="assets/demo.gif" width="85%" alt="Customer Churn Predictor Demo" />
</p>

---

## 📖 Project Overview

Customer churn is one of the **most critical problems in subscription businesses**.  
Losing customers directly impacts **revenue, growth, and acquisition costs**.

**CustomerChurnPredictor** is an end-to-end churn risk system that combines:

- ✅ Machine Learning prediction (probability scoring)
- ✅ Risk segmentation (buckets + deciles)
- ✅ ROI-based decisioning (threshold → business value)
- ✅ FastAPI inference service (single + batch prediction)
- ✅ Streamlit modern SaaS dashboard (interactive demo + analytics)
- ✅ Monitoring with Evidently (data drift report)
- ✅ Tableau dashboards for executive stakeholders

The solution is designed to answer:

> **“Who is most likely to churn next, and what should we do about it?”**

---

## 🎯 What Makes This Project Portfolio-Grade

Most churn projects stop at accuracy. This project goes further:

- **Probability → Decision Policy:** Intervene only above a chosen threshold  
- **Threshold is ROI-driven:** We simulate expected value across thresholds (not random 0.50 defaults)  
- **Explainability built-in:** SHAP + permutation importance  
- **Deployed system:** API + UI + logs + monitoring  
- **Executive dashboards:** Tableau-ready exports + dashboard suite

---

## 🧱 System Architecture

```
                ┌──────────────────────────────┐
                │   Telco Churn Dataset (CSV)  │
                └───────────────┬──────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │  Data Cleaning + Prep     │
                  │  (Pandas + sklearn pipes) │
                  └───────────────┬──────────┘
                                  │
                                  ▼
                  ┌──────────────────────────┐
                  │  Model Training + Eval    │
                  │  (LogReg vs RF + Metrics) │
                  └───────────────┬──────────┘
                                  │
                                  ▼
                  ┌──────────────────────────┐
                  │  Explainability           │
                  │  (Permutation + SHAP)     │
                  └───────────────┬──────────┘
                                  │
                                  ▼
                  ┌──────────────────────────┐
                  │ Business Simulation (ROI) │
                  │ Threshold → Expected Value│
                  └───────────────┬──────────┘
                                  │
                  ┌───────────────┴───────────────────────────────┐
                  │                                               │
                  ▼                                               ▼
       ┌──────────────────────────┐                 ┌──────────────────────────┐
       │ FastAPI Inference API     │                 │ Tableau Dashboard Layer   │
       │ /predict + /predict_batch │                 │ (Executive + ML Risk)     │
       └───────────────┬──────────┘                 └───────────────┬──────────┘
                       │                                            │
                       ▼                                            ▼
         ┌──────────────────────────┐                ┌──────────────────────────┐
         │ Streamlit SaaS Dashboard  │                │ Tableau-ready CSV Exports │
         │ Decisions + Analytics     │                │ cleaned/scored/ROI/drivers│
         └───────────────┬──────────┘                └──────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │ Monitoring (Evidently)    │
            │ Drift report from logs    │
            └──────────────────────────┘
```

---

## 🗂️ Repository File Structure

```
CustomerChurnPredictor/
├─ churn/                      # Core ML + API + monitoring modules
│  ├─ data.py                  # Download + clean dataset
│  ├─ modeling.py              # Preprocess + candidate models
│  ├─ train.py                 # Train + save best model
│  ├─ evaluate.py              # Metrics + confusion matrix + threshold scan
│  ├─ explain.py               # Permutation + SHAP explainability
│  ├─ business.py              # ROI simulation + best threshold
│  ├─ tableau_export.py        # Exports final Tableau-ready CSVs
│  ├─ api.py                   # FastAPI inference service + logging
│  ├─ monitor.py               # Evidently drift report
│  └─ config.py                # Paths, columns, business defaults
│
├─ app/
│  └─ streamlit_app.py         # Modern SaaS Streamlit UI
│
├─ data/                       # Local-only data (ignored in git except placeholder)
│  ├─ raw/
│  ├─ processed/
│  ├─ tableau/                 # Exports for Tableau dashboards
│  └─ logs/                    # API prediction logs
│
├─ models/                     # Saved model artifact (model.joblib) + metadata
├─ reports/
│  ├─ figures/                 # Explainability + ROI plots (PNG/CSV)
│  ├─ metrics/                 # Model metrics, threshold scan, best threshold
│  └─ monitoring/              # Drift report HTML
│
├─ tableau/                    # Tableau workbook (.twbx) + screenshots
│  └─ screenshots/
│
├─ tests/                      # Basic CI tests
├─ requirements.txt
├─ requirements-dev.txt
├─ Makefile
└─ README.md
```

---

## 🧠 Churn Prediction Pipeline

```
Raw Telco Dataset
        ↓
Data Cleaning & Feature Engineering (Pandas)
        ↓
Preprocessing Pipeline (Impute + Scale + OneHotEncode)
        ↓
Classification Models (LogReg, RF)
        ↓
Churn Probability Scores (0–1)
        ↓
Risk Segmentation (Buckets + Deciles)
        ↓
ROI-Based Threshold Decisioning
        ↓
FastAPI + Streamlit + Monitoring + Tableau Dashboards
```

---

## 📊 Tableau Dashboards (Two-Dashboard Workflow)

> You requested a 2-dashboard workflow:
- **Dashboard 1: Churn Overview (Executive / Business Analysis)**
- **Dashboard 2: ML Risk Intelligence (Predictive + Decision Layer)**

### Dashboard 1 — Churn Overview (Business Analysis)
Focus: historical churn patterns and segmentation insights.

**KPIs**
- Total Customers
- Churn Rate
- Avg Monthly Charges
- Avg Tenure

**Visuals**
- Churn by Contract Type
- Churn by Internet Service
- Churn by Payment Method
- Churn by Tenure Bucket
- Interactive filters (Contract, InternetService, PaymentMethod, SeniorCitizen)

---

### Dashboard 2 — ML Risk Intelligence (Predictive)
Focus: who will churn next and what to do.

**KPIs**
- Avg Churn Probability
- High Risk Count (≥ threshold)
- Targeted Customers (decision policy)

**Visuals**
- Churn probability distribution
- Risk decile breakdown (Top 10% = Decile 10)
- High-risk customers table
- ROI threshold curve (Total Expected Value vs Threshold)

---

## 📦 Tableau Data Files (Ready to Connect)

After running:
```bash
python -m churn.tableau_export
```

Tableau-ready exports are generated in:
- `data/tableau/telco_cleaned.csv`  
- `data/tableau/telco_scored.csv`  
- `data/tableau/roi_thresholds.csv`  
- `data/tableau/feature_importance.csv`  
- `data/tableau/threshold_scan.csv`  

---

## 🖥️ Streamlit SaaS Dashboard

The Streamlit app includes:
- KPI cards (model performance + threshold)
- Predict tab (decision + EV per customer)
- Analytics dashboard tab (stacked distributions + drivers + threshold tradeoffs)
- Explainability tab (Permutation + SHAP global/local)
- Business tab (ROI curve + threshold strategy)
- Logs & batch scoring tab (CSV upload + /predict_batch)

---

## 🌐 FastAPI Inference Service

Endpoints:
- `GET /health` — service status  
- `POST /predict` — score one customer  
- `POST /predict_batch` — score many rows (batch scoring)

All predictions are logged to:
- `data/logs/predictions_log.csv`

This log is used for monitoring drift.

---

## 📈 Monitoring (Evidently Drift Report)

Run:
```bash
python -m churn.monitor
```

Output:
- `reports/monitoring/data_drift_report.html`

This compares:
- reference sample (saved during training)
- current inference logs (from API)

---

## 🚀 How to Run (End-to-End)

### 1) Setup
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2) Build the full pipeline
```bash
python -m churn.data --download
python -m churn.train
python -m churn.evaluate
python -m churn.explain
python -m churn.business
python -m churn.tableau_export
```

### 3) Run API + UI
Terminal A:
```bash
python -m churn.api
```

Terminal B:
```bash
python -m streamlit run app/streamlit_app.py
```

---

## 💡 Key Insights (Examples)

- Month-to-month contracts are consistently the **highest churn risk**
- Long-term contracts (1–2 year) strongly reduce churn likelihood
- Churn risk is concentrated: a smaller segment can represent a large share of revenue exposure
- Probability-based segmentation enables targeted retention strategies instead of broad campaigns

---

## 📷 Example Dashboard Views

*(Insert screenshots here)*

### Tableau — Churn Overview
<img src="assets/dashboard_business.png" width="100%">

### Tableau — ML Risk Intelligence
<img src="assets/dashboard_ml.png" width="100%">

### Streamlit — Modern SaaS UI
<img src="assets/streamlit_ui.png" width="100%">

---

## 📈 Potential Business Applications

Companies can use this system to:
- identify high-risk customers early
- deploy targeted retention campaigns
- improve contract conversion strategies
- protect recurring revenue with ROI-optimized decisions

---

## 👤 Author

<p align="center">
<b>Mitra Boga</b><br>
<a href="https://www.linkedin.com/in/bogamitra/">
<img src="https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin" />
</a>
<a href="https://x.com/techtraboga">
<img src="https://img.shields.io/badge/X-Follow-black?logo=x" />
</a>
</p>
