PY=python

.PHONY: help setup setup-xgb data train evaluate explain business tableau api app monitor all test lint

help:
	@echo "Targets:"
	@echo "  setup       - install deps"
	@echo "  setup-xgb   - install optional XGBoost"
	@echo "  data        - download + clean dataset"
	@echo "  train       - train + save best model"
	@echo "  evaluate    - metrics + threshold scan"
	@echo "  explain     - permutation + SHAP plots"
	@echo "  business    - ROI simulation + best threshold"
	@echo "  tableau     - export Tableau-ready CSVs"
	@echo "  api         - run FastAPI locally"
	@echo "  app         - run Streamlit locally"
	@echo "  monitor     - drift report (Evidently)"
	@echo "  all         - data -> train -> evaluate -> explain -> business -> tableau"
	@echo "  test        - run pytest"
	@echo "  lint        - ruff check ."

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

setup-xgb:
	pip install -r requirements-optional.txt

data:
	$(PY) -m churn.data --download

train:
	$(PY) -m churn.train

evaluate:
	$(PY) -m churn.evaluate

explain:
	$(PY) -m churn.explain

business:
	$(PY) -m churn.business

tableau:
	$(PY) -m churn.tableau_export

api:
	$(PY) -m churn.api

app:
	streamlit run app/streamlit_app.py

monitor:
	$(PY) -m churn.monitor

all:
	$(PY) -m churn.data --download
	$(PY) -m churn.train
	$(PY) -m churn.evaluate
	$(PY) -m churn.explain
	$(PY) -m churn.business
	$(PY) -m churn.tableau_export

test:
	pytest -q

lint:
	ruff check .
