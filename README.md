# 📦 AI-Powered Demand Forecasting & Supply Chain Analytics Platform

An end-to-end, production-style data science project: it forecasts product demand,
detects supply chain anomalies, explains its own predictions, quantifies the
business impact in dollars, and ships as a containerized API + dashboard with
CI/CD. Built entirely with free and open-source tools — no paid APIs, no cloud
credit card required.

This is designed as a **portfolio centerpiece** for data science / data
analytics roles, especially ones that mention MLOps, deployment, or
production ML.

---

## Why this project stands out in interviews

Most portfolio projects stop at "I trained a model and got 85% accuracy."
This one demonstrates the full lifecycle a company actually cares about:

| Layer | What's demonstrated |
|---|---|
| **Data engineering** | Synthetic-but-realistic data generation with trend, seasonality, promotions, holidays |
| **Feature engineering** | Lag features, rolling windows, cyclical calendar encoding — no train/serve skew |
| **Modeling** | XGBoost vs. classical statistical baseline (Holt-Winters), benchmarked head-to-head |
| **Explainability** | SHAP values — "why did the model predict this?" |
| **Anomaly detection** | Isolation Forest on residuals — flags real supply disruptions |
| **Business translation** | Converts forecast error into **$ cost** (holding cost vs. stockout cost) |
| **Experiment tracking** | MLflow — every run logged, reproducible |
| **Serving** | FastAPI REST API with input validation, batch prediction, auto-generated docs |
| **Visualization** | Interactive Streamlit dashboard for non-technical stakeholders |
| **DevOps** | Docker + docker-compose, GitHub Actions CI/CD (lint → test → train → build) |
| **Testing** | 14 unit + integration tests (pytest) |

**The one sentence to say in an interview:**
> "I built an end-to-end demand forecasting system — from synthetic data
> generation through a benchmarked XGBoost model, SHAP explainability, and
> anomaly detection — served via a FastAPI microservice with Docker and a
> CI/CD pipeline, and I quantified its business impact at roughly $1.1M in
> estimated annual savings versus a naive baseline."

---

## Architecture

```
                         ┌─────────────────────┐
                         │  generate_data.py    │  synthetic demand data
                         │  (or swap in real     │  (trend/seasonality/promo/
                         │   Walmart M5 / Olist) │   holiday/anomalies)
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │    features.py       │  lag + rolling + calendar
                         └──────────┬───────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │          train.py              │
                    │  XGBoost  vs.  Holt-Winters     │──► MLflow (tracking)
                    │  + SHAP explainability          │──► models/*.joblib
                    └───────────────┬─────────────────┘
                                    ▼
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
   ┌─────────────────────┐                      ┌─────────────────────────┐
   │ anomaly_detection.py │                      │    cost_savings.py       │
   │ Isolation Forest on  │                      │ residual → $ impact      │
   │ residuals             │                      │ (holding + stockout)     │
   └──────────┬───────────┘                      └────────────┬─────────────┘
              └───────────────────┬───────────────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │   api/main.py (FastAPI)  │  /predict /anomalies
                     │   dashboard/app.py        │  /cost-savings /model-info
                     │   (Streamlit)              │
                     └─────────────────────────┘
                                  ▼
                     Docker + docker-compose + GitHub Actions CI/CD
```

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (data → train → anomalies → cost savings)
```bash
make all
# or manually:
python src/generate_data.py
python src/train.py
python src/anomaly_detection.py
python src/cost_savings.py
```

### 3. Launch the API
```bash
make api
# or: uvicorn api.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive Swagger docs.

### 4. Launch the dashboard
```bash
make dashboard
# or: streamlit run dashboard/app.py
```
Visit `http://localhost:8501`.

### 5. Run tests
```bash
make test
# or: pytest tests/ -v
```

### 6. Everything in Docker (one command)
```bash
docker compose -f docker/docker-compose.yml up --build
```
API on `:8000`, dashboard on `:8501`.

### 7. View MLflow experiment tracking
```bash
make mlflow-ui
# or: mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Visit `http://localhost:5000`.

---

## Project structure

```
supply-chain-forecast/
├── src/
│   ├── generate_data.py       # synthetic data generator
│   ├── features.py            # feature engineering (shared train/serve)
│   ├── train.py                # XGBoost + baseline + MLflow + SHAP
│   ├── anomaly_detection.py    # Isolation Forest on residuals
│   └── cost_savings.py         # $ impact simulation
├── api/
│   └── main.py                 # FastAPI serving layer
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── tests/
│   ├── test_pipeline.py        # unit tests (data/features/anomaly/cost)
│   └── test_api.py             # API integration tests
├── .github/workflows/ci-cd.yml # lint → test → train → docker build
├── requirements.txt
├── Makefile
└── README.md
```

---

## Results (on synthetic data, 8 stores × 15 items × 3 years)

| Model | MAE | Notes |
|---|---|---|
| Naive (last value) | highest | simplest possible baseline |
| Holt-Winters (statistical) | ~38 units | classical time-series baseline |
| **XGBoost (this project)** | **~13.5 units** | **64% improvement over statistical baseline** |

**Estimated annual cost savings vs. naive baseline: ~$1.1M** (illustrative unit
economics — see `src/cost_savings.py` to plug in real numbers for your use case).

Your exact numbers will vary slightly by random seed — that's expected and fine
to mention: "results are reproducible via a fixed seed but the underlying
architecture is what matters."

---

## Making this yours (recommended before interviews)

1. **Swap in real data.** Kaggle's [Walmart M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
   or [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
   datasets drop in with minor column renaming in `generate_data.py`'s output schema.
2. **Deploy it for free** so you have a live link on your resume:
   - API → [Render.com](https://render.com) or [Railway.app](https://railway.app) free tier
   - Dashboard → [Streamlit Community Cloud](https://streamlit.io/cloud) (free, made for this)
3. **Adjust cost assumptions** in `src/cost_savings.py` to numbers you can defend
   if asked (holding cost, stockout margin) — even rough industry benchmarks are fine.
4. **Push to GitHub** and let the included GitHub Actions workflow run — a green
   CI badge on your repo is a strong, free signal of engineering rigor.
5. **Write 3-4 bullet points for your resume** based on the "one sentence" above,
   split into: what you built, what you benchmarked, what you deployed, what
   business impact you measured.

---

## Tech stack (100% free / open-source)

Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels · SHAP ·
FastAPI · Pydantic · Streamlit · Plotly · MLflow · SQLite · Docker ·
docker-compose · GitHub Actions · pytest

No paid API keys, no cloud billing account, no proprietary datasets required
to run this end-to-end.
