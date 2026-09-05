"""
FastAPI Serving Layer
-----------------------
Exposes the trained XGBoost model as a REST API.

Endpoints:
  GET  /health                -> liveness check
  GET  /model-info             -> metrics, feature list, training info
  POST /predict                -> single/batch demand prediction
  GET  /anomalies              -> precomputed anomaly list
  GET  /cost-savings            -> business impact summary

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import json
import sys
import os
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import FEATURE_COLUMNS  # noqa: E402

MODEL_PATH = "models/xgboost_model.joblib"
METRICS_PATH = "models/metrics.json"
ANOMALIES_PATH = "models/anomalies.csv"
COST_SAVINGS_PATH = "models/cost_savings.json"

app = FastAPI(
    title="Demand Forecasting API",
    description="Serves XGBoost demand forecasts, anomaly flags, and cost-savings estimates.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictionRequest(BaseModel):
    store_id: int = Field(..., ge=1, description="Store identifier")
    item_id: int = Field(..., ge=1, description="Item identifier")
    lag_1: float = Field(..., description="Units sold 1 day ago")
    lag_7: float = Field(..., description="Units sold 7 days ago")
    lag_14: float = Field(..., description="Units sold 14 days ago")
    lag_28: float = Field(..., description="Units sold 28 days ago")
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_28: float
    rolling_std_7: float
    rolling_std_14: float
    rolling_std_28: float
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_promo: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    month: int = Field(..., ge=1, le=12)
    day_of_month: int = Field(..., ge=1, le=31)
    week_of_year: int = Field(..., ge=1, le=53)
    quarter: int = Field(..., ge=1, le=4)
    month_sin: float
    month_cos: float
    dow_sin: float
    dow_cos: float


class PredictionResponse(BaseModel):
    predicted_units: float
    store_id: int
    item_id: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics not found. Run training first.")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return {
        "model_type": "XGBoost Regressor",
        "features_used": FEATURE_COLUMNS,
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = get_model()
    row = request.model_dump()
    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred = float(model.predict(X)[0])
    pred = max(pred, 0.0)
    return PredictionResponse(
        predicted_units=round(pred, 2),
        store_id=request.store_id,
        item_id=request.item_id,
    )


@app.post("/predict/batch")
def predict_batch(requests: List[PredictionRequest]):
    model = get_model()
    rows = [r.model_dump() for r in requests]
    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    preds = model.predict(X)
    preds = [max(float(p), 0.0) for p in preds]
    return [
        {"predicted_units": round(p, 2), "store_id": r["store_id"], "item_id": r["item_id"]}
        for p, r in zip(preds, rows)
    ]


@app.get("/anomalies")
def get_anomalies(limit: int = 20, store_id: Optional[int] = None, item_id: Optional[int] = None):
    if not os.path.exists(ANOMALIES_PATH):
        raise HTTPException(status_code=404, detail="Anomalies not computed. Run src/anomaly_detection.py first.")
    df = pd.read_csv(ANOMALIES_PATH)
    df = df[df["is_anomaly"] == 1]
    if store_id is not None:
        df = df[df["store_id"] == store_id]
    if item_id is not None:
        df = df[df["item_id"] == item_id]
    df = df.sort_values("anomaly_score_raw").head(limit)
    return df.to_dict(orient="records")


@app.get("/cost-savings")
def get_cost_savings():
    if not os.path.exists(COST_SAVINGS_PATH):
        raise HTTPException(status_code=404, detail="Cost savings not computed. Run src/cost_savings.py first.")
    with open(COST_SAVINGS_PATH) as f:
        return json.load(f)


@app.get("/")
def root():
    return {
        "message": "Demand Forecasting API",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict", "/predict/batch", "/anomalies", "/cost-savings"],
    }
