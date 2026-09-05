"""
Integration tests for the FastAPI serving layer.
Requires a trained model (run src/train.py first).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "store_id": 1, "item_id": 1, "lag_1": 80, "lag_7": 75, "lag_14": 70, "lag_28": 65,
    "rolling_mean_7": 77, "rolling_mean_14": 74, "rolling_mean_28": 70,
    "rolling_std_7": 5, "rolling_std_14": 6, "rolling_std_28": 7,
    "day_of_week": 5, "is_weekend": 1, "is_promo": 0, "is_holiday": 0,
    "month": 12, "day_of_month": 20, "week_of_year": 51, "quarter": 4,
    "month_sin": -0.5, "month_cos": 0.87, "dow_sin": -0.97, "dow_cos": -0.22,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict():
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_units" in body
    assert body["predicted_units"] >= 0


def test_predict_batch():
    resp = client.post("/predict/batch", json=[SAMPLE_PAYLOAD, SAMPLE_PAYLOAD])
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_predict_rejects_invalid_input():
    bad_payload = dict(SAMPLE_PAYLOAD)
    bad_payload["day_of_week"] = 10  # out of range (must be 0-6)
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    assert "metrics" in resp.json()
