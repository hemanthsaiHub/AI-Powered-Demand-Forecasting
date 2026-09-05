"""
Unit tests for the demand forecasting pipeline.
Run with: pytest tests/ -v
"""

import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_data import generate_demand_data
from features import build_feature_frame, add_calendar_features, FEATURE_COLUMNS, TARGET_COLUMN
from anomaly_detection import detect_anomalies
from cost_savings import simulate_cost


@pytest.fixture(scope="module")
def sample_data():
    # Small, fast dataset for testing (not the full 3-year generator)
    import generate_data as gd
    original_start, original_end = gd.START_DATE, gd.END_DATE
    gd.START_DATE, gd.END_DATE = "2023-01-01", "2023-06-30"
    df = generate_demand_data(seed=1)
    gd.START_DATE, gd.END_DATE = original_start, original_end
    return df


def test_data_generation_shape(sample_data):
    assert len(sample_data) > 0
    expected_cols = {"date", "store_id", "item_id", "units_sold", "is_promo",
                      "is_holiday", "day_of_week", "is_weekend", "true_anomaly"}
    assert expected_cols.issubset(set(sample_data.columns))


def test_no_negative_demand(sample_data):
    assert (sample_data["units_sold"] >= 0).all()


def test_no_extreme_values(sample_data):
    # Regression guard for the exponential-trend blowup bug
    assert sample_data["units_sold"].max() < 100_000


def test_calendar_features_add_expected_columns(sample_data):
    df = add_calendar_features(sample_data)
    for col in ["month", "day_of_month", "week_of_year", "quarter",
                "month_sin", "month_cos", "dow_sin", "dow_cos"]:
        assert col in df.columns


def test_feature_frame_has_no_nulls(sample_data):
    feat_df = build_feature_frame(sample_data)
    assert feat_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum() == 0


def test_feature_frame_smaller_than_input(sample_data):
    # Lag features should drop the first N rows per group
    feat_df = build_feature_frame(sample_data)
    assert len(feat_df) < len(sample_data)


def test_anomaly_detection_runs_and_flags_subset():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "store_id": 1,
        "item_id": 1,
        "units_sold": np.concatenate([np.random.normal(50, 5, 95), [200, 1, 210, 2, 190]]),
        "predicted": 50,
    })
    result = detect_anomalies(df, contamination=0.05)
    assert "is_anomaly" in result.columns
    assert result["is_anomaly"].sum() > 0
    assert result["is_anomaly"].sum() < len(result)  # shouldn't flag everything


def test_cost_simulation_basic():
    df = pd.DataFrame({
        "units_sold": [100, 90, 110, 80],
        "predicted": [100, 100, 100, 100],
    })
    result = simulate_cost(df)
    assert result["total_cost"] >= 0
    assert result["holding_cost"] >= 0
    assert result["stockout_cost"] >= 0
    # units_sold=90,80 are overforecasts (holding); 110 is underforecast (stockout)
    assert result["stockout_cost"] > 0
    assert result["holding_cost"] > 0


def test_cost_simulation_perfect_forecast_has_zero_cost():
    df = pd.DataFrame({"units_sold": [50, 60, 70], "predicted": [50, 60, 70]})
    result = simulate_cost(df)
    assert result["total_cost"] == 0
