"""
Feature Engineering
--------------------
Builds lag, rolling-window, and calendar features for the forecasting model.
Kept as pure functions so they can be reused identically at training time
and at inference time (avoids train/serve skew, a very common MLOps bug).
"""

import pandas as pd
import numpy as np


LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    # cyclical encodings so the model understands wraparound (Dec -> Jan)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["store_id", "item_id", "date"])
    group = df.groupby(["store_id", "item_id"])["units_sold"]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = group.shift(lag)

    df["_shifted"] = group.shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df.groupby(["store_id", "item_id"])["_shifted"].transform(
            lambda x: x.rolling(window).mean()
        )
        df[f"rolling_std_{window}"] = df.groupby(["store_id", "item_id"])["_shifted"].transform(
            lambda x: x.rolling(window).std()
        )
    df = df.drop(columns=["_shifted"])

    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = (
    [f"lag_{lag_day}" for lag_day in LAG_DAYS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + ["day_of_week", "is_weekend", "is_promo", "is_holiday",
       "month", "day_of_month", "week_of_year", "quarter",
       "month_sin", "month_cos", "dow_sin", "dow_cos",
       "store_id", "item_id"]
)

TARGET_COLUMN = "units_sold"
