"""
Model Training Pipeline
------------------------
Trains two models for comparison (a resume talking point: "benchmarked
ML vs classical statistical forecasting"):

1. XGBoost regressor  -> main production model (uses lag/rolling/calendar features)
2. Exponential Smoothing (Holt-Winters) -> classical statistical baseline, per store/item

Also:
- Logs experiments to MLflow (local file store, 100% free, no cloud account needed)
- Computes SHAP values for explainability
- Saves the trained model + feature list + metrics to /models
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import shap
import warnings

from features import build_feature_frame, FEATURE_COLUMNS, TARGET_COLUMN

warnings.filterwarnings("ignore")

DATA_PATH = "data/demand_data.csv"
MODEL_DIR = "models"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feat_df = build_feature_frame(df)

    split_date = feat_df["date"].quantile(0.85, interpolation="nearest")
    train_df = feat_df[feat_df["date"] < split_date]
    test_df = feat_df[feat_df["date"] >= split_date]

    return train_df, test_df, feat_df


def train_xgboost(train_df, test_df):
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mape": float(mean_absolute_percentage_error(y_test.clip(lower=1), preds.clip(min=1))),
    }
    return model, metrics, (X_test, y_test, preds)


def train_baseline_exponential_smoothing(df, n_series=20):
    """Classical statistical baseline, sampled on a subset of store/item
    series (fitting ES per series across all 120 combos is slow and not
    the point of the demo -- the comparison is what matters)."""
    df = df.sort_values("date")
    combos = df[["store_id", "item_id"]].drop_duplicates().sample(
        n=min(n_series, df[["store_id", "item_id"]].drop_duplicates().shape[0]),
        random_state=42,
    )

    errors = []
    for _, row in combos.iterrows():
        series = df[(df.store_id == row.store_id) & (df.item_id == row.item_id)].set_index("date")["units_sold"]
        split = int(len(series) * 0.85)
        train, test = series.iloc[:split], series.iloc[split:]
        if len(train) < 30 or len(test) < 5:
            continue
        try:
            fit = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=7
            ).fit()
            preds = fit.forecast(len(test))
            mae = mean_absolute_error(test, preds)
            errors.append(mae)
        except Exception:
            continue

    return {"mae": float(np.mean(errors))} if errors else {"mae": None}


def compute_shap_summary(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    return importance


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("demand-forecasting")

    print("Loading data & building features...")
    train_df, test_df, full_feat_df = load_and_prepare_data()
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    with mlflow.start_run(run_name="xgboost_main"):
        print("\nTraining XGBoost model...")
        model, metrics, (X_test, y_test, preds) = train_xgboost(train_df, test_df)
        print(f"XGBoost  -> MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | MAPE: {metrics['mape']:.2%}")

        mlflow.log_params({"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print("\nTraining statistical baseline (Exponential Smoothing) for comparison...")
        raw_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        baseline_metrics = train_baseline_exponential_smoothing(raw_df)
        if baseline_metrics["mae"]:
            print(f"Baseline (Holt-Winters) -> MAE: {baseline_metrics['mae']:.2f}")
        else:
            print("Baseline failed")
        mlflow.log_metric("baseline_mae", baseline_metrics["mae"] or -1)

        improvement = None
        if baseline_metrics["mae"]:
            improvement = (baseline_metrics["mae"] - metrics["mae"]) / baseline_metrics["mae"]
            print(f"\nXGBoost improves MAE over statistical baseline by {improvement:.1%}")

        print("\nComputing SHAP feature importance...")
        shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
        importance = compute_shap_summary(model, shap_sample)
        importance.to_csv(f"{MODEL_DIR}/shap_importance.csv", index=False)
        print(importance.head(8).to_string(index=False))

        # Save artifacts
        joblib.dump(model, f"{MODEL_DIR}/xgboost_model.joblib")
        with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
            json.dump(FEATURE_COLUMNS, f)
        with open(f"{MODEL_DIR}/metrics.json", "w") as f:
            json.dump({
                "xgboost": metrics,
                "baseline_exponential_smoothing": baseline_metrics,
                "improvement_over_baseline": improvement,
            }, f, indent=2)

        # Save a slice of test predictions for the dashboard
        results_df = test_df[["date", "store_id", "item_id", "units_sold"]].copy()
        results_df["predicted"] = preds
        results_df.to_csv(f"{MODEL_DIR}/test_predictions.csv", index=False)

        print(f"\nAll artifacts saved to {MODEL_DIR}/")
        print("MLflow run logged. Launch UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
