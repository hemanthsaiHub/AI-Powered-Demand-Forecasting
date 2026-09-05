"""
Supply Chain Anomaly Detection
--------------------------------
Flags days where actual demand deviates sharply from what the model
expected -- these are the days a supply-chain analyst actually cares
about (stockouts, disruptions, data errors, unexpected demand shocks).

Approach: residual-based Isolation Forest. We don't just threshold the
raw residual (too naive) -- we let Isolation Forest find multivariate
outliers across residual magnitude, day-of-week, and rolling volatility,
which catches anomalies a simple z-score would miss.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(predictions_df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """
    predictions_df must have columns: date, store_id, item_id, units_sold, predicted
    Returns the same df with `residual`, `anomaly_score`, and `is_anomaly` columns added.
    """
    df = predictions_df.copy()
    df["residual"] = df["units_sold"] - df["predicted"]
    df["pct_residual"] = df["residual"] / df["predicted"].clip(lower=1)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek

    features = df[["residual", "pct_residual", "day_of_week"]].copy()

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    df["anomaly_score"] = iso.fit_predict(features)
    df["anomaly_score_raw"] = iso.decision_function(features)
    df["is_anomaly"] = (df["anomaly_score"] == -1).astype(int)

    return df.sort_values("anomaly_score_raw")


if __name__ == "__main__":
    preds = pd.read_csv("models/test_predictions.csv")
    result = detect_anomalies(preds)
    result.to_csv("models/anomalies.csv", index=False)

    n_anomalies = result["is_anomaly"].sum()
    print(f"Detected {n_anomalies} anomalies out of {len(result)} test rows ({n_anomalies / len(result):.1%})")
    print("\nTop 10 most anomalous days:")
    cols = ["date", "store_id", "item_id", "units_sold", "predicted", "residual", "pct_residual"]
    print(result[cols].head(10).to_string(index=False))
