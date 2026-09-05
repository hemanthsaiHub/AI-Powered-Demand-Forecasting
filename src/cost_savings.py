"""
Cost Savings Simulator
------------------------
Translates forecast error into a dollar figure -- the single most
important thing to have ready for an interview: "my model saves ~$X"
is a much stronger sentence than "my MAE is Y".

Simple, defensible model:
- Overforecasting -> excess inventory -> holding cost per unit/day
- Underforecasting -> stockouts -> lost margin per unit
Compares the XGBoost model's cost vs. the classical baseline's cost
vs. a "naive last value" forecast, using the same held-out test period.
"""

import pandas as pd
import numpy as np
import json

HOLDING_COST_PER_UNIT_DAY = 0.15   # $ cost to hold 1 excess unit for 1 day
STOCKOUT_MARGIN_LOSS_PER_UNIT = 4.50  # $ lost margin per unit of unmet demand


def simulate_cost(df: pd.DataFrame, actual_col="units_sold", pred_col="predicted") -> dict:
    residual = df[actual_col] - df[pred_col]  # positive = underforecast (stockout risk)
    overforecast_units = np.clip(-residual, 0, None)
    underforecast_units = np.clip(residual, 0, None)

    holding_cost = (overforecast_units * HOLDING_COST_PER_UNIT_DAY).sum()
    stockout_cost = (underforecast_units * STOCKOUT_MARGIN_LOSS_PER_UNIT).sum()
    total_cost = holding_cost + stockout_cost

    return {
        "holding_cost": round(float(holding_cost), 2),
        "stockout_cost": round(float(stockout_cost), 2),
        "total_cost": round(float(total_cost), 2),
        "n_days": len(df),
    }


def naive_last_value_forecast(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Simplest possible baseline: tomorrow's demand = today's demand."""
    df = raw_df.sort_values(["store_id", "item_id", "date"]).copy()
    df["naive_pred"] = df.groupby(["store_id", "item_id"])["units_sold"].shift(1)
    return df.dropna(subset=["naive_pred"])


def main():
    test_preds = pd.read_csv("models/test_predictions.csv", parse_dates=["date"])
    raw = pd.read_csv("data/demand_data.csv", parse_dates=["date"])

    xgb_cost = simulate_cost(test_preds, pred_col="predicted")

    naive_df = naive_last_value_forecast(raw)
    naive_test = naive_df.merge(
        test_preds[["date", "store_id", "item_id"]],
        on=["date", "store_id", "item_id"], how="inner"
    )
    naive_cost = simulate_cost(naive_test, pred_col="naive_pred")

    savings_vs_naive = naive_cost["total_cost"] - xgb_cost["total_cost"]
    savings_pct = savings_vs_naive / naive_cost["total_cost"] if naive_cost["total_cost"] else 0

    # Annualize: scale the test-period cost gap up to a full 365-day year
    days_covered = test_preds["date"].nunique()
    annual_factor = 365 / days_covered

    result = {
        "xgboost_model": xgb_cost,
        "naive_baseline": naive_cost,
        "savings_vs_naive_baseline": round(float(savings_vs_naive), 2),
        "savings_pct": round(float(savings_pct), 4),
        "estimated_annual_savings": round(float(savings_vs_naive * annual_factor), 2),
        "assumptions": {
            "holding_cost_per_unit_per_day": HOLDING_COST_PER_UNIT_DAY,
            "stockout_margin_loss_per_unit": STOCKOUT_MARGIN_LOSS_PER_UNIT,
            "note": "Illustrative unit economics -- swap in your real company's numbers.",
        },
    }

    with open("models/cost_savings.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
