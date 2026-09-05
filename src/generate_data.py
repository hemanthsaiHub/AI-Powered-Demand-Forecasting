"""
Synthetic Demand Data Generator
--------------------------------
Generates realistic multi-store, multi-item daily sales data with:
- Long-term trend
- Weekly + yearly seasonality
- Promotions (random spikes)
- Holiday effects
- Store/item-level base demand differences
- Injected anomalies (supply chain disruptions, stockouts) for the
  anomaly-detection module to find later.

No external download required — fully free, fully offline, reproducible.
Swap this out for a real dataset (e.g. Walmart M5, Olist) if you want;
the rest of the pipeline doesn't care where the data comes from as long
as the schema matches.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_STORES = 8
N_ITEMS = 15
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
OUTPUT_PATH = "data/demand_data.csv"

HOLIDAYS = [
    "2022-01-01", "2022-07-04", "2022-11-24", "2022-12-25",
    "2023-01-01", "2023-07-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-07-04", "2024-11-28", "2024-12-25",
]


def generate_demand_data(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    holiday_set = set(pd.to_datetime(HOLIDAYS))

    rows = []
    for store_id in range(1, N_STORES + 1):
        store_multiplier = rng.uniform(0.7, 1.6)  # store size/traffic factor

        for item_id in range(1, N_ITEMS + 1):
            base_demand = rng.uniform(15, 120)
            # Total drift over the whole 3-year window, expressed as a
            # fraction of base_demand (e.g. +40% or -20% by the end).
            # Linear, not compounding, so it can never blow up.
            total_drift_fraction = rng.uniform(-0.25, 0.5)
            item_volatility = rng.uniform(0.08, 0.25)

            # Random promotion days (~4% of days)
            promo_days = set(rng.choice(len(dates), size=int(len(dates) * 0.04), replace=False))

            # Random anomaly window (supply disruption / stockout), 1-2 per item
            n_anomalies = rng.integers(1, 3)
            anomaly_windows = []
            for _ in range(n_anomalies):
                start_idx = rng.integers(30, len(dates) - 30)
                length = rng.integers(3, 10)
                anomaly_windows.append((start_idx, start_idx + length))

            for i, date in enumerate(dates):
                day_of_week = date.dayofweek
                day_of_year = date.dayofyear

                # Trend component: linear drift from base_demand to
                # base_demand * (1 + total_drift_fraction) across the series
                progress = i / max(len(dates) - 1, 1)
                trend = base_demand * (1 + total_drift_fraction * progress)

                # Weekly seasonality (weekend lift for retail)
                weekly = 1.25 if day_of_week >= 5 else 1.0

                # Yearly seasonality (holiday season lift, summer dip)
                yearly = 1 + 0.3 * np.sin((day_of_year / 365) * 2 * np.pi + 1.5)

                # Holiday effect
                holiday_boost = 1.6 if date in holiday_set else 1.0

                # Promotion effect
                promo = 1.8 if i in promo_days else 1.0

                # Noise
                noise = rng.normal(1.0, item_volatility)

                demand = trend * weekly * yearly * holiday_boost * promo * noise * store_multiplier
                demand = max(demand, 0)

                # Inject anomaly (supply disruption -> demand crashes to near 0)
                is_anomaly = 0
                for (a_start, a_end) in anomaly_windows:
                    if a_start <= i < a_end:
                        demand = demand * rng.uniform(0.02, 0.15)
                        is_anomaly = 1
                        break

                rows.append({
                    "date": date,
                    "store_id": store_id,
                    "item_id": item_id,
                    "units_sold": round(demand, 1),
                    "is_promo": int(i in promo_days),
                    "is_holiday": int(date in holiday_set),
                    "day_of_week": day_of_week,
                    "is_weekend": int(day_of_week >= 5),
                    "true_anomaly": is_anomaly,  # ground truth, kept for evaluation only
                })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_demand_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {OUTPUT_PATH}")
    print(df.head())
    print(f"\nDate range: {df.date.min()} to {df.date.max()}")
    print(f"Stores: {df.store_id.nunique()}, Items: {df.item_id.nunique()}")
    print(f"Injected anomalies: {df.true_anomaly.sum()} rows")
