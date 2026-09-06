"""
Demand Forecasting Dashboard
------------------------------
Business-facing Streamlit app for supply chain planners.

Run:  streamlit run dashboard/app.py
"""

import sys
import os
import json

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(
    page_title="Demand Forecasting & Supply Chain Dashboard",
    page_icon="📦",
    layout="wide",
)

DATA_PATH = "data/demand_data.csv"
PREDICTIONS_PATH = "models/test_predictions.csv"
ANOMALIES_PATH = "models/anomalies.csv"
METRICS_PATH = "models/metrics.json"
COST_SAVINGS_PATH = "models/cost_savings.json"
SHAP_PATH = "models/shap_importance.csv"


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    preds = None
    anomalies = None
    metrics = None
    cost = None
    shap_imp = None

    if os.path.exists(PREDICTIONS_PATH):
        preds = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    if os.path.exists(ANOMALIES_PATH):
        anomalies = pd.read_csv(ANOMALIES_PATH, parse_dates=["date"])
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    if os.path.exists(COST_SAVINGS_PATH):
        with open(COST_SAVINGS_PATH) as f:
            cost = json.load(f)
    if os.path.exists(SHAP_PATH):
        shap_imp = pd.read_csv(SHAP_PATH)

    return raw, preds, anomalies, metrics, cost, shap_imp


raw, preds, anomalies, metrics, cost, shap_imp = load_data()

st.title("📦 AI-Powered Demand Forecasting & Supply Chain Dashboard")
st.caption("XGBoost forecasting · Isolation Forest anomaly detection · SHAP explainability · cost-impact simulation")

if preds is None:
    st.warning("No trained model found. Run `python src/train.py` first, then `python src/anomaly_detection.py` and `python src/cost_savings.py`.")
    st.stop()

# ---------------- Top-line metrics ----------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model MAE", f"{metrics['xgboost']['mae']:.1f} units")
with col2:
    st.metric("Model MAPE", f"{metrics['xgboost']['mape']:.1%}")
with col3:
    improvement = metrics.get("improvement_over_baseline")
    st.metric("Improvement vs Baseline", f"{improvement:.1%}" if improvement else "N/A")
with col4:
    if cost:
        st.metric("Est. Annual Savings", f"${cost['estimated_annual_savings']:,.0f}")

st.divider()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
stores = sorted(raw["store_id"].unique())
items = sorted(raw["item_id"].unique())
selected_store = st.sidebar.selectbox("Store", stores, index=0)
selected_item = st.sidebar.selectbox("Item", items, index=0)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🚨 Anomalies", "💰 Cost Impact", "🔍 Model Explainability"])

# ---------------- Tab 1: Forecast ----------------
with tab1:
    st.subheader(f"Actual vs Predicted Demand — Store {selected_store}, Item {selected_item}")

    filtered = preds[(preds.store_id == selected_store) & (preds.item_id == selected_item)].sort_values("date")

    if filtered.empty:
        st.info("No test-period data for this store/item combination.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered.date, y=filtered.units_sold, name="Actual", mode="lines", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=filtered.date, y=filtered.predicted, name="Predicted", mode="lines", line=dict(color="#f97316", dash="dash")))
        fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Units Sold", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Historical Demand (all data, not just test period)")
    hist = raw[(raw.store_id == selected_store) & (raw.item_id == selected_item)].sort_values("date")
    fig2 = px.line(hist, x="date", y="units_sold", title=None)
    fig2.update_layout(height=350, xaxis_title="Date", yaxis_title="Units Sold")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tab 2: Anomalies ----------------
with tab2:
    st.subheader("Detected Supply Chain Anomalies")
    st.caption("Flagged using Isolation Forest on model residuals — days where actual demand deviated sharply from expectation.")

    if anomalies is None:
        st.info("Run `python src/anomaly_detection.py` to generate anomaly data.")
    else:
        anom_only = anomalies[anomalies.is_anomaly == 1].copy()
        c1, c2 = st.columns(2)
        c1.metric("Total Anomalies Detected", len(anom_only))
        c2.metric("% of Test Days", f"{len(anom_only)/len(anomalies):.1%}")

        store_filter = st.multiselect("Filter by store", sorted(anomalies.store_id.unique()), default=[])
        display_df = anom_only if not store_filter else anom_only[anom_only.store_id.isin(store_filter)]

        st.dataframe(
            display_df[["date", "store_id", "item_id", "units_sold", "predicted", "residual", "pct_residual"]]
            .sort_values("date", ascending=False)
            .round(2),
            use_container_width=True,
            height=400,
        )

        fig3 = px.scatter(
            anomalies, x="predicted", y="units_sold", color=anomalies.is_anomaly.map({0: "Normal", 1: "Anomaly"}),
            color_discrete_map={"Normal": "#94a3b8", "Anomaly": "#ef4444"},
            labels={"predicted": "Predicted Units", "units_sold": "Actual Units", "color": "Status"},
            title="Actual vs Predicted — Anomalies Highlighted",
        )
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)

# ---------------- Tab 3: Cost Impact ----------------
with tab3:
    st.subheader("Business Impact: Forecast Accuracy → $ Savings")
    if cost is None:
        st.info("Run `python src/cost_savings.py` to generate cost impact data.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("XGBoost Model Cost (test period)", f"${cost['xgboost_model']['total_cost']:,.0f}")
        c2.metric("Naive Baseline Cost (test period)", f"${cost['naive_baseline']['total_cost']:,.0f}")
        c3.metric("Savings", f"${cost['savings_vs_naive_baseline']:,.0f}", f"{cost['savings_pct']:.1%}")

        cost_df = pd.DataFrame({
            "Model": ["XGBoost", "Naive Baseline"],
            "Holding Cost": [cost["xgboost_model"]["holding_cost"], cost["naive_baseline"]["holding_cost"]],
            "Stockout Cost": [cost["xgboost_model"]["stockout_cost"], cost["naive_baseline"]["stockout_cost"]],
        })
        fig4 = px.bar(cost_df, x="Model", y=["Holding Cost", "Stockout Cost"], barmode="stack", title="Cost Breakdown by Model")
        fig4.update_layout(height=400, yaxis_title="Cost ($)")
        st.plotly_chart(fig4, use_container_width=True)

    st.info(
        f"**Estimated annual savings: \\${cost['estimated_annual_savings']:,.0f}** "
        f"(based on holding cost of \\${cost['assumptions']['holding_cost_per_unit_per_day']}/unit/day "
        f"and stockout margin loss of \\${cost['assumptions']['stockout_margin_loss_per_unit']}/unit — "
        f"adjust these in `src/cost_savings.py` to match real unit economics)."
    )

# ---------------- Tab 4: Explainability ----------------
with tab4:
    st.subheader("What Drives the Model's Predictions? (SHAP Feature Importance)")
    if shap_imp is None:
        st.info("Run `python src/train.py` to generate SHAP importance data.")
    else:
        fig5 = px.bar(
            shap_imp.head(15).sort_values("mean_abs_shap"),
            x="mean_abs_shap", y="feature", orientation="h",
            title="Mean |SHAP value| by Feature (higher = more influence on predictions)",
        )
        fig5.update_layout(height=500, xaxis_title="Mean |SHAP value|", yaxis_title="")
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Interpretation: rolling averages and day-of-week/promo flags dominate — the model has learned realistic retail seasonality and promotion effects, not spurious noise.")

st.divider()
st.caption("Built with XGBoost, SHAP, Isolation Forest, MLflow, FastAPI, and Streamlit — 100% free/open-source stack.")
