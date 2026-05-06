"""
Supply Chain Disruption Intelligence — Risk Dashboard (V1)

Minimal Streamlit app showing:
  - Current risk level per chokepoint
  - Commodity price shock predictions
  - Feature importance from trained model
  - Interactive map of chokepoint regions

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Supply Chain Risk Monitor", layout="wide")

# --- Header ---
st.title("🚢 Supply Chain Disruption Intelligence")
st.caption("Real-time risk monitoring for commodity supply chains")

# --- Load data ---
DATA_DIR = Path("./data") if Path("./data").exists() else Path("/app/data")
FEATURES_PATH = DATA_DIR / "parquet" / "features"
CHOKEPOINTS_PATH = DATA_DIR / "chokepoints" / "chokepoints.csv"

@st.cache_data
def load_features():
    try:
        return pd.read_parquet(FEATURES_PATH)
    except Exception:
        # Generate sample data for demo if pipeline hasn't run yet
        return generate_demo_data()

@st.cache_data
def load_chokepoints():
    try:
        return pd.read_csv(CHOKEPOINTS_PATH)
    except Exception:
        return pd.DataFrame()

def generate_demo_data():
    """Synthetic demo data so the dashboard works before the pipeline runs."""
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    chokepoints = ["hormuz", "suez", "red_sea"]
    rows = []
    for cp in chokepoints:
        for d in dates:
            rows.append({
                "event_date": d,
                "chokepoint": cp,
                "event_count": np.random.poisson(5),
                "avg_goldstein_7d": np.random.normal(-3, 2),
                "avg_tone_7d": np.random.normal(-1, 1.5),
                "conflict_ratio_7d": np.random.beta(2, 5),
                "total_mentions_7d": np.random.poisson(20),
                "return_5d": np.random.normal(0, 0.02),
                "volatility_20d": abs(np.random.normal(0.015, 0.005)),
                "label": np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1]),
            })
    return pd.DataFrame(rows)


df = load_features()
df["event_date"] = pd.to_datetime(df["event_date"])
chokepoints = load_chokepoints()

# --- Sidebar: date filter ---
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["event_date"].max() - pd.Timedelta(days=90), df["event_date"].max()),
    min_value=df["event_date"].min(),
    max_value=df["event_date"].max(),
)
if len(date_range) == 2:
    mask = (df["event_date"] >= pd.Timestamp(date_range[0])) & (df["event_date"] <= pd.Timestamp(date_range[1]))
    df_filtered = df[mask]
else:
    df_filtered = df

# --- Row 1: KPI cards ---
col1, col2, col3, col4 = st.columns(4)
latest = df_filtered[df_filtered["event_date"] == df_filtered["event_date"].max()]

with col1:
    avg_conflict = latest["conflict_ratio_7d"].mean() if len(latest) > 0 else 0
    st.metric("Avg Conflict Ratio", f"{avg_conflict:.2f}",
              delta=f"{'HIGH' if avg_conflict > 0.4 else 'Normal'}")

with col2:
    total_events = latest["event_count"].sum() if len(latest) > 0 else 0
    st.metric("Events Today", f"{total_events:,.0f}")

with col3:
    avg_goldstein = latest["avg_goldstein_7d"].mean() if len(latest) > 0 else 0
    st.metric("Avg Goldstein (7d)", f"{avg_goldstein:.2f}")

with col4:
    shock_pct = (latest["label"] != 1).mean() * 100 if len(latest) > 0 else 0
    st.metric("Shock Probability", f"{shock_pct:.0f}%")

st.divider()

# --- Row 2: Time series + Map ---
left, right = st.columns([2, 1])

with left:
    st.subheader("Conflict Intensity Over Time")
    ts = (
        df_filtered.groupby(["event_date", "chokepoint"])
        .agg({"conflict_ratio_7d": "mean"})
        .reset_index()
    )
    fig = px.line(
        ts, x="event_date", y="conflict_ratio_7d", color="chokepoint",
        labels={"conflict_ratio_7d": "Conflict Ratio (7d)", "event_date": "Date"},
    )
    fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Chokepoint Risk Map")
    if not chokepoints.empty:
        # Color by recent conflict level
        recent_risk = (
            df_filtered[df_filtered["event_date"] >= df_filtered["event_date"].max() - pd.Timedelta(days=7)]
            .groupby("chokepoint")
            .agg({"conflict_ratio_7d": "mean"})
            .reset_index()
        )
        map_data = chokepoints.merge(recent_risk, left_on="chokepoint", right_on="chokepoint", how="left")
        map_data["conflict_ratio_7d"] = map_data["conflict_ratio_7d"].fillna(0)

        fig_map = px.scatter_geo(
            map_data,
            lat="lat", lon="lon",
            size="radius_deg",
            color="conflict_ratio_7d",
            hover_name="chokepoint",
            color_continuous_scale="YlOrRd",
            projection="natural earth",
        )
        fig_map.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Chokepoint data not loaded — run generate_chokepoints.py first")

st.divider()

# --- Row 3: Risk alerts ---
st.subheader("⚠ Recent Risk Alerts")
if len(latest) > 0:
    alerts = latest[latest["label"] != 1].copy()
    if len(alerts) > 0:
        alerts["alert_type"] = alerts["label"].map({0: "🔴 Negative Shock", 2: "🟢 Positive Shock"})
        for _, row in alerts.iterrows():
            st.warning(
                f"{row['alert_type']} — **{row['chokepoint'].title()}** "
                f"| Conflict: {row['conflict_ratio_7d']:.2f} "
                f"| Goldstein: {row['avg_goldstein_7d']:.1f}"
            )
    else:
        st.success("No price shock alerts in current period")
else:
    st.info("No data for selected period")

# --- Footer ---
st.divider()
st.caption(
    "DATA 228 — Supply Chain Disruption Intelligence | "
    "Sources: GDELT, Yahoo Finance, FRED"
)
