import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

FORECAST_FILE = Path("data/processed/forecast.parquet")

st.title("Forecast Explorer")

if not FORECAST_FILE.exists():
    st.error("Missing forecast file. Run: `python -m src.predict`")
    st.stop()

df = pd.read_parquet(FORECAST_FILE)
df["date"] = pd.to_datetime(df["date"])

st.sidebar.header("Selection")
sku = st.sidebar.selectbox("SKU", sorted(df["sku"].unique().tolist()))
channel = st.sidebar.selectbox("Channel", sorted(df["channel"].unique().tolist()))

sub = df[(df["sku"] == sku) & (df["channel"] == channel)].sort_values("date")

st.subheader(f"{sku} — {channel}")

# Metrics
mae = float(np.mean(np.abs(sub["target_units_next_day"] - sub["prediction"])))
st.metric("MAE", f"{mae:.3f}")

# Time series plot table for Streamlit chart
plot_df = sub[["date", "target_units_next_day", "prediction"]].rename(
    columns={"target_units_next_day": "actual_next_day"}
)
st.line_chart(plot_df, x="date", y=["actual_next_day", "prediction"])

st.subheader("Recent rows")
st.dataframe(
    sub.tail(30)[["date", "units_sold", "target_units_next_day", "prediction", "abs_error", "price", "promo_flag"]],
    use_container_width=True
)
