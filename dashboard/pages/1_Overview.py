import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

FORECAST_FILE = Path("data/processed/forecast.parquet")
META_FILE = Path("models/metadata.json")
DECISION_FILE = Path("data/processed/decision_report.csv")

def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return np.nan if denom == 0 else float(np.sum(np.abs(y_true - y_pred)) / denom)

st.title("Overview")

if not FORECAST_FILE.exists():
    st.error("Missing forecast file. Run: `python -m src.predict`")
    st.stop()

df = pd.read_parquet(FORECAST_FILE)
df["date"] = pd.to_datetime(df["date"])

# Sidebar filters
st.sidebar.header("Filters")
min_d, max_d = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
channels = ["All"] + sorted(df["channel"].unique().tolist())
sel_channel = st.sidebar.selectbox("Channel", channels)

filtered = df.copy()
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
if sel_channel != "All":
    filtered = filtered[filtered["channel"] == sel_channel]

# KPIs
mae = float(np.mean(np.abs(filtered["target_units_next_day"] - filtered["prediction"])))
wape_val = wape(filtered["target_units_next_day"].to_numpy(), filtered["prediction"].to_numpy())

k1, k2, k3 = st.columns(3)
k1.metric("MAE (avg abs error)", f"{mae:.3f}")
k2.metric("WAPE", f"{wape_val:.3f}")
k3.metric("Rows", f"{len(filtered)}")

st.subheader("Decision Summary (Today)")
if DECISION_FILE.exists():
    dec = pd.read_csv(DECISION_FILE)
    st.dataframe(dec.head(25), use_container_width=True)
else:
    st.warning("Decision report missing. Run: `python -m src.decision`")


# Metadata (if present)
if META_FILE.exists():
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with st.expander("Model metadata"):
        st.json(meta)

st.subheader("Top SKUs by Total Absolute Error")
sku_err = (
    filtered.assign(abs_err=lambda x: np.abs(x["target_units_next_day"] - x["prediction"]))
            .groupby("sku", as_index=False)["abs_err"].sum()
            .sort_values("abs_err", ascending=False)
            .head(10)
)
st.dataframe(sku_err, use_container_width=True)

st.subheader("Forecast Error Over Time (Total)")
daily = (
    filtered.assign(abs_err=lambda x: np.abs(x["target_units_next_day"] - x["prediction"]))
            .groupby("date", as_index=False)["abs_err"].mean()
)
st.line_chart(daily, x="date", y="abs_err")
