import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="BevOps AI Dashboard",
    layout="wide"
)

st.title("BevOps AI — Forecasting & Reorder Dashboard")

st.markdown(
    """
This dashboard reads artifacts produced by the pipeline:
- **Forecasts:** `data/processed/forecast.parquet`
- **Reorder plan (later):** `data/processed/reorder_plan.csv`
- **Model metadata:** `models/metadata.json`
"""
)

base_paths = {
    "Forecast": Path("data/processed/forecast.parquet"),
    "Reorder Plan": Path("data/processed/reorder_plan.csv"),
    "Model Metadata": Path("models/metadata.json"),
}

cols = st.columns(3)
for i, (name, p) in enumerate(base_paths.items()):
    with cols[i]:
        st.subheader(name)
        st.write("Found" if p.exists() else "Missing", "—", str(p))

st.info("Use the pages in the left sidebar to explore forecasts and (later) reorder recommendations.")
