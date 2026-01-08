import streamlit as st
import pandas as pd
from pathlib import Path

REORDER_FILE = Path("data/processed/reorder_plan.csv")

DECISION_FILE = Path("data/processed/decision_report.csv")
REORDER_FILE = Path("data/processed/reorder_plan.csv")

if DECISION_FILE.exists():
    df = pd.read_csv(DECISION_FILE)
    st.caption("Showing decision report (includes confidence + reasons + adjusted reorder).")
elif REORDER_FILE.exists():
    df = pd.read_csv(REORDER_FILE)
    st.caption("Showing reorder plan (basic).")
else:
    st.warning("Missing decision report and reorder plan. Run: `python -m src.predict`, `python -m src.reorder`, `python -m src.decision`.")
    st.stop()

st.title("Reorder Plan")

if not REORDER_FILE.exists():
    st.warning("Reorder plan not generated yet. Next step is `src/reorder.py` to create `data/processed/reorder_plan.csv`.")
    st.stop()

df = pd.read_csv(REORDER_FILE)

st.sidebar.header("Filters")
channels = ["All"] + sorted(df["channel"].unique().tolist()) if "channel" in df.columns else ["All"]
sel_channel = st.sidebar.selectbox("Channel", channels)

filtered = df.copy()
if sel_channel != "All" and "channel" in filtered.columns:
    filtered = filtered[filtered["channel"] == sel_channel]

st.dataframe(filtered, use_container_width=True)
st.download_button("Download CSV", filtered.to_csv(index=False), file_name="reorder_plan.csv", mime="text/csv")
