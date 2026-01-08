import streamlit as st
import pandas as pd
from pathlib import Path

DECISION_FILE = Path("data/processed/decision_report.csv")
REORDER_FILE = Path("data/processed/reorder_plan.csv")

st.title("Reorder Plan")

# Prefer decision report (more human), fallback to reorder plan
if DECISION_FILE.exists():
    df = pd.read_csv(DECISION_FILE)
    st.caption("Showing decision report (confidence + reasons + adjusted reorder).")
    download_name = "decision_report.csv"
elif REORDER_FILE.exists():
    df = pd.read_csv(REORDER_FILE)
    st.caption("Showing reorder plan (basic).")
    download_name = "reorder_plan.csv"
else:
    st.warning(
        "Missing decision report and reorder plan.\n\n"
        "Run:\n"
        "- `python -m src.predict`\n"
        "- `python -m src.reorder`\n"
        "- `python -m src.decision`"
    )
    st.stop()

# Optional filter if channel exists
st.sidebar.header("Filters")
if "channel" in df.columns:
    channels = ["All"] + sorted(df["channel"].dropna().unique().tolist())
    sel_channel = st.sidebar.selectbox("Channel", channels)
    if sel_channel != "All":
        df = df[df["channel"] == sel_channel]

st.dataframe(df, use_container_width=True)

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    file_name=download_name,
    mime="text/csv",
)
