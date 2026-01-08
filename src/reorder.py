import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")
FORECAST_FILE = PROCESSED / "forecast.parquet"
REORDER_FILE = PROCESSED / "reorder_plan.csv"

# Simple inventory assumptions (can be tuned later)
LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.65  # ~95% service level
MIN_ORDER_QTY = 50

def main():
    if not FORECAST_FILE.exists():
        raise FileNotFoundError("Run `python -m src.predict` first.")

    df = pd.read_parquet(FORECAST_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Use the most recent date as "today"
    today = df["date"].max()

    recent = df[df["date"] == today].copy()

    # Simulate current inventory (placeholder but realistic)
    np.random.seed(42)
    recent["inventory_on_hand"] = np.random.randint(100, 400, size=len(recent))

    # Estimate lead-time demand using prediction
    recent["lead_time_demand"] = recent["prediction"] * LEAD_TIME_DAYS

    # Estimate demand variability (use recent error as proxy)
    recent["demand_std"] = recent["abs_error"].clip(lower=1)

    # Safety stock
    recent["safety_stock"] = SERVICE_LEVEL_Z * recent["demand_std"] * np.sqrt(LEAD_TIME_DAYS)

    # Target stock level
    recent["target_stock"] = recent["lead_time_demand"] + recent["safety_stock"]

    # Reorder quantity
    recent["reorder_qty"] = (
        recent["target_stock"] - recent["inventory_on_hand"]
    ).clip(lower=0)

    # Apply minimum order quantity
    recent["reorder_qty"] = recent["reorder_qty"].apply(
        lambda x: 0 if x < MIN_ORDER_QTY else int(round(x))
    )

    reorder_cols = [
        "sku",
        "channel",
        "inventory_on_hand",
        "lead_time_demand",
        "safety_stock",
        "reorder_qty",
    ]

    reorder_df = recent[reorder_cols].sort_values("reorder_qty", ascending=False)

    reorder_df.to_csv(REORDER_FILE, index=False)

    print(f"Reorder plan saved to: {REORDER_FILE}")
    print("SKUs needing reorder:", (reorder_df["reorder_qty"] > 0).sum())

if __name__ == "__main__":
    main()
