from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
FORECAST_FILE = PROCESSED / "forecast.parquet"
REORDER_FILE = PROCESSED / "reorder_plan.csv"       # optional, if you’ve generated it
DECISION_FILE = PROCESSED / "decision_report.csv"

# Tunable policy knobs (this is YOU)
POLICY = {
    "high_wape_threshold": 0.25,      # above this, model is considered unreliable for that SKU-channel
    "volatility_cv_threshold": 0.35,  # coefficient of variation threshold on recent demand
    "regime_change_z": 2.5,           # spike detection threshold for recent demand change
    "min_history_days": 21,           # minimum data to make decisions confidently
    "buffer_low_conf": 0.25,          # add +25% buffer to reorder when low confidence
    "buffer_high_conf": 0.10,         # add +10% buffer when high confidence (optional)
}

def wape(y_true, y_pred) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def add_reason(reasons, text):
    if text and text not in reasons:
        reasons.append(text)

def main():
    if not FORECAST_FILE.exists():
        raise FileNotFoundError("Missing forecast.parquet. Run `python -m src.predict` first.")

    df = pd.read_parquet(FORECAST_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    # Define "today" as last date in file
    today = df["date"].max()
    lookback_days = 28
    cutoff = today - pd.Timedelta(days=lookback_days)

    recent = df[df["date"] >= cutoff].copy()

    # Compute per SKU-channel diagnostics
    group_cols = ["sku", "channel"]

    def compute_group_metrics(g: pd.DataFrame) -> pd.Series:
        # Use columns produced by predict.py
        y_true = g["target_units_next_day"].to_numpy()
        y_pred = g["prediction"].to_numpy()

        # Reliability metrics
        g_wape = wape(y_true, y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # Demand stability metrics
        demand = g["units_sold"].to_numpy()
        mean_d = float(np.mean(demand))
        std_d = float(np.std(demand))
        cv = float(std_d / mean_d) if mean_d != 0 else np.nan

        # Regime change heuristic: compare last 7 days to prior 7 days
        g_sorted = g.sort_values("date")
        last7 = g_sorted.tail(7)["units_sold"].to_numpy()
        prev7 = g_sorted.tail(14).head(7)["units_sold"].to_numpy() if len(g_sorted) >= 14 else np.array([])

        regime_z = np.nan
        if len(prev7) == 7 and std_d > 0:
            regime_z = float((np.mean(last7) - np.mean(prev7)) / std_d)

        # History length
        history_days = int(g_sorted["date"].nunique())

        return pd.Series({
            "wape_28d": g_wape,
            "mae_28d": mae,
            "demand_mean_28d": mean_d,
            "demand_cv_28d": cv,
            "regime_z": regime_z,
            "history_days": history_days,
        })

    metrics = recent.groupby(group_cols).apply(compute_group_metrics).reset_index()

    # Pull today's forecast row per SKU-channel (the “current” recommended demand)
    today_rows = df[df["date"] == today].copy()
    today_rows = today_rows[group_cols + ["date", "prediction", "abs_error", "units_sold"]]

    # Merge diagnostics into today snapshot
    report = today_rows.merge(metrics, on=group_cols, how="left")

    # If reorder plan exists, merge it in
    if REORDER_FILE.exists():
        reorder = pd.read_csv(REORDER_FILE)
        report = report.merge(reorder, on=group_cols, how="left")
    else:
        report["reorder_qty"] = np.nan
        report["inventory_on_hand"] = np.nan
        report["lead_time_demand"] = np.nan
        report["safety_stock"] = np.nan

    # Decision logic + human-readable reasons
    confidence = []
    recommended_action = []
    buffer_pct = []
    reasons_out = []

    for _, row in report.iterrows():
        reasons = []

        # Confidence scoring (rule-based, transparent)
        low_conf = False

        if pd.isna(row["history_days"]) or row["history_days"] < POLICY["min_history_days"]:
            low_conf = True
            add_reason(reasons, f"Not enough history (<{POLICY['min_history_days']} days)")

        if not pd.isna(row["wape_28d"]) and row["wape_28d"] >= POLICY["high_wape_threshold"]:
            low_conf = True
            add_reason(reasons, f"High recent error (WAPE {row['wape_28d']:.2f} ≥ {POLICY['high_wape_threshold']})")

        if not pd.isna(row["demand_cv_28d"]) and row["demand_cv_28d"] >= POLICY["volatility_cv_threshold"]:
            low_conf = True
            add_reason(reasons, f"High demand volatility (CV {row['demand_cv_28d']:.2f} ≥ {POLICY['volatility_cv_threshold']})")

        if not pd.isna(row["regime_z"]) and abs(row["regime_z"]) >= POLICY["regime_change_z"]:
            low_conf = True
            add_reason(reasons, f"Possible regime change (z {row['regime_z']:.2f} ≥ {POLICY['regime_change_z']})")

        if low_conf:
            conf = "LOW"
            buf = POLICY["buffer_low_conf"]
            action = "ORDER_CONSERVATIVE"
            add_reason(reasons, "Use conservative buffer due to low confidence")
        else:
            conf = "HIGH"
            buf = POLICY["buffer_high_conf"]
            action = "ORDER_BASELINE"
            add_reason(reasons, "Model appears stable on recent window")

        confidence.append(conf)
        buffer_pct.append(buf)
        recommended_action.append(action)
        reasons_out.append("; ".join(reasons) if reasons else "")

    report["confidence"] = confidence
    report["buffer_pct"] = buffer_pct
    report["recommended_action"] = recommended_action
    report["reason"] = reasons_out

    # If reorder_qty exists, produce an adjusted recommendation
    # (If reorder_qty is NaN because reorder.py hasn’t run, we still output a suggested adjustment conceptually.)
    report["reorder_qty_adjusted"] = report["reorder_qty"]

    mask_has_reorder = report["reorder_qty"].notna()
    report.loc[mask_has_reorder, "reorder_qty_adjusted"] = (
        report.loc[mask_has_reorder, "reorder_qty"] * (1.0 + report.loc[mask_has_reorder, "buffer_pct"])
    ).round().astype("Int64")

    # Output ordering: highest adjusted reorder first
    report = report.sort_values("reorder_qty_adjusted", ascending=False, na_position="last")

    # Select columns for stakeholder readability
    cols = [
        "date", "sku", "channel",
        "prediction", "units_sold",
        "confidence", "recommended_action", "buffer_pct", "reason",
        "wape_28d", "mae_28d", "demand_mean_28d", "demand_cv_28d", "regime_z", "history_days",
        "inventory_on_hand", "lead_time_demand", "safety_stock", "reorder_qty", "reorder_qty_adjusted"
    ]
    cols = [c for c in cols if c in report.columns]
    out = report[cols].copy()

    out.to_csv(DECISION_FILE, index=False)
    print(f"Decision report saved to: {DECISION_FILE}")
    print("Rows:", len(out))
    print("LOW confidence rows:", int((out["confidence"] == "LOW").sum()))

if __name__ == "__main__":
    main()
