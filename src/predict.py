import pandas as pd
import joblib
from pathlib import Path

PROCESSED = Path("data/processed")
MODELS = Path("models")

FEATURES_FILE = PROCESSED / "features.parquet"
FORECAST_FILE = PROCESSED / "forecast.parquet"
MODEL_FILE = MODELS / "model.pkl"

TARGET = "target_units_next_day"

def main():
    # Load data + model
    df = pd.read_parquet(FEATURES_FILE)
    model = joblib.load(MODEL_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    # Keep a copy of ground truth for evaluation / dashboard
    y_true = df[TARGET]

    # Drop target from features before predicting
    X = df.drop(columns=[TARGET])

    # Predict
    y_pred = model.predict(X)

    # Build forecast table
    forecast_df = df[[
        "date",
        "sku",
        "channel",
        "units_sold",
        TARGET
    ]].copy()

    forecast_df["prediction"] = y_pred
    forecast_df["abs_error"] = (forecast_df[TARGET] - forecast_df["prediction"]).abs()

    forecast_df.to_parquet(FORECAST_FILE)

    print(f"Forecast saved to: {FORECAST_FILE}")
    print("Rows:", len(forecast_df))
    print("Mean absolute error:", forecast_df["abs_error"].mean())

if __name__ == "__main__":
    main()
