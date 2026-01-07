import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib

PROCESSED = Path("data/processed")
FEATURES_FILE = PROCESSED / "features.parquet"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODELS_DIR / "model.pkl"
META_FILE = MODELS_DIR / "metadata.json"

TARGET = "target_units_next_day"

def wape(y_true, y_pred) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def time_split(df: pd.DataFrame, split_date: str = "2025-07-01"):
    """
    Train on dates < split_date, validate on dates >= split_date.
    You can move this split later once you want a bigger train set.
    """
    split_dt = pd.to_datetime(split_date)
    train = df[df["date"] < split_dt].copy()
    valid = df[df["date"] >= split_dt].copy()
    return train, valid

def baseline_naive(valid: pd.DataFrame) -> np.ndarray:
    """
    Simple baseline: predict tomorrow = yesterday (units_lag_1).
    This is intentionally dumb but surprisingly strong for time series.
    """
    return valid["units_lag_1"].to_numpy()

def main():
    df = pd.read_parquet(FEATURES_FILE)

    # Safety: ensure date type and sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    # Split
    train_df, valid_df = time_split(df, split_date="2025-07-01")
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("Time split produced empty train or valid set. Adjust split_date.")

    # Define features
    numeric_features = [
        "price",
        "promo_flag",
        "dayofweek",
        "month",
        "weekofyear",
        "is_weekend",
        "units_lag_1",
        "units_lag_7",
        "units_lag_14",
        "units_roll7_mean",
        "units_roll14_mean",
        "price_lag_1",
        "price_lag_7",
        "price_lag_14",
        "promo_lag_1",
        "promo_lag_7",
        "promo_lag_14",
    ]
    categorical_features = ["sku", "channel"]

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[TARGET].to_numpy()

    X_valid = valid_df[numeric_features + categorical_features]
    y_valid = valid_df[TARGET].to_numpy()

    # Preprocess: impute + one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    # Model: Ridge regression (strong baseline, fast, stable)
    model = Ridge(alpha=1.0, random_state=42)

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    # Baseline metrics
    y_pred_base = baseline_naive(valid_df)
    base_mae = mean_absolute_error(y_valid, y_pred_base)
    base_wape = wape(y_valid, y_pred_base)

    # Model metrics
    y_pred = pipe.predict(X_valid)
    model_mae = mean_absolute_error(y_valid, y_pred)
    model_wape = wape(y_valid, y_pred)

    print("=== Validation Metrics (Time Split) ===")
    print(f"Train rows: {len(train_df)} | Valid rows: {len(valid_df)}")
    print(f"Baseline (yesterday): MAE={base_mae:.3f} | WAPE={base_wape:.3f}")
    print(f"Ridge model        : MAE={model_mae:.3f} | WAPE={model_wape:.3f}")

    # Save model + metadata
    joblib.dump(pipe, MODEL_FILE)

    meta = {
        "target": TARGET,
        "split_date": "2025-07-01",
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "metrics": {
            "baseline": {"mae": float(base_mae), "wape": float(base_wape)},
            "ridge": {"mae": float(model_mae), "wape": float(model_wape)},
        },
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        }
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to: {MODEL_FILE}")
    print(f"Saved metadata to: {META_FILE}")

if __name__ == "__main__":
    main()
