import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
FEATURES_FILE = PROCESSED / "features.parquet"

CLEAN_PARQUET = PROCESSED / "clean_sales.parquet"
CLEAN_CSV = PROCESSED / "clean_sales.csv"

def load_clean_sales() -> pd.DataFrame:
    if CLEAN_PARQUET.exists():
        df = pd.read_parquet(CLEAN_PARQUET)
    elif CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV)
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise FileNotFoundError("Run `python -m src.clean` first.")
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek  # 0=Mon
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_lag_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["sku", "channel"]

    # Lags (yesterday, last week, two weeks)
    for lag in [1, 7, 14]:
        df[f"units_lag_{lag}"] = df.groupby(group_cols)["units_sold"].shift(lag)
        df[f"price_lag_{lag}"] = df.groupby(group_cols)["price"].shift(lag)
        df[f"promo_lag_{lag}"] = df.groupby(group_cols)["promo_flag"].shift(lag)

    # Rolling means based on past values only (shift(1) prevents leakage)
    df["units_roll7_mean"] = df.groupby(group_cols)["units_sold"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).mean()
    )

    df["units_roll14_mean"] = df.groupby(group_cols)["units_sold"].transform(
        lambda s: s.shift(1).rolling(window=14, min_periods=5).mean()
    )

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["sku", "channel"]

    # Predict next-day units
    df["target_units_next_day"] = df.groupby(group_cols)["units_sold"].shift(-1)
    return df

def main():
    df = load_clean_sales()
    df = df.sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_lag_and_rolling(df)
    df = add_target(df)

    # Drop rows where features/target are missing (first days have no lag, last day has no target)
    model_df = df.dropna().reset_index(drop=True)

    model_df.to_parquet(FEATURES_FILE)

    print(f"Saved features to: {FEATURES_FILE}")
    print("Rows before dropna:", len(df))
    print("Rows after dropna :", len(model_df))
    print("Columns:", len(model_df.columns))
    print(model_df.head(3)[["date","sku","channel","units_sold","units_lag_1","units_roll7_mean","target_units_next_day"]])


if __name__ == "__main__":
    main()


