import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("data/processed")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Update this if you used CSV fallback
CLEAN_PARQUET = PROCESSED / "clean_sales.parquet"
CLEAN_CSV = PROCESSED / "clean_sales.csv"

def load_clean_sales() -> pd.DataFrame:
    if CLEAN_PARQUET.exists():
        df = pd.read_parquet(CLEAN_PARQUET)
    elif CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV)
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise FileNotFoundError("No clean_sales.parquet or clean_sales.csv found. Run `python -m src.clean` first.")
    return df

def plot_total_sales_over_time(df: pd.DataFrame):
    daily = df.groupby("date", as_index=False)["units_sold"].sum()

    plt.figure()
    plt.plot(daily["date"], daily["units_sold"])
    plt.title("Total Units Sold (All SKUs, All Channels)")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "total_units_over_time.png", dpi=200)
    plt.close()

def plot_by_channel_over_time(df: pd.DataFrame):
    daily = df.groupby(["date", "channel"], as_index=False)["units_sold"].sum()

    plt.figure()
    for ch in sorted(daily["channel"].unique()):
        sub = daily[daily["channel"] == ch]
        plt.plot(sub["date"], sub["units_sold"], label=ch)

    plt.title("Total Units Sold by Channel")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "units_by_channel_over_time.png", dpi=200)
    plt.close()

def plot_top_skus(df: pd.DataFrame, top_n: int = 6):
    totals = df.groupby("sku", as_index=False)["units_sold"].sum()
    top = totals.sort_values("units_sold", ascending=False).head(top_n)

    plt.figure()
    plt.bar(top["sku"], top["units_sold"])
    plt.title(f"Top {top_n} SKUs by Total Units Sold")
    plt.xlabel("SKU")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"top_{top_n}_skus.png", dpi=200)
    plt.close()

def plot_promo_effect(df: pd.DataFrame):
    promo = df.groupby("promo_flag", as_index=False)["units_sold"].mean()

    plt.figure()
    plt.bar(promo["promo_flag"].astype(str), promo["units_sold"])
    plt.title("Average Units Sold: Promo vs No Promo")
    plt.xlabel("promo_flag (0=no, 1=yes)")
    plt.ylabel("Avg Units Sold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "promo_effect_avg_units.png", dpi=200)
    plt.close()

def main():
    df = load_clean_sales()

    # quick sanity prints (control/visibility)
    print("Rows:", len(df))
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("SKUs:", df["sku"].nunique(), "| Channels:", df["channel"].nunique())
    print("Promo rate:", round(df["promo_flag"].mean(), 3))

    plot_total_sales_over_time(df)
    plot_by_channel_over_time(df)
    plot_top_skus(df, top_n=6)
    plot_promo_effect(df)

    print(f"Saved figures to: {FIG_DIR.resolve()}")

if __name__ == "__main__":
    main()
