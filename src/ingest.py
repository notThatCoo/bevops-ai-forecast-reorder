import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
CHANNELS = ["retail", "online"]

np.random.seed(42)

# -----------------------------
# Product generation
# -----------------------------
def generate_products():
    skus = [
        "FB_LATTE_VAN",
        "FB_LATTE_MOCHA",
        "FB_FRAPPE_CARAMEL",
        "FB_FRAPPE_CHOC",
        "HM_MATCHA_CLASSIC",
        "HM_MATCHA_VAN"
    ]

    products = []
    for sku in skus:
        products.append({
            "sku": sku,
            "brand": "Frozen Bean" if sku.startswith("FB") else "Harmony Matcha",
            "base_price": np.round(np.random.uniform(8, 14), 2),
            "unit_cost": np.round(np.random.uniform(3, 6), 2)
        })

    return pd.DataFrame(products)

# -----------------------------
# Sales generation
# -----------------------------
def generate_sales(products: pd.DataFrame):
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows = []

    for _, product in products.iterrows():
        for channel in CHANNELS:
            base_demand = np.random.randint(20, 60)
            price_multiplier = 1.2 if channel == "online" else 1.0

            for date in dates:
                dow = date.dayofweek
                month = date.month

                # seasonality
                weekly_factor = 1.2 if dow >= 5 else 1.0
                summer_factor = 1.3 if month in [6, 7, 8] else 1.0

                promo_flag = np.random.rand() < 0.1
                promo_lift = 1.5 if promo_flag else 1.0

                expected_demand = (
                    base_demand
                    * weekly_factor
                    * summer_factor
                    * promo_lift
                )

                units_sold = np.random.poisson(expected_demand)

                price = np.round(product["base_price"] * price_multiplier, 2)
                if promo_flag:
                    price *= 0.85

                rows.append({
                    "date": date,
                    "sku": product["sku"],
                    "channel": channel,
                    "units_sold": units_sold,
                    "price": np.round(price, 2),
                    "promo_flag": int(promo_flag)
                })

    return pd.DataFrame(rows)

# -----------------------------
# Main
# -----------------------------
def main():
    products = generate_products()
    sales = generate_sales(products)

    products.to_csv(RAW_DATA_DIR / "products.csv", index=False)
    sales.to_csv(RAW_DATA_DIR / "sales.csv", index=False)

    print("Raw data generated:")
    print(f"- {len(products)} products")
    print(f"- {len(sales)} sales rows")

if __name__ == "__main__":
    main()
