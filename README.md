# BevOps AI — Decision & Risk-Aware Inventory System

## What This Project Is 

This project demonstrates how automated forecasts are **turned into business decisions**, and—more importantly—how to **detect when those decisions should NOT be trusted**.

Rather than optimizing for “best predictions,” the system is designed to:
- surface uncertainty,
- flag risky situations,
- and support **human judgment** before money is committed.

This mirrors real-world operational analytics, where silent failures and overconfidence cause financial loss.

---

## The Problem This Solves

In production systems, money is lost when:
- data pipelines break quietly,
- demand behavior shifts suddenly,
- models continue outputting numbers without warning,
- decisions are made on stale or misleading information.

Most demo projects stop at prediction.  
This project goes further by adding **decision safeguards**.

---

## System Overview

The pipeline runs end-to-end:

1. **Ingest**
   - Generates or loads sales and product data.

2. **Clean**
   - Validates data integrity (dates, prices, units, promo flags).
   - Prevents obvious data errors from propagating.

3. **Feature Engineering**
   - Adds temporal context (lags, rolling averages, calendar effects).

4. **Train**
   - Trains a baseline and a regression model.
   - Evaluates performance using a time-based split.

5. **Predict**
   - Generates next-day demand forecasts.
   - Stores predictions as artifacts, not assumptions.

6. **Reorder**
   - Converts forecasts into inventory recommendations.
   - Simulates real operational constraints (lead time, safety stock).

7. **Decision Layer (Key Contribution)**
   - Assesses **model reliability per SKU/channel**.
   - Flags high-risk situations (volatility, regime change, high error).
   - Applies conservative buffers when confidence is low.
   - Produces a human-readable decision report explaining *why*.

---

## Why This Is Not “Just Forecasting”

The model does NOT decide what to buy.

The system:
- evaluates recent performance,
- measures instability and change,
- and recommends **how cautious to be**.

Final decisions are intentionally left to humans.

This reflects real operational environments where:
- forecasts are advisory,
- accountability matters,
- and blind automation is dangerous.

---

## Outputs

The system produces explicit, auditable artifacts:

- `forecast.parquet`  
  → Model predictions + actual outcomes

- `reorder_plan.csv`  
  → Base inventory recommendations

- `decision_report.csv`  
  → Confidence levels, risk flags, explanations, and adjusted actions

These outputs are designed to be inspected, questioned, and overridden.

---

## Limitations (Intentional)

This project makes its limitations explicit:

- External factors (marketing, social trends, supply shocks) are not modeled.
- New or low-history SKUs are flagged as low confidence.
- Predictions are approximate by design.
- The system prioritizes **risk awareness over false certainty**.

These constraints reflect real-world analytics systems more accurately than “perfect” demos.

---

## How to Run the Full Pipeline

```bash
python -m src.ingest
python -m src.clean
python -m src.features
python -m src.train
python -m src.predict
python -m src.reorder
python -m src.decision
