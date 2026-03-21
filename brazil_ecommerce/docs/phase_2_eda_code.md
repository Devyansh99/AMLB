# Phase 2 EDA: What Each Code Part Does

Source notebook: `02_EDA_Continue.ipynb`

This file explains each code block in simple words.

## 1) Imports and setup
- Loads Python libraries for data work, plotting, and modeling.
- Sets plot style and warning behavior.

## 2) Data loading
- Reads cleaned merged data and raw review data.
- Prints dataset shapes and late-order rate to confirm load success.

## 3) Helper columns + review merge
- Converts timestamp columns to datetime.
- Aggregates reviews by order and joins average review score.
- Creates purchase time helper features like month, quarter, weekday, hour, and time-of-day label.

## 4) Core EDA plots (2.1)
- Builds class imbalance, month/quarter lateness, and seller-experience lateness charts.
- Purpose: complete required missing EDA checks.

## 5) Review + correlation plots (2.1)
- Creates annotated numeric correlation matrix.
- Plots delay vs review relationships and grouped trend.
- Computes Spearman correlation with `is_late` for non-normal/skewed data sensitivity.

## 6) Extra repo-based plots (2.1)
- Adds log-distribution checks, monthly late rate vs growth, time-pattern volume plots, normalized freight/time metrics, SP seller comparison, price-review relation, and distance outlier visuals.
- Purpose: deeper business understanding and stronger justification for feature choices.

## 7) Imbalance analysis code (2.2)
- Creates temporal train/validation split.
- Builds preprocessing pipeline (scaling numeric + one-hot categorical).
- Trains logistic model with class weights.
- Optionally trains SMOTE + logistic when `imblearn` is available.
- Runs threshold sweep and business cost function (`FP*15 + FN*155`) to find best threshold.
- Returns comparison table with AUC, recall, precision, F1, and cost.

## 8) Feature engineering code (2.3)
- Creates features in groups:
- Temporal: month, quarter, weekend, season, recency-style fields.
- Geographic: same city/region signals.
- Order/Product/Payment: value, freight, volume, bulky and installment-based signals.
- Seller: counts and rolling 30-day seller activity.
- Interaction: combinations like distance x weight and freight x distance.
- Optional: purchase-to-approval time.
- Also creates one-hot payment flags and final feature list.

## 9) Train-only target encoding (2.3)
- Builds late-rate encodings from training period only.
- Applies to customer state, seller state, category, route, seller key.
- Adds seller average review from training period only.
- Purpose: keep predictive signal while reducing leakage risk.

## 10) Leakage checklist (2.4)
- Defines columns that can leak future information.
- Confirms key train-only encoded features exist.
- Confirms review and delivery-outcome fields are excluded from model feature set.
- Prints status checks and final model-feature count.

## 11) Final meaning of this notebook
- It completes Phase 2 from EDA to leakage-safe feature engineering.
- It gives both business explanation (plots) and modeling-ready outputs (features and checks).
