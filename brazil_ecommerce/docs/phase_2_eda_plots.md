# Phase 2 EDA: What Each Plot Tells

Source notebook: `02_EDA_Continue.ipynb`

This file explains each plot in simple words.

## 1) Class imbalance bar
- Shows how many late vs on-time/early orders we have.
- Why this matters: if late orders are much fewer, model training can become biased to the majority class.

## 2) Late rate by month (line)
- Shows how late-delivery percent changes month by month.
- Why this matters: helps detect seasonality and bad periods.

## 3) Late rate by quarter (bar)
- Gives a higher-level seasonal view than monthly.
- Why this matters: confirms whether delay patterns are stable across larger time blocks.

## 4) Late rate by seller experience bin
- Compares seller lateness for sellers with low vs high order history.
- Why this matters: seller maturity may affect logistics performance.

## 5) Correlation heatmap with numbers
- Shows linear relationships between numeric variables, including `is_late`.
- Why this matters: quickly identifies useful signals and possible redundant features.

## 6) Review score vs delivery delay (scatter)
- Shows individual points of delay against customer review score.
- Why this matters: checks if delay hurts satisfaction in real data.

## 7) Binned trend: delay vs review (line)
- Smooths the scatter into delay bins for clearer trend.
- Why this matters: easier to explain the overall direction.

## 8) Average review by late flag (bar)
- Compares average review score for late and non-late orders.
- Why this matters: simple business message about customer impact of late delivery.

## 9) Correlation heatmap (numeric)
- Another compact relation map focused on numeric fields.
- Why this matters: a second quick check before feature engineering/modeling.

## 10) Log distributions (price, weight, distance, freight)
- Shows skewed variables after log transform.
- Why this matters: helps understand outliers and supports scaling/model decisions.

## 11) Monthly late rate vs next-month order growth (dual-axis)
- Compares service quality now with volume growth in the next month.
- Why this matters: supports discussion on business impact of late delivery.

## 12) Order volume by time of day
- Shows when customers place orders during the day.
- Why this matters: helps define operational peaks.

## 13) Order volume by day of week
- Shows weekday/weekend demand differences.
- Why this matters: useful for staffing and logistics planning.

## 14) Order volume by day of month
- Shows demand pattern within a month.
- Why this matters: detects month-end or salary-cycle spikes.

## 15) Freight per 10km and delivery hours per 10km (histograms)
- Normalizes cost/time by distance.
- Why this matters: better route-efficiency view than raw freight or raw delivery days.

## 16) SP dominance charts
- Compares SP sellers vs non-SP sellers in order share and average freight.
- Why this matters: checks geographic concentration and freight differences.

## 17) Average price vs review score (line)
- Shows if higher-priced orders get better or worse ratings.
- Why this matters: links order value to customer satisfaction.

## 18) Distance outlier plots (box + clipped violin)
- Shows spread and extreme values in distance.
- Why this matters: validates outlier handling choice for modeling.

## 19) Imbalance analysis result table (model comparison output)
- Compares class-weight logistic, SMOTE logistic (if available), and threshold tuning.
- Why this matters: selects approach with better late-order recall and lower business cost.
