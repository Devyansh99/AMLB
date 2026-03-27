# Project: Review Prediction & Seller Impact

## Status Summary
- **Data:** Created `olist_reviews_rich.csv` (94k rows) with reviews, logistics, and IDs.
- **EDA:** Completed in `notebooks/new/eda.ipynb`. Confirmed `is_late` and `delivery_days` are primary drivers of reviews.
- **Methodology:** "Seller Service Premium" analysis using Residuals (Model A vs Model B).

## To-Do
- [ ] **Baseline Model:** Train Model A (Logistics + Product features only).
- [ ] **Full Model:** Train Model B (Add Seller/Customer metadata).
- [ ] **Impact Analysis:** Calculate residuals to isolate Seller influence.
- [ ] **Re-ranking:** Rank sellers by "Service Premium" instead of raw review averages.
