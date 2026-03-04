# Project TODO — Olist Late Delivery Prediction
**Plaksha AML Spring 2026 | Due: Apr 23**

---

## Current Status (as of Mar 4)
- ✅ Interim submitted — problem defined, data loaded, EDA sketched, feature inventory done
- ✅ Business framing: Operations Manager, priority-handling decision, cost matrix (FP=R$15, FN=R$155)
- ✅ Feature inventory: 41 features catalogued across 7 groups (110,196 order-items)
- ❌ Actual merged/cleaned dataset not yet built as a notebook
- ❌ No model trained yet
- ❌ Economic analysis not yet done
- ⏳ Mid-project check-in: **Mar 17–21** (need at least one concrete result)

---

## Phase 1 — Dataset Understanding & Merge (Notebook: `01_data_cleaning.ipynb`)

### 1.1 Understand Every Table
- [ ] Print shape, dtypes, head(5) for all 8 tables
  - `olist_orders_dataset.csv` — timestamps, status (key: filter `order_status == 'delivered'`)
  - `olist_order_items_dataset.csv` — price, freight, seller_id, product_id (multi-item = aggregate per order)
  - `olist_customers_dataset.csv` — customer zip, city, state
  - `olist_sellers_dataset.csv` — seller zip, city, state
  - `olist_products_dataset.csv` — weight, dimensions, category (73 categories)
  - `olist_order_payments_dataset.csv` — payment_type, installments, value (multiple rows per order → aggregate)
  - `olist_order_reviews_dataset.csv` — review_score (don't use as a feature — it's post-delivery)
  - `olist_geolocation_dataset.csv` — zip → lat/lon (many duplicates: use median lat/lon per zip)

### 1.2 Build Target Variable
- [ ] Keep only `order_status == 'delivered'` (~96,476 rows)
- [ ] `is_late = 1` if `order_delivered_customer_date > order_estimated_delivery_date` else 0
- [ ] Verify: expected ~8.1% late (≈7,827 orders)
- [ ] Drop rows where `order_delivered_customer_date` is null (canceled/in-transit)

### 1.3 Merge Pipeline (step by step, check shape after each join)
```
orders (base)
  → LEFT JOIN order_items (aggregate per order_id: n_items, total_price, total_freight, max_weight)
  → LEFT JOIN customers (on customer_id → customer_state, customer_zip)
  → LEFT JOIN sellers (on seller_id → seller_state, seller_zip) [from items table]
  → LEFT JOIN products (on product_id → category, weight, dimensions) [heaviest item or aggregated]
  → LEFT JOIN payments (aggregate per order_id: payment_type mode, total_value, n_installments)
  → LEFT JOIN geolocation (customer_zip → customer_lat/lon, seller_zip → seller_lat/lon)
```
- [ ] Final merged shape expected: ~96,476 rows × ~35 raw columns
- [ ] Save as `datasets/olist_merged_clean.csv`

### 1.4 Missing Value Handling
| Column | Strategy |
|---|---|
| `product_category_name` (1.4%) | Fill with `"unknown"` |
| `product_weight_g` (<0.1%) | Fill with category median |
| `product_dimensions` (<0.1%) | Fill with category median |
| `geolocation` (~0.5%) | Fill with state centroid lat/lon |
| `review_score` | Not a feature — only use to validate business framing |

---

## Phase 2 — EDA & Feature Engineering (Notebook: `02_eda_features.ipynb`)

### 2.1 EDA — Must-Do Plots
- [ ] **Class imbalance bar** — 91.9% on-time vs 8.1% late (justify SMOTE/class_weight)
- [ ] **Late rate by month/quarter** — show Q2 peak, seasonal pattern
- [ ] **Late rate by customer state** — choropleth or bar (RN, PA, AM worst)
- [ ] **Late rate by product category** — furniture 14.2%, books 3.2%
- [ ] **Late rate by distance bucket** (0-100km, 100-500km, 500-1000km, 1000km+)
- [ ] **Late rate by seller experience** (seller order count bins: <10, 10-50, 50-200, 200+)
- [ ] **Review score vs delivery delay** — show exponential degradation (key non-trivial insight)
- [ ] **Avg review score by `is_late`** — bar chart showing late=1 orders score 1–2 stars vs late=0 orders score 4–5 stars (cite Mangla & Soni 2025: delivery = #1 predictor of satisfaction)
- [ ] **Payment type vs late rate** — boleto vs credit card
- [ ] **Correlation heatmap** of numerical features vs `is_late`
- [ ] **Distribution plots** for: price, weight, distance_km, freight_cost (log scale — skewed)

### 2.2 Imbalance Analysis
- [ ] Confirm class ratio: 1:11.3
- [ ] Try all three approaches and compare:
  1. `class_weight='balanced'` in sklearn models
  2. SMOTE (from `imbalanced-learn`) on training set only — never on test
  3. Threshold tuning at inference time
- [ ] Choose based on holdout performance, not training metrics

### 2.3 Feature Engineering (target: ~45–50 features)

#### Temporal (10)
- [ ] `month`, `quarter`, `day_of_week`, `hour_of_day`, `is_weekend`
- [ ] `is_peak_season` (Q2: Apr–Jun = 1)
- [ ] `week_of_year`
- [ ] `days_since_platform_launch` (2016 = early, 2018 = mature — platform reliability changed)
- [ ] `is_holiday_season` (Nov–Dec = 1; Black Friday surge confirmed by Repo 2 forecasting + Repo 3 EDA)
- [ ] `is_end_of_month` (`order_purchase_timestamp.dt.day >= 24`; Repo 3 shows end-of-month order spike overwhelms seller capacity)
- [ ] `purchase_time_of_day` bin: Dawn(0–6)/Morning(7–12)/Afternoon(13–18)/Night(19–24) — Afternoon = highest volume → next-day processing by seller

#### Geographic (7)
- [ ] `distance_km` — Haversine from seller_zip centroid to customer_zip centroid
  ```python
  # Use median lat/lon per zip from geolocation table
  # Apply Haversine: R=6371 km
  ```
- [ ] `same_state` (bool)
- [ ] `same_city` (bool)
- [ ] `customer_state` (label/target encode)
- [ ] `seller_state` (label/target encode)
- [ ] `route_late_rate` — historical late rate for (seller_state, customer_state) pair (target encode, careful of leakage — use train split only)
- [ ] `customer_region` (N/NE/SE/S/CO — Brazil's 5 macro-regions, simpler than 27 states)

#### Order (6)
- [ ] `n_items` (number of items in order)
- [ ] `total_order_value`
- [ ] `total_freight_cost`
- [ ] `avg_item_price`
- [ ] `freight_to_value_ratio` (proxy for remote location)
- [ ] `freight_per_item`

#### Product (8)
- [ ] `product_weight_g` (use max or sum across items)
- [ ] `product_volume_cm3` = length × height × width
- [ ] `is_bulky` = weight > 10,000g OR volume > 100,000 cm³
- [ ] `product_category` (target encode — 73 categories)
- [ ] `category_late_rate` (compute from training set only)
- [ ] `is_furniture` (bool — highest risk category)
- [ ] `product_photos_qty` (low photos → low seller quality signal)
- [ ] `weight_to_price_ratio`

#### Payment (5)
- [ ] `payment_value`
- [ ] `payment_installments`
- [ ] `payment_per_installment`
- [ ] `payment_type` (one-hot: credit_card, boleto, debit_card, voucher)

#### Seller (5)
- [ ] `seller_order_count` (total orders fulfilled, proxy for experience)
- [ ] `seller_late_rate` (historical rate, train set only — key leakage risk!)
- [ ] `seller_avg_review` (from training reviews only)
- [ ] `seller_state_volume` (how much a seller's state ships)
- [ ] `seller_monthly_volume` (orders to this seller in 30d window before purchase — seller stress proxy; Repo 3: seller prep time doubles during peak)

#### Time Pressure (3)
- [ ] `estimated_delivery_days` = `order_estimated_delivery_date - order_purchase_timestamp`
- [ ] `is_urgent` = estimated_delivery_days < 7
- [ ] `is_long_window` = estimated_delivery_days > 20

#### Interaction Features (7) — capture non-linear compounding risk
- [ ] `distance_x_weight` = distance_km × product_weight_g (normalized)
- [ ] `distance_x_bulky` = distance_km × is_bulky
- [ ] `peak_x_distance` = is_peak_season × distance_km
- [ ] `freight_x_distance` = freight_cost / max(distance_km, 1)
- [ ] `seller_exp_x_distance` = (1/seller_order_count) × distance_km
- [ ] `price_per_km` = `total_order_value / max(distance_km, 1)` — high-value products over long routes = worst late outcome (from paper Fig 8: expensive orders → lowest satisfaction when late)
- [ ] `is_high_value_and_high_distance` = `(total_order_value > 75th percentile) & (distance_km > 500km)` — binary risk flag for highest-stakes orders

### 2.4 Leakage Checklist — CRITICAL
- [ ] `seller_late_rate`, `category_late_rate`, `route_late_rate` must be computed from **training fold only**, then applied to val/test
- [ ] `review_score` must be **excluded** (post-delivery information)
- [ ] `order_delivered_customer_date` must be **excluded** (post-delivery)
- [ ] `shipping_limit_date` from order_items — use only if available at order placement time (it is — it's the seller's deadline, set at purchase)
- [ ] All timestamps used must be derivable from `order_purchase_timestamp`

---

## Phase 3 — Modeling (Notebook: `03_modeling.ipynb`)

### 3.1 Data Split Strategy — Temporal (avoid leakage)
```
Train:      Sep 2016 – Dec 2017   (~45,000 orders)
Validation: Jan 2018 – Jun 2018   (~27,000 orders)  ← tune threshold here
Test:       Jul 2018 – Oct 2018   (~24,000 orders)  ← final evaluation only
```
- [ ] Sort by `order_purchase_timestamp` before splitting — no random shuffle
- [ ] Recompute all target-encoded features (seller_late_rate etc.) from train split only

### 3.2 Models to Train

#### Baseline
- [ ] **Logistic Regression** (with StandardScaler, `class_weight='balanced'`)
  - Interpretable coefficients
  - Set benchmark AUC, recall, cost

#### Tree-Based (Primary)
- [ ] **Random Forest** (`class_weight='balanced'`, `n_estimators=300`)
- [ ] **XGBoost** (`scale_pos_weight = n_negative/n_positive ≈ 11.3`, `eval_metric='aucpr'`)
- [ ] **LightGBM** (faster than XGBoost; use `is_unbalance=True`)
- [ ] **HistGradientBoosting** (`sklearn.ensemble.HistGradientBoostingClassifier`) — handles NaN natively, best ROC-AUC (0.7389) in paper, fastest sklearn GBM

#### Ensembles (after individual models are trained)
- [ ] **Soft Voting Ensemble** — `VotingClassifier(voting='soft')` on all 4 tree models; need probabilities not hard votes (essential for threshold sweep)
- [ ] **Cost-Weighted MetaVote** — weight each model by how much it minimizes `FP×15 + FN×155` on validation at its optimal threshold (adapted from paper's Precision-Weighted MetaVote)
  ```python
  # Weight = 1 / min_cost_on_val for each model → normalize → weighted prob average
  weights = [1/cost_score(y_val, m.predict_proba(X_val)[:,1]) for m in models]
  weights = [w/sum(weights) for w in weights]
  final_proba = sum(w * m.predict_proba(X_test)[:,1] for w,m in zip(weights,models))
  ```

#### Optional (if time allows)
- [ ] **CatBoost** (handles high-cardinality categoricals natively — good for state/category)
- [ ] **Logistic Regression + Polynomial Features** (see if interactions help baseline)

### 3.3 Hyperparameter Tuning
- [ ] XGBoost: tune `max_depth` (3–8), `learning_rate` (0.01–0.1), `n_estimators` (100–500), `subsample`, `colsample_bytree`
- [ ] Use `Optuna` or `RandomizedSearchCV` on validation set (not test!)
- [ ] Report best params in notebook

### 3.4 Threshold Optimization — THE KEY STEP
```python
cost_fn = lambda fp, fn: fp * 15 + fn * 155
# Sweep threshold from 0.05 to 0.95
# Pick threshold that minimizes cost on VALIDATION set
# Apply that threshold to TEST set
```
- [ ] Plot: Total Economic Cost vs Threshold (validation set)
- [ ] Report: optimal threshold, corresponding precision, recall, cost
- [ ] Compare: cost at threshold=0.5 (default) vs optimal threshold
- [ ] Expected: optimal threshold ≈ 0.2–0.3 (aggressive, given FN >> FP cost)

### 3.5 Evaluation on Test Set
- [ ] Confusion matrix (with R$ cost annotations on each cell)
- [ ] Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- [ ] **Primary: Total Economic Cost (R$)** at optimal threshold
- [ ] **Secondary: Cost vs. doing nothing** (0% flagging) and **vs. flagging everything** (100% flagging)
- [ ] Compute: net savings = cost(no model) − cost(model) per 10,000 orders

### 3.6 Handle Class Imbalance — Compare Methods
| Method | Notes |
|---|---|
| `class_weight='balanced'` | Default first try |
| SMOTE (train only) | May improve recall |
| Threshold tuning | Always done regardless |
- [ ] Table: compare recall/precision/cost across imbalance strategies

---

## Phase 4 — Why the Model Works (Notebook: `04_interpretation.ipynb`)

### 4.1 Feature Importance
- [ ] XGBoost built-in feature importance (gain and weight)
- [ ] **SHAP values** — waterfall plot, summary beeswarm plot, bar plot
  ```python
  import shap
  explainer = shap.TreeExplainer(xgb_model)
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test)
  ```
- [ ] Partial Dependence Plots (PDP) for top 5 features: distance_km, weight, category, same_state, estimated_delivery_days
- [ ] SHAP interaction plot: distance × weight (verify interaction feature matters)

### 4.2 What Drives Late Deliveries — Hypothesis vs Findings
Document for each key feature:
- [ ] **distance_km**: expected #1 driver — verify. Cross-state (>500km) likely dominant
- [ ] **product_weight / is_bulky**: confirm furniture/heavy items have elevated SHAP
- [ ] **seller_late_rate**: shows seller reliability predicts future — key business insight
- [ ] **estimated_delivery_days**: shorter window → higher risk (platform over-commits)
- [ ] **is_peak_season**: Q2 surge effect — quantify seasonal SHAP contribution
- [ ] Any surprises? (e.g., payment_type boleto, seller_state, time of day)

### 4.3 Segmented Performance
- [ ] What's the model's recall for: furniture orders, cross-state orders, peak season orders?
- [ ] Where does it fail? (low recall segments = underserved by model → limitations section)
- [ ] Which routes (seller_state → customer_state) are highest risk? Top 10 table.

---

## Phase 5 — Business & Economic Analysis (Notebook: `05_business_economics.ipynb`)

### 5.1 Cost Matrix Translation
- [ ] At optimal threshold on test set: count TP, FP, TN, FN
- [ ] Compute:
  ```
  Total Cost (model)     = FP × 15 + FN × 155
  Cost if flag nothing   = all_late × 155           (baseline: do nothing)
  Cost if flag everything = all_orders × 20 + TN × 5  (flag everyone)
  Net savings vs baseline = cost(nothing) - cost(model)
  ```
- [ ] Per 10,000 orders: how much does the model save in R$?
- [ ] Annualize: ~100,000 orders/year → annual R$ savings

### 5.2 Sensitivity Analysis — 4 Scenarios
| Scenario | Review Loss | Intervention Cost | Key Question |
|---|---|---|---|
| Base case | R$150 | R$20 | Baseline |
| Conservative | R$75 | R$20 | What if churn impact is low? |
| High churn | R$300 | R$20 | What if CLV loss is high? |
| Expensive ops | R$150 | R$40 | What if premium logistics costs double? |
- [ ] For each: recompute optimal threshold, total cost, net savings
- [ ] Key finding: does the model remain profitable across all 4? (it should — asymmetry holds)
- [ ] Plot: Net Savings vs Review Loss Cost (line chart) — show the break-even point
- [ ] Show: insight survives perturbation (mirrors credit scoring lecture)

### 5.3 Seasonal Threshold Strategy
- [ ] Does optimal threshold differ in Q2 (peak) vs Q3 (off-peak)?
- [ ] If yes: recommend **dynamic threshold** — lower in peak season
- [ ] Quantify: how much extra savings from dynamic vs single threshold?

### 5.4 Segment Prioritization
- [ ] Top 5 highest-risk order profiles (distance bucket × category × season)
- [ ] What % of late orders are captured by the top 3 risk profiles?
- [ ] Can we define rule-based tiers? (always flag: furniture + cross-state + Q2)
- [ ] Compare rule-based vs model on those segments

### 5.5 Business Recommendation
- [ ] Primary recommendation: deploy model at order placement, flag top X% as priority
- [ ] Secondary: dynamic threshold by season
- [ ] Third: invest in regional fulfillment for high-risk routes (geography dominates)
- [ ] ROI calculation: cost of building system vs annual savings

---

## Phase 6 — Limitations & Next Investment

- [ ] What we can't know without being Olist:
  - Actual churn rate after late delivery (assumed, not measured)
  - Intervention effectiveness — does priority handling actually prevent delay? (need A/B test)
  - Operational capacity: can ops actually route 2,000+ orders/day through premium logistics?
  - Weather, carrier strikes, road conditions (not in data)
- [ ] Next investment:
  1. **A/B test**: randomly assign to intervention/control, measure causal effect
  2. **Real-time pipeline**: integrate model into order placement API (<100ms inference)
  3. **Feedback loop**: track which flagged orders were actually saved
  4. **External features**: weather API at seller→customer route, carrier reliability scores
  5. **Multi-seller orders**: currently assumes 1 seller per order (not always true)

---

## Phase 7 — Other Repos & Research Paper — Methods to Borrow

### ✅ From `Late_Shipment_Prediction_Olist/` → see `TODO/repo1_late_shipment_analysis.md`
- [x] Read both notebooks; analysis complete
- Key borrows: target buffer trick, threshold sweep pattern, SMOTE>ADASYN finding, counterfactual predict_proba analysis
- Key errors NOT to copy: random split (temporal leakage), `order_approved_to_delivered_to_carrier` as feature (post-purchase leakage)

### ✅ From `Olist-business-analysis/` → see `TODO/repo2_olist_business_analysis.md`
- [x] Read all 6 analysis notebooks/files; analysis complete
- Key borrows: centroid geolocation, geodesic distance, Spearman correlation for EDA, GBM > RF confirmed, payment installments as rural proxy, 5 high-risk geographic clusters in N/NE Brazil
- New features added: `is_holiday_season`, `high_installment_flag`, `is_north_northeast_customer`

### ✅ From `olists-delivery-analysis/main.ipynb` → see `TODO/repo3_olist_delivery_analysis.md`
- [x] Read full notebook; analysis complete
- Key borrows: 3-stage delivery decomposition, `not_defined` payment drop, `purchase_time_of_day` bucketing, IQR outlier code, `day_time()` function, OLS statsmodels summary
- New features added: `is_end_of_month`, `purchase_time_of_day`, `seller_monthly_volume`
- Critical error NOT to copy: MinMaxScaler fitted before split (leakage), random split

### ✅ Research Paper: Mangla & Soni (2025) → see `TODO/paper_customer_satisfaction_analysis.md`
- [x] Read full paper; analysis complete
- Key borrows: HistGradientBoosting as 5th model, Soft Voting Ensemble, Cost-Weighted MetaVote (adapted from Precision-Weighted MetaVote), `price_per_km` interaction, academic citation for delivery → satisfaction link
- Critical errors NOT to copy: leaky features (`delay_duration`, `carrier_time`), unspecified balancing method, random split
- Citation for report: Mangla & Soni (2025) achieved ROC-AUC=0.7389 with leaky features; our honest model without post-purchase data provides fairer benchmark

---

## Deliverable Checklist

### Mid-Project Check-In (Mar 17–21) — Written 2 pages
- [ ] Show: cleaned merged dataset (shape, features)
- [ ] Show: at least one trained baseline model with real numbers (AUC, recall, cost)
- [ ] Show: one EDA finding with a figure
- [ ] Revised plan: what remains, biggest risk

### Final Report (Apr 23) — 8–12 pages
- [ ] Introduction + business framing
- [ ] Data description (with sample)
- [ ] Feature engineering (include what didn't work)
- [ ] Methodology + model comparison table
- [ ] Results: confusion matrix with costs, ROC-AUC, PR-AUC
- [ ] **Economic analysis + sensitivity table** (the 30% weight criterion)
- [ ] Limitations + next investment
- [ ] Contribution statement

### GitHub Repository (Apr 23)
- [ ] Notebooks numbered: `01_data_cleaning.ipynb`, `02_eda_features.ipynb`, `03_modeling.ipynb`, `04_interpretation.ipynb`, `05_business_economics.ipynb`
- [ ] `requirements.txt` with all packages
- [ ] `README.md`: project description, how to run, data access
- [ ] No API keys or credentials in any file
- [ ] All notebooks run cleanly from top to bottom

### Presentation (Apr 23 – May 4)
- [ ] Structure: business context (2–3 min) → data + features (3–4 min) → methodology (3–4 min) → business impact (3–4 min) → limitations (1–2 min)
- [ ] Key slides: cost matrix, total cost vs threshold chart, SHAP summary, sensitivity table, savings waterfall
- [ ] Both members present

---

## Key Risks

| Risk | Mitigation |
|---|---|
| Geolocation merge reduces rows significantly | Use state centroid fallback for missing zips |
| Seller target-encoding leakage | Compute strictly from train fold, re-check |
| SMOTE on multi-feature data distorts space | Compare with/without on validation — keep only if recall improves |
| Model only learns "distance" and adds no value | If distance alone explains 80%, add seller_late_rate as key differentiator |
| Mid-check-in has no concrete result | Train logistic regression baseline this week — takes 20 min |
