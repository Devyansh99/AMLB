# Project TODO v2 â€” Olist Late Delivery Prediction (Enhanced)
**Plaksha AML Spring 2026 | Due: Apr 23**
**This file merges the original todo.md with actionable items from all 3 repo analyses + research paper**

---

## Current Status (as of Mar 5)
- âœ… Interim submitted (Feb 17) â€” problem defined, data loaded, EDA sketched, feature inventory done
- âœ… Business framing: Operations Manager, priority-handling decision, cost matrix (FP=R$15, FN=R$155)
- âœ… Feature inventory: 41 features catalogued across 7 groups (110,196 order-items)
- âœ… All 3 repos analyzed + research paper read â†’ insights catalogued
- âœ… Phase 1 complete â€” merged/cleaned dataset built (96,189 rows x 34 cols, 7.94% late rate)
- ✅ Baseline model trained (03_Baseline.ipynb): Logistic Regression with class_weight='balanced' and benchmark metrics (AUC=0.7118, Recall=0.5978, Cost@0.5=406,780)
- âŒ Economic analysis not yet done
- âŒ Business analysis not yet done
- â³ Mid-project check-in: **Mar 17â€“21** (need at least one concrete result)

## âš¡ PRIORITY: Mid-Project Check-In (Mar 17â€“21) â€” Written 2 pages

**Deadline is 13â€“17 days away. Must have concrete results.**

### What to show:
- [x] **Cleaned merged dataset** with shape, feature list, dtypes (proves data pipeline works) â€” DONE: 96,189 x 34
- [x] **At least one trained baseline model** (Logistic Regression) with real numbers: AUC, recall, cost at threshold=0.5 AND optimal threshold
- [x] **One EDA finding with a figure** â€” best candidate: "Late rate by distance bucket" or "Review score vs delivery delay" (both are strong visually)
- [ ] **Revised plan** â€” what remains, biggest risk, timeline to Apr 23

### Minimum viable path (do in order):
1. Build `01_data_cleaning.ipynb` â†’ merged dataset (Phase 1 below) â€” **~3-4 hours**
2. Build top 10 features only (distance_km, weight, volume, same_state, n_items, freight, category, month, estimated_delivery_days, payment_installments) â€” **~2 hours**
3. Train Logistic Regression baseline with `class_weight='balanced'` â€” **~30 min**
4. Compute cost at default threshold (0.5) and sweep to find optimal â€” **~30 min**
5. Generate 2-3 EDA plots â€” **~1 hour**
6. Write 2-page check-in document â€” **~1 hour**

---

## Phase 1 â€” Dataset Understanding & Merge (Notebook: `01_data_cleaning.ipynb`)

### 1.1 Understand Every Table
- [x] Print shape, dtypes, head(5) for all 8 tables + `product_category_name_translation.csv`
  - `olist_orders_dataset.csv` â€” timestamps, status (key: filter `order_status == 'delivered'`)
  - `olist_order_items_dataset.csv` â€” price, freight, seller_id, product_id (multi-item = aggregate per order)
  - `olist_customers_dataset.csv` â€” customer zip, city, state
  - `olist_sellers_dataset.csv` â€” seller zip, city, state
  - `olist_products_dataset.csv` â€” weight, dimensions, category (73 categories)
  - `olist_order_payments_dataset.csv` â€” payment_type, installments, value (multiple rows per order â†’ aggregate)
  - `olist_order_reviews_dataset.csv` â€” review_score (don't use as a feature â€” it's post-delivery)
  - `olist_geolocation_dataset.csv` â€” zip â†’ lat/lon (many duplicates: use median lat/lon per zip)

> **FROM REPO 2:** Translate category names to English early using `product_category_name_translation.csv` â€” LEFT JOIN on `product_category_name`. Use English names throughout for EDA readability and report.

### 1.2 Build Target Variable
- [x] Keep only `order_status == 'delivered'` (~96,476 rows)
- [x] **FROM REPO 1:** Add 11:59:59 buffer to estimated delivery date before comparison:
  ```python
  orders['order_estimated_delivery_date'] += pd.Timedelta(hours=11, minutes=59, seconds=59)
  ```
  This prevents false-late labels on same-day deliveries. Their late rate = 7.9% vs our strict 8.1%.
- [x] `is_late = 1` if `order_delivered_customer_date > order_estimated_delivery_date` else 0
- [x] Verify: 7.94% late (7,637 orders out of 96,189) â€” matches expected range
- [x] Drop rows where `order_delivered_customer_date` is null (canceled/in-transit)
- [x] **FROM REPO 1:** Drop rows where any timestamp delta is negative (data entry errors):
  ```python
  # Filter out impossible timestamps
  df = df[df['order_approved_at'] >= df['order_purchase_timestamp']]
  ```

### 1.3 Geolocation Pre-processing
- [x] **FROM REPO 2+3 (both confirm):** Use centroid method â€” `.groupby(zip).mean()` for lat/lng:
  ```python
  loc_centroids = geoloc[['geolocation_zip_code_prefix','geolocation_lat','geolocation_lng']]\
      .groupby('geolocation_zip_code_prefix').mean().reset_index()
  ```
  This is better than Repo 1's `.drop_duplicates()` (averages out GPS noise).

- [x] **FROM REPO 3:** Keep city/state from `.drop_duplicates(subset=['geolocation_zip_code_prefix'])` separately, then merge with centroid lat/lng.

### 1.4 Merge Pipeline (step by step, check shape after each join)
```
orders (base)
  â†’ LEFT JOIN order_items (aggregate per order_id: n_items, total_price, total_freight, max_weight)
  â†’ LEFT JOIN customers (on customer_id â†’ customer_state, customer_zip)
  â†’ LEFT JOIN sellers (on seller_id â†’ seller_state, seller_zip) [from items table]
  â†’ LEFT JOIN products (on product_id â†’ category, weight, dimensions) [heaviest item or aggregated]
  â†’ LEFT JOIN payments (aggregate per order_id: payment_type mode, total_value, max_installments)
  â†’ LEFT JOIN geolocation (customer_zip â†’ customer_lat/lon, seller_zip â†’ seller_lat/lon)
```

- [x] **FROM REPO 2:** Rename lat/lng columns immediately after merge to avoid `_x/_y` ambiguity:
  ```python
  delivery.rename(columns={
      'geolocation_lat_x': 'customer_lat', 'geolocation_lng_x': 'customer_lng',
      'geolocation_lat_y': 'seller_lat', 'geolocation_lng_y': 'seller_lng'
  }, inplace=True)
  ```

- [x] **FROM REPO 2:** For payment aggregation, use `.sum()` for payment_value but `.max()` for installments (summing installments across payment methods is meaningless):
  ```python
  pay_agg = payments.groupby('order_id').agg({
      'payment_value': 'sum',
      'payment_installments': 'max',
      'payment_type': lambda x: x.mode()[0]  # most common payment type
  }).reset_index()
  ```

- [x] **FROM REPO 3:** Drop `payment_type == 'not_defined'` rows before aggregation:
  ```python
  payments = payments[~payments['payment_type'].isin(['not_defined'])]
  ```

- [x] **Aggregate to order-level** (NOT item-level like Repo 1 did â€” our decision is per-order, not per-item):
  ```python
  items_agg = order_items.groupby('order_id').agg({
      'price': 'sum',           # total_order_value
      'freight_value': 'sum',   # total_freight_cost
      'order_item_id': 'count', # n_items
      'product_weight_g': 'max',# heaviest item
      'seller_id': 'first'      # primary seller (for single-seller orders)
  }).reset_index()
  ```

- [x] Final merged shape: 96,189 rows Ã— 34 columns (after all filters)
- [x] Save as `datasets/processed/olist_merged_clean.csv`

### 1.5 Missing Value Handling
| Column | Strategy | Source |
|---|---|---|
| `product_category_name` (1.4%) | Fill with `"unknown"` | All repos |
| `product_weight_g` (<0.1%) | Fill with category median | Repo 1 |
| `product_dimensions` (<0.1%) | Fill with category median | Repo 1 |
| `geolocation` (~0.5%) | Fill with state centroid lat/lon | Our plan |
| `review_score` | Not a feature â€” only use to validate business framing | All repos confirm |

### 1.6 Month/Date Filtering
- [x] **FROM REPO 3:** Exclude incomplete months from analysis:
  - Drop: 2016-09, 2016-10, 2016-11, 2016-12 (early launch, few orders)
  - Drop: 2018-09, 2018-10 (truncated data, incomplete orders)
  - This is consistent with our temporal split (train: Jan 2017â€“Dec 2017, val: Janâ€“Jun 2018, test: Julâ€“Aug 2018)

---

## Phase 2 â€” EDA & Feature Engineering (Notebook: `02_eda_features.ipynb`)

### 2.1 EDA â€” Must-Do Plots

#### Core Plots (from original todo):
- [x] **Class imbalance bar** â€” 91.9% on-time vs 8.1% late (justify SMOTE/class_weight)
- [x] **Late rate by month/quarter** â€” show Q2 peak, seasonal pattern
- [x] **Late rate by customer state** â€” choropleth or bar (RN, PA, AM worst)
- [x] **Late rate by product category** â€” furniture 14.2%, books 3.2%
- [x] **Late rate by distance bucket** (0-100km, 100-500km, 500-1000km, 1000km+)
- [x] **Late rate by seller experience** (seller order count bins: <10, 10-50, 50-200, 200+)
- [x] **Review score vs delivery delay** â€” show exponential degradation
- [x] **Avg review score by `is_late`** â€” bar chart showing late=1 orders score 1â€“2 stars vs late=0 orders score 4â€“5 stars
- [x] **Payment type vs late rate** â€” boleto vs credit card
- [x] **Correlation heatmap** of numerical features vs `is_late`
- [x] **Distribution plots** for: price, weight, distance_km, freight_cost (log scale â€” skewed)

#### Additional Plots from Repos:

- [x] **FROM REPO 2:** Use **Spearman correlation** (not Pearson) for skewed features:
  ```python
  from scipy.stats import spearmanr
  # All Olist features are right-skewed â€” Spearman handles monotonic non-linear relationships
  ```
  Repo 2 found: distance â†’ Spearman r â‰ˆ 0.6 with freight. Check distance vs is_late.

- [x] **FROM REPO 3:** Three-stage delivery decomposition EDA (descriptive, NOT for features):
  ```python
  # Stage 1: purchase â†’ approval (seller_cfmed)
  # Stage 2: approval â†’ carrier handoff (seller_to_carrier) â† BOTTLENECK
  # Stage 3: carrier â†’ customer delivery (carrier_to_buyer)
  ```
  Show: during Nov-Dec 2017 peak, seller_to_carrier nearly doubles (~50h â†’ ~80h). This is the insight that seller preparation, not carrier transit, is the bottleneck.

- [x] **FROM REPO 3:** Dual-axis chart â€” late_rate (monthly, red) vs order_growth_next_month (green):
  Shows causal chain: late rate climbed Nov 2017 â†’ next-month orders went flat/negative.

- [x] **FROM REPO 3:** Order volume patterns:
  - By time of day: Afternoon = highest volume
  - By day of week: Monâ€“Wed > Thuâ€“Sun
  - By day of month: Spike at days 24-28 (paycheck effect)
  - Confirms our temporal features (`purchase_time_of_day`, `is_end_of_month`)

- [x] **FROM REPO 2:** Freight normalized metrics for route comparison:
  - `freight_per_10km` = freight_value Ã— 10 / distance_km
  - `hours_per_10km` = delivery_hours Ã— 10 / distance_km
  Good for EDA comparisons across routes of different lengths.

- [x] **FROM REPO 3:** SP state dominance chart â€” 80% of orders from SP sellers; SP sellers avg 18-19 BRL freight vs 25-30+ BRL for non-SP. Confirms `same_state` feature importance.

- [x] **FROM PAPER (Fig 8):** Avg price vs review score â€” higher price â†’ slightly lower satisfaction. Supports `price_per_km` interaction feature: expensive orders over long routes = worst outcome.

- [x] **FROM REPO 3:** IQR outlier visualization (don't remove outliers for modeling, but use IQR for EDA display):
  ```python
  q1 = df['distance_km'].quantile(0.25)
  q3 = df['distance_km'].quantile(0.75)
  iqr = q3 - q1
  upper = q3 + 1.5 * iqr
  # Use for boxplot/violin plot clip, NOT for data removal
  ```

### 2.2 Imbalance Analysis
- [x] Confirm class ratio: 1:11.3
- [x] Try all three approaches and compare:
  1. `class_weight='balanced'` in sklearn models
  2. SMOTE (from `imbalanced-learn`) on training set only â€” never on test
  3. Threshold tuning at inference time
- [x] **FROM REPO 1:** SMOTE > ADASYN on this dataset (their empirical comparison). Skip testing ADASYN â€” saves one experiment.
- [x] **FROM REPO 1:** Always SMOTE inside CV fold using `imblearn.pipeline.Pipeline` â€” their lesson learned was that SMOTE before CV bleeds into folds â†’ inflated scores.
- [x] Choose based on holdout performance, not training metrics

### 2.3 Feature Engineering (target: ~50â€“55 features)

#### Temporal (12)
- [x] `month`, `quarter`, `day_of_week`, `hour_of_day`, `is_weekend`
- [x] `is_peak_season` (Q2: Aprâ€“Jun = 1)
- [x] `week_of_year`
- [x] `days_since_platform_launch` (2016 = early, 2018 = mature)
- [x] `is_holiday_season` (Novâ€“Dec = 1)
  > **FROM REPO 2 (Forecasting-Sales):** Nov-Dec revenue spikes (Black Friday). Monthly revenue: Nov 2017 = R$1.03M, Dec = R$1.05M vs Aug = R$656K. Confirms holiday logistics strain.
- [x] `is_end_of_month` (`order_purchase_timestamp.dt.day >= 24`)
  > **FROM REPO 3:** End-of-month purchase spike at days 24-28. Sellers overwhelmed by month-end rush â†’ delayed processing.
- [x] `purchase_time_of_day` â€” Dawn(0â€“6)/Morning(7â€“12)/Afternoon(13â€“18)/Night(19â€“24)
  > **FROM REPO 3:** Use their `day_time()` helper function. Afternoon = highest volume â†’ next-day processing by seller.
  ```python
  def day_time(x):
      if x <= 6:   return 'Dawn'
      if x <= 12:  return 'Morning'
      if x <= 18:  return 'Afternoon'
      return 'Night'
  ```

#### Geographic (8)
- [x] `distance_km` â€” from seller_zip centroid to customer_zip centroid
  > **FROM REPO 2:** Use `geopy.distance.geodesic` (WGS-84 ellipsoid, more accurate than haversine for Brazil's scale, ~0.3% difference). BUT vectorize â€” their `.apply()` is slow. Use numpy-vectorized haversine for speed on 100k rows, or the `haversine` package with batch arrays.
  >
  > **FROM REPO 1:** Late deliveries average 740.5 km vs on-time 584.4 km (+27%). Distance is #1 univariate predictor.
  >
  > **FROM REPO 2:** Distance explains 40% of freight cost variance (Spearman râ‰ˆ0.6). Cite in report.
- [x] `same_state` (bool)
- [x] `same_city` (bool)
- [x] `customer_state` (target encode)
- [x] `seller_state` (target encode)
  > **FROM REPO 3:** Do NOT use `pd.get_dummies` for states (54 dummy columns = high dimensionality). Use target encoding: `mean(is_late)` per state from train fold only.
- [x] `route_late_rate` â€” historical late rate for (seller_state, customer_state) pair (target encode, train split only)
- [x] `customer_region` (N/NE/SE/S/CO â€” Brazil's 5 macro-regions)
- [x] `is_north_northeast_customer`
  > **FROM REPO 2 (Route-Optimization):** The 5 highest-risk delivery clusters are all in N/NE Brazil (Manaus AM, Nordeste interior, ParÃ¡, MaranhÃ£o coast, PR coast). These regions have sparse seller coverage â†’ structurally long routes.
  ```python
  NNE_states = ['AM','PA','MA','CE','RN','PB','PE','AL','SE','BA','PI','TO','AP','RR','AC','RO']
  df['is_north_northeast_customer'] = df['customer_state'].isin(NNE_states).astype(int)
  ```

#### Order (6)
- [x] `n_items` (number of items in order)
  > **FROM REPO 1:** Their `order_item_id` feature showed each extra item = -2.6% on-time probability. We use `n_items` (count) â€” better semantics.
- [x] `total_order_value`
- [x] `total_freight_cost`
- [x] `avg_item_price`
- [x] `freight_to_value_ratio` (proxy for remote location)
  > **FROM REPO 2:** Confirmed â€” distance is the dominant driver of freight cost; high freight/value ratio = remote/rural customer.
- [x] `freight_per_item`

#### Product (8)
- [x] `product_weight_g` (use max or sum across items)
- [x] `product_volume_cm3` = length Ã— height Ã— width
  > **FROM REPO 3:** Aggregate per order with `.groupby(order_id).mean()` if multi-item.
- [x] `is_bulky` = weight > 10,000g OR volume > 100,000 cmÂ³
- [x] `product_category` (target encode â€” 73 categories)
  > **FROM REPO 1 CRITICAL:** They dropped category entirely because it's a string â€” big miss. Furniture=14.2% vs books=3.2% late rate. Always target-encode from train fold.
- [x] `category_late_rate` (compute from training set only)
- [x] `is_furniture` (bool â€” highest risk category)
- [x] `product_photos_qty` (low photos â†’ low seller quality signal)
  > **FROM REPO 2 (Images-Reviews):** Photos predict sales for beauty, fashion, appliances but NOT electronics/computers. Could create interaction: `photos_x_visual_category` for categories where listing quality matters.
- [x] `weight_to_price_ratio`

#### Payment (6)
- [x] `payment_value`
- [x] `payment_installments`
- [x] `payment_per_installment`
- [x] `payment_type` (one-hot: credit_card, boleto, debit_card, voucher)
- [x] `high_installment_flag` = `payment_installments > 6`
  > **FROM REPO 2 (Payment-Analysis):** High installments (>6) correlate with expensive categories AND remote/rural buyers. Computers avg ~8 installments, furniture ~3.2. This is a rural/remote proxy beyond just distance.

#### Seller (5)
- [x] `seller_order_count` (total orders fulfilled, proxy for experience)
- [x] `seller_late_rate` (historical rate, train set only â€” key leakage risk!)
- [x] `seller_avg_review` (from training reviews only)
- [x] `seller_state_volume` (how much a seller's state ships)
- [x] `seller_monthly_volume` (orders to this seller in 30d window before purchase)
  > **FROM REPO 3:** During peak, `seller_to_carrier` time nearly doubles (~50h â†’ ~80h). Seller stress = volume surge. Pre-purchase proxy: count orders to this seller in prior 30 days.

#### Time Pressure (3)
- [x] `estimated_delivery_days` = `order_estimated_delivery_date - order_purchase_timestamp`
  > **FROM REPO 3:** Olist deliberately over-estimates by ~13.6 days as buffer. Orders that STILL miss this generous estimate are truly problematic.
- [x] `is_urgent` = estimated_delivery_days < 7
- [x] `is_long_window` = estimated_delivery_days > 20

#### Interaction Features (7)
- [x] `distance_x_weight` = distance_km Ã— product_weight_g (normalized)
  > **FROM REPO 2 (PCA analysis):** Physical features + distance compress into one "delivery complexity" dimension. This interaction captures that latent factor.
- [x] `distance_x_bulky` = distance_km Ã— is_bulky
- [x] `peak_x_distance` = is_peak_season Ã— distance_km
- [x] `freight_x_distance` = freight_cost / max(distance_km, 1)
- [x] `seller_exp_x_distance` = (1/seller_order_count) Ã— distance_km
- [x] `price_per_km` = `total_order_value / max(distance_km, 1)`
  > **FROM PAPER (Fig 8):** Higher-priced products get slightly lower review scores â€” expectations scale with price. Expensive orders over long routes = worst satisfaction outcome.
- [x] `is_high_value_and_high_distance` = `(total_order_value > 75th pct) & (distance_km > 500km)` â€” binary risk flag

#### Optional Features (from repos, add if time allows):
- [x] `order_purchased_to_approved` = days from purchase to payment approval
  > **FROM REPO 1:** Valid pre-shipment signal (slow approval â†’ delay risk). ~5 hours normally, ~6 hours during peak.
- [x] `is_sp_seller` = `seller_state == 'SP'`
  > **FROM REPO 3:** 80% of orders from SP sellers. SP sellers have lowest freight (18-19 BRL avg).

### 2.4 Leakage Checklist â€” CRITICAL
- [x] `seller_late_rate`, `category_late_rate`, `route_late_rate` must be computed from **training fold only**, then applied to val/test
- [x] `review_score` must be **excluded** (post-delivery information)
  > **FROM ALL 3 REPOS + PAPER:** Confirmed. Paper used `delay_duration` and `carrier_time` as features (post-delivery leakage) â†’ inflated ROC-AUC to 0.7389. Our honest model without these will be lower but production-valid.
- [x] `order_delivered_customer_date` must be **excluded** (post-delivery)
- [x] `order_approved_to_delivered_to_carrier` must be **excluded**
  > **FROM REPO 1 CRITICAL:** They used this as a feature â€” it's post-purchase (carrier handoff date unknown at order placement).
- [x] `shipping_limit_date` â€” use only if available at order placement time (it is â€” seller's deadline, set at purchase)
- [x] All timestamps used must be derivable from `order_purchase_timestamp`
- [x] **FROM REPO 3:** MinMaxScaler must be fitted on train only: `scaler.fit(X_train)` then `scaler.transform(X_test)`. They fitted on all data â†’ leakage.

---

## Phase 3 â€” Modeling (Notebook: `03_modeling.ipynb`)

### 3.1 Data Split Strategy â€” Temporal (avoid leakage)
```
Train:      Jan 2017 â€“ Dec 2017   (~45,000 orders)
Validation: Jan 2018 â€“ Jun 2018   (~27,000 orders)  â† tune threshold here
Test:       Jul 2018 â€“ Aug 2018   (~24,000 orders)  â† final evaluation only
```
- [ ] Sort by `order_purchase_timestamp` before splitting â€” no random shuffle
- [ ] Recompute all target-encoded features (seller_late_rate etc.) from train split only
- [ ] **FROM ALL 3 REPOS:** They ALL used random split â†’ temporal leakage. Our temporal split is a key methodological advantage. Mention in report.
- [ ] **FROM REPO 3:** Exclude Sep-Oct 2018 (truncated data). This aligns with our test set ending at Aug 2018.

### 3.2 Models to Train

#### Baseline
- [x] **Logistic Regression** (with StandardScaler, `class_weight='balanced'`)
  - Interpretable coefficients
  - Set benchmark AUC, recall, cost
  - **FROM REPO 1:** Unscale coefficients for interpretation: `readjusted_coefs = model.coef_[0] / scaler.scale_`
  - **FROM REPO 3:** Run OLS statsmodels summary for p-values and confidence intervals:
    ```python
    import statsmodels.api as sm
    X2 = sm.add_constant(X_train)
    model_stats = sm.OLS(y_train.values.reshape(-1, 1), X2).fit()
    model_stats.summary()
    ```

#### Tree-Based (Primary)
- [ ] **Random Forest** (`class_weight='balanced'`, `n_estimators=300`)
  > **FROM REPO 2:** GradientBoosting > RF confirmed on Olist data (RÂ²=0.71 vs 0.66 for freight prediction). Expect similar for classification.
- [ ] **XGBoost** (`scale_pos_weight = n_negative/n_positive â‰ˆ 11.3`, `eval_metric='aucpr'`)
- [ ] **LightGBM** (faster than XGBoost; use `is_unbalance=True`)
- [ ] **HistGradientBoosting** (`sklearn.ensemble.HistGradientBoostingClassifier`)
  > **FROM PAPER:** Best single-model ROC-AUC (0.7389). Handles NaN natively (no imputation needed), fastest sklearn GBM.
  ```python
  from sklearn.ensemble import HistGradientBoostingClassifier
  hgb = HistGradientBoostingClassifier(
      max_iter=500, learning_rate=0.05, max_depth=6,
      l2_regularization=0.1, random_state=42,
      class_weight='balanced'
  )
  ```

#### Ensembles (after individual models are trained)
- [ ] **Soft Voting Ensemble** â€” `VotingClassifier(voting='soft')` on all 4 tree models
  > **FROM PAPER:** Soft voting > hard voting. We need probabilities for threshold optimization. Hard voting produces binary outputs â€” useless for cost-matrix sweep.
  ```python
  from sklearn.ensemble import VotingClassifier
  ensemble = VotingClassifier(
      estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm), ('hgb', hgb)],
      voting='soft'
  )
  ```
- [ ] **Cost-Weighted MetaVote** â€” weight each model by how much it minimizes `FPÃ—15 + FNÃ—155` on validation
  > **FROM PAPER:** Adapted from their Precision-Weighted MetaVote. They weighted by precision (minimize FP). We weight by cost minimization (our objective function).
  ```python
  def cost_score(y_true, y_proba, fp_cost=15, fn_cost=155):
      best_cost = float('inf')
      for thresh in np.arange(0.05, 0.50, 0.01):
          y_pred = (y_proba >= thresh).astype(int)
          tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
          cost = fp * fp_cost + fn * fn_cost
          if cost < best_cost:
              best_cost = cost
      return 1 / best_cost  # higher = better â†’ use as weight

  weights = [cost_score(y_val, m.predict_proba(X_val)[:,1]) for m in models]
  weights = [w/sum(weights) for w in weights]
  final_proba = sum(w * m.predict_proba(X_test)[:,1] for w,m in zip(weights,models))
  ```

#### Optional (if time allows)
- [ ] **CatBoost** (handles high-cardinality categoricals natively)
- [ ] **Logistic Regression + Polynomial Features**

### 3.3 Hyperparameter Tuning
- [ ] XGBoost: tune `max_depth` (3â€“8), `learning_rate` (0.01â€“0.1), `n_estimators` (100â€“500), `subsample`, `colsample_bytree`
- [ ] Use `Optuna` or `RandomizedSearchCV` on validation set (not test!)
- [ ] **FROM REPO 1:** Reasonable grid for RF: `n_estimators=[i**2 for i in range(1,10)]`, `max_depth=[9,16,25,36,49]`
- [ ] Report best params in notebook
- [ ] **FROM PAPER:** Model spread was only 0.0021 across all approaches â†’ feature engineering matters far more than hyperparameter tuning. Don't over-invest in tuning â€” invest in feature quality.

### 3.4 Threshold Optimization â€” THE KEY STEP
```python
cost_fn = lambda fp, fn: fp * 15 + fn * 155
# Sweep threshold from 0.05 to 0.95
# Pick threshold that minimizes cost on VALIDATION set
# Apply that threshold to TEST set
```
- [ ] Plot: Total Economic Cost vs Threshold (validation set)
- [ ] Report: optimal threshold, corresponding precision, recall, cost
- [ ] Compare: cost at threshold=0.5 (default) vs optimal threshold
- [ ] Expected: optimal threshold â‰ˆ 0.10â€“0.30 (aggressive, given FN >> FP cost)
  > **FROM REPO 1:** Their empirical optimal was threshold=0.10 using FBeta(Î²=2).
  > **FROM PAPER:** Their thresholds ranged 0.482â€“0.535 â€” but their task was balanced precision/recall. Ours will be much lower because FN cost >> FP cost.

- [ ] **FROM REPO 1:** Borrow their threshold sweep code pattern:
  ```python
  y_probs = model.predict_proba(X_val)[:,1]
  costs = []
  thresholds = np.arange(0.05, 0.95, 0.01)
  for t in thresholds:
      y_pred = (y_probs >= t).astype(int)
      tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
      costs.append(fp * 15 + fn * 155)
  optimal_threshold = thresholds[np.argmin(costs)]
  ```

### 3.5 Evaluation on Test Set
- [ ] Confusion matrix (with R$ cost annotations on each cell)
  > **FROM REPO 1:** Borrow and extend their labeled confusion matrix function. Add R$ per cell: FP=R$15, FN=R$155, TP=+R$130 saved, TN=R$5 standard.
  ```python
  # Extend their function with cost annotations
  cmlabels = [f'TN\nR$5 std', f'FP\nR$15 wasted', f'FN\nR$155 lost', f'TP\nR$130 saved']
  ```
- [ ] Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- [ ] **Primary: Total Economic Cost (R$)** at optimal threshold
- [ ] **Secondary: Cost vs. doing nothing** (0% flagging) and **vs. flagging everything** (100% flagging)
- [ ] Compute: net savings = cost(no model) âˆ’ cost(model) per 10,000 orders
- [ ] **FROM REPO 1:** ROC curve overlay â€” all models on one plot:
  ```python
  fig, ax = plt.subplots(figsize=(8,8))
  for name, fpr, tpr in model_curves:
      plt.plot(fpr, tpr, label=name)
  plt.plot([0,1], [0,1], 'r--', alpha=0.5)  # diagonal baseline
  ```
- [ ] **FROM PAPER:** Add cost metric column to model comparison table (not just accuracy/precision/recall/F1/AUC):
  ```
  | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Cost (R$) | Optimal Threshold |
  ```

### 3.6 Handle Class Imbalance â€” Compare Methods
| Method | Notes |
|---|---|
| `class_weight='balanced'` | Default first try |
| SMOTE (train only) | May improve recall |
| Threshold tuning | Always done regardless |
- [ ] Table: compare recall/precision/cost across imbalance strategies
- [ ] **FROM REPO 1:** Use `imblearn.pipeline.Pipeline` to keep SMOTE inside CV folds

---

## Phase 4 â€” Why the Model Works (Notebook: `04_interpretation.ipynb`)

### 4.1 Feature Importance
- [ ] XGBoost built-in feature importance (gain and weight)
- [ ] **SHAP values** â€” waterfall plot, summary beeswarm plot, bar plot
  > **FROM PAPER:** They did no SHAP analysis at all â€” "cannot know which features drove predictions." This is a key gap we fill.
  ```python
  import shap
  explainer = shap.TreeExplainer(xgb_model)
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test)
  ```
- [ ] Partial Dependence Plots (PDP) for top 5 features: distance_km, weight, category, same_state, estimated_delivery_days
- [ ] SHAP interaction plot: distance Ã— weight (verify interaction feature matters)
- [ ] **FROM REPO 1:** RF feature importance as clean DataFrame:
  ```python
  fi = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
  fi.sort_values('importance', ascending=False)
  ```

### 4.2 What Drives Late Deliveries â€” Hypothesis vs Findings
Document for each key feature:
- [ ] **distance_km**: expected #1 driver â€” verify.
  > **FROM REPO 1:** Quantified at +27% mean difference. **FROM REPO 2:** Explains 40% of freight variance.
- [ ] **product_weight / is_bulky**: confirm furniture/heavy items have elevated SHAP
- [ ] **seller_late_rate**: shows seller reliability predicts future â€” key business insight
  > **FROM REPO 3:** Seller preparation = 25% of delivery time during peak. This is why seller metrics matter.
- [ ] **estimated_delivery_days**: shorter window â†’ higher risk (platform over-commits)
  > **FROM REPO 3:** Olist buffers 13.6 days ahead. Orders that still fail are truly at-risk.
- [ ] **is_peak_season**: Q2 surge effect â€” quantify seasonal SHAP contribution
  > **FROM REPO 3:** Nov 2017 late rate spike â†’ orders went sideways for 5+ months. Causal chain confirmed.
- [ ] Any surprises? (e.g., payment_type boleto, seller_state, time of day)

### 4.3 Segmented Performance
- [ ] What's the model's recall for: furniture orders, cross-state orders, peak season orders?
- [ ] Where does it fail? (low recall segments = underserved by model â†’ limitations section)
- [ ] Which routes (seller_state â†’ customer_state) are highest risk? Top 10 table.
- [ ] **FROM REPO 2:** Check if the 5 high-risk geographic clusters (N/NE Brazil) show up as top risk routes.
- [ ] **FROM REPO 3:** Check SP-to-SP vs cross-state performance â€” model may underweight N/NE lateness because SP-SP dominates training (80% of orders).

### 4.4 Counterfactual Analysis
- [ ] **FROM REPO 1:** Borrow their counterfactual predict_proba technique:
  ```python
  # "What if this seller had average performance?"
  sample = X_test.sample(1000, random_state=42)
  pred_original = model.predict_proba(sample)[:,1]

  sample_modified = sample.copy()
  sample_modified['seller_late_rate'] = sample_modified['seller_late_rate'].mean()
  pred_modified = model.predict_proba(sample_modified)[:,1]

  delta = np.mean(pred_original - pred_modified)
  print(f"Avg probability reduction if all sellers had avg performance: {delta:.4f}")
  ```
  Report in business section: quantify the value of seller improvement.

---

## Phase 5A â€” Economic Analysis (Notebook: `05_economic_analysis.ipynb`) â€” 30% weight

> **PROJECT INSTRUCTIONS:** "show what happens when [economics] change. Which assumptions matter and which don't?"
> This is the highest-weight section. Every analysis here must produce a number, table, or figure.

### 5A.1 Cost Matrix Translation
- [ ] At optimal threshold on test set: count TP, FP, TN, FN
- [ ] Compute:
  ```
  Total Cost (model)     = FP Ã— 15 + FN Ã— 155
  Cost if flag nothing   = all_late Ã— 155           (baseline: do nothing)
  Cost if flag everything = all_orders Ã— 20 + TN Ã— 5  (flag everyone)
  Net savings vs baseline = cost(nothing) - cost(model)
  ```
- [ ] Per 10,000 orders: how much does the model save in R$?
- [ ] Annualize: ~100,000 orders/year â†’ annual R$ savings

### 5A.2 Sensitivity Analysis â€” 4 Scenarios
| Scenario | Review Loss | Intervention Cost | Key Question |
|---|---|---|---|
| Base case | R$150 | R$20 | Baseline |
| Conservative | R$75 | R$20 | What if churn impact is low? |
| High churn | R$300 | R$20 | What if CLV loss is high? |
| Expensive ops | R$150 | R$40 | What if premium logistics costs double? |
- [ ] For each: recompute optimal threshold, total cost, net savings
- [ ] Key finding: does the model remain profitable across all 4? (it should â€” asymmetry holds)
- [ ] Plot: Net Savings vs Review Loss Cost (line chart) â€” show the break-even point
- [ ] Show: insight survives perturbation (mirrors credit scoring lecture)

### 5A.3 Break-Even Analysis
- [ ] At what FN cost (review loss) does the model stop being profitable?
- [ ] At what intervention cost does it stop being profitable?
- [ ] Produce a 2D heatmap: (review_loss Ã— intervention_cost) â†’ net savings. Color red where model loses money.
- [ ] Key insight: "model is profitable for all FN costs above R$X" â€” this is the floor.

### 5A.4 Threshold-Cost Tradeoff Curve
- [ ] Plot: x = threshold (0.05 to 0.95), y = total cost (R$). One line per scenario.
- [ ] Show that optimal threshold shifts left as FN cost increases (more aggressive flagging when stakes are higher).
- [ ] Table: optimal threshold for each scenario.

### 5A.5 Value of Information Analysis
- [ ] Compare total cost for: (a) no model (flag nothing), (b) flag everything, (c) model at t=0.5, (d) model at t=optimal.
- [ ] This quantifies the economic value of having a prediction model vs naive strategies.
- [ ] Waterfall chart: cost savings decomposition (baseline â†’ model â†’ threshold optimization â†’ each step saves R$X).

### 5A.6 Seasonal Threshold Strategy
- [ ] Does optimal threshold differ in Q2 (peak) vs Q3 (off-peak)?
- [ ] If yes: recommend **dynamic threshold** â€” lower in peak season
- [ ] Quantify: extra savings from dynamic vs single threshold
  > **FROM REPO 3:** During peak, seller_to_carrier time doubles. Dynamic threshold accounts for this structural change.

### 5A.7 Freight Cost Economics
- [ ] **FROM REPO 2:** Freight cost vs distance regression â€” what is the marginal cost per km?
- [ ] Compare freight cost for same-state vs cross-state orders (cost of geographic mismatch).
- [ ] Estimate: if Olist onboarded sellers in 5 N/NE clusters, how much freight cost would decrease?
- [ ] Freight-to-value ratio analysis: orders where freight > 30% of product value â€” late rate and customer satisfaction impact.

### 5A.8 Customer Lifetime Value Impact
- [ ] **FROM PAPER (Fig 13):** Late delivery â†’ low review â†’ churn. Quantify the chain:
  - Late orders avg review: ~1.5 stars. On-time avg review: ~4.2 stars.
  - Assume 1-2 star reviews â†’ 60% churn, 4-5 star â†’ 10% churn (industry benchmarks).
  - CLV per customer: estimate from avg order value Ã— avg orders per customer Ã— margin.
- [ ] Sensitivity: CLV Ã— churn_delta Ã— orders_flagged = total revenue protected.
- [ ] This produces the annualized dollar value of the model.

---

## Phase 5B â€” Business Analysis & Recommendations (Notebook: `06_business_analysis.ipynb`) â€” 30% weight

> **PROJECT INSTRUCTIONS:** "Identify a decision-maker. Who would use this model's output, and for what decision?"
> Decision-maker: Operations Manager. Decision: flag risky orders for Priority Handling at order placement.

### 5B.1 Delivery SLA Performance Analysis
- [ ] Compute on-time delivery rate by month â€” trend chart.
- [ ] **FROM REPO 3:** Three-stage delivery decomposition (purchaseâ†’approval, approvalâ†’carrier, carrierâ†’customer).
  - Show: seller_to_carrier is the bottleneck (25% of delivery time during peak, doubles from 50h to 80h).
  - This justifies the seller-focused intervention recommendation.
- [ ] SLA compliance by state â€” which states consistently miss delivery estimates?
- [ ] SLA compliance by product category â€” which categories are structurally late?

### 5B.2 Seller Performance Ranking
- [ ] Rank sellers by late delivery rate (only sellers with >20 orders for statistical reliability).
- [ ] Top 10 worst sellers â€” what % of all late orders do they cause?
- [ ] **FROM REPO 3:** Seller dispatch time distribution â€” show the long tail of slow sellers.
- [ ] Recommendation: target the worst 5% of sellers for SLA enforcement or removal.
- [ ] **FROM REPO 2:** Seller geographic concentration â€” 80% from SP. What if non-SP sellers were onboarded?

### 5B.3 Geographic Risk Analysis
- [ ] Top 10 highest-risk routes (seller_state â†’ customer_state) â€” late rate and volume.
- [ ] **FROM REPO 2 (Route-Optimization):** 5 specific N/NE risk clusters:
  > | Cluster | Location | Products to stock |
  > |---|---|---|
  > | 1 | PR coast (-25.9Â°, -48.6Â°) | health_beauty, telephony |
  > | 2 | NE interior (-6.1Â°, -37.5Â°) | health_beauty |
  > | 3 | RO/northwest (-8.8Â°, -63.9Â°) | sports_leisure |
  > | 4 | AM/Manaus (-3.1Â°, -60.0Â°) | computers, health_beauty, sports_leisure, telephony |
  > | 5 | MA coast (-2.5Â°, -44.2Â°) | health_beauty, telephony |
- [ ] Map visualization: late rate by customer state (choropleth or bubble map).
- [ ] Estimate: orders in N/NE states â€” what % are late vs national average?
- [ ] Same-state vs cross-state delivery performance comparison.

### 5B.4 Segment Prioritization
- [ ] Top 5 highest-risk order profiles (distance bucket Ã— category Ã— season).
- [ ] What % of late orders are captured by the top 3 risk profiles?
- [ ] Define rule-based tiers: always flag (furniture + cross-state + Q2), never flag (same-city + light + off-peak).
- [ ] Compare: rule-based tiers vs ML model â€” which captures more late orders at same FP rate?
- [ ] Conclude whether ML adds value over simple rules (it should â€” interactions matter).

### 5B.5 Operational Recommendations
- [ ] **Recommendation 1:** Deploy model at order placement, flag top X% for Priority Handling.
  - Quantify: at optimal threshold, X% of orders flagged, Y% of late orders caught.
- [ ] **Recommendation 2:** Dynamic threshold by season â€” lower in peak (Nov-Dec, Q2).
- [ ] **Recommendation 3:** Regional fulfillment investment for high-risk N/NE routes.
  > **FROM REPO 2:** Onboarding sellers in 5 clusters would reduce avg distance by Z km for affected orders.
- [ ] **Recommendation 4:** Enforce 48-hour seller dispatch SLA.
  > **FROM REPO 3:** Shopee (1.5 days), Lazada (2 days) already enforce this. If dispatch drops from 80h to 48h, predicted late probability decreases.
- [ ] **Recommendation 5:** Product-category-specific handling â€” bulky/furniture gets automatic premium logistics.

### 5B.6 ROI of Model Deployment
- [ ] Cost of building and maintaining the model (engineering time, compute, monitoring).
- [ ] Annual savings from the model (from 5A.1).
- [ ] Payback period: when does cumulative savings exceed deployment cost?
- [ ] Compare: model-based flagging vs hiring 2 additional logistics coordinators.

### 5B.7 Connecting to Research Paper
- [ ] **FROM PAPER:** Cite: "Mangla & Soni (2025) demonstrated that delivery delay features are the strongest predictors of customer satisfaction on Olist (ROC-AUC=0.7389). Our work extends this by predicting delivery delay itself at order placement time â€” enabling proactive intervention."
- [ ] Argue the economic chain: model â†’ predicts late â†’ Priority Handling â†’ prevents delay â†’ protects reviews â†’ protects retention.
- [ ] **FROM PAPER (Fig 8):** Higher-priced products get lower satisfaction when late â€” justify price-weighted intervention.

### 5B.8 Counterfactual Business Scenarios
- [ ] **FROM REPO 1:** "What if this seller had average performance?" â€” counterfactual predict_proba.
  ```python
  sample_modified['seller_late_rate'] = sample_modified['seller_late_rate'].mean()
  delta = np.mean(pred_original - pred_modified)
  ```
- [ ] "What if all orders were same-state?" â€” compute how much late rate would drop.
- [ ] "What if Olist reduced estimated delivery buffer from 13.6 days to 10 days?" â€” how many more orders would be labeled late?
- [ ] Each scenario quantifies the business value of a specific intervention.

---

## Phase 6 â€” Limitations & Next Investment

- [ ] What we can't know without being Olist:
  - Actual churn rate after late delivery (assumed, not measured)
  - Intervention effectiveness â€” does priority handling actually prevent delay? (need A/B test)
  - Operational capacity: can ops actually route 2,000+ orders/day through premium logistics?
  - Weather, carrier strikes, road conditions (not in data)
  - Dynamic seller behavior â€” do sellers improve after negative feedback?
  > **FROM REPO 3:** "Freight improved delivery speed in 2018 but at higher cost â†’ orders didn't recover." Cost sensitivity matters â€” customers value BOTH speed AND price.

- [ ] Next investment:
  1. **A/B test**: randomly assign to intervention/control, measure causal effect
  2. **Real-time pipeline**: integrate model into order placement API (<100ms inference)
     > **FROM REPO 1:** They built a Flask app with sliders for distance/price/weight â†’ real-time prediction. Mention as deployment reference.
  3. **Feedback loop**: track which flagged orders were actually saved
  4. **External features**: weather API at sellerâ†’customer route, carrier reliability scores
  5. **Multi-seller orders**: currently assumes 1 seller per order (not always true)
  6. **Regional seller onboarding**: partner with sellers in 5 N/NE risk clusters (FROM REPO 2)
  7. **Seller SLA enforcement**: 48-hour dispatch mandate with penalty structure (FROM REPO 3)

- [ ] **FROM PAPER:** Note on our AUC vs theirs: "Paper achieved ROC-AUC=0.7389 using post-delivery features (leakage). Our honest model without those features provides a fairer but lower benchmark. If we achieve AUC > 0.70 without leakage, it demonstrates the model has genuine predictive power from pre-purchase signals alone."

---

## Deliverable Checklist

### Mid-Project Check-In (Mar 17â€“21) â€” Written 2 pages
- [x] Show: cleaned merged dataset (shape, features) â€” 96,189 x 34, 7.94% late rate
- [x] Show: at least one trained baseline model with real numbers (AUC, recall, cost)
- [x] Show: one EDA finding with a figure
- [ ] Revised plan: what remains, biggest risk

### Final Report (Apr 23) â€” 8â€“12 pages
- [ ] Introduction + business framing
  > **PROJECT INSTRUCTIONS:** "Identify a decision-maker. Who would use this model's output, and for what decision?" â†’ Operations Manager, Priority Handling decision.
- [ ] Data description (with sample)
- [ ] Feature engineering (include what didn't work)
  > **PROJECT INSTRUCTIONS:** "A well-chosen problemâ€”the right dataset, the right decision-maker, the right framingâ€”makes the rest of the project flow naturally."
- [ ] Methodology + model comparison table
  > Include cost metric column, not just accuracy/precision/recall/F1/AUC.
- [ ] Results: confusion matrix with costs, ROC-AUC, PR-AUC
- [ ] **Economic analysis + sensitivity table** (the 30% weight criterion)
  > **PROJECT INSTRUCTIONS:** "an insight that survives reasonable perturbation of the assumptions" â€” our 4-scenario sensitivity table demonstrates this.
- [ ] Limitations + next investment
  > **PROJECT INSTRUCTIONS:** "What would you needâ€”data, access, experiments, infrastructureâ€”to move from this analysis to a deployed system? What can't be known without being the firm?"
- [ ] Contribution statement
- [ ] **Citation:** Mangla & Soni (2025), DOI: https://doi.org/10.36676/jrps.v16.i4.337

### GitHub Repository (Apr 23)
- [ ] Notebooks numbered: `01_data_cleaning.ipynb`, `02_eda_features.ipynb`, `03_modeling.ipynb`, `04_interpretation.ipynb`, `05_economic_analysis.ipynb`, `06_business_analysis.ipynb`
- [ ] `requirements.txt` with all packages
- [ ] `README.md`: project description, how to run, data access
- [ ] No API keys or credentials in any file
- [ ] All notebooks run cleanly from top to bottom

### Presentation (Apr 23 â€“ May 4)
- [ ] Structure: business context (2â€“3 min) â†’ data + features (3â€“4 min) â†’ methodology (3â€“4 min) â†’ business impact (3â€“4 min) â†’ limitations (1â€“2 min)
- [ ] Key slides: cost matrix, total cost vs threshold chart, SHAP summary, sensitivity table, savings waterfall
- [ ] Both members present
  > **PROJECT INSTRUCTIONS:** "Both members must contribute to every phase."

---

## Key Risks

| Risk | Mitigation | Repo Source |
|---|---|---|
| Geolocation merge reduces rows significantly | Use state centroid fallback for missing zips | Repos 2+3 confirm centroid approach |
| Seller target-encoding leakage | Compute strictly from train fold, re-check | Repo 1+3 both made this error |
| SMOTE on multi-feature data distorts space | Compare with/without on validation â€” keep only if recall improves | Repo 1: SMOTE > ADASYN |
| Model only learns "distance" and adds no value | If distance alone explains 80%, add seller_late_rate as key differentiator | Repo 2: distance = 40% of freight variance, not 80% |
| Mid-check-in has no concrete result | Train logistic regression baseline this week â€” takes 30 min | Repo 1: LR baseline pattern |
| Random split used accidentally | Triple-check: `train_test_split` must NOT be used â€” temporal split only | All 3 repos + paper used random split (all wrong) |
| SP-SP routes dominate training â†’ underweight N/NE risk | Consider stratified sampling by geographic cluster | Repo 3: SP = 80% of orders |
| Scaler fitted on all data | `scaler.fit(X_train)` ONLY, then `.transform(X_val)` and `.transform(X_test)` | Repo 3 made this error |

---

## Quick Reference: What Each Repo/Paper Contributed

| Source | Key Borrows | Key Warnings |
|---|---|---|
| **Repo 1** (Late_Shipment_Prediction) | Target buffer trick, threshold sweep, SMOTE>ADASYN, counterfactual predict_proba, LR coefficient unscaling | Random split (leakage), `order_approved_to_delivered_to_carrier` as feature (post-purchase leakage), haversine in for-loop (slow) |
| **Repo 2** (Olist-business-analysis) | Centroid geolocation, geodesic distance, Spearman correlation, GBM>RF, 5 high-risk N/NE clusters, payment installments as rural proxy, category translation merge | Random split, payment installments summed (use max), Euclidean for clustering (ok for clustering, not for analysis) |
| **Repo 3** (olists-delivery-analysis) | 3-stage delivery decomposition, `not_defined` payment drop, `day_time()` bucketing, `is_end_of_month`, seller_monthly_volume, OLS statsmodels summary, month exclusion list | Random split, MinMaxScaler on all data (leakage), get_dummies for 27 states (high dim) |
| **Paper** (Mangla & Soni 2025) | HistGradientBoosting as 5th model, Soft Voting Ensemble, Cost-Weighted MetaVote (adapted), `price_per_km` interaction, academic citation | Post-delivery features as inputs (leakage), unspecified balancing method, random split, no SHAP analysis |
