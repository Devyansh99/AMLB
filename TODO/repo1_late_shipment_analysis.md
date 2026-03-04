# Repo 1 Analysis — Late_Shipment_Prediction_Olist
**Source files:** `Project_03_Data_Cleaning.ipynb` + `Project_03_Modeling.ipynb` + `README.md`

---

## What This Repo Did (Summary)
Predicted late Olist shipments using only pre-shipment data. Best model: **Random Forest** at **53% recall, 19% precision** with a custom FBeta(β=2) scorer. Used SMOTE for class imbalance, haversine distance as primary feature. No business/economic framing — pure ML exercise.

---

## Full Methodology Walkthrough

### Data Pipeline (Data_Cleaning notebook)

**Database:** Used PostgreSQL (`psycopg2`) to query tables. We're using CSVs directly — no change needed, same logic applies.

**Geolocation handling:**
```python
# They took the FIRST lat/long per zip (drop_duplicates — fast and simple)
geolocation.drop_duplicates('geolocation_zip_code_prefix', inplace=True)
# Checked that max-min lat within any zip was negligible ~0.01 degrees, so first = median ≈ OK
```
→ **Our approach:** use `median lat/lon per zip` which is slightly more robust but their approach is defensible if we verify the spread.

**Merge sequence they used:**
```
geolocation → customers → orders   (customer side)
geolocation → sellers → order_items → products   (seller/product side)
Final: orders_customers LEFT JOIN order_items_products_sellers ON order_id
```
Result: **kept multiple rows per order** when an order had multiple items (duplicated order_id rows). They left this as-is intentionally, treating each item-row separately.

→ **Our approach:** aggregate to order-level (sum price, sum freight, max weight, count items). Better for business framing since our decision is per-order, not per-item.

**Target variable definition — important detail:**
```python
# Added 11:59:59 buffer to estimated delivery date
# So that deliveries arriving on the SAME DAY as estimate are NOT counted as late
orders['order_estimated_delivery_date'] = (
    orders['order_estimated_delivery_date'] + pd.Timedelta(seconds=59, minutes=59, hours=11))

orders['is_late'] = orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']
```
→ **Borrow this.** Prevents edge-case false positives where delivery day = estimate day (timestamp < midnight = "late" without the buffer). Their late rate: **7.9%** (we get 8.1% with our method — slight difference from this buffer).

**Feature engineering in cleaning:**

| Feature | Code | Notes |
|---|---|---|
| `product_vol_cm` | `length × height × width` | Derived from products table |
| `dist_km` | Haversine in a **for loop** | Very slow on 100k rows — see fix below |
| `order_purchased_to_approved` | `(approved_at - purchase_timestamp).dt.days` | ✅ Valid pre-shipment feature |
| `order_approved_to_delivered_to_carrier` | `(carrier_date - approved_at).dt.days` | ⚠️ See leakage warning below |

**Data cleaning decisions:**
- Dropped rows where `order_approved_to_delivered_to_carrier < 0` or `order_purchased_to_approved < 0` (data entry errors / timestamp typos in the dataset)
- Dropped `product_category_name` entirely (kept numeric features only) — they wanted clean numeric-only input for scaling
- Dropped rows with null lat/long after merge (lost ~0.5% of data)
- Final df: pickled as `df.pkl` for modeling notebook

---

### Modeling (Modeling notebook)

**Features used (final set):**
- `dist_km` — haversine distance seller→customer
- `price`, `freight_value`
- `product_weight_g`, `product_vol_cm`, `product_length_cm`, `product_height_cm`, `product_width_cm`
- `order_item_id` — proxy for multi-item orders (higher ID = more items in cart)
- `order_purchased_to_approved` — approval delay in days
- `order_approved_to_delivered_to_carrier` — fulfillment speed in days

**Dropped from X:** all ID columns, all raw timestamps, both zip codes, seller/customer lat/long (replaced by dist_km), `is_late`, both raw delivery date columns.

**Train/test split:**
```python
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,
                                                     stratify=df_y,
                                                     random_state=37)
# Default 75/25 split — RANDOM, NOT temporal
```
→ **We will NOT do this.** Random split on time-series data leaks future information into training. We use temporal split.

**Scaling:**
```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# Fit on train only, transform test — correct!
```

**Imbalance handling:**
```python
x_train_smoted, y_train_smoted = SMOTE(random_state=37).fit_sample(x_train_scaled, y_train)
# KNN used SMOTE'd data; RF used original data with class_weight='balanced' — inconsistent
```
→ They tried both SMOTE and ADASYN — **SMOTE outperformed ADASYN** (their finding). Keep SMOTE as the primary oversampling strategy.

**Models trained:**

| Model | Config | Scoring |
|---|---|---|
| **Logistic Regression** | `class_weight='balanced'`, `n_jobs=-1` | fit on scaled non-SMOTE data |
| **KNN** | GridSearch `n_neighbors=[1..9]`, `cv=2`, `scoring='roc_auc'` | fit on SMOTE'd scaled data |
| **Random Forest** | GridSearch `n_estimators=[1,4,9,16,...,81]`, `max_depth=[9,16,25,36,49]`, `cv=2`, `scoring='roc_auc'`, `class_weight='balanced'` | fit on original (non-SMOTE) data |
| **RF (reduced features)** | Same but dropped dimensions, kept only volume | marginal difference |

**Threshold optimization (RF):**
```python
y_probs = rf_best.predict_proba(x_test)[:,1]
precisions = []
thresholds = []
for t in range(0, 100, 5):   # sweep 0.0 to 1.0 in 0.05 steps
    y_pred = pd.Series(y_probs).apply(lambda x: 0 if x < t/100 else 1)
    precisions.append(fbeta_score(y_test, y_pred, beta=2))
    thresholds.append(t/100)

best_threshold = thresholds[np.argmax(precisions)]  # Found: threshold = 0.10
```
→ **Borrow this pattern.** We replace `fbeta_score(beta=2)` with our economic cost function `FP×15 + FN×155` as the optimization target.

**Final results:**
```
Logistic Regression:  Recall=?, Precision=?, FBeta≈lower
KNN:                  Recall=higher, Precision=lower (SMOTE helps recall but hurts precision)
Random Forest:        Recall=53%, Precision=19%, FBeta(β=2)=best
```
RF ROC-AUC was clearly the best on the ROC curve.

**Key finding — distance:**
```python
# Late deliveries are ~27% farther on average
late_dist_mean    = 740.5 km
ontime_dist_mean  = 584.4 km
delta             = (740.5 - 584.4) / 584.4 = +26.7%
```
→ **Confirms our hypothesis.** Distance is the strongest univariate predictor.

**Logistic regression coefficients (interpretability):**
```python
# They unscaled the coefficients for interpretation
readjusted_coefs = logistic_model.coef_[0] / scaler.scale_

# Finding: customer/seller latitude and longitude were top coefficients
# order_item_id had -2.6% coefficient (each additional item slightly reduces on-time probability)
# Surprising: weight and dimensions had low coefficients
```
→ **Borrow the coefficient unscaling technique.** We can apply this to our logistic baseline for the report.

**Counterfactual analysis (interesting technique):**
```python
# "What if we set order_approved_to_delivered_to_carrier = 0?"
sample = x_test.sample(1000, random_state=37)
pred_1 = rf.predict_proba(sample)[:,1]

sample['order_approved_to_delivered_to_carrier'] = 0   # Hypothetically perfect fulfillment speed
pred_2 = rf.predict_proba(sample)[:,1]

mean_pred_delta = np.mean(pred_1 - pred_2)
# Result: 2.3% lower predicted late probability if fulfillment was instant
```
→ **Borrow this for business section.** We can do: "what if this seller had average performance instead of poor performance?" — quantify SHAP-style using predict_proba deltas.

**ROC curve comparison (all models on same plot):**
```python
fig, ax = plt.subplots(figsize=(8,8))
plt.plot(knn_fpr, knn_tpr, label='K-Nearest Neighbors')
plt.plot(logit_fpr, logit_tpr, label='Logistic Regression')
plt.plot(rf_fpr, rf_tpr, 'g', label='Random Forest')
plt.plot([0,1], 'r--', alpha=0.5)  # diagonal baseline
```
→ **Borrow this exact plot structure** for our model comparison.

**Confusion matrix function (labeled TP/TN/FP/FN):**
```python
def confusion_matrix_graphic(prediction, real_y=y_test):
    conf_matrix = metrics.confusion_matrix(real_y, prediction)
    ax = sns.heatmap(conf_matrix, cmap=plt.cm.Blues, annot=True, fmt='g',
                     xticklabels=['Not late', 'Late'],
                     yticklabels=['Not late', 'Late'])
    cmlabels = ['True Negatives','False Positives','False Negatives','True Positives']
    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
```
→ **Borrow and extend** — add R$ cost annotation to each cell (FP=R$15, FN=R$155, TP=+R$130, TN=R$5).

---

## What We Can TAKE From This Repo

### ML Perspective

| Item | What to Borrow | Why |
|---|---|---|
| **Target definition buffer** | Add `Timedelta(hours=11, min=59, sec=59)` to estimated date | Prevents false late labels on same-day deliveries |
| **Threshold sweep pattern** | `for t in range(0,100,5): fbeta_score(...)` | Replace FBeta with our cost function |
| **FBeta(β=2) scorer** | Recall-weighted scoring during GridSearch | Use as a secondary metric alongside our cost metric |
| **SMOTE > ADASYN** | Use SMOTE, skip ADASYN | Their empirical comparison saves us one experiment |
| **ROC overlay plot** | All models on one figure | Clear model comparison visual for report |
| **Labeled confusion matrix** | TP/TN/FP/FN annotation function | Extend with R$ cost labels |
| **Coefficient unscaling** | `coef_[0] / scaler.scale_` | Interpret logistic baseline in original units |
| **Counterfactual predict_proba** | Null out a feature, compare predicted probability | Business section: "what if seller had average performance?" |
| **RF feature importance DataFrame** | `pd.DataFrame(rf.feature_importances_, index=X.columns)` | Clean feature importance table |
| **GridSearch param grids** | `n_estimators=[i**2 for i in range(1,10)]` | Reasonable grid for RF |
| **`order_purchased_to_approved`** | Days from purchase to payment approval | Valid pre-shipment signal (slow approval → delay risk) |
| **`order_item_id` as feature** | Proxy for items-in-cart | They found each extra item = -2.6% on-time — use `n_items` instead |
| **Negative timedelta filter** | Drop rows where any time delta < 0 | Real data has timestamp errors — we should filter these |

### Business Perspective

| Insight | From This Repo | How We Use It |
|---|---|---|
| **Late = 7.9–8.1%** | Confirmed independently | Validates our class imbalance framing |
| **Distance = #1 predictor** | Late deliveries 27% farther | Frame as: "geography dominates over price" — key business insight |
| **Recall > Precision for this problem** | Model predicts late to warn customer → false alarm costs less than missed warning | Directly supports our R$15 (FP) vs R$155 (FN) asymmetry |
| **Flask app idea** | Sliders for distance, price, weight → real-time prediction | Mention as "next investment" — operational deployment via API |

### Economic Perspective

| Gap in Repo | What They Missed | What We Add |
|---|---|---|
| **No cost matrix** | FBeta(β=2) is arbitrary weighting | We have R$15 FP and R$155 FN from business reasoning |
| **No threshold justification** | Chose threshold=0.10 via FBeta maximization | We choose threshold by minimizing `Total Cost = FP×15 + FN×155` |
| **No sensitivity analysis** | One fixed metric | We test 4 scenarios (low churn, high churn, expensive ops) |
| **No R$ savings calculation** | Performance reported as recall/precision only | We convert to: "saves R$X per 10,000 orders vs doing nothing" |
| **No baseline comparison** | No "flag nothing" or "flag everything" benchmark | We compute: cost(model) vs cost(flag nothing) vs cost(flag all) |

---

## Critical Warnings — Do NOT Copy These

| Issue | What They Did | Why It's Wrong | Our Fix |
|---|---|---|---|
| **Temporal leakage** | Random stratified 75/25 split | Orders from 2016 in test, 2018 in train — model sees the future | Strict temporal split: train≤2017, val=H1 2018, test=H2 2018 |
| **`order_approved_to_delivered_to_carrier`** | Used as feature | Carrier handoff date is **post-purchase** — not known at order placement | Drop this feature entirely. Use `estimated_delivery_days` instead |
| **Haversine in a for loop** | `for i in range(len(df))` | ~20 minutes on 100k rows | Use vectorized: `from scipy.spatial.distance import cdist` or `haversine_distances` from sklearn, or apply with numpy arrays |
| **Scoring metric mismatch** | GridSearch used `roc_auc`, but final eval used FBeta | RF was tuned for AUC but reported on FBeta — best RF for AUC ≠ best RF for FBeta | Set GridSearch scoring = same metric you care about for final eval. For us: use `roc_auc` for comparison, but pick threshold separately via cost minimization |
| **SMOTE on KNN, not RF** | Inconsistent imbalance strategy | Can't fairly compare models if one got synthetic data and other didn't | Apply same strategy to all models OR explicitly test SMOTE vs class_weight consistently |
| **Dropped product_category_name** | Removed because it's a string | Loses a highly predictive feature (furniture=14.2% vs books=3.2%) | Target-encode the 73 categories using training fold late rates |
| **SMOTE before cross-validation** | Their explicit lesson learned | SMOTE on full train set bleeds into CV folds → inflated CV scores | Always SMOTE inside the fold: use `imblearn.pipeline.Pipeline` |

---

## Repo 1 Feature Set vs Ours

| Feature | Repo 1 Used It? | We Use It? | Notes |
|---|---|---|---|
| `dist_km` (haversine) | ✅ | ✅ | Their #1 feature — vectorize the calculation |
| `price` | ✅ | ✅ as `avg_item_price` | |
| `freight_value` | ✅ | ✅ as `total_freight_cost` | |
| `product_weight_g` | ✅ | ✅ | |
| `product_vol_cm3` | ✅ (derived) | ✅ | Same derivation |
| `product_dimensions` | ✅ (kept all 3) | ✅ → may drop if volume is enough | |
| `order_item_id` | ✅ as raw ID | ✅ as `n_items` (count) | Better semantics |
| `order_purchased_to_approved` | ✅ | ✅ | Valid signal |
| `order_approved_to_delivered_to_carrier` | ✅ (LEAKAGE) | ❌ | Post-purchase, not at order time |
| `product_category_name` | ❌ dropped | ✅ target-encoded | Big miss in their work |
| `same_state` | ❌ | ✅ | Geographic feature they skipped |
| `seller_late_rate` | ❌ | ✅ | High-value feature they missed |
| `is_peak_season` | ❌ | ✅ | Temporal feature they missed |
| `payment_type` | ❌ | ✅ | They ignored payments entirely |
| `estimated_delivery_days` | ❌ | ✅ | Time pressure signal |
| Customer/seller raw lat/long | ✅ dropped after haversine | ❌ | We only keep dist_km |
| `customer_state`, `seller_state` | ❌ | ✅ target-encoded | Geographic richness |

---

## Summary: Key Takeaways for Our Project

1. **Confirmed what works:** Distance is the dominant predictor — quantified at 27% mean difference. This validates our business narrative: "geography determines fate."

2. **Confirmed what to avoid:** Random split was their biggest technical error. Temporal split is non-negotiable for us.

3. **Threshold of 0.10** was their empirical optimum — we expect similar (aggressive threshold justified because FN >> FP cost).

4. **SMOTE beats ADASYN** on this dataset — skip testing ADASYN.

5. **FBeta(β=2)** maps to our economic framing — FN is ~10x more costly than FP, which corresponds to β ≈ √10 ≈ 3. Their β=2 is close but we can justify β=3 more precisely from our cost structure. Use as a secondary metric.

6. **Counterfactual predict_proba analysis** is an excellent "business impact" technique — set seller_late_rate to 0 (hypothetically perfect seller) and show how predicted probability drops. Report in the business section.

7. **They had no economic analysis at all.** This is the biggest gap. Our economic layer (cost matrix, sensitivity, R$ savings) is the entire differentiation.
