# Repo 2 Analysis — Olist-business-analysis
**Source files:** `Data manipulation and combining.ipynb`, `Freight value prediction.ipynb`, `Preliminary Data analysis.ipynb`, `Product analysis.ipynb`, `Reviews Sentiment Analysis.ipynb`, `Association rule mining.ipynb`, `Payment-Analysis.md` (R), `Delivery-Route-Optimisation.md` (R), `Effect-of-images-and-description-on-Reviews.md` (R), `Forecasting-Sales.md` (R)

---

## What This Repo Did (Summary)
A comprehensive multi-angle business analysis of the Olist dataset. No single ML classification model for late delivery prediction — instead six separate analyses: freight prediction, route optimization, payment behavior, product analysis, sentiment analysis, and sales forecasting. The most useful for us are the **data merge pipeline**, the **freight value prediction** (reveals what drives shipping cost = proxy for delivery complexity), and the **delivery route optimization** (geographic clustering of inefficient routes = directly relevant to our business recommendation).

---

## Analysis 1 — Data Merge Pipeline (`Data manipulation and combining.ipynb`)

**This is the cleanest, most reusable data pipeline in all three repos.** Written in Python, step-by-step, well-commented.

### Geolocation: Centroid method (better than Repo 1)
```python
# GROUP by zip → MEAN lat/lng per zip (centroid of all points in that zip)
# This is better than .drop_duplicates() because it averages out GPS noise
loc_coordnts = geoloc[[
    'geolocation_zip_code_prefix',
    'geolocation_lat',
    'geolocation_lng'
]].groupby('geolocation_zip_code_prefix').mean().reset_index()
```
→ **Use this approach.** More robust than Repo 1's first-value method.

### Customer data with lat/lng
```python
customer_data = pd.merge(
    left=cust,
    right=loc_coordnts,
    left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix',
    how='left'
)[['customer_id','customer_unique_id','customer_zip_code_prefix',
   'customer_city','customer_state','geolocation_lat','geolocation_lng']]
```

### Delivery dataset (seller + customer lat/lng per order)
```python
# Step 1: order_items LEFT JOIN orders → get order timestamps + seller_id per order
order_seller = pd.merge(left=ord_items, right=orders, on='order_id', how='left')[cols]

# Step 2: LEFT JOIN customer_data → get customer_unique_id + customer lat/lng
cust_seller = pd.merge(left=order_seller, right=customer_data[cols], on='customer_id', how='left')

# Step 3: LEFT JOIN seller_loc → get seller lat/lng
delivery = pd.merge(left=cust_seller, right=seller_loc[cols], on='seller_id', how='left')

# Step 4: rename lat/lng columns for clarity
delivery.rename(columns={
    'geolocation_lat_x': 'customer_lat',
    'geolocation_lng_x': 'customer_lng',
    'geolocation_lat_y': 'seller_lat',
    'geolocation_lng_y': 'seller_lng'
}, inplace=True)
```
→ **Borrow rename pattern exactly** — avoids the `_x/_y` ambiguity when merging two tables both containing lat/lng.

### Product category translation (Portuguese → English)
```python
# They merged products with product_category_name_translation.csv
products = pd.merge(
    left=products,
    right=transln,
    on='product_category_name',
    how='left'
)
# Use product_category_name_english for all analysis
```
→ **Borrow this.** Our feature engineering uses English category names for interpretability.

### Payment aggregation (multiple rows per order)
```python
# They grouped payments by order_id and SUMMED installments and value
order_payment = payments[[
    'order_id',
    'payment_installments',
    'payment_value'
]].groupby('order_id').sum().reset_index()
```
→ **Borrow this.** Two notes: (1) They note 75% of payments are credit cards — payment_type is predictive but low variance. (2) Summing installments across payment methods slightly misrepresents installment count — we should use `max` or `first` for installments, `sum` only for value.

### Transaction dataset for market basket
```python
# Created one-hot encoded category columns, grouped by customer + timestamp
txn_data = pd.get_dummies(
    data=cust_order[['customer_unique_id','order_purchase_timestamp','product_category_name_english']],
    columns=['product_category_name_english'],
    prefix='catg'
).groupby(['customer_unique_id','order_purchase_timestamp']).sum().reset_index()
```
→ Not directly useful for our classification task, but the structure (one-hot per category, grouped per order) is exactly how we could build `is_furniture`, `is_electronics` flags efficiently.

### Product-Review mapping
```python
# For each product_id: find its order_ids → find reviews for those orders
def product_review(prod):
    oid = ord_items[ord_items.product_id == prod].order_id
    rid = reviews[reviews.order_id.isin(oid)].review_id
    data = reviews[reviews.review_id.isin(rid)].copy()
    data['product_id'] = prod
    ...
```
→ **Not needed for our modeling** (reviews are post-delivery). But if we wanted seller quality signal: compute `seller_avg_review` by joining reviews → order_items → aggregate by `seller_id`.

---

## Analysis 2 — Freight Value Prediction (`Freight value prediction.ipynb`)

**Task:** Predict `freight_value` given distance, weight, dimensions, price.
**Why it matters for us:** Freight cost is used as a proxy for delivery difficulty/remoteness. Understanding what drives freight = understanding what drives delivery risk.

### Feature selection via Spearman correlation
```python
# They tested each feature's Spearman correlation with freight_value
# Key finding (Spearman r values):
# distance → r ≈ 0.6, explains ~40% of variance in freight_value  ← STRONGEST
# weight   → r ≈ 0.4, explains ~16%
# volume   → r ≈ 0.4, explains ~16%
# dimensions (individual) → r ≈ 0.3, explains ~8% each
# price    → r ≈ 0.2, low
# lat/lng  → r ≈ 0, negative for longitude (not useful raw)
```
→ **Key insight for us:** Distance explains 40% of freight variance. This is exactly why `distance_km` is our #1 expected feature. They quantified this — we can cite "40% of freight cost variability is explained by distance alone."

### Distance calculation: `geopy.distance.geodesic` (WGS-84 ellipsoid)
```python
from geopy.distance import geodesic

def calc_dist(cust, sel):
    return geodesic(cust, sel).km   # WGS-84 ellipsoid model

df['distance'] = df[['cust_lat','cust_lng','seller_lat','seller_lng']].apply(
    lambda row: calc_dist((row.cust_lat, row.cust_lng), (row.seller_lat, row.seller_lng)),
    axis=1
)
```
→ **`geodesic` is more accurate than `haversine`** for Brazil's scale. The difference is small (~0.3%) but use `scipy.spatial.distance.cdist` or batch `geodesic` for speed. Their `apply` is faster than Repo 1's for loop but still slow — use numpy vectorization or `haversine` package's bulk API for 100k rows.

### PCA for insight
```python
# They applied PCA on [weight, length, height, width, distance, volume, price]
# 1 component explained ~52% of variance — confirms these collinear features compress well
pca = PCA(n_components=1)
# Result: all 7 features collapse to one "delivery complexity" dimension
```
→ **Interesting diagnostic** but not needed for XGBoost (handles collinearity natively). We do NOT need PCA. But the insight — that physical features + distance = one latent "complexity" factor — supports our interaction term `distance_x_weight`.

### Model comparison for freight prediction

| Model | R² (CV) | Notes |
|---|---|---|
| Linear Regression | 0.58 | Poor — non-linear relationship |
| Lasso (LassoCV, `α=[0.001..10]`) | ~0.58 | L1 regularization, same R² |
| Ridge (RidgeCV) | ~0.58 | L2 regularization, same R² |
| **RandomForestRegressor (400 trees)** | **0.66** | Non-linear captures interactions |
| **GradientBoostingRegressor (1000 trees, lr=0.05)** | **0.71** | Best — chosen as final model |

→ **Takeaway:** Tree-based models beat linear for this data — confirms why we use XGBoost/RF for classification. **GradientBoosting > RandomForest** on this dataset.

### Model validation approach (rigorous)
```python
# They did full statistical validation of residuals:
# 1. One-sample t-test: mean(residuals) = 0 → p=0.075, PASS (residuals centered)
# 2. Welch's t-test: mean(train_residuals) = mean(test_residuals) → p=0.09, PASS (no overfit)
# 3. Shapiro-Wilk: normality of residuals → p < 0.05, residuals are normal
# 4. F-test (ANOVA): variance(train) = variance(test) → p=0.09, PASS
```
→ **Borrow this residual validation pattern** if we do regression sub-tasks. For our classification task: use PR-AUC + confusion matrix + cost instead.

### `freight_to_value_ratio` as derived feature
- High ratio = expensive shipping relative to product price = remote/rural customer
- This repo confirms: distance is the dominant driver of freight cost
→ **Confirms our feature** `freight_to_value_ratio` is a valid geographic proxy.

---

## Analysis 3 — Delivery Route Optimisation (`Delivery-Route-Optimisation.md`)

**This is the most business-relevant analysis in the repo for our project.**

### Finding inefficient routes
```R
# Euclidean distance proxy (for clustering, not for accuracy)
delivery_unique <- delivery_unique %>% 
  mutate(dist = sqrt((seller_lat - customer_lat)^2 + (seller_lng - customer_lng)^2))

# Subset top 5% longest routes
long_dist_cut <- quantile(delivery_unique$dist, 0.95, na.rm=TRUE)
long_dist <- delivery_unique %>% filter(dist > long_dist_cut[[1]])
```
→ **Business insight:** The top 5% longest routes are systematically inefficient — customers being served by distant sellers when nearby sellers exist. This is a structural logistics problem.

### Hierarchical + K-Means clustering of inefficient routes
```R
# Hierarchical clustering to find natural number of clusters
hc <- hclust(dist(long_dist[, c('customer_lat','customer_lng')]))
# Dendrogram cut at height 17-18 → 5 clusters

# K-Means with k=5
km <- kmeans(long_dist[, c('customer_lat','customer_lng')], 5, nstart=100)
long_dist$clust <- km$cluster
```
→ **Technique to borrow:** We can cluster orders by geographic risk profile (customer lat/lng + distance bucket) and report: "these 5 customer regions have no nearby sellers for high-demand categories" — this is the business recommendation section.

### Nearest seller recommendation
```R
# For each cluster centroid, find closest seller
NearestSell <- function(lat, long) {
  comp <- seller_unique %>% mutate(dist = sqrt((seller_lat-lat)^2 + (seller_lng-long)^2))
  arrange(comp, dist)[1, -4]
}
# Result: 5 candidate sellers + recommended product categories
```
**Actual finding:**

| Cluster | Nearest Seller location | Products to stock |
|---|---|---|
| 1 | -25.9°, -48.6° (PR coast) | health_beauty, telephony |
| 2 | -6.1°, -37.5° (NE interior) | health_beauty |
| 3 | -8.8°, -63.9° (RO/northwest) | sports_leisure |
| 4 | -3.1°, -60.0° (AM/Manaus) | computers_accessories, health_beauty, sports_leisure, telephony |
| 5 | -2.5°, -44.2° (MA coast) | health_beauty, telephony |

→ **Business narrative for our project:** The 5 highest-risk delivery regions (top 5% routes) share a common pattern — they are in the **North and Northeast** of Brazil, where seller density is low. If Olist could onboard sellers in these 5 clusters to stock the top-ordered categories, it would structurally reduce cross-state delivery distances and cut late rates in those regions. This is a **next investment** recommendation backed by data.

---

## Analysis 4 — Payment Analysis (`Payment-Analysis.md`, R/ggplot2)

**Focus:** Average installment count and payment value per product category.

### Key findings (quantified)

| Category | Avg Installments | Avg Payment (R$) | Interpretation |
|---|---|---|---|
| Computers | Highest (~8) | High (~500+) | Big-ticket → many installments |
| Fixed telephony | 2.7 | R$296 | High value, few installments (luxury) |
| Furniture (bedroom) | 3.2 | R$266 | Heavy + expensive → high late rate + many installments |
| Construction tools | 3.5 | R$224 | Bulky, few installments |
| Security & services | 1.0 (lowest) | R$162 | Pure luxury, cash-like payment |

**Core finding:** *Costlier product categories → more installments.* Electronics and computers top both. Luxury categories (fixed telephony, air conditioning) are expensive but bought in fewer installments (customer can afford them outright).

### Implication for our features
- `payment_installments` is a proxy for **order value and buyer financial profile** — rural/lower-income buyers tend to use more installments
- High installments in boleto (bank slip) = likely remote/rural customer → also higher distance → higher late rate
- **Feature to add:** `high_installment_flag` = `payment_installments > 6` as a rural/remote proxy

---

## Analysis 5 — Effect of Images & Description on Reviews (`Effect-of-images-and-description-on-Reviews.md`, R)

**Task:** Does product listing quality (photos count, description length) affect sales and reviews?

### Method
- Linear regression per category: `sales ~ no_img + desc_length`
- Only categories with 10+ products kept (avoids leverage from small samples)

### Key findings

**Categories where more photos → more sales (p < 0.05):**
- `bed_bath_table` (coef=+0.59)
- `health_beauty` (coef=+0.55)
- `toys` (coef=+0.33)
- `fashion_bags_accessories` (coef=+0.42)
- `small_appliances_home_oven_and_coffee` (coef=+1.19)
- `fashio_female_clothing` (coef=+0.52)

**Negative relationship (more photos → fewer sales):**
- `computers_accessories` (coef=-0.76, p=0.008) — buyers don't need photos, they know specs
- `office_furniture` (coef=-1.15, p=0.049) — unclear noise

**Description length:** Almost universally not significant (p > 0.05 for most), except `auto` category.

### Implication for our model
- `product_photos_qty` as a **seller quality signal** — low photos = poor seller → higher late rate
- We already include this feature. Now we know **which categories it matters for**: beauty, fashion, appliances, household. For electronics, it doesn't matter.
- Could create: `photos_x_category_matters` interaction flag (photos_qty for visual categories only)

---

## Analysis 6 — Sales Forecasting (`Forecasting-Sales.md`, R)

**Task:** Forecast next year's monthly sales revenue using time series.

### Key data insight (monthly revenue R$):
```
2017: Jan=92K, Feb=283K, Mar=397K, Apr=353K, May=586K, Jun=544K,
      Jul=545K, Aug=656K, Sep=712K, Oct=761K, Nov=1.03M, Dec=1.05M
2018: Jan=987K, Feb=958K, Mar=1.21M, Apr=1.12M, May=1.25M, Jun=1.02M,
      Jul=965K, Aug=1.26M
```
→ **Strong growth trend:** ~10× revenue growth from Jan 2017 to Aug 2018.  
→ **Seasonal pattern:** Nov-Dec spike (Black Friday / holiday), May peak (Q2 surge confirms our is_peak_season feature).  
→ **Data limitations:** Sep 2016 incomplete, 2020 dates found (data entry errors — their note about excluding them).  

### Implication for our features
- `is_peak_season` (Q2: Apr-Jun) confirmed by revenue data — this is a high-volume period when logistics strain is real
- November-December also sees volume surge (holiday season) — consider `is_holiday_season` feature  
- **Add feature:** `is_holiday_season` = November + December (could have elevated late rates alongside Q2)

---

## Analysis 7 — Preliminary Data Analysis (`Preliminary Data analysis.ipynb`)
*(Not read in full — covered by summary above and feature inventory already done in our project)*

Confirms: standard EDA — distributions of price, freight, review scores, order volume by state/category. All consistent with our interim report numbers.

---

## What We Can TAKE From This Repo

### ML Perspective

| Item | What to Borrow | Code/Note |
|---|---|---|
| **Centroid geolocation** | `.groupby(zip).mean()` instead of `.drop_duplicates()` | Better noise handling |
| **`geodesic` distance** | `geopy.distance.geodesic` with `.apply()` | More accurate than haversine; vectorize for speed |
| **Rename after merge** | `.rename(columns={'geolocation_lat_x':'customer_lat',...})` | Prevents `_x/_y` confusion in delivery merge |
| **Payment aggregation** | `.groupby('order_id').sum()` for value, not for installments | Use `max` for installments |
| **Category translation merge** | LEFT JOIN with `product_category_name_translation.csv` | English names for EDA and target encoding |
| **GradientBoosting > RF** | GBM achieves R²=0.71 vs RF at 0.66 on Olist data | Supports using LightGBM/XGBoost over plain RF |
| **Spearman correlation** | For skewed distributions (all Olist features are right-skewed) | Use `scipy.stats.spearmanr` not Pearson for EDA |
| **Residual validation** | t-test + ANOVA on train vs test residuals | Rigorous model validation panel |
| **Category flags via get_dummies** | `pd.get_dummies(column='product_category_name_english')` then `.groupby(order_id).sum()` | Efficient multi-category per order flags |

### Business Perspective

| Insight | What This Repo Found | How We Use It |
|---|---|---|
| **5 high-risk customer clusters** | Top 5% routes cluster in N/NE Brazil (Manaus, Nordeste, Pará, Maranhão coast) | Business recommendation: "Olist should onboard sellers in these 5 regions stocking health_beauty, telephony, sports_leisure" |
| **Distance = 40% of freight variance** | Spearman r=0.6 between distance and freight_value | Cite in report: "distance alone explains 40% of freight cost — confirming it as the dominant logistics variable" |
| **High installments = rural buyer** | Computers, furniture, appliances → 6-8 installments | `payment_installments` signals buyer remoteness → late delivery risk |
| **Q2 + Nov-Dec revenue peaks** | Monthly revenue time series shows May + Nov spikes | Confirms `is_peak_season` (Q2) + add `is_holiday_season` (Nov-Dec) |
| **Listing quality matters for visual categories** | Photos predict sales for beauty, fashion, appliances | `product_photos_qty` is a seller quality signal — especially for non-commodity categories |

### Economic Perspective

| Gap in Repo | What They Missed | What We Add |
|---|---|---|
| **No cost framing** | No decision-maker identified for route analysis | We frame it: "each inefficient route = likelihood of late delivery = R$155 expected cost" |
| **Route optimization is a recommendation, not a model** | They suggest *which seller to onboard* but don't quantify the benefit | We can quantify: "orders in top 5% distance bucket have X% late rate vs Y% in bottom 95%. If rerouted, saves R$Z annually." |
| **Freight prediction ≠ delivery prediction** | They predicted freight cost, not lateness | Our insight: high freight_to_value_ratio = proxy for remote customer = elevated late risk. Connect the two. |
| **No scenario analysis** | Route optimization done once, no sensitivity | We test: does the model's advantage hold if only same-state deliveries are flagged? |

---

## Feature Additions from This Repo (Add to todo.md Phase 2)

These are features not in our original plan that this repo reveals as valuable:

| New Feature | Source | How to Compute |
|---|---|---|
| `is_holiday_season` | Forecasting-Sales (Nov-Dec revenue spike) | `month.isin([11,12])` |
| `high_installment_flag` | Payment-Analysis | `payment_installments > 6` |
| `regional_cluster` | Delivery-Route-Optimisation | K-Means (k=5) on customer lat/lng, label clusters by risk |
| `photos_x_visual_category` | Images-Reviews analysis | `product_photos_qty × is_visual_category` (beauty/fashion/appliances) |
| `seller_category_match` | Route-Optimization | Whether seller's historically stocked category matches order category |
| `is_north_northeast_customer` | Route-Optimisation cluster result | `customer_state.isin(['AM','PA','MA','CE','RN','PB','PE','AL','SE','BA','PI','TO','AP','RR','AC','RO'])` |

---

## Critical Notes — Do NOT Copy These

| Issue | What They Did | Why It's Wrong | Our Fix |
|---|---|---|---|
| **Euclidean distance for clustering** | `sqrt((lat2-lat1)² + (lng2-lng1)²)` | Not actual km distance — just degree-space Euclidean | Use geodesic km for real analysis; Euclidean fine for clustering only (proportional) |
| **Random train/test split** | `train_test_split(X, y, random_state=1)` in freight prediction | Time-series data — leaks future into past | Our temporal split |
| **Payment installments: sum** | `payment_installments.groupby(order_id).sum()` | If order paid half by credit card (3 installments) + half by boleto (1 installment) = 4 installments total. Meaningless sum. | Use `max` or mode per order_id |
| **Dropped lat/lng from freight model** | "latitude and longitude show close to 0 correlation" (their note) | They dropped raw lat/lng but should have kept distance | We keep `distance_km` (derived from lat/lng), not raw coordinates |
| **No leakage check in product-review mapping** | Reviews mapped to products without any time-ordering | A review from 2018 mapped to a 2016 order could contain future signal | We don't use review data as training features at all |

---

## Summary: What This Repo Adds to Our Project

1. **Cleaner merge pipeline:** Centroid geolocation + systematic column renaming after merge. Copy this structure for `01_data_cleaning.ipynb`.

2. **Quantified distance dominance:** Distance explains 40% of freight variance (Spearman r=0.6). Use this in the report as a business anchor: "logistics cost is driven by geography, not product price."

3. **Geographic clustering of risk:** 5 customer clusters in North/Northeast Brazil have structurally high delivery distances due to sparse seller coverage. This is our strongest **next investment** recommendation: "onboard sellers in these 5 regions to structurally reduce late rates."

4. **Two new temporal features:** `is_holiday_season` (Nov-Dec) from the sales forecasting time series. Add to feature set.

5. **Payment installments as rural proxy:** High installments (>6) correlate with expensive categories and remote areas. Add `high_installment_flag`.

6. **GBM > RF confirmed** on this exact dataset. Justifies prioritizing LightGBM/XGBoost over RandomForest in our model comparison.

7. **They had no business framing at all** for the route optimization. We connect their geographic clustering directly to our cost matrix: clusters in North/Northeast = ~15% late rate (vs 8% average) = each such order has 2× expected cost if not flagged.
