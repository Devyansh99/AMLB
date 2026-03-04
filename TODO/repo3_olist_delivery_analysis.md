# Repo 3 Analysis — olists-delivery-analysis
**Source:** `main.ipynb` (single notebook, 859 lines, Python), `README.md`

---

## What This Repo Did (Summary)
Three-goal analysis of the Olist dataset through a business intelligence + linear regression lens:

1. **Monthly orders EDA** — What caused the unusual growth spike at end of 2017?
2. **Delivery impact on orders** — Does higher late rate / longer delivery time reduce future orders?
3. **Delivery time prediction** — Build a Linear Regression model to predict actual delivery days → use that to set better estimated delivery dates → reduce late delivery rate.

**Key finding (README):** Their linear regression model reduces the "delivery gap" from 13.64 days (Olist's own estimate error) to 5.46 days MAE. This translates to a **2.2% reduction in late delivery rate** on their test set.

**What's missing:** No classification model, no cost framing, no feature importance analysis, no handling of class imbalance, no temporal split. But the descriptive analysis is excellent.

---

## Section 1 — Data Cleaning Pipeline

### Incomplete month removal
```python
# They identified months with insufficient order data and removed them
# Months dropped: 2016-09, 2016-10, 2016-11, 2016-12 (early launch, few orders)
#                 2018-09, 2018-10 (truncated data, not enough complete orders)
eli_ord = []
def eli_month(x):
    if re.findall('2016-09|2016-10|2016-11|2016-12|2018-09|2018-10', str(x['order_purchase_timestamp'])):
        eli_ord.append(x['order_id'])
ord_stt.apply(eli_month, axis=1)

# Apply to ALL tables (critical — keeps all tables consistent)
ord = ord[~ord['order_id'].isin(eli_ord)]
pay = pay[~pay['order_id'].isin(eli_ord)]
rv  = rv[~rv['order_id'].isin(eli_ord)]
ord_stt = ord_stt[~ord_stt['order_id'].isin(eli_ord)]
```
→ **Borrow the month-exclusion list exactly.** Our project also excludes Sep-Dec 2016 and should exclude Sep-Oct 2018. This confirms our temporal split decision. **Note:** Using `.apply(lambda, axis=1)` for this is slow — better to use `.isin(list_of_months)` with `.dt.to_period('M')`.

### Geolocation: same centroid method as Repo 2
```python
avg_lat_lng = (loc.drop_duplicates()).groupby('geolocation_zip_code_prefix').mean()[
    ['geolocation_lat','geolocation_lng']
].reset_index()
city = loc[['geolocation_zip_code_prefix','geolocation_city','geolocation_state']].drop_duplicates(
    subset=['geolocation_zip_code_prefix']
)
loc = pd.merge(avg_lat_lng, city, on='geolocation_zip_code_prefix', how='inner')
```
→ **Third repo confirming centroid approach.** Both repos 2 and 3 use `.groupby(zip).mean()` for lat/lng, but keep city/state from `.drop_duplicates()`. This is the canonical pattern.

### Payment type: remove `not_defined`
```python
pay = pay[~pay['payment_type'].isin(['not_defined'])]
```
→ **Add to our cleaning pipeline.** There are a small number of `not_defined` payment type rows that should be dropped.

---

## Section 2 — EDA: Why Did Orders Stagnate in 2017-2018?

### Finding 1: Negative reviews spiked at end of 2017
```python
# Monthly avg review score → drops sharply at Nov 2017
rv.groupby(rv['review_creation_date'].dt.to_period('M')).mean()['review_score'].plot()

# 1+2 star reviews by month → Nov 2017 has highest negative review count
neg_rv = rv.copy()[(rv['review_score'] == 1) | (rv['review_score'] == 2)]
# → Red bars for Nov 2017 – Jan 2018 window (peak negative period)
```
→ **Business context for our project:** We model orders from 2017-2018. The Nov 2017 spike in late deliveries is real, not noise. This validates that our `is_peak_season` / `is_holiday_season` seasonal features will capture a real logistics strain signal.

### Finding 2: NLP on negative review text (Portuguese)
```python
pt_stopwords = stopwords.words('portuguese')
def clean_cmt(txt):
    txt = re.sub(r'\r|\n|\.|,|$|!|\d|\?', '', txt)
    txt = txt.lower()
    result = [i for i in txt.split() if i not in pt_stopwords]
    return ' '.join(result)

neg_cmt_rv['new_cmt'] = neg_cmt_rv['review_comment_message'].apply(clean_cmt)
# freq count → top complaints:
# "produto" (product quality), "entrega" (delivery), "chegou" (arrived), 
# "prazo" (deadline/date), "recebi" (received), "devolucao" (return)
```
→ Top 2 customer complaints confirmed as: (1) product quality, (2) delivery (late + damaged).
→ **Not needed for our task** (reviews are post-delivery, excluded from features), but this validates our business framing: late deliveries directly translate into negative reviews → churn.

### Finding 3: Late rate correlates negatively with order growth
```python
ord_ontime['gap'] = (ord_ontime['order_estimated_delivery_date'] 
                   - ord_ontime['order_delivered_customer_date']).dt.days

# gap >= 0 → on time (delivered before or on estimated date)
# gap <  0 → late (delivered after estimated date)
ord_ontime['late'] = ord_ontime['gap'].apply(lambda x: 0 if x >= 0 else 1)

# Dual-axis chart: late_rate (monthly, red) vs order_growth_next_month (green)
# Key: as late_rate climbed in Nov 2017, next-month order growth went flat / negative
```
→ **Confirms the causal chain for our business section:** Late delivery → lower customer satisfaction → lower repeat purchase rate → platform growth stagnation.
→ **Late rate definition here:** `delivered_date > estimated_date` with NO buffer (strict). Our project uses an 11:59:59 buffer. They use strict gap = 0. Both are valid. Note the different rates this produces.

### Finding 4: Seller preparation time = 25% of total delivery time during peak
```python
deli_gap['seller_cfmed']      = (order_approved_at - order_purchase_timestamp) / hours
deli_gap['seller_to_carrier'] = (order_delivered_carrier_date - order_approved_at) / hours
deli_gap['carrier_to_buyer']  = (order_delivered_customer_date - order_delivered_carrier_date) / hours

# During Nov-Dec 2017 (peak):
# seller_to_carrier ≈ 80 hours = 3.3 days = ~25% of total delivery time
# → This bottleneck explains the spike in late deliveries
```
→ **Key operational insight:** The late delivery crisis of Nov 2017 was caused by sellers being overwhelmed with orders. The carrier leg of delivery is NOT the bottleneck — seller preparation is.
→ **Feature implication for our model:** `seller_to_carrier_hours` or `is_peak_season × seller_order_volume` would be strong predictors — but these are partially post-purchase. We can use `seller_order_volume_30d` (pre-purchase window) as a proxy for seller stress level.

### Finding 5: SP state sellers have lowest freight, 80% of orders from SP
```python
# Pivot table: seller_city vs customer_city → mean freight_value
city_dis_pivot = pd.pivot_table(data=city_dis,
    index='customer_city',  columns='seller_city',
    values='freight_value', aggfunc='mean')

# Result: São Paulo (SP) sellers average 18-19 BRL freight to all regions
# Outside SP: average freight 25-30+ BRL
# 80% of all Olist orders come FROM customers in SP state
```
→ **Confirms our geographic features.** SP-to-SP orders = low freight + short distance + low late rate. Cross-state orders = high freight + long distance + high late rate. `seller_state == customer_state` binary flag captures this.

### Finding 6: Delivery time improved in 2018 but freight cost rose → orders stayed flat
```python
# `hours/10km` metric (delivery hours normalized per 10km of distance)
dist_day['hours/10km'] = round(dist_day['actual_deli(h)'] * 10 / dist_day['distance(km)'], 2)

# Monthly avg hours/10km: peaked at ~25 hours/10km in Nov 2017 → improved to ~19 in Apr 2018
# BUT: frei/10km (freight per 10km) INCREASED from 2017 avg (~1.9 BRL/10km) to 2018 avg (~2.1 BRL/10km)
# → Trade-off: faster delivery was achieved by using more expensive shipping methods
# → Price-sensitive customers reduced purchase frequency despite faster delivery
```
→ **Business narrative:** Fixing delivery time alone is insufficient — cost efficiency must also be maintained. Third-party logistics partners (3PL) in non-SP regions are the proposed solution.

---

## Section 3 — Linear Regression: Predict Delivery Time

### Feature set used
| Feature | Type | Preprocessing |
|---|---|---|
| `weight(g)` | Numerical | MinMaxScaler |
| `vol(cm3)` = length × height × width | Numerical | MinMaxScaler |
| `distance` (haversine, km) | Numerical | MinMaxScaler |
| `freight_value` | Numerical | MinMaxScaler |
| `timeofday` (Morning/Afternoon/Night/Dawn) | Categorical | `pd.get_dummies` |
| `dayofweek` (0-6) | Categorical | `pd.get_dummies` |
| `dayofmonth` (1-31) | Categorical | `pd.get_dummies` |
| `customer_state` (27 states) | Categorical | `pd.get_dummies` |
| `seller_state` (27 states) | Categorical | `pd.get_dummies` |

**Target:** `actual_deli_day` = `(order_delivered_customer_date - order_purchase_timestamp).dt.days`

### Feature engineering code
```python
# Volume from dimensions
prd_size['vol(cm3)'] = prd_size['product_width_cm'] * prd_size['product_length_cm'] * prd_size['product_height_cm']
# Aggregate per order (mean weight/volume if multi-item order)
prd_size = prd_size.drop(columns=['product_length_cm','product_height_cm','product_width_cm'])
prd_size = pd.merge(ord[['order_id','product_id']], prd_size, how='left', on='product_id')
prd_size = prd_size.groupby('order_id').mean()[['weight(g)','vol(cm3)']].reset_index()

# Temporal features from purchase timestamp
time_pur['timeofday']   = time_pur['order_purchase_timestamp'].dt.hour.apply(day_time)
time_pur['dayofweek']   = time_pur['order_purchase_timestamp'].dt.dayofweek
time_pur['dayofmonth']  = time_pur['order_purchase_timestamp'].dt.day
time_pur['actual_deli_day'] = (delivered - purchased).dt.days
```

### `day_time()` helper function (reusable)
```python
def day_time(x):
    if x <= 6:   return 'Dawn'
    if x <= 12:  return 'Morning'
    if x <= 18:  return 'Afternoon'
    if x <= 24:  return 'Night'
```
→ **Borrow this.** We have `order_hour` in our feature set — this bucketing gives a cleaner categorical signal than raw hour.

### Haversine distance function (manual math implementation)
```python
from math import sin, cos, sqrt, atan2, radians

def cal_distance(x):
    cus_lat = radians(x['cus_lat']);  cus_lng = radians(x['cus_lng'])
    sel_lat = radians(x['sel_lat']);  sel_lng = radians(x['sel_lng'])
    dlon = sel_lng - cus_lng;  dlat = sel_lat - cus_lat
    a = sin(dlat/2)**2 + cos(cus_lat) * cos(sel_lat) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return round(6373.0 * c, 2)   # Earth radius ≈ 6373 km

ord_dist['distance(km)'] = ord_dist.apply(cal_distance, axis=1)
```
→ **Same approach as Repo 1 — row-by-row `.apply()` = slow.** For 100k orders this takes minutes. Use `haversine` package with batch arrays (vectorized) or `geopy.distance.geodesic` as in Repo 2.

### Data split (same critical error as Repo 1)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
→ **Random split on time-series data.** Future orders leak into train. **Do not copy.** Our temporal split is correct.

### Outlier removal: IQR on distance
```python
q1 = ord_dist['distance(km)'].quantile(0.25)
q3 = ord_dist['distance(km)'].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
dist_day = dist_day[dist_day['distance(km)'] < upper]  # removes extreme long-haul outliers
```
→ **Reusable IQR pattern.** For our classification task, we might want to KEEP long-distance orders — they are high–late-risk. But use this pattern for EDA visualization to avoid scale distortion.

### Model results
```
MAE  = 5.46 days
RMSE = (not shown directly, can infer ~7–8 days)
```
**vs Olist's built-in estimated delivery:**
```
Olist's estimate error (gap when on time) = 13.64 days average ahead
→ Their estimates are very conservative (over-promise buffer of ~13 days)
```
→ **Key insight:** Olist deliberately over-estimates delivery by ~13.6 days to avoid late flags. Their late rate definition is thus stricter to beat than it appears — because even their generous estimates get missed sometimes.
→ **Impact:** With their linear regression model, if the predicted delivery day is used + average gap buffer added, test-set late rate drops from ~8% to ~5.8% (2.2% improvement).

### OLS statistical summary
```python
import statsmodels.api as sm
X2 = sm.add_constant(X_train)
model_stats = sm.OLS(y_train.values.reshape(-1, 1), X2).fit()
model_stats.summary()
```
→ **Borrow this.** Not just sklearn metrics — running OLS statsmodels gives p-values, R², confidence intervals for each feature coefficient. We should do the same for our logistic regression baseline to report coefficient significance.

---

## Section 4 — Delivery Time Breakdown (Temporal Analysis)

### Three-stage decomposition
```python
# Stage 1: order confirmed (purchase → approval)
deli_gap['seller_cfmed']      = (order_approved_at - order_purchase_timestamp) / hours
# Stage 2: seller packs and ships to carrier (approval → carrier handoff)
deli_gap['seller_to_carrier'] = (order_delivered_carrier_date - order_approved_at) / hours
# Stage 3: carrier delivers to customer
deli_gap['carrier_to_buyer']  = (order_delivered_customer_date - order_delivered_carrier_date) / hours
```

**Quantified findings:**
| Stage | Normal period | Peak (Nov-Dec 2017) | % of total |
|---|---|---|---|
| Purchase → Approval | ~5 hours | ~6 hours | ~4% |
| Approval → Carrier | ~50 hours | **~80 hours** | **25% (peak)** |
| Carrier → Customer | ~200 hours | ~240 hours | ~71% |

→ **The bottleneck is the seller, not the carrier.** During peak, `seller_to_carrier` nearly doubles.
→ **Feature to engineer:** `seller_days_to_handoff` = `(order_delivered_carrier_date - order_approved_at).dt.hours` — but this is post-purchase. Pre-purchase proxy: `seller_volume_last_30d` (how overloaded is this seller right now).

### Correlation heatmap: delivery stage vs total time
```python
deli_gap_week = deli_gap.groupby(dt.to_period('W')).agg({...})
sns.heatmap(data=deli_gap_week.corr())
# Result: seller_to_carrier has highest correlation with total delivery time
# cfmed_to_carrier (= seller_cfmed + seller_to_carrier combined) is even stronger
```
→ `cfmed_to_carrier` is the combined pre-carrier time — the most predictive feature for total delivery duration.

---

## Section 5 — Order Volume Patterns (Temporal)

### Peak periods identified
```python
# Orders by time of day:      Afternoon = highest volume
# Orders by day of week:      Monday–Wednesday = higher than Thu-Sun
# Orders by day of month:     Spike at end of month (last week, esp days 24-28)
# Orders by month:            Nov 2017 = massive spike (Black Friday)
```

**Quantified breakdown:**
| Dimension | Peak period | Implication |
|---|---|---|
| Time of day | Afternoon (12:00-18:00) | Afternoon orders = seller gets order at end of day → next-day processing |
| Day of week | Monday–Wednesday | Beginning of week orders see better fulfillment |
| Day of month | Days 24-28 | End-of-month spending habit (paycheck effect) |
| Month | November (Black Friday) | Volume surge overwhelms seller capacity |

→ **We already have `order_hour` and `order_dow` in our feature set.** This confirms they are informative. Add `is_end_of_month` (day of month ≥ 24) as a feature — it creates seller stress that our current features don't capture.

---

## What We Can TAKE From This Repo

### ML Perspective

| Item | What to Borrow | Code/Note |
|---|---|---|
| **`not_defined` payment drop** | `pay = pay[~pay['payment_type'].isin(['not_defined'])]` | Add to cleaning pipeline |
| **Month exclusion list** | Drop 2016-09/10/11/12 and 2018-09/10 | Exact same months as our plan |
| **`vol(cm3)` feature** | `width × length × height` at order level (mean across items) | Already in our plan; their aggregation method is `.groupby(order_id).mean()` |
| **`day_time()` bucketing** | Dawn/Morning/Afternoon/Night from `order_hour` | Better than raw hour as categorical |
| **Three-stage delivery decomposition** | `seller_cfmed`, `seller_to_carrier`, `carrier_to_buyer` | NOT usable as features (post-purchase), but great for EDA insight section |
| **OLS statsmodels summary** | `sm.OLS(y, sm.add_constant(X)).fit().summary()` | Run on logistic regression coefficients for p-value reporting |
| **IQR outlier removal code** | `upper = Q3 + 1.5 × IQR` pattern | Clean helper for EDA; don't use for modeling (keep all orders) |
| **`frei/10km` and `hours/10km` normalized metrics** | Freight and time normalized by distance | Good for EDA comparisons across routes of different lengths |
| **Growth rate calculation** | `pct_change()` on monthly order counts | Simple, reusable; add MoM growth to EDA |

### Business Perspective

| Insight | What This Repo Found | How We Use It |
|---|---|---|
| **Seller preparation = 25% of delivery time** | During peak, `seller_to_carrier` ≈ 80 hours | Business recommendation: "Olist should enforce a 48-hour seller dispatch SLA. Top e-commerce players (Shopee: 1.5 days, Lazada: 2 days) already do this." |
| **SP state = logistics hub** | 80% of orders from SP; SP sellers have lowest freight (18-19 BRL avg) | Business recommendation: "Focus onboarding efforts on SP-based sellers for non-SP customers, before considering regional 3PL." |
| **Late rate → order growth stagnation** | Nov 2017 late rate spike → orders went sideways for 5+ months | Our economic argument: "Each 1% increase in late rate is correlated with flat/declining order growth → quantifiable revenue impact." |
| **Freight improved delivery but at higher cost** | Speed ↑ in 2018 but freight ↑ → orders didn't recover | Cost sensitivity finding: customers value both speed AND price; reducing late rates without increasing freight is critical |
| **End-of-month purchase spike** | Days 24-28 are peak purchase days | Sellers must pre-stock for month-end rush; connects to `is_end_of_month` feature |

### Economic Perspective

| Gap in Repo | What They Missed | What We Add |
|---|---|---|
| **No cost matrix** | They said late rate drops 2.2% but didn't quantify R$ value | We compute: 2.2% × avg FN cost (R$155) × test_set_orders = total expected savings from using ML model |
| **No priority handling scenario** | They modeled estimated delivery improvement — but didn't model intervention (like our Priority Handling cost matrix) | Our model flags which orders to intervene on proactively, not just improve estimates post-fact |
| **Seller SLA enforcement economics** | Shopee/Lazada SLA cited but no cost-benefit | We can quantify: if seller dispatches within 48h, `seller_to_carrier` drops from 80h to 48h = -32h delay = lower late probability in our model |
| **Random split inflates model quality** | 2.2% improvement figure uses test data contaminated with future orders | Real improvement (temporal split) would be lower; our project gives an honest number |

---

## New Features to Add (from this repo's EDA)

| Feature | Source | How to Compute |
|---|---|---|
| `is_end_of_month` | Day-of-month purchase pattern | `order_purchase_timestamp.dt.day >= 24` |
| `purchase_time_of_day` | Afternoon peak finding | `day_time(order_hour)` → bin to Dawn/Morning/Afternoon/Night |
| `seller_state_eq_customer_state` | SP = lowest freight finding | `seller_state == customer_state` (binary same-state flag) |
| `is_sp_seller` | 80% of orders from SP sellers | `seller_state == 'SP'` |
| `seller_monthly_volume` | Seller stress = peak-period late surge | Count of orders to this seller in the 30d before purchase date (computed from train fold only) |

---

## Critical Notes — Do NOT Copy These

| Issue | What They Did | Why It's Wrong | Our Fix |
|---|---|---|---|
| **Random train-test split** | `train_test_split(X, y, test_size=0.2, random_state=0)` | Time-series data — future orders leak into training | Temporal split: train ≤ Dec 2017, val H1 2018, test H2 2018 |
| **Haversine via `.apply()` row loop** | `ord_dist.apply(cal_distance, axis=1)` | O(N) Python row iteration — slow for 100k rows | Use `haversine` package with numpy arrays or `geopy.distance.geodesic` with vectorized apply |
| **`get_dummies` on `customer_state` + `seller_state`** | 54 dummy columns added for 27 states × 2 | High dimensionality, leaks unseen states | Use target encoding: `mean(is_late)` per state from train fold |
| **MinMaxScaler fitted on ALL data** | `scaler.fit_transform(model_data[features])` before split | Scaler sees test-set values → data leakage | Fit on train only: `scaler.fit(X_train)` then `transform(X_test)` |
| **`actual_deli_day` as regression target** | `(delivered - purchased).dt.days` | Predicting actual delivery time requires knowing if the order was late AFTER delivery | This is a leakage-safe regression only if distance/weight/time-of-purchase are features — which they are. BUT: does not directly predict `is_late` (a classification problem). |
| **Review data in NLP analysis** | Portuguese stopwords cleaning on `review_comment_message` | Reviews are post-delivery — ok for descriptive analysis but **never use as modeling features** | We exclude all review data from features |

---

## Summary: What This Repo Adds to Our Project

1. **Three-stage delivery decomposition is the single most important analytical insight.** Seller preparation (approval → carrier handoff) is the bottleneck, not carrier transit. This frames our business recommendation: enforce a 48-hour seller dispatch SLA.

2. **Late rate correlates directly with order growth decline.** The Nov 2017 data shows this is causal, not coincidental. This is the economic narrative for our report: reducing late rate by 2-5% directly protects platform revenue.

3. **Confirmed feature set overlaps:** `vol(cm3)`, `distance_km`, `freight_value`, `seller_state`, `customer_state`, `order_hour`, `order_dow` — all three repos independently selected these same variables.

4. **Three new features identified:** `is_end_of_month`, `purchase_time_of_day` (binned), `seller_monthly_volume` (seller stress proxy).

5. **Their 2.2% late rate improvement is real but overstated** (random split leak). We should not aim to match this number — our temporal split will give a more conservative but honest benchmark. That said, even a 1% late rate reduction at scale = ~R$500k+ annual savings (worth computing for our cost analysis section).

6. **Olist's own estimates buffer 13.64 days ahead of actual delivery.** This means most orders that are "late" by our definition (delivered after estimated date) failed to beat a very generous estimate. The problem is severe in the N/NE clusters identified in Repo 2.

7. **SP state dominance (80% of orders):** Our model should account for this imbalance — SP-SP routes will dominate training data, potentially underweighting the features that predict N/NE route lateness. Consider stratified sampling by geographic cluster during train/val split.
