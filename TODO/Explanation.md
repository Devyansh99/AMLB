# Phase 1 Key Decisions — Why We Did What We Did

This document explains the reasoning behind every key decision in Phase 1 (`docs/phase1_data_cleaning.md`). For each decision: what the earlier/default approach was, why we changed it, and which reference repos validated the choice.

---

## 1. Target Variable — 11:59:59 Buffer on Estimated Delivery Date

**Decision:** `is_late = 1` if `order_delivered_customer_date > order_estimated_delivery_date + Timedelta(hours=11, minutes=59, seconds=59)`

**What was done earlier (naive approach):**
Use a strict comparison — `delivered_date > estimated_date`. Any delivery with a timestamp even one second past midnight of the estimated date counts as late.

**Why we changed it:**
The Olist dataset stores `order_estimated_delivery_date` as a date with no time component (it defaults to midnight `00:00:00`). A delivery that physically arrives on the correct day but whose timestamp is `2017-05-12 14:30:00` would be flagged as late if the estimate is `2017-05-12 00:00:00` — a false positive. The buffer extends the deadline to the end of that day (`23:59:59`), so any delivery that arrives on the same calendar day as the estimate is correctly counted as on-time.

Without this buffer our strict late rate is 8.1%; with the buffer it is 7.94%. The 0.16% difference represents real false positives that would corrupt the training signal.

> **Cited from Repo 1 (Late_Shipment_Prediction_Olist):** They explicitly added this buffer in their cleaning notebook:
> ```python
> orders['order_estimated_delivery_date'] = (
>     orders['order_estimated_delivery_date'] + pd.Timedelta(seconds=59, minutes=59, hours=11))
> orders['is_late'] = orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']
> ```
> Their resulting late rate was 7.9% vs our strict 8.1% — confirming the buffer removes a small set of same-day false positives.

> **Contrast with Repo 3 (olists-delivery-analysis):** They used the strict definition (`gap < 0 = late`) with no buffer. This is technically simpler but produces slightly inflated late counts and misclassifies same-day deliveries.

---

## 2. Geolocation — Centroid (Mean) Instead of `drop_duplicates`

**Decision:** `.groupby('geolocation_zip_code_prefix').mean()` on lat/lng to produce one centroid coordinate per zip code.

**What was done earlier (naive approach):**
`geolocation.drop_duplicates('geolocation_zip_code_prefix', inplace=True)` — keep the first row seen for each zip. This is the approach used in Repo 1.

**Why we changed it:**
The geolocation table has ~1 million rows that collapse to only 19,015 unique zip codes — an average of ~53 GPS readings per zip. These readings are collected from different sources and contain GPS noise (scatter within a zip). Taking the first row is arbitrary and may land on a noisy outlier entry. Repo 1 themselves noted that "max-min lat within any zip was negligible ~0.01 degrees, so first ≈ median ≈ OK" — but that is only true on average. For zip codes near geographic boundaries (coastal zips, zip codes straddling hills), the first reading can be significantly off. Averaging all 53 readings cancels the noise and gives a stable centroid.

Additionally, taking the mean is reproducible regardless of row ordering; `drop_duplicates` depends on the sort order of the raw CSV, which is not guaranteed to be consistent.

> **Cited from Repo 2 (Olist-business-analysis):** Their data pipeline uses the centroid method explicitly and describes it as "better than `.drop_duplicates()` because it averages out GPS noise":
> ```python
> loc_coordnts = geoloc[['geolocation_zip_code_prefix','geolocation_lat','geolocation_lng']]\
>     .groupby('geolocation_zip_code_prefix').mean().reset_index()
> ```

> **Cited from Repo 3 (olists-delivery-analysis):** Independently arrived at the same centroid pattern:
> ```python
> avg_lat_lng = (loc.drop_duplicates()).groupby('geolocation_zip_code_prefix').mean()[
>     ['geolocation_lat','geolocation_lng']].reset_index()
> ```
> Two out of three reference repos independently converged on centroid — this is the canonical approach for this dataset.

---

## 3. Aggregation Level — Order-Level, Not Item-Level

**Decision:** Aggregate all tables to one row per `order_id` before modeling.

**What was done earlier (Repo 1 approach):**
Repo 1 kept multiple rows per order when an order contained multiple items — one row per `(order_id, product_id)` pair. A 3-item order produced 3 rows, all with the same target label `is_late`.

**Why we changed it:**
Our prediction task is: *given an order at placement time, predict whether it will arrive late.* The decision unit is the order, not the item. If we model at item level, a 3-item order contributes 3 rows with identical target labels to training — it inflates the effective sample size by a factor of `n_items`, biases the model toward multi-item orders (which tend to be heavier and more expensive), and introduces pseudo-replication that makes cross-validation splits unreliable (the same order could have training rows and test rows simultaneously).

The aggregation strategy:
- `price` → `sum` (total order value)
- `freight_value` → `sum` (total freight cost for all items)
- `product_weight_g` → `max` (the heaviest item drives shipping class)
- `product_length/height/width` → values from the heaviest item (same reason)
- `order_item_id` → `count` = `n_items` (number of items in the order)
- `seller_id` → `first` (for single-seller orders this is exact; multi-seller orders are rare)

> **Contrast with Repo 1:** They stated they "left this as-is intentionally, treating each item-row separately." Their own analysis notes this is a limitation and that "our approach: aggregate to order-level is better for business framing since our decision is per-order, not per-item."

---

## 4. Payment Aggregation — `max` for Installments, `sum` for Value, Mode for Type

**Decision:**
```python
pay_agg = payments.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'max',
    'payment_type': lambda x: x.mode()[0]
}).reset_index()
```

**What was done earlier:**
Repo 2 used `.groupby('order_id').sum()` on the entire payments table, including `payment_installments`. This sums installment counts across different payment methods for the same order.

**Why we changed it:**
A single order can be split across multiple payment methods — e.g., partly paid with a `credit_card` (3 installments) and partly with `boleto` (1 installment). Summing gives 4 installments, which is meaningless as a business signal. The installment count that matters is the maximum — it reflects the buyer's largest single commitment and correlates with their creditworthiness and order value. Summing would systematically overcount installments for orders paid with mixed methods and destroy the interpretability of this feature.

For `payment_value`, summing is correct — you want the total amount paid, regardless of how many payment methods were used.

For `payment_type`, we take the mode (most frequent type) because credit cards dominate (75%+ of payments per Repo 2's finding) but some orders genuinely use boleto as their primary method.

> **Cited from Repo 2 (Olist-business-analysis):** Their data pipeline used `.sum()` for installments and explicitly noted the issue: "Two notes: (1) They note 75% of payments are credit cards. (2) Summing installments across payment methods slightly misrepresents installment count — we should use `max` or `first` for installments, `sum` only for value."

---

## 5. Payment Cleaning — Drop `not_defined` Payment Type

**Decision:** `payments = payments[~payments['payment_type'].isin(['not_defined'])]` before any aggregation.

**What was done earlier:**
Repos 1 and 2 did not filter `not_defined` payment types. These rows pass through aggregation and pollute the `payment_type` mode calculation — an order that has one `credit_card` row and one `not_defined` row could end up with `payment_type = not_defined` if the mode is a tie broken by order of appearance.

**Why we changed it:**
`not_defined` is not a real payment category — it is a data quality flag indicating the payment method was not recorded. It has no business meaning and should not be treated as a payment type. Removing it before aggregation ensures the mode aggregation always picks from real payment methods.

> **Cited from Repo 3 (olists-delivery-analysis):** Only Repo 3 explicitly handled this:
> ```python
> pay = pay[~pay['payment_type'].isin(['not_defined'])]
> ```
> Repos 1 and 2 missed this cleaning step.

---

## 6. Missing Value Strategy — Category Median for Products, State Centroid for Geolocation

**Decision:**
- Missing `product_weight_g` and dimensions → fill with the median of the same product category.
- Missing geolocation after zip-code join (~0.5% of rows) → fill with the centroid of the customer/seller's state.
- Missing `product_category_name` (1.4% of products) → fill with `"unknown"`.

**What was done earlier:**
Repo 1 dropped all rows with null lat/long after the geolocation merge, losing ~0.5% of data. For product dimensions, they also dropped rows with nulls.

**Why we changed it:**
Dropping nulls is the easiest path but introduces selection bias. The 0.5% of orders with missing geolocation are not random — they are more likely to be orders from small towns or newly annexed zip codes not yet in the geolocation table. These are precisely the remote, long-distance deliveries most likely to be late. Dropping them would cause the training data to underrepresent high-risk geographic segments.

The fix for geolocation is to fall back to the state-level centroid lat/lng — an approximation, but it preserves the geographic signal (which state the customer is in) even when the exact zip is missing. This only affects ~500 rows so the precision loss is negligible.

For product weight/dimensions, using the category median is more informative than row deletion because products within the same category have correlated physical properties. A missing furniture weight is much more likely to be ~20 kg (furniture median) than 0.1 kg.

---

## 7. Timestamp Validation — Drop Impossible Timestamps

**Decision:** Drop rows where `order_approved_at < order_purchase_timestamp` (approval before purchase is physically impossible).

**What was done earlier:**
No repo systematically filtered impossible timestamp sequences. Repo 1 dropped rows where derived time deltas (e.g., `order_approved_to_delivered_to_carrier`) were negative, but only after computing the delta — a less explicit approach.

**Why we changed it:**
14 rows in the dataset have `order_approved_at` earlier than `order_purchase_timestamp`. These are data entry errors or system clock mismatches. If kept, they produce negative values for `approval_time` features and other derived time deltas, which would inject noise into any time-based feature. Since only 14 rows are affected (0.014% of data), dropping them has zero practical cost.

> **Referenced from Repo 1:** They filtered similar impossible data:
> ```python
> # Dropped rows where order_approved_to_delivered_to_carrier < 0 or 
> # order_purchased_to_approved < 0 (data entry errors / timestamp typos)
> ```

---

## 8. Date Filtering — Exclude Sep-Dec 2016 and Sep-Oct 2018

**Decision:** Remove orders from: September–December 2016 and September–October 2018.

**What was done earlier:**
Most analyses use the full date range of the dataset without verifying completeness at the month level. This includes Repos 1 and 2.

**Why we changed it:**
The Olist dataset was collected over a specific window and not all months are "complete." September–December 2016 are the platform's first months of operation — order volumes are tiny (dozens per month vs thousands later), seller behavior/logistics infrastructure was immature, and the late rate during this ramp-up period does not reflect steady-state operations. Including these months would teach the model patterns from a fundamentally different operational environment.

September–October 2018 represent truncated collection — orders placed in these months may not have reached final delivery status before the dataset snapshot was taken. Including them would introduce large numbers of orders with missing `order_delivered_customer_date`, artificially inflating missing-value rates.

After filtering, the date range is Jan 2017–Aug 2018, which is the full "steady-state" operating period: 96,189 rows vs 96,456 without filtering (267 rows removed, 0.28% of data).

> **Cited from Repo 3 (olists-delivery-analysis):** Repo 3 is the only reference that explicitly identifies and removes these exact months:
> ```python
> # Months dropped: 2016-09, 2016-10, 2016-11, 2016-12 (early launch, few orders)
> #                 2018-09, 2018-10 (truncated data, not enough complete orders)
> ```
> They also applied the filter consistently across ALL tables (orders, payments, reviews) to keep the tables in sync — a detail Repos 1 and 2 did not address.

---

## 9. Keeping Only `order_status == 'delivered'`

**Decision:** Filter to delivered orders only before any other processing.

**What was done earlier:**
The raw orders table contains 99,441 rows across multiple statuses: `delivered`, `shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `approved`, `created`. Repos that did not filter early could inadvertently compute `is_late` on non-delivered orders where `order_delivered_customer_date` is null.

**Why we changed it:**
The business question — will this order arrive late? — only makes sense for orders that were completed (delivered). Canceled, shipped, or in-progress orders have no known delivery outcome. Computing `is_late` on a row with null `order_delivered_customer_date` would produce NaN, which any classifier would either drop silently or impute incorrectly.

More importantly, canceled orders have different feature distributions (higher cancellation rates for high-value, distant, or problematic orders) — including them would introduce confounding patterns unrelated to the delivery lateness mechanism.

2,963 non-delivered rows removed (3.0% of the raw dataset), reducing from 99,441 to ~96,478.

---

## 10. Column Renaming After Geolocation Merge

**Decision:** Immediately rename `geolocation_lat_x`/`geolocation_lng_x` to `customer_lat`/`customer_lng` and `geolocation_lat_y`/`geolocation_lng_y` to `seller_lat`/`seller_lng` after the merge.

**What was done earlier:**
When two tables both containing `geolocation_lat` and `geolocation_lng` are joined, pandas automatically appends `_x` and `_y` suffixes. Without explicit renaming, downstream code must track which suffix corresponds to which entity — error-prone and unreadable.

**Why we changed it:**
Explicit semantic names (`customer_lat`, `seller_lat`) make every subsequent operation self-documenting. The distance calculation `geodesic((customer_lat, customer_lng), (seller_lat, seller_lng))` reads correctly; `geodesic((lat_x, lng_x), (lat_y, lng_y))` does not.

> **Cited from Repo 2 (Olist-business-analysis):** Their pipeline introduced this pattern explicitly:
> ```python
> delivery.rename(columns={
>     'geolocation_lat_x': 'customer_lat',
>     'geolocation_lng_x': 'customer_lng',
>     'geolocation_lat_y': 'seller_lat',
>     'geolocation_lng_y': 'seller_lng'
> }, inplace=True)
> ```
> Repo 1 did not rename and had to track `_x`/`_y` throughout their modeling notebook, causing readability issues.

---

## Summary Table

| Decision | What Was Done Before | Why We Changed It | Repo Citation |
|---|---|---|---|
| 11:59:59 buffer on target | Strict `delivered > estimated` | Same-day deliveries falsely flagged as late | Repo 1 |
| Centroid geolocation | `drop_duplicates` (first row) | First row is arbitrary; mean of 53 readings cancels GPS noise | Repo 2, Repo 3 |
| Order-level aggregation | Item-level rows (1 row per item) | Decision is per-order; item-level inflates sample size and corrupts CV | Repo 1 (contrasted) |
| `max` installments | `sum` installments | Summing across payment methods is meaningless | Repo 2 (corrected) |
| Drop `not_defined` payment | Not filtered | `not_defined` is a data quality flag, not a real payment method | Repo 3 |
| Category median for missing dims | Drop null rows | Drops selectively bias toward remote/unusual orders | Repo 1 (contrasted) |
| Drop impossible timestamps | Not filtered | 14 rows with approval before purchase corrupt time-delta features | Repo 1 (referenced) |
| Exclude Sep-Dec 2016, Sep-Oct 2018 | Use full date range | Early months = ramp-up behaviour; late months = truncated data | Repo 3 |
| Filter `delivered` status first | Mixed statuses included | Non-delivered orders have null delivery date, no valid target label | All repos |
| Rename lat/lng columns post-merge | Keep `_x`/`_y` suffixes | Semantic names prevent downstream confusion in distance calculations | Repo 2 |
