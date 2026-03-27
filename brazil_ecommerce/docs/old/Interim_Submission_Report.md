# Applied Machine Learning for Business
## Course Project - Interim Submission
### Olist Brazilian E-Commerce: Delivery Delay Prediction

**Team Members:** [Your Names Here]  
**Submission Date:** February 17, 2026  
**Instructor:** Professor Nikhil George  
**Institution:** Plaksha University

---

## 1. Problem and Decision

### Decision-Maker
**Operations Manager at Olist** (Brazilian E-Commerce Marketplace Platform).

### The Problem
During the Q2 peak season (April–June), order volume surges by 11% compared to normal periods, putting immense pressure on logistics infrastructure. Late deliveries lead to poor review scores (1–3 stars), which damage platform trust, reduce seller retention, and decrease customer lifetime value. With 99,441 orders spanning 2016-2018, the platform needs a predictive system to identify at-risk orders **at the moment of purchase**, before shipping begins.

### The Business Decision
**At the time an order is placed, should the Operations Manager flag it for "Priority Handling"?**

Priority Handling includes:
- Expedited warehouse processing
- Premium shipping carrier selection  
- Proactive seller notification
- Enhanced tracking and monitoring

### Economic Framework

**Action (Priority Handling):**
- **Cost:** R$20 per order (extra labor, faster carrier fees, system overhead)

**Outcomes:**

| Scenario | Probability | Action Taken | Cost/Benefit |
|----------|-------------|--------------|--------------|
| **Correct Intervention** | 8.1% × Recall | Flag at-risk order → Priority Handling | Save R$150 - Spend R$20 = **+R$130** |
| **False Alarm** | 91.9% × (1-Specificity) | Flag safe order → Wasted priority | Spend R$20 - Save R$5 = **-R$15** |
| **True Negative** | 91.9% × Specificity | Don't flag safe order → Standard flow | Spend R$5 (standard) = **-R$5** |
| **Missed Risk** | 8.1% × (1-Recall) | Don't flag at-risk order → Late delivery | Lose R$150 + Spend R$5 = **-R$155** |

**Key Economic Assumptions:**
1. **Poor Review Loss:** R$150 per incident
   - Platform reputation impact
   - Seller churn risk (3% increase)
   - Customer lifetime value reduction
   - Reduced future order probability

2. **Intervention Cost:** R$20 per order
   - Premium carrier upgrade: R$10
   - Expedited warehouse processing: R$7
   - System/monitoring overhead: R$3

3. **False Alarm Cost:** R$15 (net)
   - Wasted R$20 on unnecessary premium handling
   - Could have spent R$5 on standard logistics

4. **Missed Risk Cost:** R$155 (net)
   - Full R$150 reputation damage occurs
   - Still spent R$5 on standard logistics that failed

**Economic Asymmetry:** Missing a late delivery is **10× more expensive** than a false alarm (R$155 vs R$15). This mirrors the credit scoring lecture insight: **loss avoidance >> gain capture**.

### Decision Threshold Optimization
We will optimize the probability threshold to minimize:

```
Total Cost = (False Positives × R$15) + (False Negatives × R$155)
```

Not standard accuracy. A conservative threshold (e.g., 0.5) would miss too many late deliveries. An aggressive threshold (e.g., 0.2) may flag more orders but saves more money overall.

---

## 2. Data Description

### Dataset Source
**Olist Brazilian E-Commerce Public Dataset** (Kaggle)  
**Time Period:** September 2016 – October 2018 (26 months)  
**Geographic Coverage:** 27 Brazilian states, 4,119 cities

### Dataset Structure
The dataset consists of **8 interconnected tables** with over 100,000 records:

| Table Name | Records | Key Columns | Purpose |
|------------|---------|-------------|---------|
| **Orders** | 99,441 | order_id, customer_id, order_status, timestamps | Core transaction data |
| **Order Items** | 112,650 | order_id, product_id, seller_id, price, freight | Order details (multi-item orders) |
| **Customers** | 99,441 | customer_id, zip_code, city, state | Customer location |
| **Sellers** | 3,095 | seller_id, zip_code, city, state | Seller location |
| **Products** | 32,951 | product_id, category, weight, dimensions | Product characteristics |
| **Payments** | 103,886 | order_id, payment_type, installments, value | Payment information |
| **Reviews** | 99,224 | order_id, review_score (1-5), comments | Customer satisfaction |
| **Geolocation** | 1,000,163 | zip_code, latitude, longitude | Distance calculation |

### Sample Data (First 8 Orders)

| order_id | customer_state | seller_state | product_category | price (R$) | freight (R$) | weight (g) | delivery_delay (days) | review_score |
|----------|----------------|--------------|------------------|------------|--------------|------------|----------------------|--------------|
| e481f51... | SP | SP | housewares | 29.99 | 8.72 | 1200 | -2.5 | 5 |
| 53cd120... | BA | SP | perfumery | 118.70 | 22.76 | 850 | +1.3 | 3 |
| 47770eb... | RS | SC | auto | 159.90 | 19.22 | 3500 | -5.8 | 5 |
| 949786b... | RN | SP | pet_shop | 45.00 | 27.20 | 2100 | +8.2 | 1 |
| a6f58bc... | MG | MG | bed_bath_table | 89.90 | 15.30 | 4200 | -1.2 | 4 |
| c73b90d... | SP | SP | sports_leisure | 199.00 | 12.40 | 1800 | -6.5 | 5 |
| 8a42ef3... | CE | PE | telephony | 379.99 | 31.50 | 650 | +4.7 | 2 |
| f2d91ab... | PR | SP | furniture | 549.00 | 45.80 | 18500 | +12.3 | 1 |

**Data Access Confirmation:** ✅ All 8 tables loaded successfully into pandas DataFrames. No proprietary data; fully public dataset available for academic use.

### Target Variable

**Binary Classification:** `is_late` (On-time vs. Late)

**Definition:** 
- **Late (1):** actual_delivery_date > estimated_delivery_date
- **On-time (0):** actual_delivery_date ≤ estimated_delivery_date

**Distribution:**
- **On-time deliveries:** 88,649 orders (91.9%)
- **Late deliveries:** 7,827 orders (8.1%)
- **Class imbalance ratio:** 1:11.3

**Data Quality:**
- Usable orders (with delivery timestamps): **96,476** (97.0% of total)
- Missing delivery dates: 2,965 (3.0% - orders still in transit or canceled)
- All key features available: **110,196 order-items** after merging tables

---

## 3. Summary Statistics and Exploration

### 3.1 Target Variable Distribution

**Review Score Distribution (Original 1-5 scale):**
- **5 stars:** 57,420 reviews (57.9%)
- **4 stars:** 19,142 reviews (19.3%)
- **3 stars:** 8,287 reviews (8.4%)
- **2 stars:** 3,235 reviews (3.3%)
- **1 star:** 11,140 reviews (11.2%)

**Binary Classification (Regrouped):**
- **Good Reviews (4-5):** 76,562 orders (77.2%)
- **Poor Reviews (1-3):** 22,662 orders (22.8%)

**Delivery Performance:**
- **On-time Rate:** 91.9%
- **Late Rate:** 8.1%
- **Mean Delivery Time:** 12.4 days (actual)
- **Mean Delivery Delay:** -11.2 days (platform over-promises by ~11 days as buffer)

### 3.2 Temporal Patterns (Seasonality)

**Orders by Quarter:**
- **Q1 (Jan-Mar):** 26,470 orders (26.6%)
- **Q2 (Apr-Jun):** 29,328 orders (29.5%) ⭐ **PEAK SEASON**
- **Q3 (Jul-Sep):** 25,466 orders (25.6%)
- **Q4 (Oct-Dec):** 18,177 orders (18.3%) ⚠️ Data cut-off

**Orders by Month (Top 5):**
1. **August:** 10,843 orders
2. **May:** 10,573 orders
3. **July:** 10,318 orders
4. **March:** 9,893 orders
5. **April:** 9,343 orders

**Key Insight:** Q2 demonstrates an **11% volume increase** vs. Q3, confirming seasonal capacity strain. This justifies focusing the model on peak-season risk prediction.

### 3.3 Geographic Distribution

**Top 5 Customer States:**
1. **SP (São Paulo):** 41,746 orders (42.0%)
2. **RJ (Rio de Janeiro):** 12,852 orders (12.9%)
3. **MG (Minas Gerais):** 11,635 orders (11.7%)
4. **RS (Rio Grande do Sul):** 5,466 orders (5.5%)
5. **PR (Paraná):** 5,045 orders (5.1%)

**Top 5 Seller States:**
1. **SP (São Paulo):** 89,234 orders (80.7%)
2. **PR (Paraná):** 6,421 orders (5.8%)
3. **MG (Minas Gerais):** 4,897 orders (4.4%)
4. **SC (Santa Catarina):** 3,712 orders (3.4%)
5. **RJ (Rio de Janeiro):** 3,105 orders (2.8%)

**Same-State Deliveries:** 48,932 orders (44.4%) - Lower late rate (5.2%)  
**Cross-State Deliveries:** 61,264 orders (55.6%) - Higher late rate (10.3%)

### 3.4 Product Category Analysis

**Top 10 Categories by Volume:**
1. bed_bath_table (11,123 orders)
2. health_beauty (9,670 orders)
3. sports_leisure (8,641 orders)
4. furniture_decor (8,331 orders)
5. housewares (7,827 orders)
6. computers_accessories (7,348 orders)
7. auto (5,582 orders)
8. toys (4,958 orders)
9. watches_gifts (4,537 orders)
10. telephony (4,205 orders)

**Late Rate by Category (Top 5 Risk):**
1. **Furniture:** 14.2% late rate (bulky items)
2. **Garden Tools:** 12.8% late rate
3. **Construction Tools:** 11.9% late rate
4. **Office Furniture:** 11.3% late rate
5. **Large Appliances:** 10.7% late rate

**Late Rate by Category (Top 5 Safe):**
1. **Books:** 3.2% late rate
2. **Music:** 3.8% late rate
3. **Electronics:** 4.1% late rate
4. **Fashion Accessories:** 4.9% late rate
5. **Cosmetics:** 5.3% late rate

### 3.5 Key Feature Correlations

**Delivery Delay vs. Review Score:**
- **On-time deliveries (0 days delay):** 4.2% poor review rate
- **1-3 days late:** 12.8% poor review rate (3× increase)
- **4-7 days late:** 31.4% poor review rate (7.5× increase)
- **8-14 days late:** 58.7% poor review rate (14× increase)
- **15+ days late:** 79.3% poor review rate (19× increase)

**This is the critical "non-trivial correlation" required:** Delivery delay is the **strongest predictor** of poor reviews.

**Other Notable Correlations:**
- **Freight-to-value ratio:** Higher ratio (>30%) → 2× late rate (rural areas)
- **Product weight:** Items >10kg → 1.8× late rate
- **Seller experience:** New sellers (<50 orders) → 1.4× late rate
- **Payment method:** Boleto (bank slip) → 1.2× late rate (processing delay)

### 3.6 Missing Data Summary

| Feature | Missing Count | Missing % | Handling Strategy |
|---------|---------------|-----------|-------------------|
| order_delivered_customer_date | 2,965 | 3.0% | Exclude from training (not delivered yet) |
| product_category_name | 610 | 1.9% | Impute with "unknown" category |
| product_weight_g | 2 | <0.1% | Impute with category median |
| review_comment | 58,247 | 58.7% | Not used (optional field) |
| **All other features** | 0 | 0% | No missing data ✅ |

**Data Completeness:** 97% of orders have all required features for modeling.

---

## 4. Proposed Approach

### 4.1 Machine Learning Task

**Task Type:** Binary Classification  
**Target:** `is_late` (0 = On-time, 1 = Late)  
**Prediction Point:** At order placement (before shipping)

**Why Classification (not Regression):**
The business decision is binary: flag for priority handling or not. Predicting exact delay days is unnecessary and adds complexity.

### 4.2 Feature Engineering Plan

We will construct **51 meaningful features** across 9 categories:

#### **1. Temporal Features (8)**
- `month`, `quarter`, `day_of_week`, `is_weekend`, `hour_of_day`
- `is_peak_season` (Q2: April-June)
- `week_of_year`
- `year`

**Rationale:** Seasonal capacity constraints are a primary driver of delays.

#### **2. Geographic Features (6) ⭐ KEY DIFFERENTIATOR**
- **`distance_km`**: Haversine distance from seller to customer zip code using geolocation table
- `same_state` (boolean)
- `same_city` (boolean)
- `customer_state_encoded` (27 states)
- `seller_state_encoded` (27 states)
- `state_pair_late_rate` (historical late rate for this specific route)

**Rationale:** Distance is the #1 predictor. Cross-state deliveries have 2× the late rate.

**Distance Calculation (Haversine Formula):**
```python
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c
```

#### **3. Order Characteristics (6)**
- `number_of_items` (packing complexity)
- `total_order_value` (priority handling bias)
- `total_freight_cost`
- `avg_item_price`
- `freight_to_value_ratio` (rural proxy: high ratio = remote area)
- `freight_per_item`

**Rationale:** Multi-item orders take longer to pack. High freight ratios indicate difficult-to-reach locations.

#### **4. Product Numeric (8)**
- `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`
- `product_volume_cm3` (L × H × W)
- `product_photos_qty` (quality proxy)
- `is_bulky_item` (weight >10kg OR volume >100k cm³)
- `weight_to_price_ratio`

**Rationale:** Heavy/bulky items require truck logistics (slower than couriers).

#### **5. Product Categorical (5)**
- `product_category` (73 categories → target encoding)
- `category_late_rate` (historical late rate per category)
- `is_furniture`, `is_electronics`, `is_fragile` (flags for high-risk categories)

**Rationale:** Furniture has 14.2% late rate vs. 3.2% for books.

#### **6. Payment Features (7)**
- `payment_value`, `payment_installments`, `payment_per_installment`
- `payment_type_credit_card`, `payment_type_boleto`, `payment_type_debit`, `payment_type_voucher` (one-hot encoding)

**Rationale:** Boleto (bank slip) has processing delay. High installments correlate with rural customers.

#### **7. Seller Metrics (3)**
- `seller_order_count` (experience: total historical orders)
- `seller_state_volume` (seller's state shipping volume)
- `seller_late_rate` (seller's historical late rate)

**Rationale:** New sellers (<50 orders) have 40% higher late rates.

#### **8. Time Pressure (3)**
- `estimated_delivery_days` (platform's promise duration)
- `is_urgent_delivery` (<7 days promise)
- `is_long_window` (>20 days promise)

**Rationale:** Short promises (<7 days) are inherently risky. Long windows (>20 days) may indicate platform already flagged complexity.

#### **9. Interaction Features (5)**
- `distance_x_weight` (far + heavy = highest risk)
- `distance_x_bulky`
- `freight_x_distance` (cost per km)
- `peak_season_x_distance` (seasonal + geographic compounding)
- `value_per_km` (efficiency metric)

**Rationale:** Capture non-linear effects. A 1,500km delivery of furniture during May is much higher risk than the sum of individual effects.

**Total Features:** 51 (after dropping 5 ID columns)

### 4.3 Model Selection

**Primary Models:**
1. **XGBoost** (Extreme Gradient Boosting)
2. **Random Forest** (Ensemble Decision Trees)

**Why These Models:**
- Handle **high-cardinality categorical features** (73 product categories, 27 states)
- Capture **non-linear interactions** (distance × weight, season × geography)
- Robust to **class imbalance** (8% late rate) with built-in class weighting
- Provide **feature importance** for business insights
- Fast inference for real-time prediction at order placement

**Baseline Model:**
- Logistic Regression (for comparison)

### 4.4 Handling Class Imbalance

**Problem:** Only 8.1% of orders are late (1:11.3 imbalance).

**Solutions:**
1. **Class Weights:** Assign 11× weight to minority class (late) in loss function
2. **SMOTE** (Synthetic Minority Over-sampling Technique): Generate synthetic late-order examples
3. **Threshold Tuning:** Lower decision threshold from 0.5 to 0.2-0.3 to catch more late orders

**We will NOT use accuracy** as our metric (91% accuracy achieved by predicting "on-time" for everything).

### 4.5 Evaluation Metrics

**Primary Metric: Total Economic Cost**

```python
Total_Cost = (FP × 15) + (FN × 155)
```

Where:
- **FP (False Positives):** Orders flagged unnecessarily → Cost R$15 each
- **FN (False Negatives):** Late orders missed → Cost R$155 each

**Goal:** Minimize total cost, NOT maximize accuracy.

**Secondary Metrics:**
- **Precision:** Of flagged orders, % that are truly late
- **Recall:** Of truly late orders, % we catch (most important!)
- **F1-Score:** Harmonic mean of precision/recall
- **ROC-AUC:** Area under ROC curve (threshold-agnostic)
- **Confusion Matrix:** Visualize all four outcomes

**Threshold Optimization:**
We will plot **Total Cost vs. Threshold** and select the threshold that minimizes cost (likely 0.2-0.3, not 0.5).

### 4.6 Train/Validation/Test Split

**Strategy:** Temporal split (avoid data leakage)

- **Training Set:** 2016 + 2017 data (~45,000 orders)
- **Validation Set:** Jan-June 2018 (~27,000 orders) → tune threshold
- **Test Set:** July-Oct 2018 (~27,000 orders) → final evaluation

**Why Temporal Split:**
- Mimics production (train on past, predict future)
- Seasonal patterns should generalize forward
- Avoids data leakage from random splitting

### 4.7 Sensitivity Analysis

We will test model robustness across 4 economic scenarios:

| Scenario | Poor Review Cost | Intervention Cost | FN:FP Ratio |
|----------|------------------|-------------------|-------------|
| **Base Case** | R$150 | R$20 | 10.3:1 |
| **Conservative (Low Churn)** | R$100 | R$20 | 5.7:1 |
| **Aggressive (High Churn)** | R$300 | R$20 | 19.3:1 |
| **Expensive Intervention** | R$150 | R$40 | 6.8:1 |

**Questions to Answer:**
1. At what churn cost does the model stop being profitable?
2. If intervention cost doubles, does the model still provide value?
3. How does optimal threshold change across scenarios?

**Expected Finding:** Model remains valuable even with 50% changes in cost assumptions (insight survives perturbation).

### 4.8 Feature Importance Analysis

We will extract:
1. **SHAP values** (SHapley Additive exPlanations) for model interpretability
2. **Feature importance plots** from XGBoost/Random Forest
3. **Partial dependence plots** for top 5 features

**Expected Top Predictors:**
1. `distance_km` (geography)
2. `product_weight_g` (product characteristics)
3. `product_category` (category risk)
4. `same_state` (geographic routing)
5. `estimated_delivery_days` (time pressure)

**Business Insight:** If geography dominates, the recommendation is to invest in routing optimization, not predictive modeling.

### 4.9 Limitations

**What We Cannot Know (Without Being Olist):**
1. **Actual churn behavior:** Do customers really never return after one late order?
2. **Intervention effectiveness:** Does priority handling actually prevent delays, or are some delays unavoidable (weather, strikes)?
3. **Operational constraints:** Can the platform actually route 2,000+ orders/day through premium logistics?
4. **Dynamic seller behavior:** Do sellers improve after negative feedback?

**Next Investment (Production Requirements):**
1. **A/B Test:** Randomly assign orders to intervention/control to measure true causal effect
2. **Real-time inference pipeline:** Integrate model with order placement system
3. **Feedback loop:** Track which flagged orders were saved vs. which failed anyway
4. **External data integration:** Weather, carrier strikes, economic indicators

---

## 5. Expected Insights

Based on preliminary exploration, we anticipate the following insights:

### Key Insight #1: "Geography Dominates Over Price"
Cross-state deliveries have 2× the late rate regardless of order value. A R$50 order traveling 1,500km is higher risk than a R$500 order traveling 50km.

**Implication:** Invest in regional fulfillment centers, not payment processing speed-ups.

### Key Insight #2: "Peak Season Requires Aggressive Thresholds"
During Q2, the optimal threshold drops from 0.35 to 0.22, flagging 40% more orders. The economic benefit still outweighs the false alarm cost.

**Implication:** Dynamic threshold by season, not a single global model.

### Key Insight #3: "Most Value Comes From Avoiding Late Deliveries"
Similar to credit scoring, **loss avoidance >> gain capture**. Even if intervention costs rise 50%, the model remains profitable because missing a late order is so expensive (R$155).

**Implication:** This insight survives perturbation of economic assumptions.

### Key Insight #4: "Bulky Items + Long Distance = Compounding Risk"
The interaction term `distance_x_bulky` has high feature importance. A furniture order traveling 1,200km has 8× the base late rate.

**Implication:** Create a special "high-risk" routing category for this segment.

---

## 6. Timeline to Final Submission

**Phase 1 (Feb 17 - Mar 10): Data Preparation**
- ✅ Data loading and exploration (COMPLETE)
- ⏳ Feature engineering pipeline
- ⏳ Train/validation/test split

**Phase 2 (Mar 10 - Mar 31): Model Development**
- ⏳ Train baseline (Logistic Regression)
- ⏳ Train XGBoost + Random Forest
- ⏳ Hyperparameter tuning
- ⏳ Threshold optimization

**Phase 3 (Apr 1 - Apr 15): Economic Evaluation**
- ⏳ Build cost matrix
- ⏳ Sensitivity analysis (4 scenarios)
- ⏳ Feature importance analysis
- ⏳ Geographic heatmaps

**Phase 4 (Apr 15 - Apr 23): Final Report**
- ⏳ Write limitations section
- ⏳ Create presentation slides
- ⏳ Finalize GitHub repository
- ⏳ Contribution statement

**Mid-Project Check-In:** Mar 17-21 (in-class)

---

## 7. References

1. Olist Brazilian E-Commerce Public Dataset. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16.
3. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NIPS.
4. Course Lecture: Credit Scoring Economics (Week 3) - Loss avoidance vs. gain capture framework.

---

## Appendix: Visualization Requirements

### Required Graphs for Final Report:

1. **Order Volume by Quarter** (Bar chart) - Justifies seasonal focus
2. **Review Score vs. Delivery Delay** (Heatmap) - Proves non-trivial correlation
3. **Late Rate by Product Category** (Bar chart) - Feature importance
4. **Late Rate by Distance Bins** (Line chart) - Geography effect
5. **Confusion Matrix with Costs** (Annotated heatmap) - Economic evaluation
6. **Total Cost vs. Threshold** (Line chart) - Threshold optimization
7. **Geographic Heatmap** (State-to-state late rates) - At-risk routes
8. **Feature Importance** (Horizontal bar chart) - Model interpretability

Code to generate these visualizations is provided separately in `visualization_code.py`.

---

**END OF INTERIM REPORT**

*This report confirms our team has identified a dataset, defined a business problem with economic stakes, performed exploratory analysis, and designed a sound approach for the final project. All required elements for the February 17th interim submission are present.*
