# OLIST Brazil E-commerce: Delivery Days Regression Task

## Executive Summary

**Objective:** Build a classical regression model to predict **actual delivery days** for orders in the Brazilian OLIST e-commerce platform.

**Current State:** Olist provides an `estimated_delivery_days` column (their internal estimate). Your task is to:
1. Predict the **actual** delivery days more accurately than Olist's estimate
2. Identify which orders are likely to be late (for potential intervention)
3. Control for data leakage from the derived `is_late` column

**Dataset:** `olist_merged_clean.csv`  
**Rows:** 96,189 completed orders  
**Target Variable:** `actual_delivery_days` (continuous, float)

---

## QUICK REFERENCE: CSV FILE & COLUMNS

### Exact File Location
```
Brazil_ecommerce/datasets/processed/olist_merged_clean.csv
```

### Reference Notebook (Look At This First)
```
Brazil_ecommerce/notebooks/something/temporal.ipynb
```
**Why:** Shows how to handle temporal splits, feature engineering, and model comparison. Your regression task uses the same dataset and logic.

### Target Column (What to Predict)
```
actual_delivery_days
```

### 17 Core Features to INCLUDE (exclude_leakage mode)
```
estimated_delivery_days
distance_km
same_state
n_items
total_price
total_freight
max_weight_g
max_length_cm
max_height_cm
max_width_cm
product_photos_qty
primary_category
payment_installments
payment_type
customer_state
seller_state
month (extract from order_purchase_timestamp)
quarter (extract from order_purchase_timestamp)
is_weekend (extract from order_purchase_timestamp)
```

### Columns to EXCLUDE with Reasons

| Column | Why Excluded |
|--------|----------|
| is_late | **LEAKAGE**: Directly derived from target (actual_delivery_days > estimated_delivery_days). Tells model the outcome it should predict. |
| delivery_delay_days | **LEAKAGE**: Encodes target directly. Computed as actual_delivery_days - estimated_delivery_days. |
| order_id | **Identifier**: Not predictive; unique per order. |
| customer_id | **Identifier**: Not predictive; unique per customer. No proven purchase history impact. |
| seller_id | **Too granular**: Use engineered aggregates (seller_avg_delivery_days, seller_ontime_rate) instead. Raw IDs don't generalize. |
| order_approved_at | **Redundant**: Temporal information already captured by month, quarter, is_weekend. Time of approval doesn't add predictive power. |
| order_delivered_carrier_date | **Post-shipment**: Not available at order placement time. Can't use future information in production. |
| order_delivered_customer_date | **Post-delivery**: This IS the date used to compute actual_delivery_days. Cannot use ground truth timestamp as a feature. |
| order_estimated_delivery_date | **Redundant**: Same information as estimated_delivery_days (both are the promised date). Use numeric column, not datetime. |
| customer_lat | **Redundant**: Distance already captured via distance_km. Raw latitude doesn't help. |
| customer_lng | **Redundant**: Distance already captured via distance_km. Raw longitude doesn't help. |
| seller_lat | **Redundant**: Distance already captured via distance_km. Raw latitude doesn't help. |
| seller_lng | **Redundant**: Distance already captured via distance_km. Raw longitude doesn't help. |
| customer_zip_code_prefix | **Too granular**: 5-digit prefix is too fine-grained (27K unique values). Use state-level aggregates instead. |
| seller_zip_code_prefix | **Too granular**: 5-digit prefix is too fine-grained. Use state or engineered seller statistics. |
| customer_city | **Redundant**: City is subsumed by state. State-level effects are stronger for logistics. |
| seller_city | **Redundant**: City is subsumed by state. State-level effects are stronger for logistics. |

---

---

## Dataset Structure

### Data Dimensions
| Metric | Value |
|--------|-------|
| Total Orders | 96,189 |
| Features Available | 34 columns |
| Date Range | 2016–2018 |
| Late Orders (is_late=1) | 7,642 (7.9%) |
| On-time Orders (is_late=0) | 88,547 (92.1%) |

### Critical Timestamps (All datetime)
| Column | Description | When Known |
|--------|-------------|-----------|
| `order_purchase_timestamp` | When customer placed order | At placement ✓ |
| `order_approved_at` | When payment cleared | At placement ✓ |
| `order_estimated_delivery_date` | Olist's promise to customer | At placement ✓ |
| `order_delivered_carrier_date` | When carrier picked up | Post-shipment ✗ |
| `order_delivered_customer_date` | **Actual delivery** | Post-delivery ✗ |

---

## Target Variable: `actual_delivery_days`

### Definition
```
actual_delivery_days = (order_delivered_customer_date - order_purchase_timestamp).days
```

### Statistics
- **Type:** Float (continuous)
- **Range:** Typically 2–90 days
- **Interpretation:** Business days from order placement to customer receipt

### Conceptual Check
```
actual_delivery_days = prep_days + shipping_days
where:
  prep_days = days from purchase to carrier pickup
  shipping_days = days from carrier to customer
```

---

## Feature Categories

### ⚠️ DATA LEAKAGE RISK: `is_late` Column

**CRITICAL:** This column is **derived from your target** and must be handled carefully.

```
is_late = 1 if (actual_delivery_days > estimated_delivery_days) else 0
```

**Problem:**
- Using `is_late` in training directly encodes the target
- It's not truly predictive; it's a transformation of the ground truth
- If you include it, you're telling the model "this order is late," which it then asks to predict

**Solution:**
Create a control variable `to_include_or_not_to_include` with two modes:

| Mode | Use `is_late`? | When | Purpose |
|------|---|---|---|
| `include_leakage` | YES | Upper-bound performance; sanity check | See theoretical maximum if model "knows" the outcome |
| `exclude_leakage` (recommended) | NO | Production & fair evaluation | Realistic regression task without cheating |

**Implementation:**
```python
# In your preprocessing
if to_include_or_not_to_include == 'include_leakage':
    feature_set = [...all features..., 'is_late']
else:  # exclude_leakage
    feature_set = [...all features...]  # Drop is_late
    
# Always keep 'is_late' in dataset for analysis, just exclude from training
```

---

## Feature Inventory (by Type)

### 1. Temporal Features (KNOWN AT ORDER TIME)
| Feature | Type | Description | Availability |
|---------|------|-------------|---------------|
| `order_purchase_timestamp` | datetime | Exact order time | ✓ |
| Month (derived) | int [1–12] | Month of order | ✓ |
| Quarter (derived) | int [1–4] | Quarter of order | ✓ |
| DayOfWeek (derived) | int [0–6] | Day of week (0=Monday) | ✓ |
| IsWeekend (derived) | binary | 1 if order on Sat/Sun | ✓ |

**Why Temporal Matters:** Weekend orders may sit longer; seasonal patterns affect carrier capacity.

---

### 2. Estimated Delivery (OLIST's Prediction)
| Feature | Type | Description |
|---------|------|-------------|
| `estimated_delivery_days` | float | **Olist's estimate** (THIS IS WHAT YOU'RE BEATING) |
| `order_estimated_delivery_date` | datetime | Promised delivery date |

**Key Insight:**  
Your regression model should learn to adjust Olist's estimate based on features they don't have (location, weight, seller capacity, etc.).

---

### 3. Geographic Features (KNOWN AT ORDER TIME)
| Feature | Type | Description | Business Logic |
|---------|------|-------------|---|
| `distance_km` | float | Haversine distance (customer to seller) | Far = slower |
| `customer_lat`, `customer_lng` | float | Customer coordinates | |
| `seller_lat`, `seller_lng` | float | Seller coordinates | |
| `customer_state` | categorical | State code (SP, RJ, MG, etc.) | Regional courier efficiency |
| `seller_state` | categorical | Seller's state | |
| `same_state` | binary | 1 if customer & seller in same state | Cross-state = slower |

**Feature Engineering Ideas:**
- `is_remote_region`: Northeast/North states are slower
- `distance_x_logistics_risk`: Interaction of distance and seller capacity
- `cross_state_distance`: Only for cross-state orders

---

### 4. Product/Order Features (KNOWN AT ORDER TIME)
| Feature | Type | Description | Business Logic |
|---------|------|-------------|---|
| `n_items` | int | Number of items in order | More items = longer pick time |
| `total_price` | float | Order value (R$) | High-value orders may ship slower (QC) |
| `total_freight` | float | Freight cost (R$) | Proxy for shipping complexity |
| `max_weight_g` | float | Heaviest item (grams) | Heavy = slower |
| `max_length_cm`, `max_height_cm`, `max_width_cm` | float | Max physical dimensions | Oversize = special handling |
| `product_photos_qty` | float | Photos on listing | Proxy for seller professionalism |
| `primary_category` | categorical | Top product category | Some categories ship faster |
| `payment_installments` | float | Number of payment installments | Installments = riskier? May be slower to ship |
| `payment_type` | categorical | boleto, credit_card, debit, voucher | Payment risk proxy |

**Feature Engineering Ideas:**
- `log_price`: Log transform of price
- `log_weight`: Log transform of weight
- `is_heavy`: Binary (weight > 75th percentile)
- `is_high_value`: Binary (price > 75th percentile)
- `is_bulky`: Binary (volume > threshold)
- `freight_to_price_ratio`: Logistics complexity signal

---

### 5. Seller Features (KNOWN AT ORDER TIME)
| Feature | Type | Description | Business Logic |
|---------|------|-------------|---|
| `seller_id` | categorical | Unique seller identifier | Seller efficiency varies |
| `seller_zip_code_prefix` | int | Seller location postal code prefix | |

**Feature Engineering Ideas:**
- Compute seller-level statistics from training data only:
  - `seller_avg_delivery_days`: Average delivery time for this seller
  - `seller_ontime_rate`: Fraction of on-time deliveries
  - `seller_std_delivery_days`: Variability in this seller's delivery
- **IMPORTANT:** Compute these from **training set only** → map to test set to avoid leakage

---

### 6. Derived Features (COMPUTED BUT NOT FOR INPUT)
| Feature | Type | Status | Why |
|---------|------|--------|-----|
| `actual_delivery_days` | float | **TARGET** | What you're predicting |
| `delivery_delay_days` | float | **DO NOT USE** | This is target - estimated, leakage |
| `is_late` | binary | **CONTROL VARIABLE** | Use only in `include_leakage` mode |

---

## Feature Engineering: Categorical Encoding

### One-Hot Encoding (Converting Categories to Binary Columns)

Four categorical columns are converted to binary dummy variables:

```python
cat_cols = ['primary_category', 'payment_type', 'customer_state', 'seller_state']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
```

**What happens:**
- Each unique value in a categorical column becomes its own binary column (0 or 1)
- Example: `primary_category` with values [electronics, furniture, books, ...] becomes:
  - `primary_category_electronics` (1 if order is electronics, 0 otherwise)
  - `primary_category_furniture` (1 if order is furniture, 0 otherwise)
  - `primary_category_books` (1 if order is books, 0 otherwise)
  - ... (one per category)

**Why:**
- Tree models (XGBoost, RandomForest) can't directly use text/categorical values
- Binary columns are interpretable and allow models to learn category-specific effects
- `drop_first=True` prevents multicollinearity (can infer last category from others)

**Categorical Columns Encoded:**

| Column | Type | Example Values | Resulting Dummy Columns |
|--------|------|---|---|
| `primary_category` | Category | electronics, furniture, books, beauty, clothing, ... | ~72 categories → ~71 binary columns |
| `payment_type` | Category | credit_card, boleto, debit_card, voucher | 4 values → 3 binary columns |
| `customer_state` | Category | SP, RJ, MG, BA, SC, ... (27 states) | 27 states → 26 binary columns |
| `seller_state` | Category | SP, RJ, MG, BA, SC, ... (27 states) | 27 states → 26 binary columns |

**Total dummy columns created: ~126 (70 + 3 + 26 + 26)**

**Final feature count after encoding:**
- Original numeric features: 15 (estimated_delivery_days, distance_km, n_items, max_weight_g×4, etc.)
- Temporal features: 3 (month, quarter, is_weekend)
- Dummy columns: ~126
- **Total: ~144 features** (before feature selection)

---

### 6. Derived Features (COMPUTED BUT NOT FOR INPUT)
| Feature | Type | Status | Why |
|---------|------|--------|-----|
| `actual_delivery_days` | float | **TARGET** | What you're predicting |
| `delivery_delay_days` | float | **DO NOT USE** | This is target - estimated, leakage |
| `is_late` | binary | **CONTROL VARIABLE** | Use only in `include_leakage` mode |

---

## Data Quality

### Missing Values
- **Minimal:** <1 missing value per 96K rows
- **Strategy:** Use median imputation for numeric; mode for categorical
- Only `order_delivered_carrier_date` has ~1 missing (drop these rows)

### Outliers
- **Actual delivery days:** Rarely exceed 90 days (likely cancelled/returned)
- **Distance:** A few orders with distance_km > 2,000 (valid, e.g., São Paulo → Salvador)
- **Strategy:** Consider capping at domain-specific limits or log transform

### Class Imbalance (for classification context)
- 92.1% on-time, 7.9% late
- Regression problem, so not an issue for MSE/RMSE metrics
- BUT: If predicting P(late) later, will need stratified splits

---

## Regression Task Setup

### Train-Test Split Strategy
The `temporal.ipynb` uses **temporal split**:
- **Train:** 2017 orders
- **Test:** H2 2018 orders

For your regression task, you can:
1. **Use same temporal split** (most realistic for deployment)
2. **Use stratified random split** by region/seller to ensure representativeness

### Baseline Model
```
Baseline MSE = mean((actual_delivery_days - estimated_delivery_days)^2)
```

Your model should beat Olist's baseline.

### Loss Functions / Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **MAE** | mean(\|y_actual - y_pred\|) | Robust to outliers; interpretable (±days) |
| **RMSE** | sqrt(mean((y_actual - y_pred)²)) | Standard; penalizes large errors |
| **MAPE** | mean(\|error / y_actual\|) | For % error interpretation |
| **Quantile Loss (0.95)** | Focus on not over-promising | Conservative; for SLA |

**Recommended:** RMSE + MAE (standard regression metrics)

---

## Leakage Control: Two-Mode Training

Use binary flag: `include_leakage = 0` or `include_leakage = 1`

### Mode 0: `include_leakage = 0` (RECOMMENDED FOR PRODUCTION)

**Features Used:**  
All temporal, geographic, product, seller features **EXCEPT** `is_late` and `delivery_delay_days`

**Rationale:**  
- Mimics real deployment: model receives features at order time
- Fair evaluation: model must generalize without outcome information
- `estimated_delivery_days` from Olist is acceptable (it's a competitor prediction)

**Expected Performance:**  
- Good RMSE: 5–15 days depending on seller capacity variance
- R² likely 0.4-0.6 (realistic)

```python
include_leakage = 0  # Set to 1 to test with leakage

FEATURES_NO_LEAKAGE = [
    'estimated_delivery_days',  # Olist's estimate (not cheating)
    'distance_km', 'same_state',
    'n_items', 'total_price', 'total_freight',
    'max_weight_g', 'max_length_cm', 'max_height_cm', 'max_width_cm',
    'product_photos_qty', 'primary_category',
    'payment_installments', 'payment_type',
    'month', 'quarter', 'is_weekend',
    'customer_state', 'seller_state',
    'seller_avg_delivery_days', 'seller_ontime_rate',
]

if include_leakage == 0:
    FEATURES_TO_USE = FEATURES_NO_LEAKAGE
```

### Mode 1: `include_leakage = 1` (FOR ANALYSIS ONLY)

**Features Used:**  
Everything above **PLUS** `is_late`

**Rationale:**  
- **Sanity check:** Establishes upper-bound performance
- **Analysis:** Shows what unlucky information would contribute
- **Not for production:** But useful for research to see algorithm ceiling

**Expected Performance:**  
- Artificially inflated (R² > 0.8 possible)
- RMSE improvement of 40-60% compared to mode 0
- Proves algorithm power, not realism

```python
include_leakage = 1  # Set to check leakage impact

FEATURES_WITH_LEAKAGE = FEATURES_NO_LEAKAGE + ['is_late']

if include_leakage == 1:
    FEATURES_TO_USE = FEATURES_WITH_LEAKAGE
```

---

## CRITICAL: Metrics Comparison Procedure

**Train and evaluate the model TWICE** — once with each mode — and compare:

### Run Both Experiments
```python
results_comparison = {}

for include_leakage in [0, 1]:
    if include_leakage == 0:
        features = FEATURES_NO_LEAKAGE
        mode_name = 'WITHOUT is_late (Production)'
    else:
        features = FEATURES_WITH_LEAKAGE
        mode_name = 'WITH is_late (Leakage)'
    
    X = df[features].fillna(0)
    y = df['actual_delivery_days']
    
    # Train-test split (TEMPORAL: 2017 train, H2-2018 test)
    train_mask = df['order_purchase_timestamp'].dt.year == 2017
    test_mask = (df['order_purchase_timestamp'].dt.year == 2018) & (df['order_purchase_timestamp'].dt.month >= 7)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Train model
    model = XGBRegressor(n_estimators=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute all metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results_comparison[mode_name] = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    print(f'\nMode: {mode_name}')
    print(f'  RMSE: {rmse:.2f} days')
    print(f'  MAE: {mae:.2f} days')
    print(f'  R²: {r2:.4f}')
    print(f'  MAPE: {mape:.2f}%')
```

### Expected Results Table

| Metric | Mode 0 (No Leakage) | Mode 1 (With Leakage) | Interpretation |
|--------|---|---|---|
| **RMSE** | ~10 days | ~4-6 days | Mode 1 is 40-60% better (artificial) |
| **MAE** | ~8 days | ~3-4 days | Mode 1 error smaller (cheating) |
| **R²** | 0.4–0.6 | 0.75–0.85 | Mode 1 explains much more variance (has outcome info) |
| **MAPE** | 20–30% | 8–15% | Mode 1 % error much lower (not realistic) |

### How to Interpret the Comparison

**If Mode 0 (No Leakage) performs well:**
- ✅ RMSE < 12 days → Model is learning real patterns
- ✅ R² > 0.4 → Fair amount of variance explained
- ✅ MAE reasonable → Can trust predictions in production

**If Mode 1 (With Leakage) is much better than Mode 0:**
- ✅ Confirms algorithm works (Mode 1 RMSE >> Mode 0 RMSE proves model learns)
- ✅ Gap between modes = value of leakage variable
- ❌ But Mode 1 cannot be deployed (cheating)

**Example Scenario:**
```
Mode 0 RMSE: 10.5 days
Mode 1 RMSE: 4.2 days  
Gap: 6.3 days (60% improvement from is_late)

Interpretation: is_late helps a lot, but we can't use it.
Still: Mode 0 RMSE of 10.5 is good enough for production.
```

---

## Implementation Roadmap

### Step 1: Data Preparation
```python
# 1. Load and clean
df = pd.read_csv('olist_merged_clean.csv')

# 2. Parse dates
date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 
             'order_estimated_delivery_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# 3. Compute target (if not already present)
if 'actual_delivery_days' not in df.columns:
    df['actual_delivery_days'] = (df['order_delivered_customer_date'] - 
                                   df['order_purchase_timestamp']).dt.days

# 4. Extract temporal features
df['month'] = df['order_purchase_timestamp'].dt.month
df['quarter'] = df['order_purchase_timestamp'].dt.quarter
df['is_weekend'] = df['order_purchase_timestamp'].dt.dayofweek.isin([5,6]).astype(int)

# 5. One-hot encode categorical features (convert text categories to binary columns)
cat_cols = ['primary_category', 'payment_type', 'customer_state', 'seller_state']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
# Result: ~80-120 new binary columns (e.g., primary_category_electronics, payment_type_credit_card, etc.)
# drop_first=True avoids multicollinearity by removing one dummy per category

# 6. Feature engineering (seller-level stats from training only)
seller_stats = df_train.groupby('seller_id').agg({
    'actual_delivery_days': ['mean', 'std'],
    'is_late': 'mean'
}).fillna(df_train['actual_delivery_days'].mean())

df['seller_avg_delivery_days'] = df['seller_id'].map(seller_stats['mean'])
df['seller_ontime_rate'] = df['seller_id'].map(seller_stats['is_late'].mean())
```

### Step 2: Feature Selection (Control Leakage)
```python
include_leakage = 0  # 0 = exclude (production), 1 = include (analysis only)

if include_leakage == 0:
    features = FEATURES_NO_LEAKAGE
    print("Training WITHOUT is_late (fair, production-ready)")
else:
    features = FEATURES_WITH_LEAKAGE
    print("Training WITH is_late (cheating, for sanity check only)")

X = df[features].fillna(0)  # or use median imputation
y = df['actual_delivery_days']
```

### Step 3: Train-Test Split
```python
# Option A: Temporal split (recommended)
X_train = X[df['order_purchase_timestamp'].dt.year == 2017]
y_train = y[df['order_purchase_timestamp'].dt.year == 2017]
X_test = X[(df['order_purchase_timestamp'].dt.year == 2018) & 
           (df['order_purchase_timestamp'].dt.month >= 7)]
y_test = y[(df['order_purchase_timestamp'].dt.year == 2018) & 
           (df['order_purchase_timestamp'].dt.month >= 7)]

# Option B: Random split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 4: Model Training & Evaluation
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Simple baseline
baseline_rmse = np.sqrt(mean_squared_error(y_test, X_test['estimated_delivery_days']))

# Train models
models = {
    'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6),
    'RF': RandomForestRegressor(n_estimators=200, max_depth=15),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {'rmse': rmse, 'mae': mae, 'predictions': y_pred}
    print(f'{name}: RMSE={rmse:.2f} days, MAE={mae:.2f} days')
    print(f'Baseline RMSE: {baseline_rmse:.2f} days')
```

---

## Key Insights for Model Development

### Feature Importance Hierarchy (Expected)
1. **`estimated_delivery_days`** (Olist's own estimate)
2. **Seller dynamics** (avg delivery time, ontime rate)
3. **Distance & geography** (cross-state esp.)
4. **Product weight & size** (handling time)
5. **Customer location** (regional courier efficiency)

---

## SHAP Analysis: Which Columns Matter Most?

### Step 1: Compute Feature Importance (Tree-Based Models)
```python
# Get feature importance from trained XGBoost or RandomForest model
import matplotlib.pyplot as plt

model = results['XGBoost']['clf']  # Get trained model

# Method A: Native feature importance (fast)
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(importance_df.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df.head(15)['feature'], importance_df.head(15)['importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.show()
```

### Step 2: SHAP Values (Detailed Feature Influence)
```python
import shap

# Initialize SHAP explainer (works with any model)
explainer = shap.TreeExplainer(model)  # For tree-based models
shap_values = explainer.shap_values(X_test)

# Summary plot: Which features matter overall?
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.title("SHAP Feature Importance (Mean |SHAP value|)")
plt.show()

# Detailed summary: Feature impact on predictions
shap.summary_plot(shap_values, X_test)
plt.title("SHAP Summary (Color = Feature Value, Position = Impact)")
plt.show()
```

### Step 3: Individual Prediction Explanation (Why Specific Order?)
```python
# Explain a single prediction
sample_idx = 0  # First order in test set
shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_test.iloc[sample_idx])

# Example interpretation:
# Base prediction: 10.5 days
# distance_km = 500km pushes +2 days (red, increases prediction)
# is_weekend = 1 pushes -0.8 days (blue, decreases prediction)
# Final prediction: ~11.7 days
```

### Step 4: Feature Interaction (How Do Two Columns Work Together?)
```python
# Find most influential feature interactions
# Plot feature A vs feature B colored by SHAP values
feature_1 = 'distance_km'
feature_2 = 'estimated_delivery_days'

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_test[feature_1], X_test[feature_2], 
                     c=shap_values[:, X_test.columns.get_loc(feature_1)],
                     cmap='coolwarm', alpha=0.6, s=50)
plt.colorbar(scatter, label='SHAP Impact')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title('SHAP Interaction: How distance & estimated days affect delivery')
plt.show()
```

### Step 5: Dependence Plots (Nonlinear Relationships)
```python
# How does each feature's value affect predictions?
for feature in importance_df.head(5)['feature']:
    shap.dependence_plot(feature, shap_values, X_test)
    plt.show()

# Interpretation:
# - Near-vertical line = strong effect (small changes matter)
# - Horizontal line = weak effect (value doesn't change output much)
# - S-curve = nonlinear (e.g., distance up to 500km hurts, then flattens)
```

### What SHAP Tells You (vs Tree Importance)

| Aspect | Tree Importance | SHAP Values |
|--------|---|---|
| **What it shows** | How often feature splits tree | Individual feature contribution to each prediction |
| **Interpretation** | Global: "distance_km is used a lot" | Per-prediction: "This order's distance pushes delivery +3 days" |
| **Positive/Negative** | Only positive (split importance) | Shows direction: increases or decreases prediction |
| **Interactions** | Not captured | Implicitly shows via dependence plots |
| **Business insight** | "distance matters" | "For this customer, distance matters; for that one, it doesn't" |

---

### Example SHAP Findings (Expected)

**Top 5 Most Influential Features (by mean SHAP):**
1. `estimated_delivery_days` — Olist's estimate is the strongest signal (high correlation with actual)
2. `distance_km` — Farther orders take longer (nonlinear; plateau after ~1000km)
3. `seller_avg_delivery_days` — Seller's historical speed is very predictive
4. `is_weekend` — Weekend orders may wait longer for Monday pickup
5. `max_weight_g` — Heavier items take longer (logistics complexity)

**Key Pattern to Look For:**
```
distance_km = 1000km  →  SHAP +5 days
distance_km = 2000km  →  SHAP +6 days (plateau)

Interpretation: Distance matters, but not linearly. 
Very far orders don't get exponentially slower.
```

### Interpretability Tips
- Plot actual vs. predicted for underperforming regions
- Check residuals by seller (are some sellers' delays systematic?)
- Analyze predictions for high-value orders (do they get special treatment?)
- Use SHAP to explain why model disagrees with Olist's estimate

### Business Applications
- **Dysfunctional Sellers:** Identify sellers consistently late (high `seller_avg_delivery_days` + model underpredicts)
- **Regional Issues:** Find bottleneck states (negative SHAP for customer_state)
- **High-Value Orders:** Do expensive orders ship faster or slower? (check price SHAP)
- **Seasonal Effects:** Do holidays/Q4 affect delivery times? (check month/quarter SHAP)

---

## File Locations Reference

| File | Purpose |
|------|---------|
| `datasets/processed/olist_merged_clean.csv` | Main working dataset (96K rows) |
| `datasets/raw/olist_orders_dataset.csv` | Raw orders (timestamps only) |
| `datasets/raw/olist_sellers_dataset.csv` | Seller metadata |
| `datasets/raw/olist_products_dataset.csv` | Product info & categories |
| `notebooks/something/temporal.ipynb` | Prior temporal analysis (reference) |

---

## Summary for AntiGravity* Model Handoff

\* _"AntiGravity" appears to be your next modeling tool._

When you're ready to shift to your next model:

**You now have:**
1. ✓ Clear target variable: `actual_delivery_days`
2. ✓ Feature set split into `exclude_leakage` vs `include_leakage` modes
3. ✓ Baseline comparison: Olist's `estimated_delivery_days`
4. ✓ Train-test strategy: Temporal (2017 vs H2-2018)
5. ✓ Data quality assessment (minimal missing, few outliers)
6. ✓ Business context (7.9% late orders, ~96K total)

**Just feed AntiGravity:**
- Dataset: `olist_merged_clean.csv`
- Target: `actual_delivery_days` (regression)
- Features: Use `exclude_leakage` list (production-ready)
- Split: Temporal (2017 train / H2-2018 test)
- Baseline RMSE: Compute `np.sqrt(MSE(y_test - estimated_delivery_days))`
- Metrics: RMSE + MAE

---

## OUTPUT FORMAT: Create a Jupyter Notebook

**AntiGravity should output a fully working `.ipynb` file with:**

1. **Cell 1 (Markdown):** Title & problem statement
2. **Cell 2 (Code):** Imports (pandas, numpy, sklearn, xgboost, shap, matplotlib)
3. **Cell 3 (Code):** Load `olist_merged_clean.csv` & check shape
4. **Cell 4 (Code):** Data preparation (parse dates, extract temporal features)
5. **Cell 5 (Code):** Feature selection with `include_leakage = 0` (no leakage)
6. **Cell 6 (Code):** Train-test split (temporal: 2017 vs H2-2018)
7. **Cell 7 (Code):** Train regression model (XGBoost + Random Forest)
8. **Cell 8 (Code):** Evaluate metrics: RMSE, MAE, R², MAPE for mode 0
9. **Cell 9 (Markdown):** Leakage test explanation
10. **Cell 10 (Code):** Re-run with `include_leakage = 1` (with is_late)
11. **Cell 11 (Code):** Compare metrics: Mode 0 vs Mode 1 (table + visualization)
12. **Cell 12 (Code):** Feature importance (tree-based)
13. **Cell 13 (Code):** SHAP summary plots (bar + force + dependence)
14. **Cell 14 (Code):** Explain 1-2 individual predictions using SHAP
15. **Cell 15 (Markdown):** Key findings & business recommendations

**Save as:** `regression_delivery_days_final.ipynb`

**And you're done with data exploration.**
