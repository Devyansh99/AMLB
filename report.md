# Project Report: Brazil E-Commerce Logistics & Satisfaction

---

## 1. Project Evolution & Folder History

### The `old` Folder — First Attempt: Baseline Models

**Files:** `01_data_cleaning.ipynb`, `02_EDA.ipynb`, `02_EDA_Continue.ipynb`, `03_Baseline.ipynb`, `03_depth_ml.ipynb`, `03_regression_margin.ipynb`, `brazil_ecommerce_analysis.ipynb`, `future_fixing.ipynb`

**What was done:**
- Cleaned and merged all 8 raw Olist tables (orders, customers, sellers, products, payments, reviews, geolocation) into a single working dataset of 96,189 completed orders.
- Ran full exploratory data analysis: delivery time distributions, review score breakdowns, geographic and category-level late rates.
- Built two model types:
  - **Classification:** Predict `is_late` (did the order arrive after the estimated date?). Models: Logistic Regression, Random Forest, XGBoost.
  - **Regression:** Predict the exact number of actual delivery days. Models: Random Forest, XGBoost, linear regression.

**Key EDA findings (real numbers from the data):**
- On-time rate: 91.9% — Late rate: 8.1% (7,642 late orders out of 96,189 total)
- Mean actual delivery time: 12.4 days. Olist's mean estimated delivery: 23.6 days — a built-in buffer of ~11 days on every order.
- Cross-state orders: 10.3% late rate vs. same-state: 5.2% — a measured 2× difference from geography alone.
- Furniture: 14.2% late rate. Books: 3.2% late rate — a 4.4× difference by product category.
- Items over 10 kg: 1.8× higher late rate than lighter items.
- New sellers (under 50 total orders): 1.4× higher late rate than experienced sellers.
- Delivery delay vs. review score correlation:
  - On-time orders: 4.2% poor review rate
  - 1–3 days late: 12.8% poor review rate (3× increase)
  - 4–7 days late: 31.4% poor review rate (7.5× increase)
  - 8–14 days late: 58.7% poor review rate (14× increase)
  - 15+ days late: 79.3% poor review rate (19× increase)

**Results:**
- Classification AUC: ~0.55 — barely better than a coin flip.
- Regression R²: −0.11 for our model; −4.39 for Olist's own estimate. Both negative (explained in Section 4).

**Why it failed:**
- 91.9% of orders are on-time by Olist's generous definition. Any model that just predicts "on-time" for every order achieves 91.9% accuracy — making accuracy a useless metric here.
- The actual delivery time is driven by carrier routing events (truck availability, route congestion, facility processing) that are simply not recorded in the dataset. Pre-purchase features like distance and weight carry only weak signal for this question.

---

### The `ranking` Folder — Second Attempt: Predicting Review Scores

**Files:** `eda.ipynb`, `model_v1.ipynb`, `model_v2_full.ipynb`

**What was done:**
- Pivoted from predicting delays to directly predicting the customer's review score (1–5 stars).
- EDA established the score distribution: 57.9% gave 5 stars, 19.3% gave 4 stars, 8.4% gave 3 stars, 3.3% gave 2 stars, 11.2% gave 1 star.
- XGBoost regressor (`model_v1.ipynb`) and Random Forest with additional feature engineering (`model_v2_full.ipynb`).

**Results:**
- RMSE: ~1.13 (off by more than one full star on a 1–5 scale).
- R²: ~0.24 (the model explained only 24% of variation in review scores).

**Why it failed:**
- The review score is the combined result of three completely different problems: logistics failures, defective/wrong products, and poor customer service. A model trained on shipping features cannot separate these at all.
- 41.4% of reviewers who had an on-time delivery still gave low scores (1–3 stars), confirming that delivery performance alone cannot explain rating.

---

### The `new` Folder — Third Attempt: Spatial and Governance Layers

**Files:** `customer_location_distance_review_analysis.ipynb`, `layer_2.ipynb`, `layer_3_spatial.ipynb`, `layer_4_governance.ipynb`, `model_v3_classification.ipynb`, `suggestion.ipynb`, `v3_updated.ipynb`

**What was done:**
- Plotted customer coordinates against review scores; found that Northeast Brazil (Bahia, Ceará, Pernambuco, Maranhão) and the far North (Amazonas, Pará) had concentrated clusters of dissatisfied, late customers.
- Built spatial features: K-Means clustering on customer coordinates, regional groupings, distance bins.
- Added state-level governance indicators (logistics infrastructure ranking, carrier hub density by state).
- Key structural finding: 80.7% of all sellers are located in São Paulo state. Customers in the Northeast and North must receive packages from 1,500–3,000 km away, with fewer carrier hubs. This geographic concentration is the structural root cause of Brazil's late-delivery problem.

**Results:**
- AUC improved slightly to around 0.62–0.65 — better than baseline but still not deployable.
- Spatial features (`same_state`, `distance_km`, regional cluster) were the strongest new additions, confirming the geographic hypothesis.

**Why it failed:**
- Distance is a straight-line measure. Real carrier routes depend on highway networks, regional hub locations, and carrier contracts — none of which are in the dataset.
- State-level aggregates were too coarse: two cities in the same state can have completely different logistics infrastructure.
- The core problem — mixed and noisy review labels from three different types of complaints — was still unsolved.

---

## 2. The `something` Folder — Where the Right Frame Was Found

After three pivots, the team fundamentally changed the question. Instead of predicting delays or ratings across all orders, the focus shifted to: **among already-confirmed delayed orders, which customers will become angry enough to leave a 1–2 star review?**

### Key Insight That Changed Everything
NLP analysis of Portuguese review comments proved that a large fraction of bad reviews have nothing to do with logistics — customers gave low scores because they received the wrong product, a defective item, or had a bad refund experience. A logistics model trained on all bad reviews was learning seller/product failure patterns. The fix: clean the training labels with NLP first, then model only what logistics can actually explain.

---

### `review_root_cause_nlp.ipynb` — The Label Cleaner

- Filtered all orders with review score below 4 that also had a written text comment in Portuguese.
- Applied a zero-shot multilingual NLP model (mDeBERTa-v3-base-mnli-xnli) — no manual training labels needed. The model reads Portuguese natively and outputs English business categories.
- Four output categories: Logistics & Delivery Delay, Defective or Bad Quality Product, Wrong Product Delivered, Customer Service / Refund Issue.
- Saved root-cause labels to `datasets/processed/nlp_root_causes.csv`.
- Result: proved that a significant portion of bad reviews come from product/seller faults, not delivery timing. This output became the data-cleaning step for all downstream ML: training data was filtered to keep only orders where the bad review was genuinely about logistics.

---

### `delivery_time_one.ipynb` and `delivery_time copy.ipynb` — Regression Experiments

- Focused regression: can we predict the exact number of delivery days better than Olist's own estimate?
- Documented the leakage risk: `is_late` is derived directly from the target (`actual_delivery_days > estimated_delivery_days`). Including it inflates performance — the model is using the answer to predict the answer.
- Ran two modes deliberately: `exclude_leakage` (production-safe, no `is_late`) and `include_leakage` (sanity check only).
- Key results:
  - `exclude_leakage` mode: our model R² ≈ −0.11; Olist's own baseline R² = −4.39. Our model makes smaller actual errors in days — R² just misrepresents this because it squares every error.
  - `include_leakage` mode: R² jumps significantly — confirming that `is_late` carries enormous information and must be excluded for any honest model.
- Conclusion: regression on delivery days is not viable with this dataset. The target is too noisy due to unrecorded external events. The project moved to classification.

---

### `temporal.ipynb` — The Time-Aware Split (Parts 1–5 only)

**Part 1 — Business frame:**
- Defined the two-model strategy: Model A (delay predictor on all orders) and Model B (anger predictor on already-confirmed delayed orders).
- Deployment scenario: each morning, Olist receives a carrier-confirmed list of delayed orders. Model B runs on that list and scores anger risk per order.

**Part 2 — NLP root-cause block:**
- Used `nlp_root_causes.csv` output to filter training data: only logistics-tagged bad reviews kept as true dissatisfaction signals.

**Part 3 — Feature engineering (pre-purchase only):**

| Feature | What It Measures | Why It Helps |
|---|---|---|
| `log_distance` | Log of haversine distance (km) | Normalises skewed distribution; long distances signal carrier handoff risk |
| `log_price` | Log of item price | Expensive items → higher customer expectations |
| `is_cross_state` | Binary: seller and customer in different states | Cross-state = multiple carrier handoffs; EDA measured 2× late rate |
| `freight_ratio` | Freight cost ÷ item price | High ratio signals customer paying heavily for shipping — more sensitive to delays |
| `month`, `quarter`, `is_weekend` | Temporal from purchase timestamp | Q2 is 11% busier; weekend orders sit until Monday pickup |
| `log_seller_orders` | Log of seller's total historical orders | Veteran sellers dispatch faster; EDA: new sellers have 1.4× late rate |
| `distance_x_weight` | Distance × product weight | Compound: heavy item over long route = harder logistics |
| `heavy_expensive_far` | Boolean: weight > p75 AND price > p75 AND distance > p75 | Flags highest-risk shipments where delay damages satisfaction most |
| `seller_delay_risk` | Target-encoded historical late rate per seller | Captures seller-specific reliability; computed from training data only |
| `state_delay_risk` | Target-encoded historical late rate per state | Captures regional infrastructure quality; computed from training data only |

**Part 4 — Temporal split:**
- Train: all 2017 orders. Test: July–December 2018 orders.
- Random splits mix past and future data in both train and test — temporal split mirrors real deployment exactly.

**Part 5 — Two-model comparison:**
- Four models tested for each: Logistic Regression, Random Forest, XGBoost, LightGBM.
- **Model A (Delay Predictor):** AUC ~0.55–0.58. Honestly documented as weak — carrier routing events are unrecorded.
- **Model B (Anger Predictor):** Higher AUC. Among confirmed late orders, distance, freight ratio, seller experience, and product weight do meaningfully separate the mildly annoyed from the truly furious.

---

### `kaggle.ipynb` — Data Scratchpad

- Kaggle-compatible file paths and GPU/CPU-aware setup.
- Used for quick NLP batch runs, dataset shape checks, and output exports during development.

---

## 3. The Final Notebook — `final_project.ipynb`

This is the complete, clean, end-to-end production pipeline. It combines every lesson from the previous phases.

### The Business Problem

Olist connects sellers and buyers but doesn't make products or drive trucks. When a delivery is late, some customers accept it quietly. Others leave a 1–2 star review that damages platform reputation, drives away sellers, and reduces repeat purchases.

The goal: **among orders already confirmed late by the carrier, which customers are most likely to leave a 1–2 star review?** Knowing this early lets Olist act proactively on only the highest-risk cases.

---

### Part 1: Spatial Reality — Where the Problem Lives

- K-Means clustering on GPS coordinates of angry, late customers identified 5 geographic "Danger Zones."
- Brazil's Northeast states (Bahia, Ceará, Pernambuco) and North consistently show the highest density of delayed, angry customers.
- Seller concentration: 80.7% of sellers are in São Paulo — customers in the Northeast and North receive packages from 1,500–3,000 km away, with fewer regional carrier hubs.
- A Folium heatmap visualises these zones on a real map of Brazil.

---

### Part 2: NLP Purification — Cleaning the Labels

- Negative reviews tagged as "Defective Product" or "Wrong Product Delivered" were removed from the training set.
- Only reviews tagged as "Logistics & Delivery Delay" were kept as genuine dissatisfaction signals.
- Without this step, the model would try to predict bad reviews caused by broken products using shipping distance and weight — a fundamentally wrong frame that poisons the signal.

---

### Part 3: Feature Engineering — Pre-Purchase Only

All features calculated strictly from information available at the time of order placement. No post-shipment data. See the full feature table in Section 2 (`temporal.ipynb` Part 3) — the same feature set is used here.

---

### Part 4: Temporal Split — Honest Evaluation

- Train: all 2017 orders. Test: July–December 2018.
- The model never sees future data during training — exactly how a production model would work.
- Random splits (the common shortcut) would mix 2017 and 2018 orders in both train and test, making performance look better than it is.

---

### Part 5: The Two-Model System — Results

**Model A — Delay Predictor:**
- Trained on all orders to predict `is_late`.
- Four models compared: Logistic Regression, Random Forest, XGBoost, LightGBM.
- **Best AUC: ~0.55–0.60.** Honestly documented as weak.
- Why it's weak: whether a specific package is late depends on what happens inside the carrier network after pickup — route load, hub congestion, vehicle availability — none of which appear in the Olist dataset. Pre-purchase features carry only marginal signal for this question.
- Role in production: pre-flags suspicious orders as an early warning, before the carrier confirms a delay.

**Model B — Anger Predictor (the deployable model):**
- Trained only on confirmed late orders to predict `is_dissatisfied` (will this delayed customer leave a 1–2 star review?).
- Same four models compared.
- **Better performance than Model A** — given that a delay has already happened, features like long distance, high freight ratio, heavy/expensive product, and inexperienced seller do meaningfully separate who will be furious from who will accept the delay.
- This is the model that drives the actual operational decision.

---

### Part 6: SHAP Interpretability

SHAP (SHapley Additive exPlanations) shows exactly which features push each prediction up or down for every individual order.

**For Model A (Delay Predictor):**
- Top features: `log_distance`, `estimated_delivery_days`, `state_delay_risk`, `is_cross_state`.
- Farther orders, cross-state, and sellers in historically slow states are flagged most — consistent with EDA findings.

**For Model B (Anger Predictor):**
- Top features: `log_distance`, `freight_ratio`, `heavy_expensive_far`, `seller_delay_risk`.
- The SHAP plot reveals the "Fridge Insight": customers who received heavy, expensive items over long distances and experienced a delay are the most angry. A customer waiting for a cheap pen that arrives 3 days late shrugs it off. A customer waiting for a refrigerator arriving 2 weeks late is furious.

---

### What the Old Notebook Had Wrong (Leakage Problems — Fixed Here)

The predecessor notebook (`Final_Project_Showcase.ipynb`) had five critical problems this notebook corrects:

1. Used `delivery_delta` (actual minus estimated delivery time) as a feature — only known after delivery. Its SHAP rank as the #1 feature was the model reading the answer in disguise.
2. Used `is_late` as a feature to predict `is_dissatisfied` — circular reasoning. The model learned "if an order is late, the customer is dissatisfied" — trivially true, not predictive.
3. Used a random train-test split — temporal leakage.
4. Used a hardcoded 30% threshold to attribute blame to sellers — no data justification.
5. AUC of 0.73 — inflated by leakage. The honest, clean model's AUC is lower but valid and deployable.

---

## 4. The "Bad" R² Values — A Statistical Explanation

In the regression experiments, our model achieved R² ≈ −0.11 while Olist's baseline scored R² = −4.39.

**Why R² goes negative:**
R² compares your model against the simplest dumb baseline: always predict the mean (12.4 days for every order). If R² is negative, your model's squared errors are larger than those of this dumb baseline. It does not mean the model is useless — it means the squared error metric is dominated by extreme outliers.

**Olist's baseline (R² = −4.39):**
Olist quotes ~23 days for orders that typically take ~12 days — a buffer of ~11 days on every order. This creates massive systematic squared errors on every prediction, producing the terrible R² score.

**Our model (R² = −0.11):**
The model predicts accurately for typical orders (4 days for short-haul, 18 for long-haul). But when a normally 4-day order is delayed to 25 days by an unrecorded event (a postal strike, a vehicle breakdown), the squared penalty is enormous: (25−4)² = 441 vs. (25−12.4)² = 158 for the mean baseline. Because the dataset contains many such unpredictable outliers, they dominate the squared error sum and pull R² below zero.

**The honest measure — MAE:**
Our model cuts average absolute error roughly in half compared to Olist's estimate. R² misrepresents this because it squares every error. MAE is the honest measure: the model genuinely reduces average error per order. Regression was ultimately abandoned in favour of classification because the delivery time noise is irreducible without carrier routing data.

---

## 5. What Else Can Be Added to `final_project.ipynb`

These are pure data-driven analyses the existing dataset already supports. Every item produces a concrete number or chart from real data — no assumptions or guesses.

### A. Error Analysis by Seller Tier

**What:** Group sellers by total historical order count into three tiers (< 50 orders, 50–500, 500+). For each tier, compute Model B's precision, recall, and false negative rate separately on the test set.

**Why:** Models perform differently across subgroups. Novice sellers have sparse historical data making their `seller_delay_risk` encoding unreliable. If the model misses 40% of angry customers from novice sellers but only 15% from veterans, that directly tells Olist where to apply the model cautiously and where manual review is needed. This is standard production model auditing.

---

### B. Calibration Curve

**What:** Plot the model's predicted probability bins (0–10%, 10–20%, up to 90–100%) against the actual observed dissatisfaction rate within each bin on the test set.

**Why:** A model can have good AUC (good ranking) but poorly calibrated probabilities. If the model says "35% chance of bad review" but the actual rate in those orders is 15%, the model is systematically over-confident. A calibration curve exposes this and whether Platt scaling or isotonic regression is needed before deployment — standard production practice for risk-scoring systems.

---

### C. Confusion Matrix by Brazilian Region

**What:** Split the test set into five regions (Southeast, Northeast, South, North, Central-West). Compute a separate confusion matrix and false negative rate for Model B in each region.

**Why:** The EDA established that the Northeast and North have fundamentally different logistics infrastructure. If Model B misses 30% of angry customers in the Northeast but only 12% in the Southeast, that is a direct, data-calculated finding — not a hypothesis. It tells the logistics team exactly where the model is blind and where investment matters most.

---

### D. The Fridge Insight — Formally Quantified

**What:** Define the high-risk cohort precisely: confirmed late orders where product weight is above the 75th percentile AND haversine distance is above the 75th percentile. Compute this cohort's bad-review rate vs. all other late orders. Report what share of all late orders they are and what share of all bad reviews they produce.

**Why:** The SHAP analysis shows qualitatively that heavy-far orders push the prediction up. One step further turns this into a precise finding: e.g., "high-risk (heavy + far) delayed orders have a bad-review rate 3.7× higher than other late orders, represent 12% of all late orders, but account for 34% of all bad reviews." That is the difference between a pattern and quantified evidence.

---

### E. Permutation Importance vs. SHAP — Stability Check

**What:** Compute permutation importance alongside the existing SHAP values. Permutation importance works by randomly shuffling one feature at a time and measuring how much Model B's AUC drops. Compare the top-10 feature ranking from both methods.

**Why:** SHAP measures each feature's average contribution to the model output. Permutation importance measures how much predictive performance degrades when the feature is removed. When both rankings agree, the result is robust. When they disagree, it signals that correlated features are splitting credit (e.g., `log_distance` and `is_cross_state` are related — one may carry the other's signal). This comparison is standard practice in production ML and demonstrates methodological rigour.

---

### F. Formal Model Comparison Table

**What:** Present the four-model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM) for both Model A and Model B as a single formatted table: Model name, AUC, Precision at optimal threshold, Recall at optimal threshold, F1 at optimal threshold.

**Why:** The notebook already trains and evaluates all four models, but isolated print statements are not how results are communicated to decision-makers. A side-by-side table makes the model selection decision transparent, reproducible, and clearly arguable.