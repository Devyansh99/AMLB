# Mid-Semester Progress Report

## Project Title
Late Shipment Risk and Delivery Margin Modeling for Brazilian E-Commerce (Olist)

## 1) Progress: What We Completed So Far

### 1.1 Data preparation and cleaning status
We have moved from raw merged data to a modeling-ready dataset and built reproducible modeling notebooks.

Completed items:
- Loaded and validated the cleaned merged dataset in the modeling pipeline.
- Converted key timestamp columns to datetime.
- Created regression targets:
  - `actual_delivery_days`
  - `estimated_delivery_days`
  - `delivery_margin_days` (positive means early/on-time margin, negative means delayed).
- Performed target safety filtering to remove invalid/extreme target values.
- Implemented a strict temporal split (train/validation/test) instead of random split.
- Added leakage controls by excluding post-outcome fields from model features.
- Explicitly removed `delivery_delay_days` from regression features after identifying it as leakage-prone for margin prediction.

Verified current dataset/modeling stats from the executed notebook:
- Loaded shape: `96,189 x 34`
- Late rate: `7.94%`
- Modeling shape after cleaning: `96,142`
- Split sizes:
  - Train: `67,299`
  - Validation: `14,421`
  - Test: `14,422`
- Feature composition in regression notebook:
  - Total features: `28`
  - Numeric: `22`
  - Categorical: `6`

### 1.2 Feature engineering and modeling work completed
Completed feature/pipeline work:
- Numeric and categorical preprocessing pipelines.
- Missing-value imputation.
- One-hot encoding for categorical variables.
- Standardization for numeric branch.
- Leakage-safe feature selection logic.

Completed model work:
- Classification path (depth notebook):
  - Multiple models and threshold optimization based on business cost.
  - Calibration and constrained-threshold logic were added.
- Regression path (new notebook):
  - ElasticNet baseline.
  - RandomForestRegressor.
  - XGBoost regressor (available in environment and executed).
  - Validation-based model selection.
  - Test evaluation with MAE/RMSE/R2.
  - Conversion of regression output into calibrated late-risk probabilities.
  - Capacity-constrained decision policy (max alert rate 20%).

## 2) Preliminary Findings (Concrete Results)

### 2.1 Classification finding (baseline direction is weak)
From the previous classification notebook outputs (`Project_03_Modeling.ipynb`), the Logistic Regression minority-class performance remains weak despite imbalance handling attempts:
- Recall (late class): `0.4917`
- Precision (late class): `0.1353`
- Approximate F1 for late class: `~0.21`

Interpretation:
- The model catches some delayed orders but raises many false alerts.
- This is not operationally reliable without stronger ranking/separation and better thresholding under capacity limits.

### 2.2 Regression finding (best current model and temporal drift)
In `03_regression_margin.ipynb`, model comparison on validation:

| Model | Val MAE | Val RMSE | Val R2 |
|---|---:|---:|---:|
| xgb_reg | 5.0666 | 7.3167 | 0.4423 |
| rf_reg | 5.8780 | 7.8629 | 0.3559 |
| elastic_net | 6.9945 | 8.2816 | 0.2855 |

Best model selected: `xgb_reg`

Test metrics for selected model:
- Test MAE: `5.1173` days
- Test RMSE: `7.3109` days
- Test R2: `0.2304`

Interpretation:
- The model predicts average delivery margin with moderate error but still misses a large portion of delivery variation.
- The drop from validation R2 (`0.4423`) to test R2 (`0.2304`) indicates meaningful temporal drift/generalization risk.

### 2.3 Economic/policy finding under operations constraint
Using calibrated risk from regression outputs and a 20% intervention cap:
- Best threshold: `0.18`
- Alert rate: `17.78%`
- Precision: `18.41%`
- Recall: `34.86%`
- Business cost (FP*15 + FN*155): `168,090`

Risk bands (quantile-based) show monotonic lift in late rate:
- Low: `1.07%`
- Medium: `4.15%`
- High: `15.51%`
- Critical: `18.99%`

Interpretation:
- Ranking quality exists (critical band is riskier than low band), but absolute precision is still low for direct action at scale.

## 3) Issues and Why Current Performance Is Still Poor

### 3.1 Severe class imbalance
- Late orders are only about 8%, so naive or weakly separated models default toward non-late predictions.
- This hurts minority precision/recall stability.

### 3.2 Feature signal is still incomplete
- Current features capture basic order/product/payment/geolocation structure.
- Missing high-signal operational context: route reliability history, seller handoff behavior, and dynamic congestion signals.

### 3.3 Temporal instability
- Regression quality degrades out-of-time (validation to test drop), indicating that patterns shift over time.
- Random-split-style assumptions would overestimate real-world performance.

### 3.4 Decision-quality gap under capacity constraints
- Even when models are calibrated, precision at practical alert caps remains low.
- This creates costly trade-offs between false alarms and missed true delays.

### 3.5 Leakage risk (identified and mitigated)
- A leakage-prone feature (`delivery_delay_days`) had to be removed from regression features.
- This correction was necessary for realistic metrics.

## 4) Revised Plan (What Remains, Scope Update, Biggest Risk)

### 4.1 What remains
1. Improve feature signal with logistics-aware engineered features:
- `product_volume_cm3`
- `is_bulky`
- `freight_ratio`
- route-level historical risk (state-to-state lane)
- seller reliability score
- temporal indicators (`day_of_week`, `is_weekend`, peak-season flags)

2. Improve classification performance for minority class:
- Compare cost-sensitive XGBoost (`scale_pos_weight`) against current baselines.
- Evaluate PR-AUC and late-class F1 as primary metrics (not only ROC-AUC).
- Re-test calibrated thresholds and top-K policy under fixed capacity.

3. Improve regression robustness:
- Test target transform `log(actual_delivery_days + 1)` for stability.
- Add rolling lag context (recent delay patterns by route/seller-state).

4. Strengthen validation protocol:
- Keep strict out-of-time test window.
- Perform rolling/temporal validation to check drift sensitivity.

### 4.2 Scope changes since interim
- Scope is now more explicitly decision-centric.
- Instead of only “predict late vs not late,” we are now doing:
  - regression for margin size,
  - calibrated risk conversion,
  - constrained intervention policy.

### 4.3 Biggest open risk
The biggest risk is insufficient predictive signal under real temporal drift. If richer route/seller/time features do not materially improve ranking quality (especially precision at capped alert volume), the final system may remain weak for operational deployment.

## 5) Best Next Actions to Take Right Now (Prioritized)
From the long list of candidate improvements, the highest-value and most feasible steps for this project are:

1. Route risk + seller reliability encoding (highest expected lift).
2. Cost-sensitive XGBoost classification with `scale_pos_weight` and PR-AUC/F1 optimization.
3. Top-K / capacity-capped triage policy with isotonic calibration.
4. Temporal rolling validation to quantify and control drift.

These four steps directly target the current bottlenecks (imbalance, weak ranking precision, and out-of-time degradation) while staying aligned with the business intervention constraint.

---

## Appendix: Evidence Sources (executed notebooks)
- `Project/brazil_ecommerce/notebooks/03_depth_ml.ipynb`
- `Project/brazil_ecommerce/notebooks/03_regression_margin.ipynb`
- `Late_Shipment_Prediction_Olist/Project_03_Modeling.ipynb`
