# Project: Olist Proactive Retention (The "Red Flag" Detector)

## Phase 1: Empirical Foundation (Layer 1) - COMPLETED
- [x] **Binary Classification:** Switched from Regression (Score 1-5) to Classification (Happy vs. Unhappy).
- [x] **Advanced Features:** Integrated Haversine distance, `delivery_delta` (Promise Gap), and `prep_to_delivery_ratio`.
- [x] **Imbalance Handling:** Implemented `scale_pos_weight` and compared with SMOTE.
- [x] **Target Encoding:** Used smoothed encoding for high-cardinality features (Seller/Category).

## Phase 2: High-Value Economic Modules (Layer 2) - IN PROGRESS
- [x] **Module A: CLV-Weighted Loss Analysis:**
    - Calculated Customer Lifetime Value (Historical Spend).
    - Weighted the loss (R$155 base) by customer CLV.
- [x] **Module B: Resource Constraint Simulation (The "500 Slots" Rule):**
    - Implemented prioritization logic.
    - Calculated "Lift" vs. random outreach.
- [x] **Module C: Responsibility Attribution (Seller vs. Carrier):**
    - Used `prep_to_delivery_ratio` to attribute blame for "Red Flags."
- [x] **Total Cost Optimization:**
    - Final tuning of the Economic Optimal Threshold.

## Phase 3: Optimization & Sensitivity Analysis
- [x] **Total Cost vs. Threshold:** Find the Economic Optimal Threshold (Minimizing weighted cost).
- [x] **Scenario Stress-Testing:** 
    - "What if the cost of a voucher doubles?" 
    - "What if the churn rate of high-value customers is 2x higher?"
    - Show that the model *still* provides a positive ROI under these perturbations.

## Phase 4: Final Delivery
- [x] **Final Showcase Notebook:** `final_business_impact_model.ipynb`.
- [x] **Visualizations:**
    - "The Money Slide": Cumulative Loss Avoided by using the model vs. baseline.
    - "Responsibility Map": Sellers with the highest "Red Flag" rates due to prep delays.
    - "Regional Risk Heatmap": States where logistics expectations are failing.

---
### Senior Insights (Grade Drivers):
1. **Expectation Gap > Speed:** Satisfaction is driven by the gap between promised and actual delivery, not just raw transit time.
2. **Economic Thresholding:** We optimize for **Total Cost** (R$), not just F1-score.
3. **Proactive Mitigation:** The model's value is in early intervention (apology/vouchers) before the 1-star review hits.
4. **Weighted Retention:** Prioritizing high-value "Whale" customers over one-time buyers.
