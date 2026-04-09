# Next Steps — What Remains to Push This to Distinction Level

## Context

`claude.ipynb` completes the core work (NLP decomposition, two-model ML pipeline, causal inference,
survival analysis, fridge insight, warehouse mapping, sensitivity ROI). This document captures what
a full Senior DS portfolio piece would add — with *specific reasoning* for why each step matters.

---

## Priority 1: Operations Research — True Facility Location Optimisation

**What we did:** K-Means on angry-late customer coordinates → 3 baseline DC locations.

**The gap:** K-Means minimises Euclidean (straight-line) distance. Road logistics don't work that way.
A DC placed at the geographic centroid of a cluster may be 3× farther by road due to highway networks.

**What to build:**
- Use **OR-Tools** (Google) or **PuLP** to formulate the *p-median* or *uncapacitated facility location* integer program.
- Inputs: seller locations (demand supply nodes), customer zip codes (demand nodes), actual O-D road distances from OpenRouteService API.
- Constraints: max budget for N new DCs, capacity per DC.
- Output: The exact GPS coordinates of the mathematically optimal N new DCs to minimise total expected delay-weighted delivery distance.

**Why this changes a decision:** The CFO asks "where do we spend $50M on the next DC?" K-Means gives a
hand-wavy answer. An OR solution gives a defensible, optimal coordinate with expected delay reduction in %.

---

## Priority 2: A/B Test Design — Validate the Voucher Assumption

**What we assumed:** Proactive R$ 15 voucher prevents churn with 40% probability.

**The gap:** This is an industry benchmark, not measured in Olist's market. If the true effectiveness
is 20%, the ROI flips negative for some thresholds.

**What to build:**
- Design a proper randomised experiment: among orders flagged by the model, randomly assign 50% to
  receive the voucher and 50% to a holdout.
- Primary metric: 90-day repeat purchase rate.
- Secondary: NPS score on next purchase, review score on the flagged order.
- Sample size calculation: with α=0.05, power=0.80, and a baseline repeat rate of 35%, we need ~1,200
  orders per arm to detect a 5pp lift.

**Why this changes a decision:** Without this, every ROI number we presented is an assumption. With it,
we replace R$ 155 and 40% with measured values — and can re-optimise the threshold in real time.

---

## Priority 3: Carrier-Level Features — Fix Model A's Low AUC

**What we have:** Model A ROC-AUC ~0.55–0.60.

**The gap:** Whether an order is late depends mostly on the *carrier* (Correios route, third-party
logistics partner) and the specific *origin–destination pair*, not just the straight-line distance.

**What to build:**
- Integrate Correios API (rastreamento) to get historical per-route delay rates.
- Feature: `carrier_route_historical_delay_rate` (the most predictive feature for delivery timing).
- Feature: `seller_fulfillment_speed_days` — how many hours between order and carrier pickup.
- Expected AUC improvement: empirically, route-level features push delay AUC to 0.70–0.75 in similar
  e-commerce contexts.

**Why this changes a decision:** A Model A with AUC 0.70 flags fewer false positives → fewer unnecessary
vouchers → direct cost saving (estimated R$ 30–50 per 1,000 orders at scale).

---

## What Is Already Complete in `claude.ipynb`

| Component | Status | Location in Notebook |
|-----------|--------|----------------------|
| Business framing & cost structure | ✅ Done | Part 1 (markdown) |
| NLP root-cause decomposition | ✅ Done | Part 2 |
| Feature engineering (leakage-free) | ✅ Done | Part 3 |
| Temporal train/test split | ✅ Done | Part 4 |
| 4-model comparison (Model A) | ✅ Done | Part 5 |
| 4-model comparison (Model B) | ✅ Done | Part 5 |
| SHAP interpretability | ✅ Done | Part 6 |
| ROI & threshold optimisation | ✅ Done | Part 7 |
| Sensitivity analysis (3 scenarios) | ✅ Done | Part 7 |
| Fridge Insight empirical proof | ✅ Done | Part 8 |
| Causal inference (IPW + bootstrap CI) | ✅ Done | Part 9 |
| Survival analysis (KM + Cox PH) | ✅ Done | Part 10 |
| Warehouse heatmap (Folium) | ✅ Done | Part 11 |
| Decision-maker action plan | ✅ Done | Part 12 |
| Honest limitations table | ✅ Done | Part 12 |

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `claude.ipynb` | The definitive notebook — everything above in one place |
| `final.ipynb` | Original single-model attempt (AUC 0.58, ROI R$ -5) — kept for comparison |
| `final_v2.ipynb` | Predecessor to claude.ipynb (two-model pipeline, no causal/survival) |
| `review_root_cause_nlp.ipynb` | Standalone NLP exploration used to build Part 2 |
| `kaggle.ipynb` | Dataset exploration scratch work |
| `next_steps.md` | This file |