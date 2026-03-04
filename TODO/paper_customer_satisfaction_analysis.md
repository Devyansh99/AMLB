# Research Paper Analysis — ML for Predicting Customer Satisfaction
**Source:** "Machine Learning for Predicting Customer Satisfaction in E-Commerce"  
**Authors:** Archi Mangla (Shiv Nadar University) & Ashu Soni (Bharati Vidyapeeth College of Engineering)  
**Published:** Dec 2025 — International Journal for Research Publication and Seminar, Vol. 16, Issue 4  
**DOI:** https://doi.org/10.36676/jrps.v16.i4.337  

---

## What This Paper Did (Summary)

Uses the same 8-table Olist dataset to **predict customer satisfaction** (binary: satisfied vs not) using RF, HistGradientBoosting, XGBoost, and LightGBM, plus two ensemble strategies. Trained on 100k+ balanced records, best accuracy reached **82.53%** (Soft Voting Ensemble).

**Key distinction from our project:** Their target is `review_score` (satisfied/not satisfied). Our target is `is_late` (delivered after estimated date). But the paper explicitly confirms that **delivery delay is the #1 driver of their satisfaction target** — so our model predicts the proximate cause of their output. Our project sits one step upstream.

---

## Section 1 — Framework and Data

### Merge pipeline (same 8 tables, same keys)
- Joined on `order_id`, `customer_id`, `seller_id`
- Added English category names via `product_category_name_translation.csv`
- Used Haversine for customer–seller distance (same choice as Repo 3)

### Feature engineering (from the paper)

| Feature Group | Features Used | Overlap with Our Project |
|---|---|---|
| Delivery timings | `delay_duration`, `approval_time`, `carrier_time` | ⚠️ LEAKAGE — all post-purchase; we must not use these |
| Payment behavior | `payment_value`, `installment_count` | ✅ We use both |
| Product characteristics | `category`, `dimensions (vol/weight)` | ✅ We use both |
| Distance | Haversine customer–seller distance | ✅ We use `distance_km` |
| Temporal | `year`, `month`, `weekday`, `hour` | ✅ We use all |

→ **Critical note:** The paper lists `delay_duration` and `carrier_time` as features. Both are computed from post-delivery timestamps. This is direct leakage — you cannot know actual delivery delay at order placement time. This inflates their results. **We do not use any post-purchase timestamps as features.**

### "Balanced 100k records" — what this means
The paper says "more than 100,000 balanced records." The raw Olist dataset has ~75% 4-5 star reviews (satisfied class). To get balanced data they must have used oversampling (SMOTE or similar) or undersampling. They don't specify which method. This is a methodology gap — their 82.53% accuracy on balanced test data would be lower on the real-world imbalanced distribution.

---

## Section 2 — EDA Observations (Figures 2–13)

| Figure | What It Shows | Relevant to Us |
|---|---|---|
| Fig 2 — Orders per Month | Monthly growth + Nov 2017 spike | ✅ Confirms seasonal features |
| Fig 3 — Top Categories by Revenue | Furniture, electronics dominate revenue | ✅ Confirms category encoding |
| Fig 4 — Top Categories by Items Purchased | Health & beauty, toys dominate volume | ≠ Revenue categories — both signals needed |
| Fig 5 — Price vs Freight Value | Positive correlation, non-linear | ✅ Confirms `freight_to_value_ratio` |
| Fig 6 — Payment Method | 75%+ credit card | ✅ Confirms `payment_type` as low-variance feature |
| Fig 7 — Review Score Distribution | Left-skewed toward 5-stars (~50% score=5) | Explains why their task is hard — imbalanced |
| Fig 8 — Avg Price vs Review Score | Higher price → slightly lower satisfaction | New: `high_value_order` flag may be relevant |
| Fig 9 — Order Status Distribution | Majority delivered, small % cancelled | ✅ We filter to `delivered` only |
| Fig 10 — Delivery Delay Distribution | Left-skewed — most delivered early vs estimate | ✅ Confirms our buffer logic; most are on-time |
| Fig 11 — Orders per Customer | Most customers buy once (1 order) | Repeat purchase proxy useful |
| Fig 12 — Recency vs Number of Orders | Newer customers order less frequently | Behavioral signal for satisfaction, not for lateness |
| Fig 13 — Customer Cohort Heatmap | Retention drops sharply after first month | Economic context: reducing late rates improves cohort retention |

### Key EDA insight from Fig 8 (Avg Price vs Review Score):
Higher-priced products get **slightly lower review scores** on average — likely because customer expectations scale with price. This is relevant: expensive orders that arrive late → worst satisfaction outcome. Feature to derive: `is_high_value_and_late` interaction — but for our classification, simply: flag high-value orders for priority treatment with lower threshold.

---

## Section 3 — Model Setup

### Split: 60% train / 20% val / 20% test
```
60% training → 20% validation → 20% test
```
→ **This is a RANDOM split.** For their customer satisfaction task (predicting review scores), random split is slightly more defensible because reviews are not strongly time-dependent. For our task (late delivery prediction), temporal split is mandatory — seller and route behavior changes over time.

### Models and why chosen
- **Random Forest (1000 trees):** Ensemble baseline, handles non-linear interactions
- **HistGradientBoosting:** Scikit-learn native GBM, handles missing values natively (no NaN imputation needed)
- **XGBoost:** Standard GBM with L1/L2 regularization
- **LightGBM:** Leaf-wise growth, faster on large datasets, handles categorical natively

→ **HistGradientBoosting** (sklearn's `HistGradientBoostingClassifier`) is a model we haven't planned. It's essentially gradient boosting with histogram binning — nearly as fast as LightGBM without the LightGBM installation complexity. Worth adding as our 5th model.

### Threshold tuning (per model)
```
Random Forest:        threshold = 0.500 (default)
LightGBM:             threshold = 0.526
HistGradientBoosting: threshold = 0.496
XGBoost:              threshold = 0.482
Voting Ensembles:     threshold = 0.510 – 0.535
```
→ **They swept thresholds** — same as our plan (cost-matrix-based threshold sweep). Their thresholds range from 0.482 to 0.535 — very close to 0.5. Our thresholds will be much lower (~0.10–0.20) because FN cost (R$155) >> FP cost (R$15) means we want high recall → always push threshold down. Their task is balanced precision/recall so most thresholds stay near 0.5.

---

## Section 4 — Full Results Table

### Table A.1 — Default Models (threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (1000 trees) | 0.8249 | 0.8287 | 0.9743 | 0.8956 | 0.7302 |
| HistGradientBoosting | 0.8244 | 0.8282 | 0.9742 | 0.8953 | **0.7389** |
| XGBoost | 0.8234 | 0.8286 | 0.9719 | 0.8945 | 0.7319 |
| LightGBM | 0.8248 | 0.8290 | 0.9735 | 0.8955 | 0.7321 |

### Table A.3 — Tuned Voting Ensembles (best entries)

| Ensemble | Accuracy | Precision | Recall | F1 | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| Voting (1,2,5,6) | 0.8252 | 0.8291 | 0.9740 | 0.8957 | 0.7370 | 0.510 |
| Voting (1,3,6,7) | **0.8254** | 0.8287 | **0.9750** | **0.8959** | **0.7373** | 0.514 |
| Voting (1,4,5,6) | 0.8247 | **0.8298** | 0.9720 | 0.8953 | 0.7383 | 0.525 |

### Table A.4 — Precision-Weighted MetaVote

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| MetaVote (top 3, equal weights) | 0.8251 | 0.8293 | 0.9735 | 0.8956 |

**Key takeaway:** All models cluster between 0.82 and 0.83 accuracy. The **spread is only 0.0021** across all approaches. This is NOT because the models are equivalent — it's because the feature set is so strong (delivery delay as a feature is ~perfectly correlated with review score) that all models hit the ceiling. The 0.73 ROC-AUC is actually the honest discriminative measure, and it shows the problem is harder than 82% accuracy suggests.

---

## Section 5 — Ensemble Strategies

### Strategy 1: Soft Voting Ensemble
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('hgb', hgb_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft'   # average probabilities, not majority vote
)
ensemble.fit(X_train, y_train)
proba = ensemble.predict_proba(X_test)
```
→ **Soft voting > hard voting** for our task: we need probabilities for threshold optimization. Hard voting produces binary outputs — useless for the cost-matrix sweep.

### Strategy 2: Precision-Weighted MetaVote
```python
# Step 1: train all base models on train set
# Step 2: compute precision score for each on validation set
# Step 3: use precision scores as weights in soft voting

val_precisions = {
    'rf':   precision_score(y_val, rf.predict(X_val)),
    'hgb':  precision_score(y_val, hgb.predict(X_val)),
    'xgb':  precision_score(y_val, xgb.predict(X_val)),
    'lgbm': precision_score(y_val, lgbm.predict(X_val)),
}
total = sum(val_precisions.values())
weights = [v / total for v in val_precisions.values()]

# weighted average of probabilities
final_proba = sum(w * model.predict_proba(X_test)[:,1] 
                  for w, model in zip(weights, models))
```
→ **This is the novel contribution of the paper.** Precision-weighted MetaVote emphasizes models that are more precise on validation → reduces false positives. For **our task we want high recall** (minimize FN), so adapt this to: **Recall-Weighted MetaVote** — weight each model by its validation recall score. This ensures the ensemble is dominated by the model that best catches actual late deliveries.

**Adaptation for our cost matrix:**
```python
# Cost-weighted MetaVote: weight by cost-minimizing threshold performance on val
def cost_score(y_true, y_proba, fp_cost=15, fn_cost=155):
    best_cost = float('inf')
    for thresh in np.arange(0.05, 0.50, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost = cost
    return 1 / best_cost  # higher is better → use as weight

weights = [cost_score(y_val, model.predict_proba(X_val)[:,1]) for model in models]
weights = [w / sum(weights) for w in weights]
```
→ **This is our improved MetaVote.** Instead of precision-weighting, weight by which model minimizes our specific cost on validation. This is directly aligned with our business objective.

---

## Section 6 — Key Insights from the Paper

### Insight 1: Delivery is the #1 predictor of satisfaction
> "Features that affect delivery are most likely to be predictive. There is a direct correlation between logistic delays and low scores on the reviews."

→ **This is the academic citation we need for our report.** Cite this paper (published Dec 2025, DOI above) when claiming: "Late delivery is the primary driver of negative customer experience on the Olist platform."

### Insight 2: Ensemble gives +0.3% accuracy over single models
> "Ensemble learning gives an extra +0.3 accuracy and recall rate over single models."

→ For our project, the gain from ensembling will likely be in **cost reduction**, not raw accuracy. Worth trying but don't over-invest — the paper confirms the improvement is real but marginal.

### Insight 3: All four boosting/ensemble models are nearly equivalent
→ The 0.0021 spread across all models means feature engineering matters far more than model selection. Invest time in feature quality (distance, temporal, seller history) not in hyperparameter tuning of each individual model.

### Insight 4: HistGradientBoosting has the highest ROC-AUC (0.7389)
→ Among single models, HGB has the best discriminative power (ROC-AUC = 0.7389 vs RF = 0.7302). This is the metric that matters for threshold optimization, not accuracy. **Add HistGradientBoosting to our model suite.**

---

## What We Can Take From This Paper

### ML Perspective

| Item | Paper's Approach | What We Add / Adapt |
|---|---|---|
| **Soft Voting Ensemble** | Average probabilities of 4 models | Implement after individual models trained |
| **Precision-Weighted MetaVote** | Weight by val precision | Adapt to **Recall-Weighted or Cost-Weighted MetaVote** (aligns with FN > FP cost) |
| **HistGradientBoosting** | Added as 5th model candidate | `sklearn.ensemble.HistGradientBoostingClassifier` — handles NaN natively |
| **60/20/20 split structure** | Random split | We use temporal split with same proportions (~60/20/20 by time) |
| **Threshold tuning per model** | Sweep 0.45–0.55 | We sweep 0.05–0.50 (cost-matrix weighted, much lower range expected) |
| **4-model comparison table** | Accuracy / Precision / Recall / F1 / AUC | Add **cost metric** column: `FP×15 + FN×155` per threshold |
| **Balanced training data** | 100k balanced records (method unspecified) | We use class_weight='balanced' or SMOTE on train fold only; evaluate on TRUE imbalanced distribution |

### Business Perspective

| Paper's Claim | How We Use It |
|---|---|
| "Delivery delay is the most important predictor of satisfaction" | **Direct citation in our business section.** Motivates Priority Handling: if we flag at-risk deliveries proactively and prevent delays, we protect review scores and customer retention. |
| "Ensemble models are applicable to operational dashboards" | Our recommendation: deploy LightGBM or Voting Ensemble as a live scoring API — each new order gets scored at placement → triggers Priority Handling if P(late) > threshold |
| "100k balanced records → 82.53% accuracy" | Provides a ceiling reference: with delivery delay AS a feature (i.e., with leakage), 82.53% is achievable. Without that leakage (our setup), our models will have lower accuracy but are honest. |
| Customer cohort heatmap (Fig 13) — retention drops sharply | Economic narrative: one late delivery reduces the probability of repeat purchase. FN cost (R$155) should be understood as: cost of Priority Handling avoided + expected lifetime-value loss from failed retention. |

### Economic Perspective

| Gap in Paper | What They Missed | What We Add |
|---|---|---|
| **No cost matrix** | No mention of FP/FN costs | We frame every prediction in R$: `FP×15 + FN×155`. The paper gives us the ceiling — we give the monetization layer. |
| **No late delivery rate reduction quantified** | Shows satisfaction prediction accuracy but not how their model would reduce late rates | We directly predict `is_late` and report: "flagging X% of orders for Priority Handling prevents Y% of late deliveries, saving R$Z annually" |
| **No explanation of how business acts on prediction** | They mention "operational dashboard" but no decision flow | We define the decision: if P(late) > threshold → assign Priority Handling → R$15 cost; if missed → R$155 cost. |
| **Random split hides temporal decay** | Model tested on random sample from same time window as training | Our temporal split reveals whether model generalizes to future orders — more honest business deployment scenario |

---

## New Items to Add to Our Project (Action Items)

### New model: HistGradientBoosting (5th model)
```python
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    l2_regularization=0.1,
    random_state=42,
    class_weight='balanced'   # handles imbalance natively
)
```
**Why:** HGB has the highest ROC-AUC in the paper (0.7389), handles missing values without imputation, and runs faster than XGBoost on scikit-learn pipelines. It's the most underrated model in the comparison.

### New ensemble: Cost-Weighted MetaVote
After training all base models, derive a final ensemble weighted by which model minimizes cost on the validation set. This bridges the paper's "Precision-Weighted MetaVote" to our FP/FN cost framework.

### New EDA chart: Avg review score vs delivery lateness
The paper shows `avg price vs review score` (Fig 8). We should add: `avg review score by is_late` — quantitatively showing late deliveries score 1-2 stars vs on-time deliveries scoring 4-5. This is the "so what" chart for our project report.

### New feature: `is_high_value_and_high_distance`
From Fig 8 (high price → lower satisfaction) + our geographic analysis: expensive products shipped long distances are the worst-outcome orders. Add interaction: `price_per_km = price / distance_km` — high price/km orders have highest expected dissatisfaction if late.

### Citation to add in report
> Mangla & Soni (2025) demonstrated that delivery delay features are the strongest predictors of customer satisfaction on the same Olist dataset, achieving ROC-AUC of 0.7389 using gradient-boosted models. Our work extends this by predicting the delivery delay itself at order placement time — before any delivery timestamps are available — enabling proactive intervention rather than retrospective analysis.

---

## Critical Gaps in the Paper (Do Not Replicate)

| Issue | What Paper Did | Why It's Wrong | Our Fix |
|---|---|---|---|
| **Post-delivery features** | Used `delay_duration`, `carrier_time`, `approval_time` as features | These are known only after delivery — cannot be used at order placement time | We engineer only pre-delivery features: distance, product dimensions, purchase time, seller history |
| **Dataset balancing (unspecified)** | "100k balanced records" — method not stated | If SMOTE was applied before split, test set contains synthetic data → inflated metrics | Apply SMOTE ONLY to training fold; validate on real imbalanced test distribution; report metrics on imbalanced data |
| **Random split** | `train_test_split(test_size=0.2)` | Temporal dependency ignored; 2017 orders in test set with 2018 orders in train = leakage | Temporal split |
| **No SHAP analysis** | No feature importance reported | Cannot know which features drove predictions | We run SHAP values for all models + SHAP beeswarm and waterfall plots for report |
| **No leakage audit** | `carrier_time` is post-purchase — no mention of this risk | Any model trained with this achieves inflated results that won't hold in production | Our leakage checklist (from todo.md) forbids all post-purchase timestamps |
| **No cost analysis** | Binary accuracy metrics only | Business cannot act on 82% accuracy without knowing the cost implications | We report cost per 1000 orders at each threshold |

---

## Summary: What This Paper Contributes to Our Project

1. **Academic citation:** Published Dec 2025, directly confirms that delivery delay is the #1 predictor of customer satisfaction on Olist. **Cite in introduction/business justification section.**

2. **ROC-AUC benchmark:** Their best single-model AUC = 0.7389 (HGB) with leaky features. Our model without those features will have lower AUC — but will be production-valid. Reference point: if we achieve AUC > 0.75 without leakage, we exceed their leaky benchmark in an honest sense.

3. **5th model: HistGradientBoosting.** Not in our original plan. HGB has best AUC among single models, handles NaN natively, is fast. Add to model comparison.

4. **Novel ensemble adaptation: Cost-Weighted MetaVote.** Adapt their "Precision-Weighted MetaVote" to use cost-minimization on validation as the weighting criterion. This is our methodological contribution to the ensemble approach.

5. **Confirms all our planned features.** Their full feature set (distance, dimensions, payment value, installments, temporal) is a subset of our feature set. We go further with: seller historical rates, route-level metrics, seasonal flags.

6. **Exposes the leakage risk clearly.** Three of their top features (`delay_duration`, `approval_time`, `carrier_time`) would be unavailable at prediction time. Our project avoids this. The gap in AUC between their results and ours is the "honest" cost of preventing data leakage — worth explaining in the report.

7. **Economic argument chain:** Paper → late delivery causes low reviews → our project → predict & prevent late delivery → protect review scores → protect cohort retention (Fig 13 → one lost delivery = lost customer). This is the full chain for our economic analysis.
