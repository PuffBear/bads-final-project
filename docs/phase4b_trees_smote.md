# Phase 4b: Decision trees, Gini, SMOTE and confusion matrices

Script: `analysis/phase4b_trees_smote.py` (numpy / pandas / scikit-learn only). Outputs are `outputs/tables/phase4b_*.csv`, figures `outputs/figures/p4b_*.png` and the log `phase4b_log.txt`.

This phase applies the course's classification toolkit to the Phase 4 problem: *will an exposure lead to a purchase within 14 days?* The design matches Phase 4 so the results compare directly:
- pre-contact features only;
- training on campaigns C001–C010 and testing on the later C011–C014 (5,933 exposures, 370 buyers);
- hyper-parameters tuned by 5-fold cross-validation **grouped by customer**, inside the training campaigns only.

## Three different "Gini"s (kept apart on purpose)

| Term | Formula | Range | What it answers here |
|---|---|---|---|
| **Gini impurity** (decision trees) | G = 1 − Σ pₖ² | 0 (pure) to 0.5 (50/50) | How mixed buyers and non-buyers are inside a group |
| **Gini coefficient of a model** | 2 × AUC − 1 | 0 (random) to 1 (perfect) | How well a model *ranks* buyers above non-buyers |
| **Gini coefficient of concentration** (Lorenz) | area between the Lorenz curve and equality | 0 (equal) to 1 (one customer has everything) | How concentrated customer value is |

## 1. Impurity of the purchase outcome (`phase4b_gini_single_splits.csv`, `phase4b_gini_by_segment.csv`, `p4b_01`)

- **Root impurity is low because the outcome is rare.** With p = 5.57%, G = 1 − 0.0557² − 0.9443² = **0.105** (the maximum is 0.5). A rule that says "nobody buys" is already **94.4% accurate**. That is why accuracy is useless here, and why the rest of this phase uses recall, precision, AUC and Gini instead.
- **Gini impurity inside each K-means segment:**

| Segment | Exposures | Purchase rate | Gini impurity |
|---|---|---|---|
| Recent actives | 3,503 | 11.4% | **0.202** |
| Lapsing regulars | 3,552 | 6.4% | 0.121 |
| Low-basket buyers | 3,560 | 4.6% | 0.088 |
| Occasional buyers | 7,532 | 4.1% | 0.079 |
| Registered, never purchased | 2,853 | 2.2% | **0.044** |

  **How to read this.** When the outcome is rare, the *most valuable* segment is the *least pure*: its buy rate moves toward 50%, so its impurity rises. Low impurity in "never purchased" means "almost certainly won't buy", not "well understood". The weighted impurity after splitting by segment is 0.1035, so the segmentation removes **1.6% of root impurity**.
- **Single splits ranked by impurity removed** (share of the root Gini removed):

| Split | Share removed |
|---|---|
| RFM recency quintile | **1.8%** |
| Recency band | 1.7% |
| K-means segment | 1.6% |
| Order-count band | 1.2% |
| Category match | 0.5% |
| Campaign | 0.1% |
| Discount | 0.07% |
| Channel | 0.05% |
| Income, region, consent, gender | < 0.02% each |

  This is the same story as the Phase 4 logit, told through impurity: **recency and history first, category match second, campaign mechanics a distant third, demographics nothing.** The K-means segments carry nearly all of recency's information (1.6% vs 1.8%) while also being usable for planning.
- **No single variable removes more than 2% of impurity.** This is the honest ceiling on how predictable a single 14-day purchase is from pre-contact data.

## 2. Decision tree with Gini splits (`phase4b_tree_cv_grid.csv`, `phase4b_tree_nodes.csv`, `p4b_02`, `p4b_03`)

**Tuning.** 56 settings (depth 2–none × minimum leaf size 50–400 × class weights none/balanced), each scored by grouped CV AUC. The 1-standard-error rule picks the **simplest tree within 1 SE of the best**: depth 4, minimum leaf 400, 13 leaves, CV AUC 0.664 (the best was depth 5 at 0.667).

**Overfitting check.** An unpruned tree grows to depth 27 with 1,360 leaves. It has training AUC **1.000** but hold-out AUC **0.536**, barely better than chance. Hold-out AUC peaks at depth 3–5 and falls after that (`p4b_03`).

**Impurity in the chosen tree.** Root Gini on the training data is 0.100; the sample-weighted Gini of the 13 leaves is 0.097. The tree removes **3.1% of impurity**, about twice the best single split.

**The tree's rules.** Thresholds are converted back from logs.

| Path | Exposures | Buy rate | Leaf Gini |
|---|---|---|---|
| Last order ≤ 36 days **and** featured = preferred category **and** 4+ past orders | 435 | **18.9%** | 0.306 |
| Last order ≤ 36 days, category match, ≤ 3 orders | 724 | 12.2% | 0.214 |
| Last order ≤ 36 days, no match, past spend > ₹4,455 | 574 | 12.5% | 0.219 |
| Last order 37–59 days, category match | 501 | 10.4% | 0.186 |
| … | | | |
| Last order > 214 days **and** no category match | 1,725 | **1.0%** | 0.020 |

- **Every leaf predicts "no buy".** No leaf reaches a 50% buy rate, so `tree.predict()` labels everyone a non-buyer. The tree is only useful through its **leaf probabilities**, used for ranking.
- **Gini importance** (mean decrease in impurity): recency 58%, category match 23%, log spend 9%, log orders 5%, website visits 3%, email engagement 2%. Every campaign, channel and demographic feature is **0**.
- **Permutation importance on the hold-out** gives the same top two: recency causes an AUC drop of 0.105, category match 0.024. MDI is biased toward continuous variables, so we checked it this way; the ranking holds.

## 3. SMOTE (KNN oversampling) + logistic regression (`phase4b_model_comparison.csv`, `p4b_05`)

**SMOTE as implemented.** For each synthetic buyer, pick a real buyer, pick one of its **k = 5 nearest buyer neighbours**, and create a point at a random position on the line between them. One-hot categories are copied from the seed buyer (the SMOTE-NC convention), so there are no "0.4 of a region" customers. Buyers are oversampled to a 50/50 balance, **inside each training fold only**.

| Model | Hold-out AUC | **Model Gini** | Avg precision | Brier | Mean predicted prob. (actual 6.2%) | Grouped CV AUC |
|---|---|---|---|---|---|---|
| **Logistic regression (Phase 4 model)** | **0.680** | **0.360** | 0.129 | **0.057** | **5.6%** | **0.680** |
| Logistic, class_weight = balanced | 0.679 | 0.359 | 0.129 | 0.232 | 46.3% | 0.678 |
| Random oversampling + logistic | 0.677 | 0.355 | 0.129 | 0.232 | 46.3% | 0.678 |
| SMOTE (k = 5) + logistic | 0.673 | 0.345 | 0.126 | 0.210 | 43.3% | 0.677 |
| KNN classifier alone (k = 401, chosen by CV) | 0.663 | 0.326 | 0.120 | 0.057 | 5.1% | 0.668 |
| Decision tree (Gini, pruned) | 0.656 | 0.312 | 0.109 | 0.057 | 5.5% | 0.664 |
| *WRONG: SMOTE before the CV split* | | | | | | *0.701* |

**What this shows:**
1. **SMOTE does not improve ranking.** AUC is unchanged or slightly lower (0.673 vs 0.680). Rebalancing changes the *intercept*, not the *ordering*. Any model's cut-off can already be moved to catch more buyers.
2. **SMOTE wrecks calibration.** It predicts an average 43% purchase probability against a real 6.2%, and Brier gets 4× worse. The Phase 4 economics (expected contribution per contact, break-even incrementality) need *true* probabilities. A SMOTE model could only be used after recalibration (Platt or isotonic), so for this problem it adds a step and gains nothing.
3. **Where SMOTE sits in the workflow matters.** Oversampling the whole training set and *then* cross-validating gives CV AUC **0.701**, 0.024 too high. Synthetic buyers built from a validation buyer's neighbours leak into training. The correct pipeline (SMOTE inside each training fold) gives 0.677. This is a validation check we report explicitly.
4. **KNN needs a very large k.** CV AUC rises steadily from k = 25 (0.590) to k = 401 (0.668). With 5% events, a small neighbourhood holds 0–2 buyers, so its probability estimate is mostly noise.
5. **Logistic regression stays the targeting model.** It has the best AUC and Gini, the best calibration, it is interpretable, and it is the cheapest to run.

## 4. Confusion matrices: the cut-off is a business decision (`phase4b_confusion_matrices.csv`, `p4b_04`)

Hold-out: 5,933 exposures, 370 buyers.

| Model and cut-off | TP | FP | FN | TN | Recall | Precision | Specificity | F1 | Share contacted |
|---|---|---|---|---|---|---|---|---|---|
| Logistic, default 0.5 | 0 | 0 | 370 | 5,563 | **0%** | n/a | 100% | 0 | 0% |
| SMOTE + logistic, default 0.5 | 207 | 1,686 | 163 | 3,877 | 56% | 10.9% | 70% | 0.18 | 32% |
| **Logistic, top 30% by score** | **200** | 1,580 | 170 | 3,983 | **54%** | **11.2%** | 72% | 0.19 | 30% |
| SMOTE + logistic, top 30% by score | 198 | 1,582 | 172 | 3,981 | 54% | 11.1% | 72% | 0.18 | 30% |
| Logistic, F1-optimal cut-off (0.094, set on training CV) | 125 | 686 | 245 | 4,877 | 34% | 15.4% | 88% | 0.21 | 14% |

- **The "default 0.5" trap.** The calibrated logistic regression (and the tree, and KNN) never predicts a probability of 0.5, so at the default cut-off it flags **nobody**: 93.8% accuracy, 0% recall. This is the usual reason people reach for SMOTE.
- **But SMOTE only moved the cut-off.** At an equal contact volume (top 30%), plain logistic and SMOTE produce virtually identical confusion matrices: 200 vs 198 buyers caught. SMOTE's apparent gain at 0.5 comes entirely from shifting probabilities upward.
- **Choose the cut-off from the business, not from 0.5:**
  - **Capacity rule:** contact the top x%. The top 30% catches 54% of buyers for 30% of contacts, 1.8× the base rate.
  - **Economic rule:** contact when p × contribution per buyer > contact cost (needs calibrated p, another reason to avoid SMOTE).
  - **F1 rule:** a statistical compromise, chosen on training out-of-fold predictions so the test set stays untouched.
- **Precision tops out at about 11–15%.** Even the best-targeted contacts mostly don't buy within 14 days. That fits the low impurity removal above: the outcome is noisy at individual level, and targeting shifts odds rather than finding certain buyers.

## 5. Concentration of customer value (`phase4b_gini_concentration.csv`, `p4b_06`)

| Measure | Gini coefficient |
|---|---|
| Pre-2026 spend, all 6,000 customers | **0.46** |
| Pre-2026 spend, 5,156 buyers only | 0.37 |
| 2026 campaign-attributed revenue, all customers | **0.90** |
| Share of pre-2026 spend held by the top 20% of customers | 47% |

Historical value is moderately concentrated: the top 20% hold 47% of spend, not the textbook 80%. Campaign revenue is extremely concentrated (Gini 0.90) because only about 1 in 6 customers bought from a campaign at all. This supports the Phase 1/4 conclusion that contact effort should follow value and recency rather than be spread evenly.

## 6. Implications
1. **The class methods confirm the Phase 4 conclusions independently.** Impurity splits, the Gini tree and permutation importance all rank **recency > category match >> campaign mechanics > demographics (≈0)**.
2. **Keep calibrated logistic regression as the scoring model** (model Gini 0.36). Use the pruned tree as the **communication device**: three readable rules (recent? category-matched? 4+ orders?) that a CRM team can apply without a model.
3. **Don't use SMOTE for this problem.** It adds leakage risk and breaks the probabilities the budget maths needs, without improving ranking. Handle the imbalance by **choosing the cut-off** on capacity or economics.
4. **Report recall and precision at the operating cut-off, never accuracy.**
