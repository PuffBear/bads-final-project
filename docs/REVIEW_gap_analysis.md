# Review: the repo against the brief and the course

Reviewed 18 Nov 2026 against `MARKETING_INTELLIGENCE_PROJECT.pdf` and the methods taught in class (k-means, association rules, MDS, cross-validation, SMOTE/KNN + logistic regression, decision trees with Gini, confusion matrices).

## 1. What the brief asks for, and where it stands

| Brief requirement | Covered by | Status |
|---|---|---|
| Understand **customers** | Phase 1 (RFM + k-means), Phase 4b (Gini by segment, Lorenz) | Strong |
| **Buying behaviour** | Phase 2 | Strong |
| **Product relationships** | Phase 3 (Apriori + Fisher/FDR, bundles), Phase 6b (product MDS) | Strong |
| **Marketing effectiveness** | Phase 4 (economics, logit, conditional logit, targeting), Phase 4b, Phase 5 | Strong |
| **Competitive brand perception** | Phase 6 (paired tests, PCA), Phase 6b (MDS) | Strong |
| **Demand over time** | Phase 7 (decomposition, back-tested forecasts) | Strong |
| **Concise management report** with tables/figures, conclusions, recommendations | Not written (11k words across 11 phase docs, 70+ figures) | **Missing: the actual deliverable** |
| State **assumptions** | Spread across the docs (60% margin, bundle contents, attribution) | Needs one consolidated table |
| **Data limitations** | Phase 0 has 13 flagged issues | Needs a short version in the report |
| **Validation checks** | Phase 0 (82 checks), time hold-outs, grouped CV, bootstraps, nulls | Strong; needs a one-page summary |

## 2. Course methods

| Method | Before this review | Now |
|---|---|---|
| K-means clustering | Done well: k chosen by gates, null silhouette, bootstrap Jaccard, external validity | Unchanged |
| Association rules | Done well: Apriori, Fisher exact, BH-FDR, stability across years and segments | Unchanged |
| Cross-validation | Done: grouped by customer, plus a time-based hold-out | Unchanged |
| **Multidimensional scaling** | **Missing.** The perceptual map used PCA | **Added (Phase 6b):** brand MDS with stress, property fitting, bootstrap, Procrustes vs PCA; product MDS from baskets |
| **SMOTE (KNN) + logistic regression** | **Missing** | **Added (Phase 4b):** SMOTE inside the CV folds, compared with class weights, random oversampling, plain logistic and KNN; leakage demonstration |
| **Decision tree / Gini impurity** | **Missing** | **Added (Phase 4b):** root and segment impurity, single-split Gini decrease, CV-tuned pruned tree, node-level impurity table, Gini (MDI) vs permutation importance |
| **Confusion matrix** | **Missing** (only AUC, AP, Brier) | **Added (Phase 4b):** at a 0.5 cut-off, a top-30% capacity cut-off and an F1-optimal cut-off; recall, precision, specificity, F1 |
| Gini coefficient | Not used | **Added:** model Gini = 2·AUC − 1 for every model; Lorenz Gini of customer value |
| Naive Bayes, gradient boosting | Done (beyond the syllabus) | Unchanged |

## 3. What is done well (keep and highlight in the report)
1. **Data validation before analysis.** 82 checks, including exact arithmetic reconciliation, one-to-one attribution and the ₹999 shipping rule.
2. **No leakage.** delivered, opened and clicked are excluded, and the excluded "leaky" AUC is reported to show why.
3. **Honest causal framing.** There is no control group, so the work reports **break-even incrementality** (~35%) instead of claiming lift. This is the most valuable idea in the project.
4. **Validation that mimics use:** train on earlier campaigns, test on later ones; customer-grouped CV; rolling-origin forecast back-tests.
5. **Defensible k for k-means.** It is chosen by pre-stated rules, not by eyeballing an elbow.
6. **Cross-source triangulation.** Platform channel data is checked against customer-level results (ρ = 0.71), and never joined to them.

## 4. Errors found and fixed
- `eda_findings.md` said "1.76 distinct products per order" and "52% of orders have 2+ products". Correct: **1.65 distinct products (1.76 lines)**, and **48%** have 2+ distinct products (52% are single-product, matching Phase 2). Fixed.
- `README.md` said 80 validation checks; Phase 0 reports 82. Fixed.

## 5. Remaining risks to address
1. **Write the report (Phase 8).** It is the only thing that gets graded as the submission.
2. **Reproducibility.** Scripts need `statsmodels` and `mlxtend` (pinned in requirements). Run the full pipeline once from a clean environment before submitting. The new Phase 4b and 6b scripts need only numpy, pandas, scipy, scikit-learn and matplotlib.
3. **Report length.** The brief says *concise*. Aim for a report of about 10–12 pages plus an appendix, with roughly 10 main figures. The other ~60 go to the appendix or the repo.
4. **The margin assumption (60%) drives every ₹ contribution figure.** Show the 40–70% sensitivity in one line wherever a ₹ figure appears.

## 6. Suggested report structure
1. Executive summary: 5 findings, 6 recommendations, the 2027 number.
2. Data and validation: one table of checks, one of limitations, one of assumptions.
3. Customers: segments (k-means), value concentration (Lorenz Gini), what does and doesn't separate them.
4. Buying behaviour and products: basket shape, ₹999 threshold, rules, product MDS, bundles.
5. Campaign effectiveness: economics, break-even incrementality, drivers (logit + Gini tree agree), targeting (confusion matrix at top 30%), SMOTE verdict.
6. Channels: platform vs customer-level, elasticities, ±20% reallocation.
7. Brand: MDS/PCA map, head-to-heads, perception doesn't predict purchase.
8. Demand and 2027 plan: seasonality, forecast, budget calendar.
9. Recommendations and test plan: holdouts, A/B tests on discount and bundles.
10. Appendix: methods detail, robustness, full tables.
