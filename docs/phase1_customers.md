# Phase 1: Customer segmentation

Script: `analysis/phase1_customers.py`. It writes `outputs/tables/customer_segments.csv`, one row per customer (segment, R/F/M/E scores, AOV, tenure), which Phases 2–7 reuse. Figures are `outputs/figures/p1_*.png`; tables are `outputs/tables/phase1_*.csv`; the run log is `phase1_log.txt`.

## 1. Approach

| Step | Choice | Why |
|---|---|---|
| Never-buyers | 844 customers with `orders_before_2026 = 0` → a rule-based segment, "Registered, never purchased" | Their recency and AOV are undefined (Phase 0, issue 6). If they went into k-means, they would form a trivial cluster and distort the scaling. |
| RFM + E scores | R = recency quintile (lower days → 5). F = order bands 1 / 2 / 3 / 4–5 / 6+. M = spend quintile. E = quintile of (Dec visits + Dec email engagement) | Orders are heavily tied integers, so F quintiles collapse to 4 levels. Explicit bands fix that. |
| Clustering features (5,156 buyers) | log recency, log orders, log AOV, website visits, email engagement, all z-scored | Spend is left out because spend ≈ orders × AOV (ρ = 0.93 with orders). Including it would double-weight frequency. Logs tame the skew (1.3–1.7). |
| Algorithm | k-means, `n_init = 50`, `random_state = 42` | Simple and reproducible. Its centroids read directly as segment profiles. |
| Outcomes | 2026 campaign results are **not** used to build segments | Kept back as an out-of-sample test that the segments matter. |

## 2. Choosing k: the evidence and the reasoning

Full grid: `phase1_k_selection.csv`. Figures: `p1_01_k_selection.png`, `p1_01b_silhouette_profiles.png`.

| k | Silhouette | Permuted-null silhouette | z vs null | Calinski-H. | Davies-B. | Weakest-cluster Jaccard | Smallest cluster | CV log-loss gain (2026) | Passes gates |
|---|---|---|---|---|---|---|---|---|---|
| 2 | **0.270** | 0.200 | 37.6 | **1,911** | 1.54 | **0.97** | 37% | 0.0203 | yes |
| 3 | 0.242 | 0.174 | 81.3 | 1,517 | 1.56 | 0.84 | 20% | **0.0247** | no (Jaccard) |
| **4** | 0.210 | 0.180 | 21.3 | 1,427 | 1.54 | 0.93 | 19% | 0.0245 | **yes** |
| 5 | 0.195 | 0.185 | 5.6 | 1,313 | 1.48 | 0.91 | 11% | 0.0239 | yes |
| 6 | 0.183 | 0.188 | −2.3 | 1,215 | 1.48 | 0.67 | 7% | 0.0293 | no |
| 7 | 0.181 | 0.182 | −0.6 | 1,141 | 1.46 | 0.64 | 7% | 0.0277 | no |

### What the silhouette score does and doesn't tell us
1. **The silhouette is low at every k (0.17–0.27).** On the Kaufman and Rousseeuw scale (>0.70 strong, 0.50–0.70 reasonable, 0.25–0.50 weak, <0.25 no substantial structure), only k = 2 reaches "weak". **NexaGlow's buyers form a behavioural continuum, not natural clusters.** We present the segments as a *useful partition* of that continuum, not a discovery of hidden types.
2. **Picking the silhouette maximum would be the wrong rule.** Silhouette usually falls as k rises, so its maximum is almost always k = 2. Two groups here are "recent and engaged" vs "everyone else", which is too coarse to plan campaigns around.
3. **The raw silhouette needs a baseline, so we built one.** We shuffled each feature column independently, 10 times. That keeps every distribution (including the skew and the recency heap) but removes any relationship between features. We then re-ran k-means on the shuffled data. Even this structureless data scores 0.17–0.20, because skewed, lumpy variables create apparent separation on their own. What matters is the **lift over that null**:
   - At k = 4 the real silhouette beats the null by 0.031, which is **z = 21**.
   - At k = 5 the lift shrinks to 0.010 (z = 5.6).
   - From k = 6 upward the real data scores the same as or **worse** than shuffled data. Splitting further captures nothing beyond the shape of the individual variables.

### Supporting criteria
4. **Elbow, Calinski-Harabasz and gap** all decline smoothly, with no sharp elbow. The Tibshirani gap rule is already satisfied at k = 2. This matches the "continuum" reading and doesn't single out any larger k.
5. **Stability.** We used Hennig's bootstrap Jaccard (20 resamples): each cluster is matched to its best counterpart after refitting. The usual reading is ≥0.85 highly stable, 0.75–0.85 stable, <0.75 doubtful.
   - At k = 4 **every** cluster scores ≥0.93.
   - k = 3 dips to 0.84.
   - k ≥ 6 has clusters at 0.64–0.67 that dissolve on resampling. k = 6's better 2026 prediction therefore comes from clusters that don't reproduce, and isn't usable.
6. **External validity.** We measured how well segment membership predicts the 2026 14-day purchase flag, using 5-fold cross-validation grouped by customer. This outcome was never used to fit the clusters. k = 3, 4 and 5 are tied (0.0239–0.0247); k = 2 is clearly worse (0.0203).

### Decision rule and result
We set the rule up front and computed the result in code:
1. **Gates.** Silhouette z vs null ≥ 3, every cluster's Jaccard ≥ 0.85, and every cluster ≥ 10% of buyers.
2. **Selection.** Among the k values that pass, choose the one with the best cross-validated prediction of 2026 purchase. A difference under 0.001 counts as a tie and goes to the smaller k.

k = 2, 4 and 5 pass the gates. k = 4 and k = 5 tie on prediction, k = 2 is clearly worse, so **k = 4**.

**Why not k = 5?** It passes the gates, but k = 4 does at least as well on every measure:
- higher silhouette (0.210 vs 0.195);
- four times the lift over the null (z 21 vs 5.6);
- stronger weakest-cluster stability (0.93 vs 0.91);
- fewer buyers with negative silhouette (6.2% vs 9.0%);
- a larger smallest segment (19% vs 11%);
- equal 2026 predictive power.

The extra k = 5 cluster ("Loyal champions", 545 buyers, defined by email engagement) is its least cohesive: mean silhouette 0.095, with 28% of its members negative. The engagement score E keeps that signal available without making it a separate segment.

**Why not k = 3?** It just misses the stability gate (0.84). It also nests cleanly inside k = 4: going from 3 to 4 only splits k = 3's large low-value group (n = 2,794) into "Occasional buyers" (1,876) and "Low-basket buyers" (915), separated by AOV (`phase1_k3_vs_k4.csv`). That split matters for margin and discount planning, so we keep it.

### Robustness (`phase1_robustness.csv`)
| Check | ARI vs chosen solution |
|---|---|
| Gaussian mixture (full covariance), same k | 0.63 |
| RFM features only (drop engagement) | 0.59 |
| Refit excluding the recency = 2 heap, heap reassigned | 0.38 |
| log spend instead of log AOV | 0.32 |

**Interpretation.** The solution is very stable under resampling (Jaccard ≥ 0.93). It is only moderately stable when the modelling choices change, which is expected when the data has no strong cluster structure. **76% of "Recent actives" sit on the recency = 2 heap** (Phase 0, issue 7), so that segment is partly defined by the floor in the data. Those customers really are the most recent buyers, though, and they convert best in 2026. A Gaussian-mixture BIC keeps falling up to k = 9 because it fits point masses on the integer and heaped variables, so it doesn't help choose k.

**Implication for later phases.** Use the segments for communication and planning. For targeting (Phase 4), use continuous scores (R/F/M/E or a propensity model) rather than segment labels alone.

## 3. The segments (`phase1_segment_profile.csv`, `p1_02`–`p1_04`)

| Segment | Customers | Share of pre-2026 spend | Median recency | Mean orders | Median AOV | Dec visits / email engagement | 2026 conversion per exposure | 2026 attributed revenue per exposure |
|---|---|---|---|---|---|---|---|---|
| **Recent actives** | 981 (16%) | **32%** | 2 days | 4.1 | ₹937 | 9.6 / 5.8 | **11.4%** | **₹140** |
| **Lapsing regulars** | 1,035 (17%) | 28% | 96 days | 3.3 | ₹967 | **11.4 / 7.5** | 6.4% | ₹84 |
| **Low-basket buyers** | 1,010 (17%) | 9% | 83 days | 1.9 | **₹558** | 7.1 / 3.3 | 4.6% | ₹63 |
| **Occasional buyers** | 2,130 (36%) | 31% | 96 days | 1.7 | ₹958 | 6.9 / 3.1 | 4.1% | ₹52 |
| **Registered, never purchased** | 844 (14%) | 0% | n/a | 0 | n/a | 6.5 / 2.5 | 2.2% | ₹31 |

**How each segment is defined** (`p1_03_snake_plot.png`, centroid z-scores):
- **Recent actives**: very recent (z −1.7) and frequent (+1.0). Highest RFM total.
- **Lapsing regulars**: about 3 months since their last order, yet the **most engaged** group online and on email (+0.9 / +1.0). They are still paying attention but have stopped buying, the classic win-back target.
- **Low-basket buyers**: AOV is the defining trait (z −1.4, ~₹560 vs ~₹950), with below-average engagement.
- **Occasional buyers**: few orders, low engagement, normal AOV. The largest group.

## 4. What it means

1. **Value is concentrated, and 2026 contact spend ignored that.** Recent actives are 16% of customers, 32% of historical spend and 33% of 2026 attributed revenue, and they convert at 11.4%. Yet every segment received a similar contact budget (₹4.00–4.22 per exposure). Occasional buyers took 36% of exposures for 26% of revenue. Never-purchased took 14% of exposures for 6%.
2. **Demographics do not separate segments.** Gender, region, city, income, preferred category, age and consent all fail to differ meaningfully across segments: every test is non-significant or Cramér's V < 0.06 (`phase1_segment_tests.csv`, `p1_05_demographic_index.png`). **Behaviour, not demographics, should drive targeting.** Demographic targeting on platforms (e.g. income or region) buys little.
3. **The segments predict 2026 outcomes they were not built on.** Conversion steps down from 11.4% to 6.4%, 4.6%, 4.1% and 2.2% (χ² = 329, p < 0.001). Revenue per exposure ranges 4.6× (₹140 vs ₹31).
4. **Category match roughly doubles conversion in every segment.** Featured = preferred category vs not: 15.8% vs 9.7% for Recent actives, and 3.2% vs 1.9% for never-purchased. This is a cheap targeting win to test in Phase 4.
5. **The RFM grid** (`p1_06_rfm_grid.png`) gives a finer ranking within segments. R5 × F4–F5 converts at 12–18%, while R1 cells sit at 1.8–2.7% regardless of frequency.
6. **Consent gaps are spread evenly.** Non-consented contacts appear in every segment (580–1,309 exposures each), so fixing them is a send-time suppression issue, not a segment strategy.
7. **For Phase 6**, survey respondents are spread across segments (110–275 per segment), so we can compare brand perception by segment.

## 5. Caveats
- The segments come from a pre-2026 snapshot. Customers will have moved between segments during 2026.
- The 2026 conversion differences are associations. Segments were not randomly assigned contacts, and better customers may simply have bought anyway (Phase 0, issue 9).
- Recency has quality issues (it exceeds tenure for 794 customers, and has the heap at 2 days). We use it as a ranking, not as exact days.
