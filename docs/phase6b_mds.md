# Phase 6b: Multidimensional scaling (MDS)

Script: `analysis/phase6b_mds.py`. Outputs are `outputs/tables/phase6b_*.csv`, figures `outputs/figures/p6b_*.png` and the log `phase6b_log.txt`.

MDS places objects on a map so that the distances on the map reproduce a matrix of dissimilarities. We use it twice: for **brands** (from the perception survey, to complement the Phase 6 PCA map) and for **products** (from basket co-occurrence, as a visual companion to the Phase 3 association rules).

## A. Brand perceptual map (`p6b_01_brand_mds.png`)

**Method.**
- **Dissimilarity.** Euclidean distance between brands' standardised mean profiles on the 5 attributes (8 × 8 matrix).
- **Algorithms.** Classical (Torgerson) MDS, which is analytic, and metric MDS (SMACOF, 50 random starts).
- **Fit.** Kruskal stress-1: below 0.05 excellent, 0.05–0.10 good, 0.10–0.20 fair, above 0.20 poor.
- **Attribute arrows** come from *property fitting*: each attribute's brand means regressed on the map coordinates.
- **Stability.** 200 bootstrap resamples of the 800 respondents; each rebuilt map is Procrustes-aligned to the original.

| Dimensions | Classical stress-1 | Metric (SMACOF) stress-1 |
|---|---|---|
| 1 | 0.275 (poor) | 0.268 |
| **2** | **0.056 (good)** | **0.044 (excellent)** |
| 3 | 0.016 | 0.009 |
| 4 | 0.008 | 0.005 |

**Results:**
- **Two dimensions are enough.** Stress drops from 0.27 to 0.05 going from 1 to 2 dimensions, then flattens (an elbow at 2). The first two dimensions carry 94% of the distance structure: 73% and 22%.
- **The map agrees with Phase 6.** Classical MDS on Euclidean distances is mathematically equivalent to PCA, and the Procrustes disparity with the Phase 6 PCA map is **0.000000**. A map built from individual respondents' brand-to-brand distances gives the same configuration (disparity 0.017). The perceptual structure doesn't depend on the method.
- **Every attribute is well represented** (property-fitting R² 0.89–0.97):
  - Dimension 1 runs **premium–natural** (Aster & Oak, PureRoot) **↔ value–trendy** (BlushBloom, RadiantCo, DewDrop).
  - Dimension 2 is **innovation**, with ClariSkin at the top.
- **NexaGlow sits at the centre of the map.** Its nearest rival is **UrbanLeaf** (map distance 1.0; every other brand is 2.2–2.7 away). UrbanLeaf was the nearest neighbour in **200 of 200** bootstrap maps.
- **Positions are very stable.** Each brand's bootstrap position varies by only about 0.06 map units on a map about 5 units wide. The brand differences are not sampling noise.
- **Why non-metric MDS isn't used here.** It returns a stress near 0, but with only 8 brands (28 distances) almost any rank order can be matched exactly. That near-zero stress is a degenerate fit, not a better one.

**What it means for NexaGlow.** The centre position is "accessible quality": rated well on everything, owning no corner of the map. The only brand competing for that space is UrbanLeaf. Differentiation has to come from one of the arrows that point away from NexaGlow's position, and **trendiness** is the cheapest to move (Phase 6: it is NexaGlow's weakest relative score).

## B. Product map from baskets (`p6b_02_product_mds.png`)

**Method.**
- **Dissimilarity.** Jaccard distance between products across 14,274 orders: 1 − (orders containing both) / (orders containing either).
- **Algorithm.** Non-metric MDS. Jaccard distances are bunched near 1 and only their *order* is meaningful, so a rank-based fit is appropriate here, unlike for brands. With 18 products and 153 distances the fit is not degenerate.

| | Value |
|---|---|
| Stress-1 (2 dimensions) | **0.13 (fair)** |
| Mean Jaccard distance, same category | 0.941 |
| Mean Jaccard distance, different category | 0.978 |
| Silhouette of the 4 categories on the map | **0.88** |

**Results:**
- **The products separate into four tight islands, one per category.** Skincare, Haircare and Bodycare each form their own cluster, and the four **bundles form a separate fourth island**. A category silhouette of 0.88 means the categories are almost perfectly separated on the map, purely from what customers put in their baskets. Product category was never used to build the map.
- **This is the association-rule finding, made visible.** In Phase 3, all 117 cross-category pairs had lift below 1. Here the categories sit far apart. Phase 3 found bundles co-purchased with other bundles (lift 1.3–1.45); here they cluster together, away from their own component products.
- **Don't over-read positions inside an island.** With stress 0.13 the map is reliable for the large distances (between islands) but only approximate for the small ones. For which products to pair within a category, use the Phase 3 lift and confidence tables, not map distances.

**What it means.** Cross-sell and recommendation design should work *within* an island, plus kit-to-kit. Moving a customer across islands, for example from Skincare into Haircare, is a separate acquisition problem for that category, not a cart add-on.
