# Phase 6: Brand perception

Script: `analysis/phase6_brand.py`. Outputs are `outputs/tables/phase6_*.csv`, figures `outputs/figures/p6_*.png` and the log `phase6_log.txt`.

## Method
- **Survey.** 800 respondents each rated 8 brands on 5 attributes (1–5). Because everyone rated every brand, these are repeated measures.
- **Overall differences.** Friedman test per attribute.
- **NexaGlow vs each competitor.** Paired within each respondent (NexaGlow rating minus competitor rating). Standard errors are clustered by customer, since 38 customers answered twice. Holm correction is applied across the 35 comparisons.
- **Scale-use check.** Some respondents rate everything higher than others (SD of personal means 0.28). Every comparison is rerun on respondent-centred ratings: **all 35 verdicts are unchanged.**
- **Perceptual map.** PCA of the brand × attribute means, with attributes standardised.
- **Link to behaviour.** Survey `customer_id`s join to the customer file, the Phase 1 segments and the 2026 campaign outcomes. 760 customers are in the survey and 738 of them received campaigns.

## 1. Where NexaGlow stands (`phase6_brand_means.csv`, `p6_01_brand_ratings.png`)

| Brand | Quality | Affordability | Innovation | Naturalness | Trendiness | Overall |
|---|---|---|---|---|---|---|
| **NexaGlow** | 4.07 (4th) | 3.97 (4th) | 3.87 (4th) | 3.92 (4th) | 3.69 (5th) | **3.90 (1st)** |
| BlushBloom | 3.47 | **4.48** | 4.21 | 2.99 | **4.11** | 3.85 |
| RadiantCo | 3.39 | 4.34 | **4.38** | 3.10 | 3.84 | 3.81 |
| PureRoot | 4.25 | 3.15 | 3.22 | **4.51** | 3.54 | 3.73 |
| UrbanLeaf | 3.67 | 3.56 | 3.38 | 4.05 | 3.74 | 3.68 |
| DewDrop | 3.15 | 4.13 | 3.49 | 3.28 | 4.04 | 3.62 |
| Aster & Oak | **4.32** | 3.04 | 3.00 | 4.42 | 3.29 | 3.62 |
| ClariSkin | 4.15 | 2.97 | 4.35 | 3.47 | 3.10 | 3.61 |

- **Highest overall, first on nothing.** NexaGlow has the best average across attributes, but it isn't the leader on any single one. It is 4th on four attributes and 5th on trendiness. The brands really do differ: the Friedman test gives p < 0.001 on every attribute (Kendall's W 0.17–0.43).
- **Paired head-to-heads.** Of 35 comparisons, NexaGlow is significantly **ahead in 19**, **behind in 15**, and level in one (trendiness vs UrbanLeaf).
- **Its strengths depend on the rival:**
  - Against the natural/premium brands (Aster & Oak, PureRoot), NexaGlow wins on **affordability (+0.8 to +0.9)**, **innovation (+0.7 to +0.9)** and trendiness, but trails on **naturalness (−0.5 to −0.6)** and quality (−0.2 to −0.3).
  - Against the clinical brand (ClariSkin), it wins on affordability (+1.0), naturalness and trendiness, but trails on **innovation (−0.5)** and slightly on quality (−0.09).
  - Against the value/trendy brands (BlushBloom, RadiantCo, DewDrop), it wins on **quality (+0.6 to +0.9)** and **naturalness (+0.6 to +0.9)**, but trails on affordability and trendiness, and on innovation against BlushBloom and RadiantCo.
  - Against UrbanLeaf, its nearest neighbour on the map, it is ahead on quality, affordability and innovation, slightly behind on naturalness (−0.12), and level on trendiness.

## 2. Perceptual map (`phase6_pca_*.csv`, `p6_02_perceptual_map.png`)
- **PC1 (73% of variance): "premium–natural" vs "value–trendy".** Quality and naturalness load on one side, affordability and trendiness on the other. This is the trade-off consumers actually perceive: across all ratings, quality correlates with affordability at −0.16 (−0.28 when respondent-centred), and naturalness with affordability at −0.20.
- **PC2 (22%): innovation.** ClariSkin is the outlier, positioned as the clinical/innovative brand.
- **NexaGlow sits almost exactly at the centre** (PC1 −0.12, PC2 +0.08). It is the only brand rated ≥ 4.0 on quality and also ≈ 4.0 on affordability. Every other brand gives up one to get the other.
- **Nearest rival:** UrbanLeaf (distance 1.57 in standardised attribute space). Every other brand is 2.3 or more away.

**Reading.** The centre position is a **genuine "accessible quality" niche**, but it carries a risk: NexaGlow is *liked* without being *distinctive*. No attribute is clearly its own, and trendiness is its weakest relative score.

## 3. Who rates NexaGlow how? (`phase6_nexaglow_by_segment.csv`, `phase6_who_rates_tests.csv`, `p6_03`)
- **Perception is the same everywhere.** NexaGlow's overall rating, and its lead over the competitor average, don't differ by Phase 1 segment (p = 0.86–0.94), gender, region, income, preferred category, or by age, spend, recency and email engagement (all p ≥ 0.15, R² ≤ 0.009).
- **Even never-buyers agree.** Customers who registered but never purchased rate NexaGlow just as highly (3.91 overall, +0.22 vs competitors) as Recent actives (3.90, +0.19).

## 4. Does perception predict buying? (`phase6_perception_predicts_purchase.csv`)
- **Model.** Logit of 2026 campaign purchase on NexaGlow ratings, controlling for recency, frequency and email engagement (2,621 exposures, 138 buyers, SEs clustered by customer).
- **Result.** No measure predicts purchase: overall rating OR 0.92 (p = 0.66), lead vs competitors OR 0.99 (p = 0.96), and every single attribute p ≥ 0.09.
- **Interpretation.** Perception is uniformly good and doesn't separate buyers from non-buyers. **The barrier to buying isn't image; it's recency and relevance** (Phases 1 and 4: category match doubles conversion). Caveats: the sample is small (138 buyers), and the survey has no purchase-intent or preference question, so attribute *importance* can't be derived directly.

## 5. Implications for the recommendations
1. **Own "accessible quality" explicitly.** It's the one position no competitor holds: quality on par with the natural brands at value-brand prices. Messaging and bundle pricing (Phase 3) should signal both together.
2. **Fix trendiness, the weakest relative score** (5th; behind BlushBloom, DewDrop, RadiantCo and UrbanLeaf). Use Social and creator content for *brand* building, but judge it on reach and brand metrics, not campaign ROAS (Phase 5 shows Social is weak as a direct-response channel).
3. **Don't segment brand messaging.** Every customer group sees NexaGlow the same way. Spend segment effort on offers and timing instead.
4. **Since image isn't the barrier, put conversion budget into relevance** (category match) and recency-based contact, not image campaigns aimed at existing customers.
5. **Add to the next survey:** overall preference or purchase intent (to estimate attribute importance), non-customers (all 800 respondents are existing registrants), and a survey date so perception can be tracked over time.
