# Detailed EDA – file by file

Script: `analysis/eda.py` → figures `outputs/figures/eda_*.png`, tables `outputs/tables/eda_*.csv`, full printout `outputs/tables/eda_summary.txt`.

## Whole dataset (`eda_00_date_coverage.png`, `eda_overview.csv`)
Customer snapshot at 31 Dec 2025 (signups Jan 2024 – Dec 2025). Orders run Jan 2025 – Oct 2026, campaigns Jan – Sep 2026, and weekly channel data Jan 2024 – Oct 2026. Only the weekly table covers all three years, so it is the only valid source for demand trends.

## customers.csv (6,000 × 14)
- **Skewed value fields** (`eda_01`): spend skew 1.34, recency 1.74, email engagement 1.17. Use log transforms. The 844 zero-spend customers form a spike in log(1 + spend).
- **Profile** (`eda_02`): 59% female; North is the largest region (30%); Delhi NCR, Chandigarh, Jaipur and Lucknow lead the cities; the modal income band is ₹8–15L; preferred category is Skincare 47%, Haircare 25%, Bodycare 20%, Bundles 8%.
- **Consent** (`eda_03`): 81.1% overall and flat across region (χ² p = 0.54) and income (p = 0.86). It is **not flat by age**: 45–54-year-olds consent at 69.3% vs ~82% for under-45s (χ² p < 0.001, n = 531). The gap is tied to age, not to value.
- **Correlations** (`eda_04`, Spearman): orders↔spend 0.93 (near-collinear); orders↔email engagement 0.45; visits↔orders 0.35; recency↔orders −0.39. **Age and consent are uncorrelated with every behaviour field** (|ρ| ≤ 0.04). The age–consent ρ is only −0.04 because the 45–54 dip is non-monotonic.
- **Spend by income/region** (`eda_05`): the medians barely move (₹1.48k in Under-4L vs ₹1.59k in 25L+). Income does not drive value here.

## campaign_catalog.csv (14 × 8)
- **Calendar** (`eda_06`): back-to-back 12-day campaigns, one at a time, Jan–Sep. Each channel runs 2–3 campaigns.
- **Mix**: Skincare is featured 5× (all Retention), Bundles 3× (2 Conversion), Haircare 4×, Bodycare 2×. Search is used only for Awareness. Discounts range 10–25%.
- **Reach**: ~1,430–1,550 exposures per campaign, so campaigns are comparable in size. Conversion ranges from 4.3% (C002 SMS, C009 Social) to 7.0% (C013 Email Payday Picks).

## campaign_exposures.csv (21,000 × 10)
- **Exposure volume**: 5,831 customers exposed, a median of 3 exposures each (max 12); 169 never exposed.
- **Contact cost** (`eda_07`): flat within each channel, and channels differ ~30×. Email ₹0.42, SMS ₹0.94, WhatsApp ₹1.36, Display ₹4.00, Social ₹7.84, Search ₹13.14.
- **Funnel** (`eda_08`): delivery is 92% (Email, SMS, WhatsApp) to 99% (paid media). Open rate is 34–45% and click rate 19–31%. Purchase rate is 4.9–6.3%, a narrow spread given the 30× cost range.
- **Conversion drivers** (`eda_09`, overall 5.57%):
  - Recency: 0–30 days 10.2%, 31–90 days 5.3%, 91–180 days 4.0%, 181–365 days 2.3%, 365+ days 0.8%.
  - Order history: 0 orders 2.2%, 1–2 orders 4.6%, 3–5 orders 7.3%, 6+ orders 14.6%.
  - **Featured category = preferred category: 8.1% vs 4.6%.**
  - Consent: 5.7% vs 5.2%, so consent does not predict response.
- **Repeat sends**: 2–5 sends of the same campaign do not raise conversion (5.5% / 6.0% / 5.0% for 1 / 2 / 3 sends).

## orders.csv (14,274 × 9)
- **Monthly volume** (`eda_10`): ~900/month in 2025 vs ~230/month in 2026, which reflects the sample design and not demand. Campaign-attributed orders appear only in 2026 (8.2% of all orders).
- **Distributions** (`eda_11`): mean GMV is ₹1,519 and the median ₹1,199. Only 8.2% of orders carry a discount, and **every discounted order is campaign-attributed**. 40% of orders pay ₹49 shipping (below the ₹999 threshold).
- **Payment**: COD is 23.3% of orders, with average value ₹1,551 vs ₹1,509 for prepaid (no meaningful gap).
- **Attributed vs non-attributed**: attributed orders average ₹1,278 vs ₹1,540 for the rest, and every attributed order is discounted (16.1% on average).
- **Orders per customer**: 2.65 for the 5,383 customers who appear in the sample.
- **Attribution lag**: 1–14 days, median 7, roughly uniform.

## order_items.csv (25,086 × 6)
- **Basket shape** (`eda_12`): 1.65 distinct products (1.76 lines) and 2.13 units per order; 48% of orders have 2+ distinct products (52% are single-product). Quantity per line is 1–3 (82% are 1).
- **Category revenue**: Skincare 38%, Haircare 26%, Bundles 25%, Bodycare 11%. The top products are Hair Growth Serum (10.6%), Glow Essentials Kit (9.7%) and Night Repair Cream (7.6%); the bottom are Lip Balm and Daily Cleanser (2.5–2.7%).
- **Co-occurrence** (`eda_13`, seeds Phase 3): strong **within-category** affinity, with pair lift around 1.4–1.6. Examples: Body Wash + Lip Balm 1.62; Shampoo + Scalp Scrub 1.49; Body Lotion + Body Wash 1.48; Shampoo + Hair Growth Serum 1.46. Cross-category pairs sit around 0.3 (Night Cream + Scalp Scrub 0.30), so customers rarely mix categories in one basket.

## product_catalog.csv (18 × 4)
- **Price ladder** (`eda_14`): ₹249 (Lip Balm) to ₹1,799 (Glow Essentials Kit).
- **Implied bundle savings**: bundles sit 9–22% below their plausible components, e.g. Glow Kit ₹1,799 vs Vitamin C + Niacinamide + Moisturiser at ₹2,297. The component mapping is our assumption, since the catalogue doesn't list contents.

## weekly_channel_performance.csv (888 rows)
- **Spend vs return** (`eda_15`, `eda_channel_kpis.csv`): Search (36%) and Social (31%) take two-thirds of spend. Reported ROAS: Email 32.7×, WhatsApp 22.3×, SMS 18.9×, Search 4.6×, Social 3.3×, Display 2.8×. CPA runs from ₹41 (Email) to ₹478 (Display).
- **CTR** is ~4.2% on every channel, suspiciously uniform. CVR varies more: 1.1% for Display vs 4.5% for Search.
- **Growth and seasonality** (`eda_16`): weeks 1–43 revenue grew +20% (2024→25) and +19% (2025→26). Peak weeks cluster in late May–August 2026, with the top week starting 25 May 2026.
- **Naive log-log slopes** (`eda_17`): 0.93–1.28, i.e. no visible diminishing returns at current spend. This is confounded by trend and seasonality, and Phase 5 adds controls.

## brand_perception_survey.csv (6,400 rows)
- **NexaGlow's position** (`eda_18`): the all-rounder. It is never last and never first on any attribute: quality 4.07, affordability 3.97, innovation 3.87, naturalness 3.92, trendiness 3.69. It ranks 4th on quality (behind Aster & Oak, PureRoot and ClariSkin) and trails BlushBloom and DewDrop on trendiness.
- **Rating distribution**: ratings centre on 4; trendiness has the fewest 5s.
- **Attribute trade-offs** (respondent level): quality↔naturalness +0.28; affordability↔trendiness +0.28; affordability↔quality −0.16.
- **Respondents** (760 distinct customers) match the base on age, spend, orders and consent. Female share is 55% vs 59% in the base.
