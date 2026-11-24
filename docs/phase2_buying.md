# Phase 2 – Buying behaviour

Script: `analysis/phase2_buying.py` → `outputs/tables/phase2_*.csv`, `order_level.csv` (one row per order, reused in Phase 3), figures `outputs/figures/p2_*.png`, log `phase2_log.txt`.

**Data caveats**
- The order table is a **sample** (Phase 0, issue 3). Shares and averages are valid; totals are not.
- Customer-level spend and AOV in the snapshot **do not match** the same customers' orders in the sample (AOV ρ = 0.02; orders ρ = 0.25). The order table therefore cannot confirm or refute the Phase 1 value split. "Low-basket buyers" average ₹558 per order in the snapshot but ₹1,550 in the sample.

## 1. The typical order (`phase2_headline_kpis.csv`, `p2_01_basket_anatomy.png`)

| | All orders | 2025 | 2026 organic | 2026 campaign |
|---|---|---|---|---|
| Orders (sample) | 14,274 | 11,005 | 2,100 | 1,169 |
| Average order total | ₹1,519 | ₹1,545 | ₹1,515 | **₹1,278** |
| GMV before discount | ₹1,519 | ₹1,526 | ₹1,496 | **₹1,496** |
| Distinct products / units | 1.65 / 2.13 | 1.66 / 2.14 | 1.62 / 2.09 | 1.62 / 2.08 |
| Single-product orders | 52% | 51% | 53% | 54% |
| Single-category orders | 71% | 71% | 72% | 73% |
| Contains a bundle | 22% | 22% | 22% | 21% |
| Pays ₹49 shipping | 40% | 40% | 40% | **48%** |
| Cash on delivery | 23% | 23% | 23% | 24% |
| Revenue mix: Skin / Hair / Body / Bundles | 38 / 26 / 11 / 25% | | | |

- **Orders are small and focused.** Half are a single product and 71% stay within one category.
- **The preferred category is a reliable label.** 82% of orders include the customer's stated preferred category; each other category appears in only 16–18% (`phase2_preferred_vs_bought.csv`).
- **The cycle is roughly two months.** Median time between orders in 2025 is 60 days for Recent actives and 69–79 days for the other segments. The sample's spacing between orders is still useful for timing, even though its counts are not.
- **Day of week makes no difference** (χ² p = 0.06; Monday is marginally lowest).

## 2. Who buys how: almost no differences (`phase2_tests.csv`, `p2_02`, `p2_06`)

| Question | Result |
|---|---|
| Order value by segment (2025) | Kruskal p = 0.23; ₹1,522–1,589 |
| Category mix by segment | Skincare 37–40%, Haircare 24–28%, Bodycare 10–12%, Bundles 23–25% |
| Bundle / COD / shipping share by segment | Cramér's V ≤ 0.025 |
| Order value by region; by city (20) | p = 0.83; p = 0.30 |
| Main basket category by region | p = 0.30 |
| COD share by city | p = 0.24 |

**Interpretation.** Segments differ in **how often and how recently** they buy and in how they respond to campaigns (Phase 1). They do **not** differ in what they put in the basket. Basket design (cross-sell, bundles, thresholds) can be the same for everyone; contact frequency and offers should vary by segment.

## 3. Cash on delivery (`phase2_cod_logit.csv`, `p2_04_cod_drivers.png`)
- **Size.** COD is 23.3% of orders, stable across 2025 and 2026. COD orders average ₹1,551 vs ₹1,509 for prepaid.
- **Predictability.** A logit with order value, region, income, segment, gender, age, campaign attribution and year explains almost nothing (pseudo-R² = 0.001). Standard errors are clustered by customer. The only significant term is East region (OR 0.84, p = 0.04), which is marginal across 18 coefficients.
- **COD is not a customer habit.** Among customers with 3 or more orders, 0.4% always pay COD; if each order were an independent 23% draw we'd expect 0.7%. Likewise 35% never use COD vs 36% expected.
- **Implication.** COD can't be targeted to a "COD customer" type. Nudges toward prepaid (prepaid-only perks, COD fee) have to work at checkout, for everyone.

## 4. Free shipping at ₹999 (`phase2_shipping_threshold.csv`, `p2_03_shipping_threshold.png`)
- **The threshold rule is exact.** Shipping is free when (GMV − discount) ≥ ₹999; otherwise ₹49 (Phase 0).
- **Customers don't stretch to reach it.** We compared real orders with baskets rebuilt by randomly shuffling products across order lines (300 shuffles, organic orders only). The ratio of orders at ₹999–1,198 to orders at ₹799–998 is 0.79 observed vs 0.80 in the shuffles (p = 0.60). **No bunching above the threshold**, so it currently generates no upsell.
- **A sizeable group is close.** 40% of orders pay shipping. Of those, 89% are single-product orders and 27% are within ₹249 of the threshold (1,126 single-product orders in the sample). A Lip Balm (₹249), Daily Cleanser (₹349) or Body Wash (₹399) would cover the gap.
- **Discounts erode the threshold.** Campaign discounts pushed 93 attributed orders (8%) below ₹999, so they paid shipping they would otherwise have avoided. This partly explains why 48% of campaign orders pay shipping vs 40% of organic orders.

## 5. Discounts and campaigns (`phase2_campaign_baskets.csv`, `phase2_featured_uptake.csv`, `p2_05`)
- **Discounting is campaign-only.** All discounting is order-level and only on campaign-attributed orders: 8.2% of orders, average 16.1% off, ₹282k given away in the sample.
- **Discounts do not enlarge baskets:**
  - Campaign GMV ₹1,496 = organic GMV ₹1,496.
  - **Within the same customers** (312 customers with both a campaign and an organic order in 2026), median GMV is ₹1,235 vs ₹1,247 (Wilcoxon p = 0.90).
  - **Deeper discounts don't buy bigger baskets.** Across campaigns, an extra 5 percentage points of discount is associated with +0.8% GMV, controlling for featured category and segment (p = 0.76). The 25%-off Weekend Flash Sale (C007) produced an average order of ₹1,097 after discount, the lowest of any campaign.
- **Campaigns do steer category choice.** Campaign orders contain the featured category more often than 2026 organic orders: Skincare 57% vs 43%, Haircare 41% vs 33%, Bodycare 40% vs 31%, Bundles 28% vs 22% (all p < 0.05). Still, roughly half of campaign buyers take the discount on something other than the featured category.

**Interpretation.** The discount cuts order value by ~16% without increasing basket size. Whether it is worth paying therefore depends entirely on whether it creates **extra orders**. That is a Phase 4 question, and without a holdout it can only be partly answered.

## 6. Implications to carry into the recommendations
1. **Turn the ₹999 threshold into an active upsell.** Show "add ₹X for free shipping" with a cheap, same-category add-on at checkout. Prioritise single-product carts at ₹750–998. Make discounts threshold-aware: apply them after the free-shipping check, or offer free shipping instead of a percentage.
2. **Cap discount depth.** No evidence that 20–25% buys more basket than 10–12%. Test lower discounts or non-price offers.
3. **Keep basket design universal; vary frequency and offers by segment.**
4. **COD: use universal prepaid nudges**, not COD-customer targeting.
5. **Time retention contacts to the ~60–75-day repurchase cycle.**
