# Phase 3 – Product relationships

Script: `analysis/phase3_products.py` → `outputs/tables/phase3_*.csv`, figures `outputs/figures/p3_*.png`, log `phase3_log.txt`.

## Method
- **Baskets.** 14,274 orders × 18 products, as product presence; the 1,567 repeated-product lines are collapsed. Single-product orders (52%) are kept, because dropping them would inflate support and confidence.
- **Apriori** (`mlxtend`): min support 0.4% (~57 orders), itemsets up to 3. This gives 103 frequent itemsets and 174 rules, each with support, confidence, lift, leverage and conviction.
- **Significance.** Every rule gets a one-sided Fisher exact test with Benjamini–Hochberg FDR control at 5%. Apriori leaves out rare pairs, and those are exactly the strong negatives. So **all 153 product pairs** are also tested directly, two-sided (`phase3_all_pairs.csv`).
- **Bundle contents** are not in the catalogue. We assume: Travel Skincare Kit = Cleanser + Moisturiser + SPF; Haircare Duo = Shampoo + Conditioner; Glow Essentials Kit = Vitamin C + Niacinamide + Moisturiser; Bodycare Combo = Body Lotion + Body Wash. Each kit is priced 9–22% below its assumed parts.

## 1. Customers buy one category per order (`p3_02_product_lift_matrix.png`, `p3_04_category_lift.png`)

| All 153 product pairs | Significantly positive | Not significant | Significantly negative | Lift range |
|---|---|---|---|---|
| Within a category (36) | 19 | 17 | **0** | 0.94–1.62 |
| Across categories (117) | **0** | 0 | **117** | 0.30–0.76 |

- **The cross-category result is unambiguous.** Every cross-category pair appears together *less* often than chance, and every significant positive association is within one category. Category-level lift is 0.45–0.67 for every pair of categories.
- **Single-category baskets are far above chance.** Among multi-product orders, 41% stay in one category, against 18% if products were picked independently by popularity (95th percentile of 200 simulations: 18.7%).
- **The pattern is stable.** The top 10 pairs keep lift above 1 in 2025 and 2026, across all five segments, and in campaign orders. The few exceptions are small-sample cells (`phase3_rule_stability.csv`).

## 2. The rules worth acting on (`phase3_rules_significant.csv`, `phase3_cross_sell.csv`, `p3_01`)

36 significant positive rules: 30 pairs and 6 triples.

| Rule | Orders together | Confidence | Lift |
|---|---|---|---|
| Body Lotion + Lip Balm → **Body Wash** | 65 | 23% | **1.98** |
| Body Lotion + Body Wash → **Lip Balm** | 65 | 22% | 1.89 |
| Lip Balm → Body Wash | 318 | 19% | 1.62 |
| Gentle Shampoo → Scalp Scrub | 218 | 15% | 1.49 |
| Body Lotion → Body Wash | 297 | 18% | 1.48 |
| Gentle Shampoo → Hair Growth Serum | 214 | 15% | 1.46 |
| Repair Conditioner → Scalp Scrub | 216 | 15% | 1.45 |
| Gentle Shampoo → Repair Conditioner | 208 | 15% | 1.44 |
| Travel Skincare Kit → Glow Essentials Kit | 75 | 9% | 1.45 |
| Night Repair Cream / Vitamin C Serum → Face Mask | 146 / 151 | 11% | 1.23 / 1.22 |
| Daily Cleanser → Hydrating Moisturiser | 155 | 12% | 1.20 |

**What stands out:**
- **Bodycare** is the tightest cluster. Lotion, Wash and Lip Balm form a three-way set: confidence reaches 23% with two in the cart.
- **Haircare** is a regimen. Shampoo, Conditioner, Scalp Scrub and Growth Serum are all mutually positive, but the existing Haircare Duo (Shampoo + Conditioner) leaves out the strongest partners (Scalp Scrub, Growth Serum).
- **Skincare** links are weaker (lift 1.2). The serums and creams pair with Face Mask; Cleanser pairs with Moisturiser. Lift is low because Skincare is the largest category, so co-purchase is already common.
- **Bundles go with other bundles.** Travel Skincare Kit, Glow Essentials Kit and Bodycare Combo are mutually positive (lift 1.30–1.45). There is a "kit shopper" who buys kits across categories, which is the only cross-category buying in the data.
- **Confidence tops out at 15–23%.** These are useful recommendation signals, not certainties.

## 3. Bundles: do they add to sales or cannibalise? (`p3_03_bundle_cannibalisation.png`)

| Evidence | Result | Reading |
|---|---|---|
| **Same basket**: bundle with its own components (`phase3_bundle_component_basket.csv`) | Lift 0.35–0.63, all significantly below 1 | Expected: nobody buys the kit and its parts together. Within one order, the bundle *replaces* the singles. |
| **Customer level**: do bundle buyers also buy the components in their *other* orders? Logit with controls for number of non-bundle orders and preferred category, customers with 2+ orders (`phase3_bundle_customer_substitution.csv`) | Raw: 30–37% vs 39–52%. Adjusted odds ratio **0.85–0.94, all n.s.** (CIs span 1) | The raw gap comes from bundle buyers having fewer other orders and different category preferences. **No significant customer-level cannibalisation.** |
| **Sequence**: what's in the *next* order after a bundle order (`phase3_bundle_sequence.csv`) | Components: 15–21% vs 19–26% otherwise (significantly lower for 3 of 4 kits). Same bundle again: **11–12% vs 5–6%** (p < 0.001, all 4) | Bundle buyers come back for the **bundle**, not the singles. They are bundle loyalists. |
| **Basket value** | Bundle orders ₹2,378 vs ₹1,278; they still include 0.85 other products on average | Bundle baskets are larger and not "kit only". |
| **Bundle-campaign windows** (C007, C009, C012, +14 days; 2026 sample, descriptive) | Bundle units/day +18% and component units/day +5% vs outside the windows | No visible drop in component sales while bundles were promoted. |

**Verdict: bundles mostly add, with a mild repeat-substitution effect.**
- Bundles don't reduce how many customers buy the single products, and promoting them didn't depress component sales.
- They do create a repeat-bundle habit. Customers who buy a kit tend to re-buy the kit rather than the parts, which costs the 9–22% bundle price gap on those repeat purchases.
- Bundle baskets are larger and kit buyers buy across kits, so the net effect looks positive. Without a randomised test this remains an association.

## 4. Implications for the recommendations
1. **Recommend within the category.** Use the rule table for "frequently bought together" and cart add-ons: Lip Balm ↔ Body Wash ↔ Body Lotion; Shampoo → Scalp Scrub / Growth Serum; Serum or Night Cream → Face Mask. Cross-category recommendations have no basis in the data (all 117 cross pairs are negative), except kit → kit.
2. **Redesign bundles around the strongest pairs.**
   - A **Bodycare trio** (Lotion + Wash + Lip Balm). This is the strongest triple; the current combo leaves out Lip Balm.
   - A **Haircare regimen** (Shampoo + Scalp Scrub + Growth Serum). Its pairs are stronger than the existing Shampoo + Conditioner duo.
   - **Kit-to-kit offers** for kit buyers (Travel ↔ Glow ↔ Bodycare).
3. **Combine with the ₹999 threshold** (Phase 2). Single-product Bodycare and Haircare carts are the natural place to suggest the cheap partner: a ₹249 Lip Balm with Body Wash, or a ₹499 Shampoo with Scalp Scrub.
4. **Manage the repeat-bundle effect.** Kit buyers re-buy kits, so consider a slightly smaller kit discount, or a replenishment offer on singles 60–75 days after a kit purchase (Phase 2 repurchase cycle).
5. **Test it.** Run an A/B test of cart recommendations and of a redesigned bundle before scaling. The rules are associations from a sampled order table.
