# Phase 4 – Campaign effectiveness and targeting

Script: `analysis/phase4_campaigns.py` → `outputs/tables/phase4_*.csv`, figures `outputs/figures/p4_*.png`, log `phase4_log.txt`.

## Assumptions and design
- **Unit and outcome.** The unit is the exposure (21,000); the outcome is `purchased_within_14_days` (5.57%).
- **Revenue.** Revenue per converted exposure = the attributed order's GMV − discount. Shipping is treated as pass-through.
- **Margin.** Gross margin is **60% of list-price GMV**, with a sensitivity check at 40–70%. Contribution = margin × GMV − discount − contact cost.
- **No control group** (Phase 0, issue 9). Attributed results are an **upper bound** on what campaigns caused. Instead of an unmeasurable lift, we report **break-even incrementality**: the share of attributed orders that must be genuinely *caused* by the contact for the campaign to pay off. Buyers who would have bought anyway still get the discount, so f* = (discount + contact cost) / (margin × GMV).
- **Models use only pre-contact information.** That means the customer snapshot plus what is known when a send is planned (channel, discount, featured category, recent contacts). `delivered`, `opened` and `clicked` happen *after* the contact. Including them raises hold-out AUC from 0.68 to 0.73, which is pure leakage, so they are excluded.

## 1. Funnel and economics (`phase4_campaign_economics.csv`, `phase4_channel_economics.csv`, `p4_01`)

| Channel | Exposures | Opened | Clicked | **Purchase rate** | Cost / exposure | **Cost / attributed order** | Contribution / exposure | Break-even incrementality |
|---|---|---|---|---|---|---|---|---|
| Email | 4,407 | 45% | 30% | 6.0% | ₹0.42 | **₹7** | ₹37.5 | 30% |
| SMS | 4,576 | 34% | 25% | 5.2% | ₹0.94 | ₹18 | ₹34.1 | 32% |
| WhatsApp | 3,003 | 43% | 31% | 6.3% | ₹1.36 | ₹22 | ₹36.2 | 34% |
| Display | 2,956 | 37% | 28% | 5.9% | ₹4.00 | ₹68 | ₹33.3 | 36% |
| Social | 3,012 | 36% | 22% | 4.9% | ₹7.84 | ₹160 | ₹26.2 | 38% |
| Search | 3,046 | 37% | 19% | 5.1% | ₹13.14 | **₹258** | ₹24.2 | 46% |
| **Total** | 21,000 | 39% | 26% | 5.6% | ₹4.08 | ₹73 | ₹32.4 | **35%** |

- **Totals (attributed, upper bound).** 1,169 orders, ₹14.7 lakh net revenue, contribution ₹6.8 lakh at a 60% margin.
- **Discount is the main cost, not contact.** Discounts total ₹2.82 lakh vs ₹0.86 lakh in contact costs (3.3×).
- **Break-even incrementality is ~35% overall.** It is 53% at a 40% margin and 30% at 70%.
- **By campaign** it ranges from 18% (C001 Email, 10% off) to 49% (C010 Search, 10% off). Deep-discount campaigns need 39–42% even on cheap channels (C007 Email 25%, C012 WhatsApp 22%).
- **Purchase rate ranges 4.3%–7.0%.** Best: C013 Payday Picks (Email, Skincare, 17%). Worst: C009 VIP Early Access (Social, Bundles) and C002 Winter Skin Rescue (SMS, Haircare).

## 2. What drives purchase? (`phase4_logit_drivers.csv`, `phase4_conditional_logit.csv`, `p4_03`)

| Driver | Cross-sectional logit OR (95% CI) | Within-customer logit OR (95% CI) |
|---|---|---|
| **Featured category = customer's preferred category** | **1.97 (1.72–2.25)** | **2.05 (1.74–2.42)** |
| Discount, per percentage point | 1.047 (1.019–1.075) | 1.037 (1.000–1.074), p = 0.049 |
| Channel vs Email (5 channels) | all n.s. (0.6–2.4 CIs span 1) | all n.s. |
| Featured category itself | n.s. | n.s. |
| Contacts in prior 30 days | 1.01, n.s. | 1.02, n.s. |
| log days since last order | 0.76 (0.73–0.79) | (absorbed) |
| log orders before 2026 | 1.41 (1.01–1.96) | (absorbed) |
| Email engagement Dec-25, per point | 1.035 (1.013–1.058) | (absorbed) |
| Consent, age, gender, region, income, preferred category | all n.s. | (absorbed) |

- **About the within-customer model.** It is a conditional logit with customer fixed effects: 982 customers with both buying and non-buying exposures, 4,354 exposures in total. It compares the *same customer* across different campaigns, so every stable customer trait is held constant.
- **Model fit:** pseudo-R² 0.061, of which 0.048 comes from customer features alone.

**What it means:**
1. **Matching the offer to the customer's category is the biggest controllable lever.** It roughly doubles purchase odds in both designs.
2. **Channel does not change *whether* people buy.** Once the audience is controlled for, all channels convert alike. Channel choice is therefore a **cost decision**: Email/SMS/WhatsApp deliver an order for ₹7–22, Search for ₹258.
3. **Recency and frequency carry most of the targeting signal**, consistent with Phase 1.
4. **No fatigue, and no benefit, from extra contacts** within 30 days (5.5% → 6.0% for 0 → 3+ prior contacts, n.s.).

## 3. Is it the campaign or the audience? (`phase4_mix_adjusted_conversion.csv`, `p4_02`)
- **Method.** We refit with campaign fixed effects and predict every campaign on the *same* 21,000-exposure audience.
- **Result.** Adjusted rates are within 0.2 points of the raw ones. The best (C013) and the two worst (C009, C002) keep their places; six mid-table campaigns shift, and C012 moves furthest (5th → 2nd). Each campaign's audience was expected to convert at 5.4–5.7%, so **audiences were essentially random across campaigns, and differences between campaigns are real campaign effects.**
- **Discount across campaigns.** Adjusted conversion rises 0.09 points per extra percentage point of discount across the 14 campaigns (R² 0.28, p = 0.054), consistent with the exposure-level estimate.

## 4. Does a deeper discount pay? (`phase4_discount_economics.csv`, `p4_06`)
Contribution per exposure = p(d) × (margin − d) × GMV. Deeper discounts win extra buyers but give the extra discount to *every* buyer.

| At 60% margin | 10% | 15% | 20% | 25% |
|---|---|---|---|---|
| Point estimate, cross-sectional | ₹37.4 | ₹41.8 | ₹46.0 | ₹49.6 |
| Point estimate, within-customer | ₹37.4 | ₹39.9 | ₹41.9 | ₹43.3 |
| Lower 95% CI, within-customer | ₹37.4 | ₹33.7 | ₹30.0 | ₹26.2 |

**Reading.** The answer depends on assumptions:
- At a 40% margin, 10% is best.
- At 60–70%, deeper discounts win *on point estimates*.
- At the lower confidence bound, 10% is best at every margin.

Pull-forward (people buying earlier, not more) is not captured, and Phase 2 showed deeper discounts don't enlarge baskets. **The evidence can't support deeper discounts. Test 10% vs 20% head to head before changing policy.**

## 5. Targeting model (`phase4_model_performance.csv`, `phase4_gains_deciles.csv`, `p4_04`)
**Validation design.** Train on campaigns C001–C010 (15,067 exposures); test on **later** campaigns C011–C014 (5,933 exposures, 370 buyers). This mimics scoring next quarter's audience.

| Model | Hold-out AUC (95% bootstrap CI) | Grouped 5-fold CV AUC | Avg precision | Brier | Mean predicted vs actual 6.2% | Top-30% buyer capture |
|---|---|---|---|---|---|---|
| **Logistic regression** (chosen) | **0.680 (0.650–0.707)** | **0.685** | **0.129** | **0.057** | 5.6% | **54.1%** |
| Gradient boosting | 0.673 (0.642–0.700) | 0.672 | 0.127 | 0.057 | 5.4% | 51.9% |
| Gaussian Naive Bayes (13 numeric features) | 0.673 (0.643–0.701) | 0.672 | 0.123 | 0.087 | **15.2%** | 53.8% |
| Gaussian NB, redundant features removed | 0.674 (0.645–0.702) | 0.674 | 0.123 | 0.067 | 8.4% | 53.2% |
| Naive Bayes, categorical (all 19 features, numeric binned into quintiles) | 0.659 (0.631–0.686) | 0.657 | 0.114 | 0.062 | 8.5% | 53.0% |
| RFM score only (R×10 + F×3 + M) | 0.667 (0.635–0.694) | n/a | 0.122 | n/a | n/a | 53.5% |
| *Leaky: + delivered / opened / clicked* | *0.730, not usable* | | *0.160* | | | *60.5%* |

**How to read this (`p4_07_model_comparison.png`):**
- **Ranking is nearly a tie.** Every fitted model sits within the logistic regression's confidence interval (0.65–0.71), and all capture 52–54% of buyers in the top 30%. The signal is mostly recency and frequency, which every method picks up. Even a hand-made RFM score is close.
- **Naive Bayes ranks well but gets the probabilities wrong.** Gaussian NB predicts an average 15.2% purchase probability when the real rate is 6.2% (Brier 0.087 vs 0.057).
  - *Why:* NB assumes features are independent within each class. Here `log_spend` and `buyer_before` correlate 0.86–0.97 with `log_orders`, so NB counts the same "frequent buyer" evidence three times and pushes probabilities to the extremes.
  - *Test:* removing those two features halves the overconfidence (8.4%) and improves Brier to 0.067, with AUC unchanged.
  - The categorical NB is binned and so less extreme (8.5%), but loses some ranking power (AUC 0.659) because binning throws away detail.
- **Why logistic regression is the chosen model.** It has the best AUC, the best calibration and the best top-30% capture. Its coefficients can be read directly as log-odds, which `phase4_logit_model_coefficients.csv` stores. Good calibration matters here: the economics (expected contribution per contact, break-even incrementality) use the *probability itself*, not just the ranking. **NB would be acceptable for ranking a contact list, but not for budget maths without recalibration.**

| Contact only the top… | Share of buyers captured | Share of contact cost | Purchase rate |
|---|---|---|---|
| 10% | 25% | 9% | 15.5% |
| **30%** | **54%** | **29%** | 11.2% |
| 50% | 75% | 49% | 9.3% |
| 100% | 100% | 100% | 6.2% |

- **Diminishing returns down the ranking.** The bottom half of the ranking yields 25% of buyers for 51% of the contact cost, converting at 1.9–4.7%.
- **But contact costs are tiny next to margin.** In attributed terms even the bottom decile has positive contribution. Break-even incrementality is about 32–37% in *every* decile, because the discount dominates the cost.
- **Use the model to choose *channel and offer*, not just *whether* to contact:**
  - Low scorers: the cheapest channel, and no or a small discount.
  - High scorers: category-matched offers.
  - Lowest decile: the first candidates for a holdout group.

## 6. Segments × channels (`phase4_segment_channel_conversion.csv`, `phase4_segment_economics.csv`, `p4_05`)
| Segment | Purchase rate | Cost / order | Contribution / exposure | Break-even incrementality |
|---|---|---|---|---|
| Recent actives | 11.4% (WhatsApp 15.0%) | ₹35 | ₹67.8 | 31% |
| Lapsing regulars | 6.4% (**Email 8.9%**) | ₹65 | ₹38.8 | 34% |
| Low-basket buyers | 4.6% | ₹86 | ₹28.5 | 36% |
| Occasional buyers | 4.1% | ₹98 | ₹22.8 | 38% |
| Registered, never purchased | 2.2% | ₹186 | ₹11.6 | **46%** |

The email-engaged Lapsing regulars respond best to Email. Never-purchased customers need nearly half of their attributed orders to be truly incremental just to break even.

## 7. Consent (`phase4_consent.csv`)
- **Scale.** 3,957 exposures (18.8%) went to customers without marketing consent, producing 204 attributed orders (17.5% of buyers) and ₹2.79 lakh in attributed revenue.
- **Conversion is no different** (5.2% vs 5.7%; OR 1.11, n.s.).
- **Implication.** Suppressing non-consented customers is a legal requirement, not an optimisation. In attributed terms it costs at most about 17% of campaign orders, and less in reality, since some of those customers would have bought anyway.

## 8. Implications for the recommendations
1. **Category-match every send.** Use `preferred_category` (validated in Phase 2) to choose the featured category or creative per customer. Roughly doubles the odds of purchase.
2. **Move campaign volume to Email, SMS and WhatsApp.** Same conversion, 10–35× cheaper per order. Keep paid channels for reach or acquisition (Phase 5).
3. **Hold discounts at 10–15% and test 20% in an A/B.** Discounts are the largest campaign cost.
4. **Score and tier contacts:** top 30% get full treatment; the bottom half gets a cheap channel and a small or no discount.
5. **Suppress non-consented customers at send time.**
6. **Build a randomised holdout of 10% per campaign** into the 2027 plan. Break-even incrementality is ~35%, and only a holdout can show whether campaigns clear it.
