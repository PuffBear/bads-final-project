# Phase 5 – Channel performance (weekly, platform-reported)

Script: `analysis/phase5_channels.py` → `outputs/tables/phase5_*.csv`, figures `outputs/figures/p5_*.png`, log `phase5_log.txt`.

**Caveat.** The data are **platform-reported**: each platform credits itself, so the six revenue lines overlap and can't be added up as truth. They are never joined to customer data. Instead, the channel ranking is cross-checked against the customer-level campaign results from Phase 4.

## 1. Where the money goes (`phase5_channel_kpis.csv`, `p5_01`)

| Channel | Spend share | Reported revenue share | ROAS | Cost / conversion | CTR | Conversion rate (CVR) | CPM |
|---|---|---|---|---|---|---|---|
| Search | **36%** | 22% | 4.6× | ₹297 | 4.3% | 4.5% | ₹569 |
| Social | **31%** | 14% | 3.3× | ₹407 | 4.2% | 2.1% | ₹365 |
| Display | 18% | 7% | 2.8× | ₹478 | 4.2% | 1.1% | ₹220 |
| Email | 7% | **31%** | **32.7×** | ₹41 | 4.2% | 3.0% | ₹52 |
| WhatsApp | 5% | 16% | 22.3× | ₹60 | 4.3% | 3.9% | ₹101 |
| SMS | 4% | 9% | 18.9× | ₹71 | 4.2% | 2.6% | ₹79 |

- **Spend and reported revenue are mismatched.** Paid media (Search, Social, Display) takes **84% of spend** but is credited with 43% of reported revenue. The owned channels (Email, WhatsApp, SMS) take 16% of spend and are credited with 57%.
- **Platform CTR is ~4.2% for every channel.** That is implausibly uniform and is a data-quality flag. Conversion rate and cost per thousand impressions are what actually differ.
- **Stable and improving.** ROAS ranking is identical in 2024, 2025 and 2026, and cost per conversion has fallen slightly on most channels (Social ₹431 → ₹382). Over weeks 1–43, revenue grew +20% (2025) and +19% (2026) on spend growth of +16% and +13%.

## 2. Are there diminishing returns? (`phase5_elasticities.csv`, `p5_02`, `p5_03`)
**Model.** Response curve: log revenue = a + β log spend, so β < 1 means diminishing returns. Three estimators:
- **Naive:** no controls.
- **Trend + seasonality:** linear trend plus three seasonal harmonics; standard errors corrected for week-to-week correlation.
- **Week fixed effects (preferred):** a panel across all 6 channels with a dummy for every week, so anything common to all channels that week is absorbed (seasonal demand, promotions, sales days). Each channel gets its own β; standard errors are clustered by week. Panel R² = 0.89.

| Channel | β naive | β trend + seasonality | **β week FE (95% CI)** | Average ROAS | **Marginal ROAS** (β × avg) |
|---|---|---|---|---|---|
| Email | 0.93 | 0.89 | 0.89 (0.63–1.15) | 32.7 | **29.0** |
| WhatsApp | 1.28 | 1.27 | 1.18 (0.95–1.42) | 22.3 | **26.4** |
| SMS | 1.12 | 1.06 | 1.07 (0.90–1.25) | 18.9 | **20.3** |
| Search | 1.10 | 1.03 | 1.05 (0.84–1.27) | 4.6 | 4.8 |
| Display | 1.17 | 1.18 | 1.09 (0.87–1.32) | 2.8 | 3.1 |
| Social | 0.96 | 0.73 | 0.89 (0.62–1.16) | 3.3 | 3.0 |

**Reading:**
- **No channel shows statistically significant diminishing returns.** Every interval includes 1. Within the spend range NexaGlow has used (roughly ±50% around the mean), **the next rupee earns about what the average rupee earns.** The data cannot show where saturation starts.
- **Controls move the estimates only modestly.** The largest shift is Social, from 0.96 to 0.73 with trend and seasonality. Week effects explain only 3.8% of the variance in channel spend, so the common budget cycle is not distorting the estimates much.
- **Marginal ROAS keeps the same ranking:** the owned channels at roughly 20–29×, paid media at roughly 3–5×.

## 3. Reallocation at constant budget (`phase5_budget_scenarios.csv`, `phase5_budget_moves_pct.csv`, `p5_04`)
**Method.** We maximise Σ R0_c·(s_c/s0_c)^β_c with total spend fixed at the last 52 weeks' level. β is capped at 1, so no increasing returns are extrapolated. Each channel can move at most ±20% (preferred) or ±50%.

| Bound | Email | SMS | WhatsApp | Search | Social | Display | Reported revenue change |
|---|---|---|---|---|---|---|---|
| **±20%** | +20% | +20% | +20% | +18% | −20% | −20% | **+4.8% to +10.8%** |
| ±50% | +50% | +50% | +50% | +45% | −50% | −50% | +11.7% to +26.6% |

The revenue range reflects how much credit the owned channels get: 25%, 50% or 100% of their platform-reported revenue.

**Reading:**
- **Direction is robust:** move money **from Social and Display to Email, SMS, WhatsApp and Search.** This holds even if the owned channels' reported revenue is cut to 25%.
- **Size is not robust.** With β ≈ 1, the optimiser pushes every channel to its bound, so the gain is set by the bound we choose, not discovered by the model. **Treat ±20% as the actionable step and the ±50% figure as an untested extrapolation.**
- **Owned channels can't be scaled freely.** Email/SMS/WhatsApp volume is capped by the consented customer list (81% consent, Phase 0). Phase 4 found no fatigue up to 3+ contacts in 30 days, but that range is narrow.

## 4. Cross-check against customer-level campaigns (`phase5_crosscheck_customer_level.csv`)
| Channel | Platform cost / conversion | Customer-level cost / order (Phase 4) | Platform CVR | Customer-level purchase rate |
|---|---|---|---|---|
| Email | ₹41 | ₹7 | 3.0% | 6.0% |
| SMS | ₹71 | ₹18 | 2.6% | 5.2% |
| WhatsApp | ₹60 | ₹22 | 3.9% | 6.3% |
| Display | **₹478** | **₹68** | 1.1% | 5.9% |
| Social | ₹407 | ₹160 | 2.1% | 4.9% |
| Search | ₹297 | ₹258 | 4.5% | 5.1% |

- **The two sources agree on ranking** (Spearman ρ = 0.71 for cost and for return). Owned channels are cheapest per order in both; Social and Search are most expensive.
- **The difference is in the audience.** Customer-level campaigns reach **existing customers**, and every channel converts them at 5–6%. Platform figures for Search, Social and Display also include **prospecting** audiences, which convert at 1–4.5%. Paid media's weak platform ROAS partly reflects acquisition work that the customer-level data can't see.
- **Display is the outlier.** It does well when aimed at existing customers (₹68 per order) and badly on the platform overall. That points to retargeting known customers rather than broad display.

## 5. Implications for the recommendations
1. **Shift about 20% of the Social and Display budget** to Email, SMS, WhatsApp (as far as the consented list allows) and Search. Expected reported-revenue gain is +5–11%. Phase in over a quarter and re-estimate.
2. **Don't cut paid media to zero.** It is the only route to new customers in this data. Re-aim Display at retargeting existing customers.
3. **Measure incrementality, not platform ROAS.** Run geo or time-split holdout tests per channel. Email's 33× is almost certainly over-credited: it reaches customers who are already engaged (Phase 4: email engagement predicts purchase).
4. **Push spend outside the past range in a controlled test** (e.g. +40% Search in some weeks or regions) to find where saturation begins. Current data can't locate it.
