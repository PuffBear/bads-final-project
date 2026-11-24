# Phase 7: Demand over time

Script: `analysis/phase7_demand.py`. Outputs are `outputs/tables/phase7_*.csv`, figures `outputs/figures/p7_*.png` and the log `phase7_log.txt`.

**The demand index.** Weekly totals across the six channels of the weekly table (Jan 2024 – Oct 2026, 148 complete weeks). Revenue is **platform-reported**, so credit overlaps between channels. It works as a **demand index** for trend and seasonality, not as booked revenue. Conversions are modelled alongside as a second index. The order table is a sample (Phase 0) and is not used for trends.

## 1. Decomposition (`phase7_decomposition.csv`, `p7_01_decomposition.png`)

**Model.** log demand = linear trend + smooth annual seasonality (6 Fourier pairs) + remainder. R² = 0.70.

**Why not STL?** STL was tried first and rejected. With fewer than 3 years of data, each week of the year has only 3 observations. STL's seasonal smoother then simply passes through them: one-off spikes become "seasonality", and the remainder drops to zero outside 2025.

| Component | Share of variance in log demand |
|---|---|
| Trend | 55% |
| Seasonality | 15% |
| Week-to-week noise | 30% |

- **Steady growth.** Trend growth is **+23% a year** for both reported revenue and conversions, and it is steady: the trend is linear in logs, with no slowdown visible. Comparing the same weeks (1–43) year on year: revenue +20% (2025) and +19% (2026); conversions +19% and +21%; ad spend +16% and +13%.
- **About two-thirds of the growth moves with ad spend.** Adding log total spend to the model (elasticity 0.78 ± 0.14) cuts the unexplained trend from **+23% to +8% a year**. Blended ROAS has crept up from 7.0 to 7.3 to 7.6.
- **Single-week spikes dominate the noise.** The largest is the week of 25 May 2026 at 1.9× trend, which overlaps the start of the VIP Early Access campaign (C009). Weekly patterns repeat only weakly from year to year (r = 0.32–0.48), so **plan at monthly resolution**.

## 2. Seasonality (`phase7_monthly_seasonality.csv`, `p7_02_seasonality_growth.png`)

| Month | Jan | Feb | Mar | Apr | **May** | **Jun** | Jul | Aug | **Sep** | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Index (100 = trend) | 90 | 96 | 100 | 108 | **116** | **112** | 104 | 101 | **88** | 95 | 93 | 101 |

- **One clear peak season: April–June** (8–16% above trend). Each year's actual May sits 13–27% above trend.
- **Two troughs:** **September** (all three years at 82–96, about 12% below trend) and **January** (86–88).
- **December is volatile.** It ran 14% above trend in 2025 but 3% below in 2024. October–December can't be checked for 2026 yet.
- **The monthly shape is consistent** across years (pairwise r = 0.62–0.93), so it is a reliable base for timing the calendar.

## 3. Forecast accuracy (`phase7_forecast_evaluation.csv`, `p7_03_holdout.png`)

**Back-test design.** Rolling origin with 3 start points and a 26-week horizon. Every training window has at least 104 weeks, so each model sees two full seasons.

| Model | Revenue weekly MAPE | Revenue 26-week total error | Conversions weekly MAPE |
|---|---|---|---|
| Seasonal naive × trailing growth | 14.2% | −2.7% | 12.4% |
| Holt-Winters (damped trend, multiplicative season) | 15.6% | −0.3% | 13.8% |
| **Harmonic regression + AR(1)** (log scale) | **12.9%** | −1.1% | **10.5%** |
| *Harmonic regression + AR(1) + actual ad spend* | *10.9%* | *−1.7%* | *9.2%* |

- **Single weeks are hard; totals are not.** Weekly errors of 13–16% reflect the one-off spikes. Errors on the 26-week *total* are 3% or less for every model, and that is the level the annual plan needs.
- **Knowing the spend plan helps.** Including actual spend cuts weekly error by about 2 points. Revenue should therefore be forecast *conditional on the spend plan* once it is set.

## 4. Forecast to December 2027 (`phase7_forecast_revenue.csv`, `phase7_forecast_conversions.csv`, `phase7_forecast_2027_monthly.csv`, `p7_04_forecast.png`)

**Method.** The point forecast is the average of Holt-Winters and the harmonic regression. Weekly 80%/90% intervals come from the log-scale harmonic model.

| Reported revenue (demand index) | ₹ lakh | vs previous year |
|---|---|---|
| 2024 actual | 1,000 | |
| 2025 actual | 1,223 | +22% |
| 2026 (Jan–Oct actual + Nov–Dec forecast) | 1,484 | +21% |
| **2027 forecast** | **1,771** (model range 1,712–1,831) | **+19%** |

- **Conversions:** about 134,000 platform-reported conversions in 2027, against 90,000 in 2025.
- **Monthly profile for 2027** (₹ lakh, ensemble): Jan 110, Feb 120, Mar 162, Apr 142, **May 195**, Jun 153, Jul 139, Aug 174, Sep 122, Oct 135, Nov 168, Dec 152.
- **Seasonal shares:** April–June is about 28% of the year; September is the weakest month at about 7%.
- **Assumptions:**
  - Spend keeps growing roughly as in 2024–26 (+13–18% a year), since most of the trend moves with spend.
  - The platforms' attribution rules don't change.
  - Uncertainty grows with the horizon. The model range (±3.4%) and the back-tested total error (≤3%) cover *model* risk, not shocks such as a pricing change or a new competitor.

## 5. Implications for the annual plan
1. **Plan for ~+19% growth in 2027.** Set the budget conditional on spend: underlying growth net of spend is about **+8% a year**, so holding spend flat would mean roughly +8%, not +19%.
2. **Front-load into April–June**, the peak that holds every year. Prepare stock, creative and campaigns from March. Two of the best-converting 2026 campaigns (C006 SPF Drive, C007 Weekend Flash Sale) ran in April–May. The July–August campaigns (C011, C012, C013) converted just as well, so the shoulder months after the peak are worth keeping.
3. **Don't pour money into September and January.** Use them for low-cost retention via owned channels (Phase 4/5), or test whether well-timed offers can lift the trough. Discount-led campaigns there need a holdout to show they are incremental.
4. **Re-forecast monthly** with actual spend as an input. That model was the most accurate in back-testing.
5. **Measure booked revenue, not just platform claims.** The demand index overlaps across platforms; a booked-sales series (full order table, not a sample) would make the forecast directly usable for finance.
