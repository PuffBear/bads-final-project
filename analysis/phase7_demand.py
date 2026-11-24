"""Phase 7: demand over time - trend, seasonality, growth and a forecast for the 2027 plan.

Series : weekly totals across the 6 channels from weekly_channel_performance (Jan 2024 - Oct 2026, 148 weeks).
         Revenue is platform-reported and overlaps across channels, so it is a DEMAND INDEX, not booked revenue.
         Conversions are modelled alongside as a second index. The order table is a sample and is not used for trend.
Models : rolling-origin evaluation (3 origins, 26-week horizon, training >= 104 weeks so every model sees 2 seasons):
           1. seasonal naive with growth: same week last year x trailing 52-week growth
           2. Holt-Winters: additive damped trend, multiplicative seasonality (52)
           3. dynamic harmonic regression: log y ~ trend + Fourier(K=6) with AR(1) errors (SARIMAX)
           4. as 3 + log total ad spend (conditional on a spend plan) - shows how much of demand spend explains
Outputs: outputs/tables/phase7_*.csv, outputs/figures/p7_*.png
"""
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from common import load, save, plt, TAB, SERIES, INK2, MUTED, AXIS

warnings.filterwarnings("ignore")
wk = load("weekly_channel_performance")
tot = wk.groupby("week_start_date")[["ad_spend_inr", "attributed_revenue_inr", "conversions"]].sum().asfreq("7D")
assert tot.isna().sum().sum() == 0
P, K, H = 52, 6, 26
res = {}

# ------------------------------------------------------------------ 1. decomposition, growth, seasonality
# NOTE: STL was tried first and rejected. With ~2.85 years, each week-of-year has only 3 observations, so STL's
# seasonal smoother interpolates them exactly and the remainder collapses to 0 outside the middle year (one-off
# spikes become "seasonality"). A regression decomposition is used instead:
#     log y_t = a + b*t + sum_k [s_k sin(2 pi k t/52.18) + c_k cos(...)] + e_t ,  K = 6 (smooth seasonal shape)
from statsmodels.formula.api import ols as _ols
def decompose(y):
    d = pd.DataFrame({"ly": np.log(y.values), "t": np.arange(len(y))})
    for k in range(1, K + 1):
        d[f"s{k}"] = np.sin(2 * np.pi * k * d.t / 52.18); d[f"c{k}"] = np.cos(2 * np.pi * k * d.t / 52.18)
    m = _ols("ly ~ t + " + " + ".join(f"s{k} + c{k}" for k in range(1, K + 1)), data=d).fit()
    trend = m.params.Intercept + m.params.t * d.t
    seas = m.fittedvalues - trend
    return pd.DataFrame({"trend": trend.values, "seasonal": seas.values, "resid": m.resid.values}, index=y.index), m
dec = {c: decompose(tot[c]) for c in ["attributed_revenue_inr", "conversions", "ad_spend_inr"]}
r, rm = dec["attributed_revenue_inr"]
comp = pd.DataFrame({"revenue": tot.attributed_revenue_inr, "trend": np.exp(r.trend), "seasonal_factor": np.exp(r.seasonal),
                     "resid_factor": np.exp(r.resid)})
res["decomposition_fit"] = pd.Series({"R2": rm.rsquared, "trend growth per year": np.exp(rm.params.t * 52.18) - 1,
                                      "seasonal swing (max/min factor)": comp.seasonal_factor.max() / comp.seasonal_factor.min(),
                                      "remainder sd (log)": r.resid.std()}).round(3)
comp.round(4).to_csv(TAB / "phase7_decomposition.csv")
res["variance_shares_log"] = pd.Series({"trend": r.trend.var(), "seasonal": r.seasonal.var(), "residual": r.resid.var()}).div(np.log(tot.attributed_revenue_inr).var()).round(3)
for c_ in ["conversions", "ad_spend_inr"]:
    res[f"trend_growth_per_year_{c_}"] = round(float(np.exp(dec[c_][1].params.t * 52.18) - 1), 4)
tr = comp.trend
res["trend_growth"] = pd.Series({"Jan24->Jan25": tr.iloc[52] / tr.iloc[0] - 1, "Jan25->Jan26": tr.iloc[104] / tr.iloc[52] - 1,
                                 "Oct25->Oct26 (last 52w)": tr.iloc[-1] / tr.iloc[-53] - 1}).round(3)
yrs = tot.assign(woy=tot.index.isocalendar().week.values, year=tot.index.year)
same = yrs[yrs.woy <= 43].groupby("year")[["attributed_revenue_inr", "conversions", "ad_spend_inr"]].sum()
res["yoy_weeks_1_43"] = same.pct_change().mul(100).round(1)
roas = tot.attributed_revenue_inr / tot.ad_spend_inr
res["blended_roas_by_year"] = roas.groupby(roas.index.year).mean().round(2)
# monthly seasonal index (average seasonal factor by calendar month) and year-to-year stability
sf = pd.DataFrame({"sf": np.exp(r.seasonal + r.resid), "month": tot.index.month, "year": tot.index.year})  # detrended actuals, per year
sf_smooth = pd.DataFrame({"sf": np.exp(r.seasonal), "month": tot.index.month})
mon = sf_smooth.groupby("month").sf.mean()
mon_by_year = sf.pivot_table(index="month", columns="year", values="sf")
res["monthly_seasonal_index"] = (mon * 100).round(0)
detr = pd.DataFrame({"v": np.log(tot.attributed_revenue_inr).values - r.trend.values, "month": tot.index.month, "year": tot.index.year})
mcorr = detr.pivot_table(index="month", columns="year", values="v").corr()
res["monthly_pattern_corr_between_years"] = mcorr.round(2)
woy_corr = detr.assign(woy=tot.index.isocalendar().week.values).pivot_table(index="woy", columns="year", values="v").corr()
res["weekly_pattern_corr_between_years"] = woy_corr.round(2)
pk = comp.assign(index_vs_trend=comp.revenue / comp.trend).sort_values("index_vs_trend", ascending=False)
res["peak_weeks_vs_trend"] = pk.head(10)[["revenue", "trend", "index_vs_trend"]].round(2)
res["trough_weeks_vs_trend"] = pk.tail(6)[["revenue", "trend", "index_vs_trend"]].round(2)
mon_tab = pd.DataFrame({"seasonal_index": mon * 100, **{f"index_{y}": mon_by_year[y] * 100 for y in mon_by_year.columns}}).round(0)
mon_tab.to_csv(TAB / "phase7_monthly_seasonality.csv")
# how much of the demand index moves with spend?
from statsmodels.formula.api import ols
dd = pd.DataFrame({"lr": np.log(tot.attributed_revenue_inr), "ls": np.log(tot.ad_spend_inr), "t": np.arange(len(tot))})
for k in range(1, K + 1):
    dd[f"s{k}"] = np.sin(2 * np.pi * k * dd.t / 52.18); dd[f"c{k}"] = np.cos(2 * np.pi * k * dd.t / 52.18)
fx = " + ".join([f"s{k} + c{k}" for k in range(1, K + 1)])
m_a = ols(f"lr ~ t + {fx}", data=dd).fit(); m_b = ols(f"lr ~ t + {fx} + ls", data=dd).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
res["spend_vs_demand"] = pd.Series({"R2 trend+season": m_a.rsquared, "R2 + log spend": m_b.rsquared, "spend elasticity (total)": m_b.params.ls,
                                    "se": m_b.bse.ls, "trend per year without spend": np.exp(m_a.params.t * 52.18) - 1,
                                    "trend per year net of spend": np.exp(m_b.params.t * 52.18) - 1}).round(4)

# ------------------------------------------------------------------ 2. forecasting models + rolling-origin evaluation
def fourier(idx_t, k=K):
    return np.column_stack([f(2 * np.pi * j * idx_t / 52.18) for j in range(1, k + 1) for f in (np.sin, np.cos)])

def fc_snaive(y, h):
    g = y.iloc[-P:].sum() / y.iloc[-2 * P:-P].sum()
    return np.array([y.iloc[-P + (i % P)] * g for i in range(h)]), None

def fc_hw(y, h):
    m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=P, initialization_method="estimated").fit()
    f = m.forecast(h).values
    sims = m.simulate(h, repetitions=500, error="mul", anchor="end").values
    return f, (np.percentile(sims, [5, 10, 90, 95], axis=1))

DHR_PARAMS = {}
def fc_dhr(y, h, exog_hist=None, exog_fut=None):
    t0 = np.arange(len(y)); t1 = np.arange(len(y), len(y) + h)
    # intercept carried as an exog column with trend='n': with phi near 1 a SARIMAX 'c' constant acts as a second drift
    X = np.column_stack([np.ones(len(t0)), t0 / 52.18, fourier(t0)]); Xf = np.column_stack([np.ones(h), t1 / 52.18, fourier(t1)])
    if exog_hist is not None:
        X = np.column_stack([X, np.log(exog_hist)]); Xf = np.column_stack([Xf, np.log(exog_fut)])
    m = SARIMAX(np.log(y.values), exog=X, order=(1, 0, 0), trend="n").fit(disp=False)
    DHR_PARAMS[len(y)] = dict(phi=m.params[-2], growth_per_year=np.exp(m.params[1]) - 1)
    p = m.get_forecast(h, exog=Xf)
    ci90, ci80 = p.conf_int(alpha=.10), p.conf_int(alpha=.20)
    sig2 = p.var_pred_mean
    mean = np.exp(p.predicted_mean + sig2 / 2)  # log-normal bias correction
    return mean, np.exp(np.vstack([ci90[:, 0], ci80[:, 0], ci80[:, 1], ci90[:, 1]]))

def metrics(a, f):
    e = f - a
    return dict(MAPE=np.mean(np.abs(e) / a) * 100, RMSE=np.sqrt(np.mean(e ** 2)), bias_pct=e.sum() / a.sum() * 100,
                total_error_pct=(f.sum() - a.sum()) / a.sum() * 100)

ev, holdout_preds = [], {}
origins = [len(tot) - H - 18, len(tot) - H - 9, len(tot) - H]
for target in ["attributed_revenue_inr", "conversions"]:
    for o in origins:
        y_tr, y_te = tot[target].iloc[:o], tot[target].iloc[o:o + H]
        sp_tr, sp_te = tot.ad_spend_inr.iloc[:o], tot.ad_spend_inr.iloc[o:o + H]
        for nm, fn in [("Seasonal naive x growth", lambda: fc_snaive(y_tr, len(y_te))), ("Holt-Winters (damped, mult.)", lambda: fc_hw(y_tr, len(y_te))),
                       ("Harmonic regression + AR(1)", lambda: fc_dhr(y_tr, len(y_te))),
                       ("Harmonic regression + AR(1) + actual spend", lambda: fc_dhr(y_tr, len(y_te), sp_tr, sp_te))]:
            f, _ = fn()
            ev.append(dict(target=target, origin=str(tot.index[o].date()), model=nm, **metrics(y_te.values, f)))
            if o == origins[-1]: holdout_preds[(target, nm)] = f
ev = pd.DataFrame(ev)
ev.round(3).to_csv(TAB / "phase7_forecast_evaluation.csv", index=False)
summ = ev.groupby(["target", "model"])[["MAPE", "RMSE", "bias_pct", "total_error_pct"]].mean().round(2)
res["forecast_eval_mean_over_3_origins"] = summ
res["forecast_eval_last_origin"] = ev[ev.origin == str(tot.index[origins[-1]].date())].set_index(["target", "model"])[["MAPE", "bias_pct", "total_error_pct"]].round(2)
uni = summ.loc["attributed_revenue_inr"].drop(index="Harmonic regression + AR(1) + actual spend")
best = uni.MAPE.idxmin()
res["best_univariate_revenue"] = best

# ------------------------------------------------------------------ 3. forecast Nov 2026 - Dec 2027 (61 weeks)
fut_idx = pd.date_range(tot.index[-1] + pd.Timedelta(days=7), "2027-12-27", freq="7D")
h = len(fut_idx)
fcs = {}
for target in ["attributed_revenue_inr", "conversions"]:
    y = tot[target]
    f_hw, pi_hw = fc_hw(y, h); f_dhr, pi_dhr = fc_dhr(y, h)
    f_sn, _ = fc_snaive(y, h)
    ens = (f_hw + f_dhr) / 2
    fcs[target] = pd.DataFrame({"holt_winters": f_hw, "harmonic_ar1": f_dhr, "seasonal_naive": f_sn, "ensemble": ens,
                                # weekly intervals from the log-scale harmonic model (positive by construction);
                                # Holt-Winters simulation intervals with additive trend widen without bound
                                "lo90": pi_dhr[0], "lo80": pi_dhr[1], "hi80": pi_dhr[2], "hi90": pi_dhr[3]}, index=fut_idx)
    fcs[target].round(1).to_csv(TAB / f"phase7_forecast_{'revenue' if 'rev' in target else 'conversions'}.csv")
fr = fcs["attributed_revenue_inr"]
hist_m = tot.attributed_revenue_inr.resample("MS").sum()
fut_m = fr[["ensemble", "holt_winters", "harmonic_ar1", "seasonal_naive"]].resample("MS").sum()
fut_m["model_low"] = fut_m[["holt_winters", "harmonic_ar1", "seasonal_naive"]].min(axis=1)
fut_m["model_high"] = fut_m[["holt_winters", "harmonic_ar1", "seasonal_naive"]].max(axis=1)
y27 = fr.loc["2027"]
y26_actual_plus_fc = tot.attributed_revenue_inr.loc["2026"].sum() + fr.loc[:"2026-12-31", "ensemble"].sum()
annual = pd.Series({"2024 actual": tot.attributed_revenue_inr.loc["2024"].sum(), "2025 actual": tot.attributed_revenue_inr.loc["2025"].sum(),
                    "2026 actual Jan-Oct + forecast Nov-Dec": y26_actual_plus_fc, "2027 forecast (ensemble)": y27.ensemble.sum(),
                    "2027 Holt-Winters": y27.holt_winters.sum(), "2027 harmonic+AR1": y27.harmonic_ar1.sum(), "2027 seasonal naive": y27.seasonal_naive.sum()})
res["annual_totals_lakh"] = (annual / 1e5).round(1)
res["growth_2027_vs_2026"] = round(y27.ensemble.sum() / y26_actual_plus_fc - 1, 4)
m27 = fut_m.loc["2027"].copy(); m27["share_of_year"] = m27.ensemble / m27.ensemble.sum()
m27.index = m27.index.strftime("%b")
m27_out = m27[["ensemble", "model_low", "model_high"]].div(1e5).assign(share_of_year=m27.share_of_year, seasonal_index=mon.values * 100)
m27_out.round(3).to_csv(TAB / "phase7_forecast_2027_monthly.csv")
res["forecast_2027_monthly_lakh"] = m27_out.round(2)
bt = summ.loc["attributed_revenue_inr", "total_error_pct"].abs().max()
res["annual_2027_range_lakh"] = pd.Series({"model low": annual[["2027 Holt-Winters", "2027 harmonic+AR1", "2027 seasonal naive"]].min() / 1e5,
                                           "ensemble": annual["2027 forecast (ensemble)"] / 1e5,
                                           "model high": annual[["2027 Holt-Winters", "2027 harmonic+AR1", "2027 seasonal naive"]].max() / 1e5,
                                           "max backtest 26-wk total error %": bt}).round(1)
fc_conv = fcs["conversions"]
res["conversions_2027"] = pd.Series({"2025 actual": tot.conversions.loc["2025"].sum(), "2027 forecast": fc_conv.loc["2027", "ensemble"].sum()}).round(0)

# ================================================================== figures
# p7_01 decomposition
fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1, .7]))
axs[0].plot(tot.index, tot.attributed_revenue_inr / 1e5, color=SERIES[0], lw=1.4, label="weekly reported revenue")
axs[0].plot(tot.index, comp.trend / 1e5, color=SERIES[1], lw=2, label="trend (log-linear)")
axs[0].set_ylabel("₹ lakh"); axs[0].legend(fontsize=7.5, loc="upper left"); axs[0].set_title("Demand index: total platform-reported revenue, 6 channels", fontsize=9)
axs[1].plot(tot.index, (comp.seasonal_factor - 1) * 100, color=SERIES[2], lw=1.4); axs[1].axhline(0, color=INK2, lw=.8)
axs[1].set_ylabel("% vs trend"); axs[1].set_title("Seasonal component (smooth, 6 Fourier pairs)", fontsize=9)
axs[2].vlines(tot.index, 0, (comp.resid_factor - 1) * 100, color=MUTED, lw=2.2); axs[2].axhline(0, color=INK2, lw=.8); axs[2].set_ylabel("%"); axs[2].set_title("Remainder", fontsize=9)
save(fig, "p7_01_decomposition")

# p7_02 monthly seasonal index by year + YoY
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
ax = axs[0]; mn = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for i_, yv in enumerate(mon_by_year.columns):
    ax.plot(range(1, 13), mon_by_year[yv] * 100, marker="o", ms=5, lw=1.4, color=SERIES[i_], label=str(yv))
ax.plot(range(1, 13), mon * 100, color=INK2, lw=2.2, ls="--", label="fitted seasonal")
ax.axhline(100, color=AXIS, lw=1); ax.set_xticks(range(1, 13), mn); ax.set_ylabel("index (100 = trend)"); ax.legend(fontsize=7, ncol=4)
ax.set_title("Monthly index vs trend: each year's actuals and the fitted seasonal shape", fontsize=9)
ax = axs[1]; g = res["yoy_weeks_1_43"].loc[[2025, 2026]]
x = np.arange(2); w = .26
for i_, (c, lab) in enumerate([("ad_spend_inr", "ad spend"), ("attributed_revenue_inr", "reported revenue"), ("conversions", "conversions")]):
    ax.bar(x + (i_ - 1) * w, g[c], width=w * .92, color=SERIES[i_], label=lab)
    for j_, v_ in enumerate(g[c]): ax.text(x[j_] + (i_ - 1) * w, v_, f"{v_:+.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, ["2025 vs 2024", "2026 vs 2025"]); ax.set_ylim(0, 27); ax.legend(fontsize=7, ncol=3, loc="upper center"); ax.grid(axis="x", visible=False)
ax.set_title("Year-on-year growth, weeks 1-43", fontsize=9)
save(fig, "p7_02_seasonality_growth")

# p7_03 hold-out comparison at last origin
o = origins[-1]
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(tot.index[o - 40:], tot.attributed_revenue_inr.iloc[o - 40:] / 1e5, color=INK2, lw=1.6, label="actual")
for i_, nm in enumerate(["Seasonal naive x growth", "Holt-Winters (damped, mult.)", "Harmonic regression + AR(1)"]):
    mp = summ.loc[("attributed_revenue_inr", nm), "MAPE"]
    ax.plot(tot.index[o:o + H], holdout_preds[("attributed_revenue_inr", nm)] / 1e5, color=SERIES[i_], lw=1.5, label=f"{nm} (mean MAPE {mp:.1f}%)")
ax.axvline(tot.index[o], color=MUTED, ls="--", lw=1); ax.set_ylabel("₹ lakh"); ax.legend(fontsize=7.5, loc="upper left")
ax.set_title(f"Hold-out check: last 26 weeks ({tot.index[o]:%d %b %Y} – {tot.index[-1]:%d %b %Y}) forecast from earlier data", fontsize=9)
save(fig, "p7_03_holdout")

# p7_04 forecast
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(tot.index, tot.attributed_revenue_inr / 1e5, color=INK2, lw=1.2, label="actual")
ax.fill_between(fr.index, fr.lo90 / 1e5, fr.hi90 / 1e5, color=SERIES[0], alpha=.12, lw=0, label="90% interval")
ax.fill_between(fr.index, fr.lo80 / 1e5, fr.hi80 / 1e5, color=SERIES[0], alpha=.22, lw=0, label="80% interval")
ax.plot(fr.index, fr.ensemble / 1e5, color=SERIES[0], lw=1.8, label="forecast (average of Holt-Winters and harmonic regression)")
ax.text(fr.index[-1], fr.ensemble.iloc[-1] / 1e5, "", fontsize=7)
ax.text(.995, .02, "intervals: harmonic regression (log scale)", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
ax.set_ylabel("₹ lakh per week"); ax.legend(fontsize=7.5, loc="upper left")
ax.set_title(f"Weekly demand forecast to Dec 2027: 2027 total ≈ ₹{y27.ensemble.sum()/1e7:.1f} crore reported revenue ({res['growth_2027_vs_2026']:+.0%} vs 2026)", fontsize=9)
save(fig, "p7_04_forecast")

pd.set_option("display.width", 250, "display.max_columns", 30)
for k, v in res.items():
    print(f"\n### {k}\n{v}")
