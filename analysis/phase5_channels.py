"""Phase 5: channel performance from the weekly platform-reported table (148 weeks x 6 channels).

Outputs: outputs/tables/phase5_*.csv, outputs/figures/p5_*.png
Notes  : * PLATFORM-REPORTED. Each platform credits itself; the six revenue figures overlap and are not additive
           truth. Never joined to customer-level data. We cross-check the channel ranking against Phase 4's
           customer-level campaign economics instead.
         * Spend moves together across channels (r = 0.50-0.66): budgets follow common seasonal demand. A per-channel
           regression of revenue on spend would credit spend for demand that would have come anyway. The preferred
           model is a panel with WEEK FIXED EFFECTS, so elasticities come only from a channel's spend moving relative
           to the other channels in the same week.
         * Response curve: revenue_c = A_c * spend_c ^ beta_c. beta < 1 = diminishing returns.
           Marginal ROAS = beta x average ROAS (extra revenue from the next rupee).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import spearmanr
from common import load, save, plt, TAB, SERIES, INK2, MUTED, AXIS, CHANNEL_COLORS

wk = load("weekly_channel_performance")
CH = list(CHANNEL_COLORS)
res = {}
wk["year"] = wk.week_start_date.dt.year
wk["t"] = (wk.week_start_date - wk.week_start_date.min()).dt.days / 7
wk["woy"] = wk.week_start_date.dt.isocalendar().week.astype(int)
for k in (1, 2, 3):
    wk[f"s{k}"] = np.sin(2 * np.pi * k * wk.t / 52.18); wk[f"c{k}"] = np.cos(2 * np.pi * k * wk.t / 52.18)
wk["ls"] = np.log(wk.ad_spend_inr); wk["lr"] = np.log(wk.attributed_revenue_inr); wk["lconv"] = np.log(wk.conversions)
wk["week"] = wk.week_start_date.dt.strftime("%Y-%m-%d")

# ------------------------------------------------------------------ 1. KPIs by channel and by year
def kpi(g):
    sp, rv = g.ad_spend_inr.sum(), g.attributed_revenue_inr.sum()
    return pd.Series({"spend": sp, "revenue": rv, "conversions": g.conversions.sum(), "ROAS": rv / sp,
                      "CPA": sp / g.conversions.sum(), "CTR": g.clicks.sum() / g.impressions.sum(), "CVR": g.conversions.sum() / g.clicks.sum(),
                      "CPC": sp / g.clicks.sum(), "CPM": sp / g.impressions.sum() * 1000, "AOV": rv / g.conversions.sum()})
k_all = wk.groupby("channel").apply(kpi).reindex(CH)
k_all["spend_share"] = k_all.spend / k_all.spend.sum(); k_all["revenue_share"] = k_all.revenue / k_all.revenue.sum()
k_all.round(4).to_csv(TAB / "phase5_channel_kpis.csv")
res["kpis"] = k_all.round(3)
k_year = wk.groupby(["channel", "year"]).apply(kpi)
k_year.round(4).to_csv(TAB / "phase5_channel_kpis_by_year.csv")
res["roas_by_year"] = k_year.ROAS.unstack().reindex(CH).round(2)
res["cpa_by_year"] = k_year.CPA.unstack().reindex(CH).round(1)
yr = wk[wk.woy <= 43].groupby("year")[["ad_spend_inr", "attributed_revenue_inr", "conversions"]].sum()
res["growth_wk1_43"] = (yr.pct_change() * 100).round(1)

# ------------------------------------------------------------------ 2. response models
rows = []
for ch in CH:
    d = wk[wk.channel == ch]
    m0 = smf.ols("lr ~ ls", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    m1 = smf.ols("lr ~ ls + t + s1 + c1 + s2 + c2 + s3 + c3", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    m1c = smf.ols("lconv ~ ls + t + s1 + c1 + s2 + c2 + s3 + c3", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    rows.append(dict(channel=ch, beta_naive=m0.params.ls, beta_naive_se=m0.bse.ls,
                     beta_trend_season=m1.params.ls, beta_trend_season_se=m1.bse.ls, r2_trend_season=m1.rsquared,
                     beta_conversions_trend_season=m1c.params.ls))
el = pd.DataFrame(rows).set_index("channel")
# week fixed effects panel with channel-specific slopes; SEs clustered by week
pm = smf.ols("lr ~ C(channel) + C(week) + " + " + ".join(f"I(ls * (channel == '{c}'))" for c in CH), data=wk).fit(
    cov_type="cluster", cov_kwds={"groups": wk.week.astype("category").cat.codes})
for c in CH:
    nm = f"I(ls * (channel == '{c}'))"
    el.loc[c, "beta_week_fe"] = pm.params[nm]; el.loc[c, "beta_week_fe_se"] = pm.bse[nm]
    el.loc[c, "ci_low"] = pm.conf_int().loc[nm, 0]; el.loc[c, "ci_high"] = pm.conf_int().loc[nm, 1]
    el.loc[c, "p_beta_lt_1"] = __import__("scipy").stats.norm.cdf((pm.params[nm] - 1) / pm.bse[nm])
el["avg_ROAS"] = k_all.ROAS
el["marginal_ROAS"] = el.beta_week_fe * el.avg_ROAS
el["marginal_ROAS_low"] = el.ci_low * el.avg_ROAS
el["marginal_ROAS_high"] = el.ci_high * el.avg_ROAS
el.round(4).to_csv(TAB / "phase5_elasticities.csv")
res["elasticities"] = el.round(3)
res["panel_r2"] = round(pm.rsquared, 4)

# common-trend check: share of variance in channel log-spend explained by week effects
sp_fe = smf.ols("ls ~ C(channel) + C(week)", data=wk).fit()
res["spend_variance_explained_by_week_effects"] = round(sp_fe.rsquared - smf.ols("ls ~ C(channel)", data=wk).fit().rsquared, 4)

# ------------------------------------------------------------------ 3. budget reallocation (same total, bounded moves)
# Use the last 52 weeks as the 'current' plan. Revenue model per channel: R_c(s) = R0_c * (s / s0_c)^beta_c,
# anchored at current average weekly spend s0 and revenue R0 (so the curve passes through today's position).
last = wk[wk.week_start_date > wk.week_start_date.max() - pd.Timedelta(weeks=52)]
s0 = last.groupby("channel").ad_spend_inr.mean().reindex(CH)
R0 = last.groupby("channel").attributed_revenue_inr.mean().reindex(CH)
def optimise(beta, R0_, lo=0.5, hi=1.5):
    S = s0.sum()
    f = lambda s: -np.sum(R0_.values * (s / s0.values) ** beta.values)
    cons = ({"type": "eq", "fun": lambda s: s.sum() - S},)
    bnds = [(lo * v, hi * v) for v in s0.values]
    r = minimize(f, s0.values, bounds=bnds, constraints=cons, method="SLSQP")
    return pd.Series(r.x, index=CH), -r.fun
scen = {}
beta = el.beta_week_fe.clip(upper=1.0)  # no extrapolation of increasing returns
for nm, R0_ in [("as reported", R0),
                ("owned channels' revenue credited at 50%", R0 * np.where(np.isin(CH, ["Email", "SMS", "WhatsApp"]), .5, 1)),
                ("owned channels' revenue credited at 25%", R0 * np.where(np.isin(CH, ["Email", "SMS", "WhatsApp"]), .25, 1))]:
    for lo, hi in [(0.5, 1.5), (0.8, 1.2)]:
        s_opt, rev_opt = optimise(beta, pd.Series(R0_, index=CH), lo, hi)
        scen[(nm, f"±{round((hi-1)*100)}%")] = dict(**{f"spend_{c}": s_opt[c] for c in CH},
                                                  revenue_now=float(np.sum(R0_)), revenue_opt=rev_opt,
                                                  uplift_pct=(rev_opt / float(np.sum(R0_)) - 1) * 100)
scen = pd.DataFrame(scen).T
scen.round(1).to_csv(TAB / "phase5_budget_scenarios.csv")
res["budget_scenarios"] = scen[["uplift_pct"] + [f"spend_{c}" for c in CH]].round(1)
res["current_weekly_spend"] = s0.round(0)
moves = pd.DataFrame({k: (scen.loc[k, [f"spend_{c}" for c in CH]].values / s0.values - 1) * 100 for k in scen.index}, index=CH).T.round(0)
moves.to_csv(TAB / "phase5_budget_moves_pct.csv")
res["budget_moves_pct"] = moves

# ------------------------------------------------------------------ 4. cross-check vs customer-level campaign data (Phase 4)
p4 = pd.read_csv(TAB / "phase4_channel_economics.csv", index_col=0).reindex(CH)
xc = pd.DataFrame({"platform_CPA": k_all.CPA, "customer_level_cost_per_order": p4.cost_per_order,
                   "platform_ROAS": k_all.ROAS, "customer_level_rev_per_contact_rupee": p4.net_rev_per_exposure / p4.cost_per_exposure,
                   "platform_CVR": k_all.CVR, "customer_level_conv_rate": p4.conv_rate})
rho_cpa = spearmanr(xc.platform_CPA, xc.customer_level_cost_per_order)[0]
rho_roas = spearmanr(xc.platform_ROAS, xc.customer_level_rev_per_contact_rupee)[0]
xc.round(3).to_csv(TAB / "phase5_crosscheck_customer_level.csv")
res["crosscheck"] = xc.round(3); res["crosscheck_rank_corr"] = pd.Series({"CPA vs cost/order": rho_cpa, "ROAS vs revenue/contact-rupee": rho_roas}).round(3)

# ================================================================== figures
# p5_01 spend share vs revenue share + ROAS/CPA
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
ax = axs[0]; x = np.arange(len(CH))
ax.bar(x - .2, k_all.spend_share * 100, width=.4, color=SERIES[0], label="% of spend")
ax.bar(x + .2, k_all.revenue_share * 100, width=.4, color=SERIES[1], label="% of platform-reported revenue")
ax.set_xticks(x, CH, fontsize=8); ax.legend(fontsize=7); ax.grid(axis="x", visible=False); ax.set_title("Spend vs reported revenue, 2024–26", fontsize=9)
for ax, col, t, fmt in [(axs[1], "ROAS", "Reported ROAS (×)", "{:.1f}×"), (axs[2], "CPA", "Reported cost per conversion (₹)", "₹{:,.0f}")]:
    ax.bar(CH, k_all[col], color=[CHANNEL_COLORS[c] for c in CH], width=.6)
    for i_, v_ in enumerate(k_all[col]): ax.text(i_, v_, fmt.format(v_), ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_title(t, fontsize=9); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", labelsize=8)
save(fig, "p5_01_channel_kpis")

# p5_02 elasticities: three estimators
fig, ax = plt.subplots(figsize=(8, 3.8))
y_ = np.arange(len(CH))
for off, col, lab, c_ in [(-.22, "beta_naive", "naive (no controls)", MUTED), (0, "beta_trend_season", "trend + seasonality", SERIES[1]),
                          (.22, "beta_week_fe", "week fixed effects (preferred)", SERIES[0])]:
    se = el[col + "_se"] if col + "_se" in el else None
    ax.errorbar(el[col], y_ + off, xerr=1.96 * se, fmt="o", ms=6, color=c_, ecolor=c_, elinewidth=1.2, capsize=0, label=lab)
ax.axvline(1, color=INK2, lw=1, ls="--"); ax.text(1.005, len(CH) - .45, "constant returns", fontsize=7, color=INK2)
ax.set_yticks(y_, CH); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3)
ax.set_xlabel("spend elasticity of reported revenue (95% CI)")
ax.set_title("Diminishing returns? Elasticity < 1 means each extra rupee earns less")
save(fig, "p5_02_elasticities")

# p5_03 marginal vs average ROAS
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(x - .2, el.avg_ROAS, width=.4, color=MUTED, label="average ROAS (reported)")
ax.bar(x + .2, el.marginal_ROAS, width=.4, color=SERIES[0], label="marginal ROAS (next rupee)")
ax.errorbar(x + .2, el.marginal_ROAS, yerr=[el.marginal_ROAS - el.marginal_ROAS_low, el.marginal_ROAS_high - el.marginal_ROAS], fmt="none", ecolor=INK2, lw=1)
for i_, v_ in enumerate(el.marginal_ROAS): ax.text(i_ + .42, v_, f"{v_:.1f}", ha="left", va="center", fontsize=7.5, color=INK2)
ax.axhline(1, color=SERIES[7], lw=1, ls=":"); ax.set_xticks(x, CH); ax.set_yscale("log"); ax.legend(fontsize=7.5); ax.grid(axis="x", visible=False)
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 40])); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.set_title("Average vs marginal ROAS (platform-reported; log scale)")
save(fig, "p5_03_marginal_roas")

# p5_04 budget moves (moves do not depend on the owned-channel haircut; only the revenue gain does)
fig, ax = plt.subplots(figsize=(9, 3.6))
for i_, bound in enumerate(["±20%", "±50%"]):
    v = moves.loc[("as reported", bound)]
    ups = scen.xs(bound, level=1).uplift_pct
    ax.bar(x + (i_ - .5) * .38, v.values, width=.36, color=SERIES[i_],
           label=f"bounds {bound}: reported revenue {ups.min():+.1f}% to {ups.max():+.1f}% (owned channels credited 25-100%)")
    for j_, vv in enumerate(v.values): ax.text(x[j_] + (i_ - .5) * .38, vv + (2 if vv > 0 else -2), f"{vv:+.0f}%", ha="center", va="bottom" if vv > 0 else "top", fontsize=7, color=INK2)
ax.axhline(0, color=INK2, lw=1); ax.set_xticks(x, CH); ax.set_ylabel("% change in weekly spend"); ax.grid(axis="x", visible=False); ax.set_ylim(-62, 62)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.1))
ax.set_title("Budget reallocation at constant total spend (baseline: last 52 weeks)")
save(fig, "p5_04_budget_reallocation")

# p5_05 yearly ROAS trend
fig, ax = plt.subplots(figsize=(8, 3.2))
for c in CH:
    ax.plot([2024, 2025, 2026], res["roas_by_year"].loc[c].values, marker="o", ms=6, color=CHANNEL_COLORS[c], label=c)
ax.set_yscale("log"); ax.set_xticks([2024, 2025, 2026], ["2024", "2025", "2026 (to Oct)"])
ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 35])); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.12)); ax.set_ylabel("ROAS (×, log)")
ax.set_title("Reported ROAS by year: stable channel ranking")
save(fig, "p5_05_roas_trend")

pd.set_option("display.width", 250, "display.max_columns", 30)
for k, v in res.items():
    print(f"\n### {k}\n{v}")
