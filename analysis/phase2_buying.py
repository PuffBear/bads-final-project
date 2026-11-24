"""Phase 2: buying behaviour (orders + order_items + product_catalog + Phase 1 segments).

Outputs: outputs/tables/phase2_*.csv, outputs/figures/p2_*.png, outputs/tables/order_level.csv (reused by Phase 3)
Notes  : * the order table is a SAMPLE (2025 ~11k orders, 2026 ~3.3k); shares and averages are fine, totals are not.
         * segments come from the 31-Dec-2025 snapshot. 2025 orders describe how those segments bought before
           they were assigned (descriptive); 2026 orders are post-assignment.
         * order lines can repeat a product (1,567 lines) -> 'distinct products' counts de-duplicate.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu, wilcoxon
from common import (load, save, plt, TAB, SERIES, INK2, MUTED, AXIS, CATEGORY_COLORS, SEQ_CMAP, DIV_CMAP)

SEED = 42
od, oi, pc, cu, cc, ex = (load(n) for n in ["orders", "order_items", "product_catalog", "customers", "campaign_catalog", "campaign_exposures"])
seg = pd.read_csv(TAB / "customer_segments.csv")
CAT = list(CATEGORY_COLORS)
SEGS = ["Recent actives", "Lapsing regulars", "Low-basket buyers", "Occasional buyers", "Registered, never purchased"]
SEG_COL = dict(zip(SEGS, SERIES[:4] + [MUTED]))
INC = ["Under 4L", "4L-8L", "8L-15L", "15L-25L", "25L+"]
THRESH, FEE = 999, 49
res = {}

# ------------------------------------------------------------------ order-level table
li = oi.merge(pc, on="product_id")
cat_rev = li.pivot_table(index="order_id", columns="category", values="line_gross_value_inr", aggfunc="sum", fill_value=0)[CAT]
basket = li.groupby("order_id").agg(lines=("order_item_id", "size"), products=("product_id", "nunique"), units=("quantity", "sum"),
                                    categories=("category", "nunique"))
o = (od.merge(basket, left_on="order_id", right_index=True)
       .merge(cat_rev.add_prefix("rev_"), left_on="order_id", right_index=True)
       .merge(cu[["customer_id", "city", "region", "annual_income_band_inr", "age", "gender", "preferred_category"]], on="customer_id")
       .merge(seg[["customer_id", "segment"]], on="customer_id"))
o["year"] = o.order_date.dt.year
o["attributed"] = o.attributed_exposure_id.notna()
o["net_value"] = o.gross_merchandise_value_inr - o.discount_inr
o["cod"] = (o.payment_mode == "COD").astype(int)
o["paid_shipping"] = (o.shipping_fee_inr > 0).astype(int)
o["discount_pct"] = o.discount_inr / o.gross_merchandise_value_inr
o["has_bundle"] = (o.rev_Bundles > 0).astype(int)
o["main_category"] = o[[f"rev_{c}" for c in CAT]].idxmax(axis=1).str.replace("rev_", "")
o["buys_preferred"] = o.apply(lambda r: r[f"rev_{r.preferred_category}"] > 0, axis=1)
o["segment"] = pd.Categorical(o.segment, SEGS, ordered=True)
o.to_csv(TAB / "order_level.csv", index=False)

def kpis(g):
    return pd.Series({
        "orders": len(g), "customers": g.customer_id.nunique(),
        "aov_total": g.order_total_inr.mean(), "median_total": g.order_total_inr.median(),
        "gmv_mean": g.gross_merchandise_value_inr.mean(), "products_per_order": g.products.mean(), "units_per_order": g.units.mean(),
        "multi_product_share": (g.products > 1).mean(), "multi_category_share": (g.categories > 1).mean(),
        "bundle_share": g.has_bundle.mean(), "discounted_share": (g.discount_inr > 0).mean(),
        "avg_discount_pct_when_discounted": g.loc[g.discount_inr > 0, "discount_pct"].mean(),
        "paid_shipping_share": g.paid_shipping.mean(), "cod_share": g.cod.mean(),
        **{f"rev_share_{c}": g[f"rev_{c}"].sum() / g.gross_merchandise_value_inr.sum() for c in CAT}})

# ------------------------------------------------------------------ 1. headline KPIs: overall, by year, attributed vs organic
head = pd.concat({"All orders": kpis(o), "2025 (sample)": kpis(o[o.year == 2025]), "2026 (sample)": kpis(o[o.year == 2026]),
                  "2026 organic": kpis(o[(o.year == 2026) & ~o.attributed]), "2026 campaign-attributed": kpis(o[o.attributed])}, axis=1)
head.round(4).to_csv(TAB / "phase2_headline_kpis.csv")
res["headline"] = head.round(3)

# ------------------------------------------------------------------ 2. by segment (2025 = behaviour that defined them; 2026 = after)
by_seg = pd.concat({y: o[o.year == y].groupby("segment", observed=True).apply(kpis) for y in (2025, 2026)}, names=["year"])
by_seg.round(4).to_csv(TAB / "phase2_by_segment.csv")
res["by_segment_2025"] = by_seg.loc[2025].T.round(3)
res["by_segment_2026"] = by_seg.loc[2026][["orders", "aov_total", "products_per_order", "bundle_share", "cod_share", "discounted_share"]].round(3)
tests = []
o25 = o[o.year == 2025]
h, p = kruskal(*[g.order_total_inr for _, g in o25.groupby("segment", observed=True)])
tests.append(dict(question="2025 order value differs by segment", test="Kruskal-Wallis", stat=h, p_value=p, n=len(o25)))
for var in ["cod", "has_bundle", "paid_shipping"]:
    ct = pd.crosstab(o.segment, o[var]); chi2, p, dof, _ = chi2_contingency(ct)
    tests.append(dict(question=f"{var} share differs by segment", test="chi-square", stat=chi2, p_value=p, n=len(o),
                      effect=f"Cramer's V {np.sqrt(chi2 / len(o)):.3f}"))

# ------------------------------------------------------------------ 3. geography
by_city = o.groupby(["region", "city"]).apply(kpis)[["orders", "aov_total", "products_per_order", "cod_share", "paid_shipping_share", "bundle_share",
                                                    "rev_share_Skincare", "rev_share_Haircare", "rev_share_Bodycare", "rev_share_Bundles"]]
by_city.round(4).to_csv(TAB / "phase2_by_city.csv")
by_region = o.groupby("region").apply(kpis)
by_region.round(4).to_csv(TAB / "phase2_by_region.csv")
res["by_region"] = by_region[["orders", "aov_total", "products_per_order", "cod_share", "bundle_share"]].round(3)
for var, lab in [("region", "region"), ("city", "city")]:
    h, p = kruskal(*[g.order_total_inr for _, g in o.groupby(var)])
    tests.append(dict(question=f"order value differs by {lab}", test="Kruskal-Wallis", stat=h, p_value=p, n=len(o)))
    ct = pd.crosstab(o[var], o.cod); chi2, p, dof, _ = chi2_contingency(ct)
    tests.append(dict(question=f"COD share differs by {lab}", test="chi-square", stat=chi2, p_value=p, n=len(o), effect=f"Cramer's V {np.sqrt(chi2/len(o)):.3f}"))
ct = pd.crosstab(o.region, o.main_category); chi2, p, dof, _ = chi2_contingency(ct)
tests.append(dict(question="main basket category differs by region", test="chi-square", stat=chi2, p_value=p, n=len(o), effect=f"Cramer's V {np.sqrt(chi2/(len(o)*3)):.3f}"))

# ------------------------------------------------------------------ 4. COD drivers (logistic regression, customer-clustered SEs)
m = smf.logit("cod ~ np.log(gross_merchandise_value_inr) + C(region) + C(annual_income_band_inr, Treatment('8L-15L')) + "
              "C(segment, Treatment('Occasional buyers')) + C(gender) + age + attributed + C(year)", data=o).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": o.customer_id.astype("category").cat.codes})
codtab = pd.DataFrame({"odds_ratio": np.exp(m.params), "ci_low": np.exp(m.conf_int()[0]), "ci_high": np.exp(m.conf_int()[1]), "p_value": m.pvalues})
codtab.round(4).to_csv(TAB / "phase2_cod_logit.csv")
res["cod_logit"] = codtab.round(3)
res["cod_pseudo_r2"] = round(m.prsquared, 4)
cod_cust = o.groupby("customer_id").cod.agg(["mean", "size"])
cod_cust = cod_cust[cod_cust["size"] >= 3]
res["cod_customer_mix"] = pd.cut(cod_cust["mean"], [-.01, 0, .34, .67, .99, 1], labels=["never COD", "<1/3", "1/3-2/3", ">2/3", "always COD"]).value_counts(normalize=True).sort_index().round(3)
# is COD habitual? compare share of customers 'always COD' with a binomial expectation at the same base rate
p_cod = o.cod.mean()
exp_always = (cod_cust["size"].apply(lambda n: p_cod ** n)).mean()
exp_never = (cod_cust["size"].apply(lambda n: (1 - p_cod) ** n)).mean()
res["cod_habit"] = pd.Series({"observed_always": (cod_cust["mean"] == 1).mean(), "expected_always_if_random": exp_always,
                              "observed_never": (cod_cust["mean"] == 0).mean(), "expected_never_if_random": exp_never}).round(4)

# ------------------------------------------------------------------ 5. free-shipping threshold: is there bunching above Rs 999?
# Counterfactual: shuffle product_id across all order lines (keeping each line's quantity and each order's line count),
# recompute order GMV and count orders just above vs just below the threshold. Organic (undiscounted) orders only,
# so net value = GMV. 300 permutations.
org = o[~o.attributed]
lines = oi[oi.order_id.isin(org.order_id)][["order_id", "product_id", "quantity"]].reset_index(drop=True)
price = pc.set_index("product_id").list_price_inr
codes = lines.order_id.astype("category").cat.codes.values
def window_shares(gmv):
    return (((gmv >= THRESH) & (gmv < THRESH + 200)).mean(), ((gmv >= THRESH - 200) & (gmv < THRESH)).mean())
obs_gmv = np.bincount(codes, weights=lines.product_id.map(price).values * lines.quantity.values)
obs_above, obs_below = window_shares(obs_gmv)
rng = np.random.default_rng(SEED); sims = []
pvals = lines.product_id.map(price).values
for _ in range(300):
    sims.append(window_shares(np.bincount(codes, weights=rng.permutation(pvals) * lines.quantity.values)))
sims = np.array(sims)
ratio_obs = obs_above / obs_below; ratio_sim = sims[:, 0] / sims[:, 1]
res["threshold_bunching"] = pd.Series({"obs_share_999_1199": obs_above, "sim_share_999_1199": sims[:, 0].mean(),
                                       "obs_share_799_999": obs_below, "sim_share_799_999": sims[:, 1].mean(),
                                       "ratio_above_below_obs": ratio_obs, "ratio_above_below_sim_mean": ratio_sim.mean(),
                                       "ratio_sim_p95": np.percentile(ratio_sim, 95),
                                       "perm_p_value": (ratio_sim >= ratio_obs).mean()}).round(4)
# who pays shipping and how close are they?
ship = o[o.paid_shipping == 1].copy()
ship["gap_to_free"] = THRESH - ship.net_value
gap = pd.cut(ship.gap_to_free, [0, 100, 250, 400, 600, 800], labels=["<=100", "101-250", "251-400", "401-600", "601-800"]).value_counts(normalize=True).sort_index()
res["shipping_gap"] = gap.round(3)
res["shipping_stats"] = pd.Series({"orders_paying_shipping": len(ship), "share_of_orders": len(ship) / len(o),
                                   "shipping_revenue_sample": ship.shipping_fee_inr.sum(),
                                   "share_within_250_of_threshold": (ship.gap_to_free <= 250).mean(),
                                   "share_single_product": (ship.products == 1).mean(),
                                   "median_net_value": ship.net_value.median()}).round(3)
pd.concat({"bunching": res["threshold_bunching"], "stats": res["shipping_stats"]}).to_csv(TAB / "phase2_shipping_threshold.csv")
# add-on candidates: cheapest products that would close the gap
res["addon_candidates"] = pc.sort_values("list_price_inr").head(6)[["product_name", "category", "list_price_inr"]]

# ------------------------------------------------------------------ 6. discounts: deeper discount -> bigger or smaller basket?
a = o[o.attributed].merge(ex[["exposure_id", "campaign_id"]], left_on="attributed_exposure_id", right_on="exposure_id").merge(
    cc[["campaign_id", "campaign_name", "channel", "featured_category", "discount_pct"]].rename(columns={"discount_pct": "camp_discount_pct"}), on="campaign_id")
a["bought_featured"] = a.apply(lambda r: r[f"rev_{r.featured_category}"] > 0, axis=1)
camp = a.groupby(["campaign_id", "campaign_name", "channel", "featured_category", "camp_discount_pct"]).agg(
    orders=("order_id", "size"), gmv=("gross_merchandise_value_inr", "mean"), total=("order_total_inr", "mean"),
    products=("products", "mean"), bought_featured=("bought_featured", "mean"), bundle_share=("has_bundle", "mean")).reset_index()
camp.round(3).to_csv(TAB / "phase2_campaign_baskets.csv", index=False)
res["campaign_baskets"] = camp.round(2)
md = smf.ols("np.log(gross_merchandise_value_inr) ~ camp_discount_pct + C(featured_category) + C(segment)", data=a).fit(cov_type="HC1")
res["discount_vs_gmv"] = pd.Series({"coef_per_pct_point": md.params["camp_discount_pct"], "p_value": md.pvalues["camp_discount_pct"],
                                    "pct_change_gmv_per_+5pp": (np.exp(5 * md.params["camp_discount_pct"]) - 1) * 100}).round(4)
# within-customer: 2026 campaign orders vs the same customers' 2026 organic orders
o26 = o[o.year == 2026]
pair = o26.groupby(["customer_id", "attributed"]).gross_merchandise_value_inr.mean().unstack().dropna()
w = wilcoxon(pair[True], pair[False])
res["within_customer_gmv"] = pd.Series({"customers_with_both": len(pair), "median_campaign_gmv": pair[True].median(),
                                        "median_organic_gmv": pair[False].median(), "mean_diff": (pair[True] - pair[False]).mean(),
                                        "wilcoxon_p": w.pvalue}).round(4)
fc = a.groupby("featured_category").bought_featured.mean()
base_cat = o[~o.attributed & (o.year == 2026)][[f"rev_{c}" for c in CAT]].gt(0).mean()
base_cat.index = CAT
from statsmodels.stats.proportion import proportions_ztest
org26 = o[~o.attributed & (o.year == 2026)]
fu_p = {}
for c in fc.index:
    k1, n1 = a.loc[a.featured_category == c, "bought_featured"].sum(), (a.featured_category == c).sum()
    k2, n2 = (org26[f"rev_{c}"] > 0).sum(), len(org26)
    fu_p[c] = proportions_ztest([k1, k2], [n1, n2])[1]
res["featured_uptake"] = pd.DataFrame({"campaign_orders_containing_featured": fc, "organic_2026_orders_containing": base_cat.reindex(fc.index),
                                       "p_value": pd.Series(fu_p)}).round(4)
res["featured_uptake"].to_csv(TAB / "phase2_featured_uptake.csv")

# ------------------------------------------------------------------ 7. category mix & preference consistency
mix = o.groupby("segment", observed=True)[[f"rev_{c}" for c in CAT]].sum()
mix = mix.div(mix.sum(axis=1), axis=0); mix.columns = CAT
mix.round(4).to_csv(TAB / "phase2_category_mix_by_segment.csv")
res["mix_by_segment"] = mix.round(3)
pref = o.groupby("preferred_category").apply(lambda g: pd.Series({c: (g[f"rev_{c}"] > 0).mean() for c in CAT}))[CAT]
pref.round(4).to_csv(TAB / "phase2_preferred_vs_bought.csv")
res["preferred_vs_bought"] = pref.round(3)
res["buys_preferred_share"] = o.buys_preferred.mean()
res["single_category_share"] = (o.categories == 1).mean()

# ------------------------------------------------------------------ 8. repeat timing (2025 sample, customers with 2+ orders)
s25 = o[o.year == 2025].sort_values(["customer_id", "order_date"])
s25["gap"] = s25.groupby("customer_id").order_date.diff().dt.days
gaps = s25.dropna(subset=["gap"])
res["interpurchase_days"] = gaps.groupby("segment", observed=True).gap.describe()[["count", "25%", "50%", "75%"]].round(0)
res["dow_orders"] = o.order_date.dt.day_name().value_counts()
chi_dow = chi2_contingency(np.vstack([res["dow_orders"].values, np.full(7, res["dow_orders"].mean())]))
tests.append(dict(question="orders evenly spread over weekdays", test="chi-square GOF", stat=np.nan, p_value=__import__("scipy").stats.chisquare(res["dow_orders"].values).pvalue, n=len(o)))
tests.append(dict(question="campaign GMV vs same customers' organic GMV (2026)", test="Wilcoxon signed-rank", stat=w.statistic, p_value=w.pvalue, n=len(pair)))
tests.append(dict(question="discount % predicts log GMV of campaign orders", test="OLS HC1", stat=md.tvalues["camp_discount_pct"], p_value=md.pvalues["camp_discount_pct"], n=len(a)))
tests.append(dict(question="orders just above Rs 999 over-represented vs random baskets", test="permutation (300)", stat=ratio_obs, p_value=res["threshold_bunching"]["perm_p_value"], n=len(org)))
tests = pd.DataFrame(tests); tests.to_csv(TAB / "phase2_tests.csv", index=False)
res["tests"] = tests.round(4)

# ================================================================== figures
# p2_01: basket anatomy
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
ax = axs[0]
bins = np.arange(0, 6001, 100)
ax.hist(o.net_value.clip(upper=6000), bins=bins, color=SERIES[0])
ax.axvline(THRESH, color=SERIES[7], lw=1.2, ls="--"); ax.text(THRESH, ax.get_ylim()[1] * .92, " free shipping ≥ ₹999", fontsize=7.5, color=INK2)
ax.set_title("Order value after discount (₹, capped at 6,000)", fontsize=9)
ax = axs[1]
vc = o.products.value_counts().sort_index(); ax.bar(vc.index, vc.values / len(o) * 100, color=SERIES[0], width=.6)
for x_, v_ in zip(vc.index, vc.values / len(o) * 100): ax.text(x_, v_, f"{v_:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(vc.index); ax.set_title("Distinct products per order", fontsize=9); ax.set_ylabel("% of orders"); ax.grid(axis="x", visible=False)
ax = axs[2]
vc = o.categories.value_counts().sort_index(); ax.bar(vc.index, vc.values / len(o) * 100, color=SERIES[0], width=.6)
for x_, v_ in zip(vc.index, vc.values / len(o) * 100): ax.text(x_, v_, f"{v_:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(vc.index); ax.set_title("Distinct categories per order", fontsize=9); ax.grid(axis="x", visible=False)
save(fig, "p2_01_basket_anatomy")

# p2_02: segments - value, size, mix
s25k = by_seg.loc[2025].reindex([s for s in SEGS if s in by_seg.loc[2025].index])
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
for ax, col, t, fmt in [(axs[0], "aov_total", "Average order value, 2025 (₹)", "₹{:,.0f}"), (axs[1], "products_per_order", "Distinct products per order, 2025", "{:.2f}")]:
    ax.barh(s25k.index, s25k[col], color=[SEG_COL[s] for s in s25k.index], height=.6); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for i_, v_ in enumerate(s25k[col]): ax.text(v_, i_, " " + fmt.format(v_), va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, s25k[col].max() * 1.25); ax.set_title(t, fontsize=9); ax.tick_params(axis="y", labelsize=8)
axs[1].set_yticklabels([])
ax = axs[2]; left = np.zeros(len(mix))
for c in CAT:
    ax.barh(mix.index.astype(str), mix[c] * 100, left=left, color=CATEGORY_COLORS[c], height=.6, label=c, edgecolor="#fcfcfb", linewidth=1)
    for i_, (l_, v_) in enumerate(zip(left, mix[c] * 100)):
        if v_ > 7: ax.text(l_ + v_ / 2, i_, f"{v_:.0f}%", ha="center", va="center", fontsize=7, color="#0b0b0b" if c == "Bundles" else "white")
    left += mix[c].values * 100
ax.invert_yaxis(); ax.set_yticklabels([]); ax.set_xlim(0, 100); ax.grid(False); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.08))
ax.set_title("Revenue mix by category (all sampled orders)", fontsize=9)
save(fig, "p2_02_segment_baskets")

# p2_03: shipping threshold permutation + gap
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
ax = axs[0]
ax.hist(ratio_sim, bins=30, color=MUTED); ax.axvline(ratio_obs, color=SERIES[1], lw=2)
ax.text(ratio_obs, ax.get_ylim()[1] * .85, f"  observed {ratio_obs:.2f}", color=INK2, fontsize=8)
ax.set_title("Bunching above ₹999? Observed vs 300 shuffled baskets", fontsize=9)
ax.set_xlabel("orders ₹999–1,198 ÷ orders ₹799–998")
ax = axs[1]
ax.bar(gap.index.astype(str), gap.values * 100, color=SERIES[0], width=.6)
for i_, v_ in enumerate(gap.values * 100): ax.text(i_, v_, f"{v_:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_title(f"Orders paying ₹49 shipping ({len(ship)/len(o):.0%}): ₹ short of free shipping", fontsize=9)
ax.set_ylabel("% of shipping-paying orders"); ax.grid(axis="x", visible=False)
save(fig, "p2_03_shipping_threshold")

# p2_04: COD odds ratios
show = codtab.drop("Intercept").copy()
show.index = (show.index.str.replace(r"C\((\w+)(?:, Treatment\('[^']*'\))?\)\[T\.", r"\1 = ", regex=True).str.replace("]", "")
              .str.replace("np.log(gross_merchandise_value_inr)", "log order value").str.replace("attributed[T.True", "campaign-attributed", regex=False)
              .str.replace("annual_income_band_inr", "income"))
show = show.sort_values("odds_ratio")
fig, ax = plt.subplots(figsize=(7.5, 5.5))
y = np.arange(len(show))
sig = show.p_value < .05
ax.errorbar(show.odds_ratio, y, xerr=[show.odds_ratio - show.ci_low, show.ci_high - show.odds_ratio], fmt="none", ecolor=AXIS, lw=1.5)
ax.scatter(show.odds_ratio[~sig], y[~sig], color=MUTED, s=30, zorder=3, label="p ≥ 0.05")
ax.scatter(show.odds_ratio[sig], y[sig], color=SERIES[1], s=30, zorder=3, label="p < 0.05")
ax.axvline(1, color=INK2, lw=1); ax.set_yticks(y, show.index, fontsize=7.5); ax.grid(axis="y", visible=False)
ax.set_xlabel("odds ratio of paying COD (95% CI, customer-clustered)"); ax.legend(loc="lower right")
ax.set_title(f"What predicts cash on delivery? (logit, n = {len(o):,}, pseudo-R² {m.prsquared:.3f})")
save(fig, "p2_04_cod_drivers")

# p2_05: campaign discount vs basket
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axs[0]
for fcat, g in camp.groupby("featured_category"):
    ax.scatter(g.camp_discount_pct, g.gmv, s=g.orders * 1.2, color=CATEGORY_COLORS[fcat], label=fcat, alpha=.85, edgecolor="#fcfcfb")
for r in camp.itertuples(): ax.annotate(r.campaign_id, (r.camp_discount_pct, r.gmv), fontsize=7, color=INK2, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("campaign discount %"); ax.set_ylabel("mean GMV of attributed orders (₹)"); ax.legend(fontsize=7)
ax.set_title("Campaign discount depth vs basket value (bubble = orders)", fontsize=9)
ax = axs[1]
fu = res["featured_uptake"]
x = np.arange(len(fu))
ax.bar(x - .2, fu.iloc[:, 0] * 100, width=.4, color=SERIES[0], label="campaign orders containing featured category")
ax.bar(x + .2, fu.iloc[:, 1] * 100, width=.4, color=MUTED, label="2026 organic orders containing it")
ax.set_xticks(x, fu.index); ax.set_ylabel("%"); ax.legend(fontsize=7, loc="upper left"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 100)
ax.set_title("Do campaign buyers buy the featured category?", fontsize=9)
save(fig, "p2_05_campaign_baskets")

# p2_06: city AOV and COD
bc = by_city.reset_index().sort_values("aov_total")
fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
axs[0].barh(bc.city, bc.aov_total, color=SERIES[0], height=.65); axs[0].set_xlim(bc.aov_total.min() * .9, bc.aov_total.max() * 1.03)
axs[0].axvline(o.order_total_inr.mean(), color=INK2, ls="--", lw=1); axs[0].set_title("Average order value by city (₹)", fontsize=9)
axs[1].barh(bc.city, bc.cod_share * 100, color=SERIES[1], height=.65); axs[1].axvline(o.cod.mean() * 100, color=INK2, ls="--", lw=1)
axs[1].set_title("COD share by city (%)", fontsize=9)
for ax in axs: ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8)
save(fig, "p2_06_city")

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
for k, v in res.items():
    print(f"\n### {k}\n{v}")
