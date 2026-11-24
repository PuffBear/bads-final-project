"""Detailed EDA, file by file. Figures -> outputs/figures/eda_*.png, tables -> outputs/tables/eda_*.csv.
Prints a compact numeric summary used in docs/phase0_eda_findings.md."""
import numpy as np
import pandas as pd
from common import (load_all, save, plt, TAB, SERIES, INK2, MUTED, SEQ_CMAP, DIV_CMAP,
                    CHANNEL_COLORS, CATEGORY_COLORS)

d = load_all()
cu, cc, ex, od, oi, pc, wk, bs = (d[k] for k in
    ["customers", "campaign_catalog", "campaign_exposures", "orders",
     "order_items", "product_catalog", "weekly_channel_performance",
     "brand_perception_survey"])
CH = list(CHANNEL_COLORS)
CAT = list(CATEGORY_COLORS)
INC = ["Under 4L", "4L-8L", "8L-15L", "15L-25L", "25L+"]
out = {}  # headline numbers


def hbar(ax, s, color=SERIES[0], fmt="{:,.0f}"):
    ax.barh(s.index.astype(str), s.values, color=color, height=0.65)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for i, v in enumerate(s.values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=7.5, color=INK2)


# ======================================================== 1. whole-dataset overview
ov = []
for k, df in d.items():
    dcols = df.select_dtypes("datetime").columns
    rng = "; ".join(f"{c}: {df[c].min():%Y-%m-%d} to {df[c].max():%Y-%m-%d}" for c in dcols)
    ov.append(dict(file=k, rows=len(df), cols=df.shape[1], date_coverage=rng or "n/a",
                   dtypes=", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())))
pd.DataFrame(ov).to_csv(TAB / "eda_overview.csv", index=False)

# timeline of coverage per file
fig, ax = plt.subplots(figsize=(8, 2.6))
spans = [("weekly_channel_performance", wk.week_start_date), ("orders", od.order_date),
         ("campaign_catalog", pd.concat([cc.start_date, cc.end_date])), ("campaign_exposures", ex.exposure_date),
         ("customers (signup)", cu.signup_date)]
for i, (n, s) in enumerate(spans):
    ax.plot([s.min(), s.max()], [i, i], lw=6, color=SERIES[0], solid_capstyle="round")
    ax.text(s.max(), i, f"  {s.min():%b %Y} – {s.max():%b %Y}", va="center", fontsize=7.5, color=INK2)
ax.axvline(pd.Timestamp("2025-12-31"), color=MUTED, ls="--", lw=1)
ax.text(pd.Timestamp("2025-12-31"), len(spans) - 0.4, " customer snapshot 31 Dec 2025", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(spans)), [n for n, _ in spans]); ax.grid(axis="y", visible=False)
ax.set_title("Date coverage by file"); ax.set_xlim(right=pd.Timestamp("2027-09-01"))
save(fig, "eda_00_date_coverage")

# ======================================================== 2. customers
num = ["age", "orders_before_2026", "spend_before_2026_inr", "days_since_last_order_at_2025_12_31",
       "website_visits_dec_2025", "email_engagement_dec_2025"]
desc = cu[num].describe().T
desc["skew"] = cu[num].skew()
q1, q3 = cu[num].quantile(.25), cu[num].quantile(.75)
desc["iqr_outliers"] = ((cu[num] > q3 + 1.5 * (q3 - q1)) | (cu[num] < q1 - 1.5 * (q3 - q1))).sum()
desc.round(2).to_csv(TAB / "eda_customers_numeric.csv")
out["customers_numeric"] = desc.round(1)

fig, axs = plt.subplots(2, 4, figsize=(12, 5.2))
labels = ["Age", "Orders before 2026", "Spend before 2026 (INR)", "Days since last order",
          "Website visits Dec-25", "Email engagement Dec-25"]
for ax, c, l in zip(axs.flat, num, labels):
    disc = cu[c].nunique() < 30
    if disc:
        vc = cu[c].value_counts().sort_index(); ax.bar(vc.index, vc.values, color=SERIES[0], width=0.8)
    else:
        ax.hist(cu[c], bins=40, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    ax.set_title(l, fontsize=9); ax.text(.98, .9, f"skew {cu[c].skew():.2f}", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
ax = axs.flat[6]
ax.hist(np.log1p(cu.spend_before_2026_inr), bins=40, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
ax.set_title("log(1 + spend)", fontsize=9)
ax = axs.flat[7]
ten = (pd.Timestamp("2025-12-31") - cu.signup_date).dt.days
ax.scatter(ten, cu.days_since_last_order_at_2025_12_31, s=3, alpha=.3, color=SERIES[0])
ax.plot([0, 730], [0, 730], color=SERIES[7], lw=1)
ax.set_title("Recency vs tenure (days)", fontsize=9); ax.set_xlabel("days since signup"); ax.set_ylabel("days since last order")
fig.suptitle("Customers: numeric distributions", x=0.01, ha="left", fontweight="bold")
save(fig, "eda_01_customer_distributions")

# categorical counts
fig, axs = plt.subplots(1, 5, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.4, 1, 1]))
for ax, c, order in zip(axs, ["gender", "region", "city", "annual_income_band_inr", "preferred_category"],
                        [None, None, None, INC, CAT]):
    vc = cu[c].value_counts()
    if order: vc = vc.reindex(order)
    hbar(ax, vc); ax.set_title(c.replace("_", " "), fontsize=9); ax.tick_params(axis="y", labelsize=7.5)
fig.suptitle("Customers: categorical profile (n = 6,000)", x=0.01, ha="left", fontweight="bold")
save(fig, "eda_02_customer_categoricals")

# consent by region and income
cons_r = cu.groupby("region").marketing_consent.mean().sort_values()
cons_i = cu.groupby("annual_income_band_inr").marketing_consent.mean().reindex(INC)
cons_a = cu.groupby(pd.cut(cu.age, [17, 24, 34, 44, 54, 100], labels=["18-24", "25-34", "35-44", "45-54", "55+"]), observed=True).marketing_consent.mean()
out["consent_rate"] = cu.marketing_consent.mean()
fig, axs = plt.subplots(1, 3, figsize=(11, 2.8))
for ax, s, t in zip(axs, [cons_r, cons_i, cons_a], ["by region", "by income band", "by age"]):
    hbar(ax, s * 100, fmt="{:.1f}%"); ax.set_xlim(0, 100); ax.set_title(f"Consent rate {t}", fontsize=9)
    ax.axvline(cu.marketing_consent.mean() * 100, color=MUTED, ls="--", lw=1)
save(fig, "eda_03_consent")
pd.concat([cons_r, cons_i, cons_a], keys=["region", "income", "age"]).to_csv(TAB / "eda_consent_rates.csv")

# correlation heatmap
corr = cu[num + ["marketing_consent"]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.4, 5.2))
im = ax.imshow(corr, cmap=DIV_CMAP, vmin=-1, vmax=1)
short = ["age", "orders", "spend", "recency", "visits", "email eng.", "consent"]
ax.set_xticks(range(len(short)), short, rotation=45, ha="right"); ax.set_yticks(range(len(short)), short)
ax.grid(False)
for i in range(len(short)):
    for j in range(len(short)):
        v = corr.iloc[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                                     color="white" if abs(v) > .55 else INK2)
fig.colorbar(im, ax=ax, shrink=.8); ax.set_title("Customer variables: Spearman correlation")
save(fig, "eda_04_customer_correlation")
corr.round(3).to_csv(TAB / "eda_customer_correlation.csv")
out["corr"] = corr.round(2)

# spend by income and region
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, c, order in zip(axs, ["annual_income_band_inr", "region"], [INC, sorted(cu.region.unique())]):
    data = [cu.loc[cu[c] == g, "spend_before_2026_inr"] for g in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False, widths=.55,
                    medianprops=dict(color="white", lw=1.5))
    for b in bp["boxes"]: b.set_facecolor(SERIES[0])
    ax.set_title(f"Pre-2026 spend by {c.split('_')[0]}", fontsize=9); ax.set_ylabel("INR")
save(fig, "eda_05_spend_by_income_region")
out["spend_by_income"] = cu.groupby("annual_income_band_inr").spend_before_2026_inr.agg(["median", "mean"]).reindex(INC).round(0)
out["spend_by_region"] = cu.groupby("region").spend_before_2026_inr.agg(["median", "mean"]).round(0)
out["pref_by_gender"] = pd.crosstab(cu.gender, cu.preferred_category, normalize="index").round(3)

# ======================================================== 3. campaign catalog
fig, ax = plt.subplots(figsize=(10, 4.2))
for i, r in enumerate(cc.itertuples()):
    ax.barh(i, (r.end_date - r.start_date).days + 1, left=r.start_date, color=CHANNEL_COLORS[r.channel], height=.6)
    ax.text(r.end_date + pd.Timedelta(days=2), i, f"{r.discount_pct}% · {r.featured_category} · {r.campaign_objective}",
            va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(cc)), [f"{r.campaign_id} {r.campaign_name}" for r in cc.itertuples()], fontsize=7.5)
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-01"))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v, label=k) for k, v in CHANNEL_COLORS.items()], ncol=6, loc="upper center",
          bbox_to_anchor=(.5, -.08))
ax.set_title("2026 campaign calendar (colour = channel; label = discount · category · objective)")
save(fig, "eda_06_campaign_gantt")
out["catalog_crosstab"] = pd.crosstab(cc.channel, cc.campaign_objective)
out["catalog_cat"] = pd.crosstab(cc.featured_category, cc.campaign_objective)
reach = ex.groupby("campaign_id").agg(exposures=("exposure_id", "size"), customers=("customer_id", "nunique"),
                                      cost=("contact_cost_inr", "sum"), buyers=("purchased_within_14_days", "sum"))
reach = cc.set_index("campaign_id")[["campaign_name", "channel", "featured_category", "discount_pct", "campaign_objective"]].join(reach)
reach["conv_rate"] = reach.buyers / reach.exposures
reach["cost_per_exposure"] = reach.cost / reach.exposures
reach.round(4).to_csv(TAB / "eda_campaign_reach.csv")
out["reach"] = reach.round(3)

# ======================================================== 4. campaign exposures
epc = ex.groupby("customer_id").size()
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
vc = epc.reindex(cu.customer_id, fill_value=0).value_counts().sort_index()
axs[0].bar(vc.index, vc.values, color=SERIES[0], width=.8); axs[0].set_title("Exposures per customer (incl. 0)", fontsize=9)
hbar(axs[1], reach.set_index(reach.index + " " + reach.channel).exposures.sort_values(ascending=False))
axs[1].set_title("Exposures per campaign", fontsize=9); axs[1].tick_params(axis="y", labelsize=7)
for ch in CH:
    axs[2].hist(ex.loc[ex.channel == ch, "contact_cost_inr"], bins=40, histtype="step", lw=1.6, color=CHANNEL_COLORS[ch], label=ch)
axs[2].set_title("Contact cost per exposure (INR)", fontsize=9); axs[2].legend()
save(fig, "eda_07_exposure_volume_cost")
out["cost_by_channel"] = ex.groupby("channel").contact_cost_inr.describe()[["mean", "50%", "min", "max"]].round(2)

funnel = ex.groupby("channel")[["delivered", "opened", "clicked", "purchased_within_14_days"]].mean().reindex(CH)
funnel["n"] = ex.channel.value_counts()
funnel["clicked_no_open"] = ex.assign(x=(ex.clicked == 1) & (ex.opened == 0)).groupby("channel").x.mean()
funnel.round(4).to_csv(TAB / "eda_funnel_by_channel.csv")
out["funnel"] = funnel.round(3)
fig, ax = plt.subplots(figsize=(9, 3.4))
stages = ["delivered", "opened", "clicked", "purchased_within_14_days"]
w = .13
for i, ch in enumerate(CH):
    ax.bar(np.arange(4) + (i - 2.5) * w, funnel.loc[ch, stages] * 100, width=w * .9, color=CHANNEL_COLORS[ch], label=ch)
ax.set_xticks(range(4), ["Delivered", "Opened", "Clicked", "Purchased ≤14d"]); ax.set_ylabel("% of exposures")
ax.legend(ncol=6, loc="upper right"); ax.grid(axis="x", visible=False)
ax.set_title("Funnel rates by channel (stages measured separately: 1,446 clicks have no open)")
save(fig, "eda_08_funnel_by_channel")

# conversion drivers: recency bucket, consent, category match
e = ex.merge(cu, on="customer_id").merge(cc[["campaign_id", "featured_category", "discount_pct"]], on="campaign_id")
e["recency_band"] = pd.cut(e.days_since_last_order_at_2025_12_31, [-1, 30, 90, 180, 365, 10_000],
                           labels=["0-30", "31-90", "91-180", "181-365", "365+"])
e["cat_match"] = np.where(e.preferred_category == e.featured_category, "match", "no match")
e["orders_band"] = pd.cut(e.orders_before_2026, [-1, 0, 2, 5, 100], labels=["0", "1-2", "3-5", "6+"])
drivers = {}
fig, axs = plt.subplots(1, 4, figsize=(13, 3))
for ax, c, t in zip(axs, ["recency_band", "orders_band", "marketing_consent", "cat_match"],
                    ["Days since last order", "Orders before 2026", "Marketing consent", "Featured = preferred category"]):
    g = e.groupby(c, observed=True).purchased_within_14_days.agg(["mean", "size"])
    drivers[c] = g
    ax.bar(g.index.astype(str), g["mean"] * 100, color=SERIES[0], width=.6)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m * 100, f"{m:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_title(t, fontsize=9); ax.set_ylim(0, g["mean"].max() * 135); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("14-day purchase rate (%)")
save(fig, "eda_09_conversion_drivers")
pd.concat(drivers).round(4).to_csv(TAB / "eda_conversion_drivers.csv")
out["drivers"] = pd.concat(drivers).round(4)
out["conv_overall"] = ex.purchased_within_14_days.mean()
rep = ex.groupby(["customer_id", "campaign_id"]).size()
out["repeat_pairs"] = (rep > 1).sum()
ex2 = ex.merge(rep.rename("sends").reset_index(), on=["customer_id", "campaign_id"])
out["conv_by_sends"] = ex2.groupby("sends").purchased_within_14_days.agg(["mean", "size"]).round(4)

# ======================================================== 5. orders
od["month"] = od.order_date.dt.to_period("M").dt.to_timestamp()
od["attributed"] = od.attributed_exposure_id.notna()
m = od.groupby(["month", "attributed"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(m.index, m[False], width=25, color=SERIES[0], label="Not attributed")
ax.bar(m.index, m[True], width=25, bottom=m[False], color=SERIES[1], label="Campaign-attributed")
ax.set_title("Orders per month in the order table (a sample: 2025 and 2026 are sampled differently)")
ax.legend(); ax.grid(axis="x", visible=False)
save(fig, "eda_10_orders_monthly")
out["orders_month"] = m

fig, axs = plt.subplots(1, 4, figsize=(13, 3))
axs[0].hist(od.gross_merchandise_value_inr, bins=50, color=SERIES[0]); axs[0].set_title("GMV per order (INR)", fontsize=9)
dd = od.discount_inr / od.gross_merchandise_value_inr
axs[1].hist(dd[dd > 0] * 100, bins=40, color=SERIES[0]); axs[1].set_title(f"Discount % ({(dd>0).mean():.1%} of orders)", fontsize=9)
vc = od.shipping_fee_inr.value_counts().sort_index()
axs[2].bar(vc.index.astype(str), vc.values, color=SERIES[0], width=.6); axs[2].set_title("Shipping fee (INR)", fontsize=9)
pm = od.groupby("payment_mode").order_total_inr.agg(["size", "mean"])
axs[3].bar(pm.index, pm["mean"], color=[SERIES[0], SERIES[1]], width=.6)
for i, (n, v) in enumerate(zip(pm["size"], pm["mean"])):
    axs[3].text(i, v, f"INR {v:,.0f}\n{n/len(od):.1%} of orders", ha="center", va="bottom", fontsize=7.5, color=INK2)
axs[3].set_title("Avg order total by payment mode", fontsize=9); axs[3].set_ylim(0, pm["mean"].max() * 1.3)
save(fig, "eda_11_order_distributions")
out["orders_desc"] = od[["gross_merchandise_value_inr", "discount_inr", "shipping_fee_inr", "order_total_inr"]].describe().round(1)
out["payment"] = pm.round(1)
out["discounted_share"] = (dd > 0).mean()
out["discount_by_attr"] = od.groupby("attributed").apply(lambda g: pd.Series({
    "orders": len(g), "aov": g.order_total_inr.mean(), "disc_share": (g.discount_inr > 0).mean(),
    "disc_pct": (g.discount_inr / g.gross_merchandise_value_inr).mean()})).round(3)
ship = od.groupby("shipping_fee_inr").gross_merchandise_value_inr.agg(["min", "max", "size"])
out["shipping_rule"] = ship
opc = od.groupby("customer_id").size()
out["orders_per_customer"] = opc.describe().round(2)
out["customers_with_orders"] = opc.size
# attributed lag
att = od[od.attributed].merge(ex, left_on="attributed_exposure_id", right_on="exposure_id", suffixes=("", "_e"))
lag = (att.order_date - att.exposure_date).dt.days
out["lag"] = lag.describe().round(2)
# orders vs customer snapshot: does 2025 sample relate to orders_before_2026?
o25 = od[od.order_date.dt.year == 2025].groupby("customer_id").size().reindex(cu.customer_id, fill_value=0)
out["corr_2025_orders_vs_snapshot"] = np.corrcoef(o25.values, cu.orders_before_2026.values)[0, 1]
out["zero_order_customers_with_2025_orders"] = int(((cu.orders_before_2026.values == 0) & (o25.values > 0)).sum())

# ======================================================== 6. order items
ipo = oi.groupby("order_id").agg(lines=("order_item_id", "size"), units=("quantity", "sum"))
li = oi.merge(pc, on="product_id")
prod = li.groupby(["product_id", "product_name", "category"]).agg(units=("quantity", "sum"), revenue=("line_gross_value_inr", "sum"),
                                                                    orders=("order_id", "nunique")).reset_index()
prod["rev_share"] = prod.revenue / prod.revenue.sum()
prod.sort_values("revenue", ascending=False).round(4).to_csv(TAB / "eda_product_sales.csv", index=False)
out["products"] = prod.sort_values("revenue", ascending=False).round(3)
out["items"] = ipo.describe().round(2)
out["qty"] = oi.quantity.value_counts().sort_index()
catsum = li.groupby("category").agg(units=("quantity", "sum"), revenue=("line_gross_value_inr", "sum"))
catsum["rev_share"] = catsum.revenue / catsum.revenue.sum()
out["cat_sales"] = catsum.round(3)

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[.7, .7, 1.4]))
vc = ipo.lines.value_counts().sort_index()
axs[0].bar(vc.index, vc.values, color=SERIES[0], width=.7); axs[0].set_title("Distinct products per order", fontsize=9)
vc = oi.quantity.value_counts().sort_index()
axs[1].bar(vc.index, vc.values, color=SERIES[0], width=.6); axs[1].set_title("Quantity per line", fontsize=9)
for a_ in axs[:2]: a_.set_xticks(a_.get_xticks()[a_.get_xticks() == a_.get_xticks().round()]); a_.grid(axis="x", visible=False)
p = prod.sort_values("revenue")
axs[2].barh(p.product_name, p.revenue / 1e5, color=[CATEGORY_COLORS[c] for c in p.category], height=.7)
axs[2].set_xlabel("Revenue in order sample (INR lakh)"); axs[2].grid(axis="y", visible=False); axs[2].tick_params(axis="y", labelsize=7.5)
axs[2].legend(handles=[Patch(color=v, label=k) for k, v in CATEGORY_COLORS.items()], loc="lower right")
axs[2].set_title("Revenue by product (colour = category)", fontsize=9)
save(fig, "eda_12_items_products")

# co-occurrence matrix (seeds Phase 3)
basket = pd.crosstab(oi.order_id, oi.product_id).clip(upper=1)
co = basket.T @ basket
names = pc.set_index("product_id").product_name
sup = np.diag(co) / len(basket)
lv = (co.values / len(basket)) / np.outer(sup, sup)
np.fill_diagonal(lv, np.nan)
lift = pd.DataFrame(lv, index=co.index, columns=co.columns)
lift.index = lift.columns = [f"{i} {names[i]}" for i in lift.index]
lift.round(3).to_csv(TAB / "eda_product_pair_lift.csv")
fig, ax = plt.subplots(figsize=(8.5, 7.2))
im = ax.imshow(lift, cmap=DIV_CMAP, vmin=0, vmax=2)
ax.set_xticks(range(18), lift.columns, rotation=90, fontsize=7); ax.set_yticks(range(18), lift.index, fontsize=7); ax.grid(False)
fig.colorbar(im, ax=ax, shrink=.7, label="pair lift (1 = independent)")
ax.set_title("Product co-occurrence lift within orders (multi-line baskets drive this)")
save(fig, "eda_13_product_lift")
lt = lift.stack().reset_index(); lt.columns = ["a", "b", "lift"]
lt = lt[lt.a < lt.b]
cnt = co.copy(); cnt.index = cnt.columns = lift.index
lt["orders_together"] = [cnt.loc[a, b] for a, b in zip(lt.a, lt.b)]
out["top_pairs"] = lt.sort_values("lift", ascending=False).head(10)
out["bottom_pairs"] = lt.sort_values("lift").head(6)
out["multi_line_share"] = (ipo.lines > 1).mean()

# ======================================================== 7. product catalog
fig, ax = plt.subplots(figsize=(8, 4))
p = pc.sort_values(["category", "list_price_inr"])
ax.barh(p.product_name, p.list_price_inr, color=[CATEGORY_COLORS[c] for c in p.category], height=.7)
for i, v in enumerate(p.list_price_inr): ax.text(v, i, f" {v}", va="center", fontsize=7.5, color=INK2)
ax.grid(axis="y", visible=False); ax.set_xlabel("List price (INR)")
ax.legend(handles=[Patch(color=v, label=k) for k, v in CATEGORY_COLORS.items()], loc="lower right")
ax.set_title("Price ladder by category")
save(fig, "eda_14_price_ladder")
bundle_map = {"P15": ["P01", "P02", "P03"], "P16": ["P07", "P08"], "P17": ["P04", "P05", "P02"], "P18": ["P11", "P12"]}
lp = pc.set_index("product_id").list_price_inr
out["bundle_check"] = pd.DataFrame({b: {"bundle_price": lp[b], "plausible_components": " + ".join(names[c] for c in comps),
                                        "components_sum": lp[comps].sum()} for b, comps in bundle_map.items()}).T

# ======================================================== 8. weekly channel performance
wk["ctr"] = wk.clicks / wk.impressions
wk["cvr"] = wk.conversions / wk.clicks
wk["cpa"] = wk.ad_spend_inr / wk.conversions
wk["roas"] = wk.attributed_revenue_inr / wk.ad_spend_inr
kpi = wk.groupby("channel").agg(spend=("ad_spend_inr", "sum"), revenue=("attributed_revenue_inr", "sum"),
                               impressions=("impressions", "sum"), clicks=("clicks", "sum"), conversions=("conversions", "sum"))
kpi["ROAS"] = kpi.revenue / kpi.spend; kpi["CPA"] = kpi.spend / kpi.conversions
kpi["CTR"] = kpi.clicks / kpi.impressions; kpi["CVR"] = kpi.conversions / kpi.clicks
kpi["spend_share"] = kpi.spend / kpi.spend.sum(); kpi["AOV"] = kpi.revenue / kpi.conversions
kpi = kpi.reindex(CH)
kpi.round(4).to_csv(TAB / "eda_channel_kpis.csv")
out["kpi"] = kpi.round(3)

fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
for ch in CH:
    s = wk[wk.channel == ch].set_index("week_start_date")
    for ax, c in zip(axs, ["ad_spend_inr", "attributed_revenue_inr", "conversions"]):
        ax.plot(s.index, s[c].rolling(4, center=True).mean(), color=CHANNEL_COLORS[ch], lw=1.6, label=ch)
for ax, t in zip(axs, ["Ad spend (INR, 4-wk avg)", "Platform-attributed revenue (INR, 4-wk avg)", "Conversions (4-wk avg)"]):
    ax.set_title(t, fontsize=9)
axs[0].legend(ncol=6, loc="upper left")
fig.suptitle("Weekly channel performance, Jan 2024 – Oct 2026 (platform-reported)", x=.01, ha="left", fontweight="bold")
save(fig, "eda_15_weekly_trends")

tot = wk.groupby("week_start_date")[["ad_spend_inr", "attributed_revenue_inr", "conversions"]].sum()
tot["year"] = tot.index.year; tot["woy"] = tot.index.isocalendar().week.values
fig, ax = plt.subplots(figsize=(10, 3.4))
for i, y in enumerate([2024, 2025, 2026]):
    s = tot[tot.year == y]; ax.plot(s.woy, s.attributed_revenue_inr / 1e5, color=SERIES[i], label=str(y))
ax.set_xlabel("ISO week"); ax.set_ylabel("INR lakh"); ax.legend()
ax.set_title("Seasonality: total platform-attributed revenue by week of year")
save(fig, "eda_16_seasonality")
yy = tot.groupby("year")[["ad_spend_inr", "attributed_revenue_inr", "conversions"]].sum()
yy["weeks"] = tot.groupby("year").size()
yy_same = tot[tot.woy <= 43].groupby("year")[["ad_spend_inr", "attributed_revenue_inr", "conversions"]].sum()
out["yearly"] = yy.round(0); out["yearly_wk1_43"] = yy_same.round(0)
out["peak_weeks"] = tot.attributed_revenue_inr.nlargest(8)

fig, axs = plt.subplots(2, 3, figsize=(11, 6))
el = {}
for ax, ch in zip(axs.flat, CH):
    s = wk[wk.channel == ch]
    x, y = np.log(s.ad_spend_inr), np.log(s.attributed_revenue_inr)
    b, a = np.polyfit(x, y, 1); el[ch] = b
    ax.scatter(x, y, s=8, alpha=.6, color=CHANNEL_COLORS[ch])
    xs = np.linspace(x.min(), x.max(), 10); ax.plot(xs, a + b * xs, color=INK2, lw=1.2)
    ax.set_title(f"{ch}: slope {b:.2f}", fontsize=9)
for ax in axs[1]: ax.set_xlabel("log spend")
for ax in axs[:, 0]: ax.set_ylabel("log revenue")
fig.suptitle("log revenue vs log spend by channel (slope ≈ naive spend elasticity; Phase 5 adds controls)", x=.01, ha="left", fontweight="bold")
save(fig, "eda_17_loglog_spend_revenue")
out["elasticity_naive"] = pd.Series(el).round(3)

# ======================================================== 9. brand perception survey
attrs = [c for c in bs.columns if c.endswith("_1_to_5")]
an = [a.replace("perceived_", "").replace("_1_to_5", "") for a in attrs]
means = bs.groupby("brand")[attrs].mean(); means.columns = an
means = means.loc[means.mean(axis=1).sort_values(ascending=False).index]
means.round(3).to_csv(TAB / "eda_brand_attribute_means.csv")
out["brand_means"] = means.round(2)
fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1.1, 1]))
im = axs[0].imshow(means, cmap=SEQ_CMAP, aspect="auto")
axs[0].set_xticks(range(5), an); axs[0].set_yticks(range(len(means)), means.index); axs[0].grid(False)
for i in range(means.shape[0]):
    for j in range(5):
        v = means.iloc[i, j]; axs[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                                          color="white" if v > means.values.mean() else "#0b0b0b",
                                          fontweight="bold" if means.index[i] == "NexaGlow" else "normal")
axs[0].set_title("Mean rating by brand × attribute (1–5)")
fig.colorbar(im, ax=axs[0], shrink=.8)
for i, (a, n) in enumerate(zip(attrs, an)):
    vc = bs[a].value_counts(normalize=True).sort_index()
    axs[1].plot(vc.index, vc.values * 100, marker="o", ms=5, color=SERIES[i], label=n)
axs[1].set_xticks(range(1, 6)); axs[1].set_ylabel("% of ratings"); axs[1].legend()
axs[1].set_title("Rating distribution per attribute (all brands)")
save(fig, "eda_18_brand_heatmap")
rc = bs[attrs].corr(); rc.index = rc.columns = an
out["attr_corr"] = rc.round(2)
# respondents vs customer base
resp = cu[cu.customer_id.isin(bs.customer_id)]
cmp_ = pd.DataFrame({
    "base": [cu.age.mean(), (cu.gender == "Female").mean(), cu.spend_before_2026_inr.mean(), cu.orders_before_2026.mean(), cu.marketing_consent.mean()],
    "respondents": [resp.age.mean(), (resp.gender == "Female").mean(), resp.spend_before_2026_inr.mean(), resp.orders_before_2026.mean(), resp.marketing_consent.mean()]},
    index=["age", "female_share", "spend", "orders", "consent"])
out["resp_vs_base"] = cmp_.round(3)
out["n_resp_customers"] = resp.customer_id.nunique()

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)
for k, v in out.items():
    print(f"\n### {k}\n{v}")
