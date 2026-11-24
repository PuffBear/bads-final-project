"""Phase 3: product relationships - market basket analysis and bundle cannibalisation.

Inputs : order_items, product_catalog, orders, campaigns, outputs/tables/order_level.csv (Phase 2)
Outputs: outputs/tables/phase3_*.csv, outputs/figures/p3_*.png
Notes  : * baskets = distinct products per order (1,567 repeated-product lines collapsed).
         * single-product orders (52%) are kept: they carry information about support and
           independence, and dropping them would inflate support/confidence.
         * every rule gets a one-sided Fisher exact test (antecedent vs consequent, 2x2) with
           Benjamini-Hochberg FDR control, so 'lift > 1' is not reported from noise.
         * bundle contents are NOT in the catalogue; the component mapping below is our assumption,
           based on product names and prices (each bundle is priced 9-22% under its assumed parts).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from mlxtend.frequent_patterns import apriori, association_rules
from common import load, save, plt, TAB, SERIES, INK2, MUTED, AXIS, CATEGORY_COLORS, DIV_CMAP, SEQ_CMAP

oi, pc, od, cc, ex = (load(n) for n in ["order_items", "product_catalog", "orders", "campaign_catalog", "campaign_exposures"])
o = pd.read_csv(TAB / "order_level.csv", parse_dates=["order_date"])
CAT = list(CATEGORY_COLORS)
name = pc.set_index("product_id").product_name
pcat = pc.set_index("product_id").category
price = pc.set_index("product_id").list_price_inr
BUNDLE_PARTS = {"P15": ["P01", "P02", "P03"], "P16": ["P07", "P08"], "P17": ["P04", "P05", "P02"], "P18": ["P11", "P12"]}
res = {}

# ------------------------------------------------------------------ basket matrix
B = pd.crosstab(oi.order_id, oi.product_id).gt(0)
B.columns = [f"{p} {name[p]}" for p in B.columns]
lab2id = {c: c.split(" ")[0] for c in B.columns}
N = len(B)
res["n_baskets"] = N
res["multi_product_baskets"] = int((B.sum(axis=1) > 1).sum())

# ------------------------------------------------------------------ 1. Apriori (product level)
MIN_SUP = 0.004  # ~57 orders: lets every pair of the 18 products with >= 57 co-purchases through
fi = apriori(B, min_support=MIN_SUP, use_colnames=True, max_len=3)
rules = association_rules(fi, metric="lift", min_threshold=0.0)
rules = rules[["antecedents", "consequents", "antecedent support", "consequent support", "support", "confidence", "lift", "leverage", "conviction"]].copy()
def fisher_p(ante, cons):
    a_ = B[list(ante)].all(axis=1); c_ = B[list(cons)].all(axis=1)
    t = [[(a_ & c_).sum(), (a_ & ~c_).sum()], [(~a_ & c_).sum(), (~a_ & ~c_).sum()]]
    return fisher_exact(t, alternative="greater")[1], (a_ & c_).sum()
fp = [fisher_p(a, c) for a, c in zip(rules.antecedents, rules.consequents)]
rules["p_value"] = [x[0] for x in fp]; rules["orders_together"] = [x[1] for x in fp]
rules["q_value"] = multipletests(rules.p_value, method="fdr_bh")[1]
rules["n_items"] = rules.antecedents.apply(len) + rules.consequents.apply(len)
fmt = lambda s: " + ".join(sorted(s))
rules["antecedent"] = rules.antecedents.apply(fmt); rules["consequent"] = rules.consequents.apply(fmt)
cats_of = lambda s: {pcat[lab2id[x]] for x in s}
rules["within_category"] = [len(cats_of(a) | cats_of(c)) == 1 for a, c in zip(rules.antecedents, rules.consequents)]
rules["categories"] = [" / ".join(sorted(cats_of(a) | cats_of(c))) for a, c in zip(rules.antecedents, rules.consequents)]
out_cols = ["antecedent", "consequent", "categories", "within_category", "n_items", "orders_together", "support", "confidence",
            "lift", "leverage", "conviction", "p_value", "q_value"]
rules = rules.sort_values("lift", ascending=False)
rules[out_cols].round(5).to_csv(TAB / "phase3_rules_all.csv", index=False)
sig = rules[(rules.q_value < 0.05) & (rules.lift > 1)]
sig[out_cols].round(5).to_csv(TAB / "phase3_rules_significant.csv", index=False)
res["rule_counts"] = pd.Series({"frequent_itemsets": len(fi), "rules": len(rules), "significant_positive (q<0.05, lift>1)": len(sig),
                                "of which pairs": int((sig.n_items == 2).sum()), "of which triples": int((sig.n_items == 3).sum()),
                                "significant within-category share": sig.within_category.mean(),
                                "significant cross-category rules": int((~sig.within_category).sum()),
                                "rules with lift < 1 and q(lower) < 0.05": None})
# negative associations (substitutes): one-sided 'less' test on pairs
pairs = rules[rules.n_items == 2].copy()
def fisher_less(ante, cons):
    a_ = B[list(ante)].all(axis=1); c_ = B[list(cons)].all(axis=1)
    return fisher_exact([[(a_ & c_).sum(), (a_ & ~c_).sum()], [(~a_ & c_).sum(), (~a_ & ~c_).sum()]], alternative="less")[1]
pairs["p_less"] = [fisher_less(a, c) for a, c in zip(pairs.antecedents, pairs.consequents)]
pairs["q_less"] = multipletests(pairs.p_less, method="fdr_bh")[1]
neg = pairs[(pairs.q_less < 0.05) & (pairs.lift < 1)]
res["rule_counts"]["rules with lift < 1 and q(lower) < 0.05"] = len(neg)
neg_u = neg[neg.antecedent < neg.consequent]
res["negative_pairs_within_vs_cross"] = neg_u.within_category.value_counts()
res["top_pair_rules"] = sig[(sig.n_items == 2) & (sig.antecedent < sig.consequent)].head(12)[["antecedent", "consequent", "orders_together", "support", "confidence", "lift", "q_value"]].round(4)
res["top_triples"] = sig[sig.n_items == 3].sort_values("lift", ascending=False).head(8)[["antecedent", "consequent", "orders_together", "confidence", "lift", "q_value"]].round(4)
res["cross_category_sig"] = sig[~sig.within_category][["antecedent", "consequent", "categories", "orders_together", "confidence", "lift", "q_value"]].round(4).head(15)

# ------------------------------------------------------------------ 1b. ALL 153 product pairs (Apriori drops rare pairs, which are
#      exactly the strong negatives) - two-sided Fisher exact + BH
from itertools import combinations
allp = []
for a_, c_ in combinations(B.columns, 2):
    x, y_ = B[a_], B[c_]
    t = [[(x & y_).sum(), (x & ~y_).sum()], [(~x & y_).sum(), (~x & ~y_).sum()]]
    allp.append(dict(a=a_, b=c_, cat_a=pcat[lab2id[a_]], cat_b=pcat[lab2id[c_]], together=t[0][0],
                     expected=x.sum() * y_.sum() / N, lift=(x & y_).mean() / (x.mean() * y_.mean()), p_value=fisher_exact(t)[1]))
allp = pd.DataFrame(allp)
allp["q_value"] = multipletests(allp.p_value, method="fdr_bh")[1]
allp["within_category"] = allp.cat_a == allp.cat_b
allp["direction"] = np.where(allp.q_value >= .05, "n.s.", np.where(allp.lift > 1, "positive", "negative"))
allp.sort_values("lift", ascending=False).round(5).to_csv(TAB / "phase3_all_pairs.csv", index=False)
res["all_pairs_summary"] = pd.crosstab(allp.within_category.map({True: "within category", False: "across categories"}), allp.direction, margins=True)
res["all_pairs_lift_by_type"] = allp.groupby("within_category").lift.describe()[["count", "min", "50%", "max"]].round(3)

# ------------------------------------------------------------------ 2. cross-sell table: best partner per product
cs = sig[sig.n_items == 2].sort_values(["antecedent", "lift"], ascending=[True, False])
cross_sell = cs.groupby("antecedent").head(2)[["antecedent", "consequent", "confidence", "lift", "orders_together"]]
cross_sell.round(3).to_csv(TAB / "phase3_cross_sell.csv", index=False)
res["cross_sell"] = cross_sell.round(3)

# ------------------------------------------------------------------ 3. category-level associations
cat_B = pd.DataFrame({c: B[[col for col in B.columns if pcat[lab2id[col]] == c]].any(axis=1) for c in CAT})
cat_lift = pd.DataFrame(index=CAT, columns=CAT, dtype=float)
for a_ in CAT:
    for c_ in CAT:
        if a_ != c_:
            cat_lift.loc[a_, c_] = (cat_B[a_] & cat_B[c_]).mean() / (cat_B[a_].mean() * cat_B[c_].mean())
res["category_incidence"] = cat_B.mean().round(3)
res["category_pair_lift"] = cat_lift.round(3)
# within-category diversity: P(2nd product of same category | basket has >= 2 products)
mb = B[B.sum(axis=1) > 1]
mbl = mb.stack()
mbl = mbl[mbl].reset_index().iloc[:, :2]; mbl.columns = ["order_id", "prod"]
mbl["cat"] = mbl["prod"].map(lambda x: pcat[lab2id[x]])
same = mbl.groupby("order_id").cat.nunique().eq(1).mean()
# expected share of single-category multi-product baskets if products were drawn independently by popularity
pop = B.mean(); sims = []
rng = np.random.default_rng(42)
sizes = mb.sum(axis=1).values
probs = (pop / pop.sum()).values; pcats = np.array([pcat[lab2id[c]] for c in B.columns])
for _ in range(200):
    sims.append(np.mean([len(set(pcats[rng.choice(len(probs), s, replace=False, p=probs)])) == 1 for s in sizes]))
res["single_category_multi_baskets"] = pd.Series({"observed": same, "expected_if_independent": np.mean(sims), "sim_p95": np.percentile(sims, 95)}).round(4)

# ------------------------------------------------------------------ 4. stability of top pairs across years and segments
top_pairs = sig[(sig.n_items == 2)].copy()
top_pairs = top_pairs[top_pairs.antecedent < top_pairs.consequent].head(10)
def pair_lift(mask, a_, c_):
    Bm = B[mask]; return (Bm[a_] & Bm[c_]).mean() / (Bm[a_].mean() * Bm[c_].mean())
ol = o.set_index("order_id").reindex(B.index)
stab = []
for r in top_pairs.itertuples():
    a_, c_ = r.antecedent, r.consequent
    row = {"pair": f"{a_} & {c_}", "lift_all": r.lift}
    for y in (2025, 2026): row[f"lift_{y}"] = pair_lift((ol.year == y).values, a_, c_)
    for sname in ["Recent actives", "Lapsing regulars", "Low-basket buyers", "Occasional buyers", "Registered, never purchased"]:
        row[sname] = pair_lift((ol.segment == sname).values, a_, c_)
    row["campaign_orders"] = pair_lift(ol.attributed.values, a_, c_)
    stab.append(row)
stab = pd.DataFrame(stab).round(2); stab.to_csv(TAB / "phase3_rule_stability.csv", index=False)
res["rule_stability"] = stab

# ------------------------------------------------------------------ 5. bundles: add or cannibalise?
idc = lambda p: f"{p} {name[p]}"
bund = []
# (a) same basket: bundle with its own components
for bnd, parts in BUNDLE_PARTS.items():
    for p in parts:
        a_, c_ = B[idc(bnd)], B[idc(p)]
        t = [[(a_ & c_).sum(), (a_ & ~c_).sum()], [(~a_ & c_).sum(), (~a_ & ~c_).sum()]]
        bund.append(dict(bundle=name[bnd], component=name[p], together=t[0][0], lift=(a_ & c_).mean() / (a_.mean() * c_.mean()),
                         p_less=fisher_exact(t, alternative="less")[1], p_greater=fisher_exact(t, alternative="greater")[1]))
bund = pd.DataFrame(bund); bund.round(4).to_csv(TAB / "phase3_bundle_component_basket.csv", index=False)
res["bundle_vs_components_same_basket"] = bund.round(4)

# (b) customer level (whole sample): are bundle buyers less likely to buy the components in OTHER orders?
#     logit: buys component in a non-bundle order ~ bought bundle + log(#orders) + preferred category, customers with 2+ orders
cu = load("customers")
ob = oi.merge(pc, on="product_id")[["order_id", "product_id"]].drop_duplicates()
ob = ob.merge(od[["order_id", "customer_id"]], on="order_id")
n_orders = od.groupby("customer_id").size()
cust = cu.set_index("customer_id")[["preferred_category"]].join(n_orders.rename("n_orders"), how="inner")
cust = cust[cust.n_orders >= 2]
cust_res = []
for bnd, parts in BUNDLE_PARTS.items():
    bundle_orders = set(ob.loc[ob.product_id == bnd, "order_id"])
    buyer = ob.loc[ob.product_id == bnd].groupby("customer_id").size().reindex(cust.index, fill_value=0).gt(0).astype(int)
    comp_other = (ob[ob.product_id.isin(parts) & ~ob.order_id.isin(bundle_orders)].groupby("customer_id").size()
                  .reindex(cust.index, fill_value=0).gt(0).astype(int))
    # non-bundle orders available to buy components in
    nb_orders = (od[~od.order_id.isin(bundle_orders)].groupby("customer_id").size().reindex(cust.index, fill_value=0))
    d = cust.assign(bundle_buyer=buyer, comp=comp_other, log_nb=np.log1p(nb_orders))
    d = d[nb_orders.reindex(d.index) > 0]
    m = smf.logit("comp ~ bundle_buyer + log_nb + C(preferred_category)", data=d).fit(disp=0)
    cust_res.append(dict(bundle=name[bnd], components=" + ".join(name[p] for p in parts), customers=len(d), bundle_buyers=int(d.bundle_buyer.sum()),
                         comp_rate_bundle_buyers=d.loc[d.bundle_buyer == 1, "comp"].mean(), comp_rate_others=d.loc[d.bundle_buyer == 0, "comp"].mean(),
                         odds_ratio=np.exp(m.params["bundle_buyer"]), ci_low=np.exp(m.conf_int().loc["bundle_buyer", 0]),
                         ci_high=np.exp(m.conf_int().loc["bundle_buyer", 1]), p_value=m.pvalues["bundle_buyer"]))
cust_res = pd.DataFrame(cust_res); cust_res.round(4).to_csv(TAB / "phase3_bundle_customer_substitution.csv", index=False)
res["bundle_customer_level"] = cust_res.round(3)

# (c) sequence: after an order containing bundle X, is the customer's NEXT order more likely to contain X's components?
seq = od[["order_id", "customer_id", "order_date"]].sort_values(["customer_id", "order_date", "order_id"])
seq["next_order"] = seq.groupby("customer_id").order_id.shift(-1)
seq = seq.dropna(subset=["next_order"])
has = {p: set(ob.loc[ob.product_id == p, "order_id"]) for p in pc.product_id}
seqr = []
for bnd, parts in BUNDLE_PARTS.items():
    prev_b = seq.order_id.isin(has[bnd])
    nxt_comp = seq.next_order.isin(set().union(*[has[p] for p in parts]))
    nxt_bnd = seq.next_order.isin(has[bnd])
    from statsmodels.stats.proportion import proportions_ztest
    p_comp = proportions_ztest([nxt_comp[prev_b].sum(), nxt_comp[~prev_b].sum()], [prev_b.sum(), (~prev_b).sum()])[1]
    p_rep = proportions_ztest([nxt_bnd[prev_b].sum(), nxt_bnd[~prev_b].sum()], [prev_b.sum(), (~prev_b).sum()])[1]
    seqr.append(dict(bundle=name[bnd], transitions_after_bundle=int(prev_b.sum()), p_components=p_comp, p_repeat_bundle=p_rep,
                     next_has_components_after_bundle=nxt_comp[prev_b].mean(), next_has_components_otherwise=nxt_comp[~prev_b].mean(),
                     next_has_same_bundle_after_bundle=nxt_bnd[prev_b].mean(), next_has_same_bundle_otherwise=nxt_bnd[~prev_b].mean()))
seqr = pd.DataFrame(seqr); seqr.round(4).to_csv(TAB / "phase3_bundle_sequence.csv", index=False)
res["bundle_sequence"] = seqr.round(3)

# (d) basket value: do bundle baskets carry extra products, and are they worth more?
ol2 = o.copy()
ol2["n_non_bundle_products"] = ol2.products - B.reindex(ol2.order_id)[[idc(b) for b in BUNDLE_PARTS]].sum(axis=1).values
bv = ol2.groupby("has_bundle").agg(orders=("order_id", "size"), gmv=("gross_merchandise_value_inr", "mean"),
                                   products=("products", "mean"), extra_singles=("n_non_bundle_products", "mean"))
res["bundle_basket_value"] = bv.round(2)

# (e) bundle-featured campaign windows (2026): component vs bundle units in window vs outside (descriptive; sample)
li26 = oi.merge(od[["order_id", "order_date"]], on="order_id")
li26 = li26[li26.order_date.dt.year == 2026]
win = cc[cc.featured_category == "Bundles"]
li26["in_bundle_window"] = False
for r in win.itertuples():
    li26.loc[li26.order_date.between(r.start_date, r.end_date + pd.Timedelta(days=14)), "in_bundle_window"] = True
comps = sorted({p for v in BUNDLE_PARTS.values() for p in v})
li26["type"] = np.where(li26.product_id.isin(BUNDLE_PARTS), "bundle", np.where(li26.product_id.isin(comps), "component", "other single"))
days_in = li26.loc[li26.in_bundle_window, "order_date"].dt.normalize().nunique(); days_out = li26.loc[~li26.in_bundle_window, "order_date"].dt.normalize().nunique()
wtab = li26.groupby(["in_bundle_window", "type"]).quantity.sum().unstack()
wtab["days"] = [days_out, days_in]
for c in ["bundle", "component", "other single"]: wtab[f"{c}_units_per_day"] = wtab[c] / wtab.days
wtab["bundle_share_of_units"] = wtab.bundle / wtab[["bundle", "component", "other single"]].sum(axis=1)
res["bundle_campaign_windows"] = wtab.round(3)
wtab.round(3).to_csv(TAB / "phase3_bundle_campaign_windows.csv")

# ------------------------------------------------------------------ figures
# p3_01: significant rules scatter
fig, ax = plt.subplots(figsize=(9, 4.6))
nonsig = rules[~rules.index.isin(sig.index)]
ax.scatter(nonsig.support * 100, nonsig.confidence * 100, s=10, color=AXIS, label=f"not significant / lift ≤ 1 (n = {len(nonsig)})")
for within, col, lab_ in [(True, SERIES[0], "within one category"), (False, SERIES[1], "across categories")]:
    s_ = sig[sig.within_category == within]
    ax.scatter(s_.support * 100, s_.confidence * 100, s=12 + (s_.lift - 1) * 120, color=col, alpha=.8, edgecolor="#fcfcfb",
               label=f"significant, {lab_} (n = {len(s_)})")
ax.set_xlabel("support (% of all orders)"); ax.set_ylabel("confidence (%)"); ax.legend(fontsize=7.5, loc="upper right")
ax.set_title(f"Association rules: {len(sig)} of {len(rules)} are significant positive (FDR 5%; bubble = lift)")
save(fig, "p3_01_rules_scatter")

# p3_02: product lift heatmap ordered by category, with significance marks
order_cols = sorted(B.columns, key=lambda c: (CAT.index(pcat[lab2id[c]]), lab2id[c]))
L = pd.DataFrame(np.nan, index=order_cols, columns=order_cols)
S = pd.DataFrame("", index=order_cols, columns=order_cols)
for r in allp.itertuples():
    for a_, c_ in [(r.a, r.b), (r.b, r.a)]:
        L.loc[a_, c_] = r.lift
        S.loc[a_, c_] = {"positive": "+", "negative": "−"}.get(r.direction, "")
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(L.values.astype(float), cmap=DIV_CMAP.reversed(), vmin=0, vmax=2)
ax.set_xticks(range(18), [c[4:] for c in order_cols], rotation=90, fontsize=7.5); ax.set_yticks(range(18), [c[4:] for c in order_cols], fontsize=7.5)
ax.grid(False)
for i in range(18):
    for j in range(18):
        if S.iloc[i, j]: ax.text(j, i, S.iloc[i, j], ha="center", va="center", fontsize=8, color="white" if abs(L.iloc[i, j] - 1) > .45 else "#0b0b0b")
bounds = np.cumsum([sum(pcat[lab2id[c]] == k for c in order_cols) for k in CAT])[:-1] - .5
for b_ in bounds: ax.axhline(b_, color=INK2, lw=1); ax.axvline(b_, color=INK2, lw=1)
fig.colorbar(im, ax=ax, shrink=.7, label="pair lift (1 = independent)")
ax.set_title("Pair lift, all 153 product pairs (+ / − = significant, Fisher, FDR 5%). Blocks = categories")
save(fig, "p3_02_product_lift_matrix")

# p3_03: bundle evidence
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axs[0]
lbl = [f"{r.bundle} ↔ {r.component}" for r in bund.itertuples()]
cols = [SERIES[7] if p_ < .05 else (SERIES[2] if q_ < .05 else MUTED) for p_, q_ in zip(bund.p_less, bund.p_greater)]
ax.barh(lbl, bund.lift, color=cols, height=.6); ax.axvline(1, color=INK2, lw=1); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
for i_, (v_, n_) in enumerate(zip(bund.lift, bund.together)): ax.text(v_, i_, f" {v_:.2f} (n={n_})", va="center", fontsize=7, color=INK2)
ax.tick_params(axis="y", labelsize=7); ax.set_xlim(0, max(1.3, bund.lift.max() * 1.25))
ax.set_title("Same basket: bundle with its own components\n(lift; red = significantly below 1)", fontsize=9)
ax = axs[1]
y = np.arange(len(cust_res))
ax.errorbar(cust_res.odds_ratio, y, xerr=[cust_res.odds_ratio - cust_res.ci_low, cust_res.ci_high - cust_res.odds_ratio], fmt="o", color=SERIES[0], ecolor=AXIS, ms=6)
ax.axvline(1, color=INK2, lw=1); ax.set_yticks(y, cust_res.bundle, fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("odds ratio (95% CI)"); ax.set_title("Customer level: bundle buyers buying\nthe components in other orders", fontsize=9)
ax = axs[2]
x = np.arange(len(seqr))
ax.bar(x - .2, seqr.next_has_components_after_bundle * 100, width=.4, color=SERIES[0], label="next order, after a bundle order")
ax.bar(x + .2, seqr.next_has_components_otherwise * 100, width=.4, color=MUTED, label="next order, after any other order")
ax.set_xticks(x, [b.replace(" ", "\n", 1) for b in seqr.bundle], fontsize=7.5); ax.set_ylabel("% containing the components"); ax.legend(fontsize=7)
ax.grid(axis="x", visible=False); ax.set_title("Sequence: does a bundle lead to buying\nthe single products next time?", fontsize=9)
save(fig, "p3_03_bundle_cannibalisation")

# p3_04: category pair lift
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cat_lift.values.astype(float), cmap=DIV_CMAP.reversed(), vmin=0, vmax=2)
ax.set_xticks(range(4), CAT); ax.set_yticks(range(4), CAT); ax.grid(False)
for i in range(4):
    for j in range(4):
        if i != j: ax.text(j, i, f"{cat_lift.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK2)
fig.colorbar(im, ax=ax, shrink=.8); ax.set_title("Category pair lift within orders")
save(fig, "p3_04_category_lift")

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
for k, v in res.items():
    print(f"\n### {k}\n{v}")
