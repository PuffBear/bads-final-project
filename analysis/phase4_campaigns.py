"""Phase 4: campaign effectiveness and targeting.

Inputs : campaign_exposures, campaign_catalog, orders (attributed), customers, customer_segments (Phase 1)
Outputs: outputs/tables/phase4_*.csv, outputs/figures/p4_*.png
Design :
  * Unit = exposure (21,000). Outcome = purchased_within_14_days.
  * Revenue of a converted exposure = its attributed order's merchandise value after discount (GMV - discount);
    shipping is treated as a pass-through and excluded.
  * Profit needs a margin assumption: GROSS MARGIN = 60% of list-price GMV (sensitivity 40-70%).
      contribution = margin x GMV - discount - contact cost
  * There is NO control group. Attributed results are an UPPER BOUND on what campaigns caused. We therefore also
    report BREAK-EVEN INCREMENTALITY: the share of attributed orders that must be truly caused by the campaign
    for it to break even, given that buyers who would have bought anyway still receive the discount:
        f* = (discount + contact cost) / (margin x GMV)
  * Models use PRE-CONTACT information only (customer snapshot + what is known when the send is planned:
    channel, discount, featured category, prior contacts). delivered / opened / clicked are post-contact
    outcomes and would leak the answer (demonstrated below).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from common import load, save, plt, TAB, SERIES, INK2, MUTED, AXIS, CHANNEL_COLORS, CATEGORY_COLORS

SEED = 42
MARGIN = 0.60
MARGINS = [0.40, 0.50, 0.60, 0.70]
ex, cc, od, cu = (load(n) for n in ["campaign_exposures", "campaign_catalog", "orders", "customers"])
seg = pd.read_csv(TAB / "customer_segments.csv")
CH = list(CHANNEL_COLORS)
SEGS = ["Recent actives", "Lapsing regulars", "Low-basket buyers", "Occasional buyers", "Registered, never purchased"]
res = {}

# ------------------------------------------------------------------ analysis table
att = od.dropna(subset=["attributed_exposure_id"])[["attributed_exposure_id", "gross_merchandise_value_inr", "discount_inr"]]
e = (ex.merge(cc.drop(columns="channel"), on="campaign_id")
       .merge(att, left_on="exposure_id", right_on="attributed_exposure_id", how="left")
       .merge(cu, on="customer_id").merge(seg[["customer_id", "segment", "R", "F", "M", "E"]], on="customer_id"))
e[["gross_merchandise_value_inr", "discount_inr"]] = e[["gross_merchandise_value_inr", "discount_inr"]].fillna(0)
e["net_rev"] = e.gross_merchandise_value_inr - e.discount_inr
e["y"] = e.purchased_within_14_days
e = e.sort_values(["customer_id", "exposure_date", "exposure_id"]).reset_index(drop=True)
e["prior_contacts"] = e.groupby("customer_id").cumcount()
e["prior_contacts_30d"] = e.groupby("customer_id").exposure_date.transform(lambda s: [((s < d) & (s >= d - pd.Timedelta(days=30))).sum() for d in s])
e["cat_match"] = (e.preferred_category == e.featured_category).astype(int)
e["buyer_before"] = (e.orders_before_2026 > 0).astype(int)
e["log_recency"] = np.log1p(e.days_since_last_order_at_2025_12_31)
e["log_orders"] = np.log1p(e.orders_before_2026)
e["log_spend"] = np.log1p(e.spend_before_2026_inr)
e["tenure_days"] = (pd.Timestamp("2025-12-31") - e.signup_date).dt.days
e["month"] = e.exposure_date.dt.month
e["segment"] = pd.Categorical(e.segment, SEGS)

# ------------------------------------------------------------------ 1. funnel + economics per campaign and channel
def econ(g, m=MARGIN):
    n, buyers = len(g), g.y.sum()
    gmv, disc, cost = g.gross_merchandise_value_inr.sum(), g.discount_inr.sum(), g.contact_cost_inr.sum()
    contrib = m * gmv - disc - cost
    return pd.Series({
        "exposures": n, "delivered": g.delivered.mean(), "opened": g.opened.mean(), "clicked": g.clicked.mean(),
        "conv_rate": buyers / n, "conv_per_delivered": buyers / g.delivered.sum(), "buyers": buyers,
        "contact_cost": cost, "cost_per_exposure": cost / n, "cost_per_order": cost / max(buyers, 1),
        "net_revenue": g.net_rev.sum(), "net_rev_per_exposure": g.net_rev.sum() / n, "avg_order_net": g.net_rev.sum() / max(buyers, 1),
        "discount_cost": disc, "contribution": contrib, "contribution_per_exposure": contrib / n,
        "roi_on_contact": contrib / cost, "revenue_per_contact_rupee": g.net_rev.sum() / cost,
        "breakeven_incrementality": (disc + cost) / (m * gmv) if gmv else np.nan,
        "nonconsent_share": (g.marketing_consent == 0).mean()})
camp = e.groupby(["campaign_id", "campaign_name", "channel", "featured_category", "discount_pct", "campaign_objective"]).apply(econ).reset_index()
chan = e.groupby("channel").apply(econ).reindex(CH)
tot = econ(e)
camp.round(4).to_csv(TAB / "phase4_campaign_economics.csv", index=False)
chan.round(4).to_csv(TAB / "phase4_channel_economics.csv")
res["campaign_economics"] = camp[["campaign_id", "channel", "featured_category", "discount_pct", "exposures", "conv_rate", "cost_per_order",
                                  "avg_order_net", "contribution", "roi_on_contact", "breakeven_incrementality"]].round(3)
res["channel_economics"] = chan[["exposures", "delivered", "opened", "clicked", "conv_rate", "cost_per_exposure", "cost_per_order",
                                 "net_rev_per_exposure", "contribution_per_exposure", "roi_on_contact", "breakeven_incrementality"]].round(3)
res["total"] = tot.round(3)
# margin sensitivity of break-even incrementality and contribution
sens = pd.DataFrame({f"margin_{int(m*100)}": e.groupby("campaign_id").apply(lambda g: econ(g, m)["breakeven_incrementality"]) for m in MARGINS})
sens["channel"] = cc.set_index("campaign_id").channel
sens.round(3).to_csv(TAB / "phase4_breakeven_sensitivity.csv")
res["breakeven_sensitivity"] = sens.round(3)
sens_tot = pd.DataFrame({f"margin_{int(m*100)}": econ(e, m)[["contribution", "breakeven_incrementality"]] for m in MARGINS})
res["total_sensitivity"] = sens_tot.round(3)

# ------------------------------------------------------------------ 2. what drives purchase? (pre-contact features; customer-clustered SEs)
f_base = ("y ~ log_recency + log_orders + log_spend + buyer_before + website_visits_dec_2025 + email_engagement_dec_2025"
          " + marketing_consent + age + C(gender) + C(region) + C(annual_income_band_inr) + C(preferred_category) + np.log1p(tenure_days)")
f_full = f_base + " + cat_match + C(channel, Treatment('Email')) + discount_pct + C(featured_category) + prior_contacts_30d"
groups = e.customer_id.astype("category").cat.codes
m_full = smf.logit(f_full, data=e).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
tab = pd.DataFrame({"odds_ratio": np.exp(m_full.params), "ci_low": np.exp(m_full.conf_int()[0]), "ci_high": np.exp(m_full.conf_int()[1]),
                    "p_value": m_full.pvalues}).drop("Intercept")
tab.round(4).to_csv(TAB / "phase4_logit_drivers.csv")
res["logit_drivers"] = tab.round(3)
res["logit_fit"] = pd.Series({"n": m_full.nobs, "pseudo_r2": m_full.prsquared, "pseudo_r2_customer_only": smf.logit(f_base, data=e).fit(disp=0).prsquared}).round(4)

# 2b. within-customer (conditional logit, customer fixed effects): identifies campaign attributes from customers
#     who received several campaigns, holding every stable customer trait constant.
cl_df = e[e.groupby("customer_id").y.transform("nunique") == 2].copy()  # only customers with variation in outcome inform the model
X_cl = pd.get_dummies(cl_df[["channel", "featured_category"]], drop_first=False).astype(float)
X_cl = X_cl.drop(columns=["channel_Email", "featured_category_Skincare"])
X_cl["discount_pct"] = cl_df.discount_pct; X_cl["cat_match"] = cl_df.cat_match; X_cl["prior_contacts_30d"] = cl_df.prior_contacts_30d
X_cl["month"] = cl_df.month
cl = ConditionalLogit(cl_df.y.values, X_cl.values, groups=cl_df.customer_id.values).fit(disp=0)
cl_tab = pd.DataFrame({"odds_ratio": np.exp(cl.params), "ci_low": np.exp(cl.conf_int()[:, 0]), "ci_high": np.exp(cl.conf_int()[:, 1]),
                       "p_value": cl.pvalues}, index=X_cl.columns)
cl_tab.round(4).to_csv(TAB / "phase4_conditional_logit.csv")
res["conditional_logit"] = cl_tab.round(3)
res["conditional_logit_n"] = pd.Series({"customers": cl_df.customer_id.nunique(), "exposures": len(cl_df)})

# ------------------------------------------------------------------ 3. raw vs targeting-mix-adjusted campaign conversion
#     logit with customer features + campaign fixed effects, then predict every campaign on the SAME population (all 21k exposures).
m_fe = smf.logit(f_base + " + cat_match + prior_contacts_30d + C(campaign_id)", data=e).fit(disp=0)
adj = {}
for cid in cc.campaign_id:
    d = e.copy(); d["campaign_id"] = cid
    d["cat_match"] = (d.preferred_category == cc.set_index("campaign_id").featured_category[cid]).astype(int)
    adj[cid] = m_fe.predict(d).mean()
mix = camp.set_index("campaign_id")[["campaign_name", "channel", "featured_category", "discount_pct", "conv_rate"]].copy()
mix["adjusted_conv_rate"] = pd.Series(adj)
mix["expected_from_audience"] = e.assign(p=smf.logit(f_base, data=e).fit(disp=0).predict(e)).groupby("campaign_id").p.mean()
mix["rank_raw"] = mix.conv_rate.rank(ascending=False).astype(int); mix["rank_adjusted"] = mix.adjusted_conv_rate.rank(ascending=False).astype(int)
mix.round(4).to_csv(TAB / "phase4_mix_adjusted_conversion.csv")
res["mix_adjusted"] = mix.round(4)
dm = smf.ols("adjusted_conv_rate ~ discount_pct", data=mix).fit()
res["adjusted_rate_vs_discount"] = pd.Series({"slope_per_pp": dm.params.discount_pct, "p_value": dm.pvalues.discount_pct, "r2": dm.rsquared}).round(5)

# ------------------------------------------------------------------ 4. targeting model (pre-contact only), time-based validation
num = ["log_recency", "log_orders", "log_spend", "buyer_before", "website_visits_dec_2025", "email_engagement_dec_2025",
       "marketing_consent", "age", "tenure_days", "discount_pct", "cat_match", "prior_contacts", "prior_contacts_30d"]
cat = ["gender", "region", "annual_income_band_inr", "preferred_category", "channel", "featured_category"]
train = e[e.campaign_id <= "C010"]; test = e[e.campaign_id > "C010"]
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
models = {
    "Logistic regression": make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0)),
    "Gradient boosting": make_pipeline(ColumnTransformer([("num", "passthrough", num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]),
                                       HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=80,
                                                                      l2_regularization=1.0, random_state=SEED)),
    # Gaussian NB: numeric features only (one-hot dummies are not Gaussian). Each feature ~ Normal within class,
    # features assumed independent given the class. Scaling does not change GaussianNB but keeps the pipeline uniform.
    "Gaussian Naive Bayes": make_pipeline(ColumnTransformer([("num", StandardScaler(), num)]), GaussianNB()),
    # Same, without log_spend and buyer_before (r = 0.86-0.97 with log_orders): NB's independence assumption makes it
    # count the same 'frequent buyer' evidence three times, which inflates its probabilities.
    "Gaussian NB, redundant features removed": make_pipeline(ColumnTransformer([("num", StandardScaler(), [c for c in num if c not in ("log_spend", "buyer_before")])]), GaussianNB()),
    # Categorical NB: ALL features. Numeric ones binned into (up to) 5 quantile bins learned on the training data;
    # categorical ones ordinal-coded. alpha = 1 (Laplace smoothing) so unseen bins never get zero probability.
    "Naive Bayes (categorical, binned)": make_pipeline(
        ColumnTransformer([("num", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf"), num),
                           ("cat", OrdinalEncoder(), cat)]),  # every test category is seen in training
        CategoricalNB(alpha=1.0, min_categories=None)),
    "RFM score only": None,
}
def boot_auc(y, p, B=500):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); out = []
    for _ in range(B):
        s = rng.choice(idx, len(idx));
        if y[s].min() != y[s].max(): out.append(roc_auc_score(y[s], p[s]))
    return np.percentile(out, [2.5, 97.5])
perf, preds = [], {}
for nm, mdl in models.items():
    if mdl is None:
        p_test = (test.R.fillna(0) * 10 + test.F.fillna(0) * 3 + test.M.fillna(0)).values  # simple RFM ranking baseline
    else:
        mdl.fit(train[num + cat], train.y); p_test = mdl.predict_proba(test[num + cat])[:, 1]
    preds[nm] = p_test
    lo, hi = boot_auc(test.y.values, p_test)
    perf.append(dict(model=nm, auc_test=roc_auc_score(test.y, p_test), auc_ci_low=lo, auc_ci_high=hi,
                     avg_precision=average_precision_score(test.y, p_test),
                     brier=brier_score_loss(test.y, p_test) if mdl is not None else np.nan))
# leakage demonstration
leak = make_pipeline(ColumnTransformer([("num", StandardScaler(), num + ["delivered", "opened", "clicked"]), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
                     LogisticRegression(max_iter=2000)).fit(train[num + cat + ["delivered", "opened", "clicked"]], train.y)
p_leak = leak.predict_proba(test[num + cat + ["delivered", "opened", "clicked"]])[:, 1]
perf.append(dict(model="LEAKY: + delivered/opened/clicked (not usable)", auc_test=roc_auc_score(test.y, p_leak), auc_ci_low=np.nan, auc_ci_high=np.nan,
                 avg_precision=average_precision_score(test.y, p_leak), brier=np.nan))
# customer-grouped 5-fold CV on all campaigns as a second check, every fitted model
from sklearn.base import clone
cv = {nm: [] for nm, mdl in models.items() if mdl is not None}
for tr, te in GroupKFold(5).split(e, groups=e.customer_id):
    for nm, mdl in models.items():
        if mdl is None: continue
        mm = clone(mdl).fit(e.iloc[tr][num + cat], e.iloc[tr].y)
        cv[nm].append(roc_auc_score(e.iloc[te].y, mm.predict_proba(e.iloc[te][num + cat])[:, 1]))
perf = pd.DataFrame(perf)
perf["cv_auc_grouped_5fold"] = perf.model.map({k: np.mean(v) for k, v in cv.items()})
perf["cv_auc_sd"] = perf.model.map({k: np.std(v, ddof=1) for k, v in cv.items()})
# calibration: mean predicted probability vs actual rate on the hold-out (NB's independence assumption inflates confidence)
perf["mean_pred_prob"] = perf.model.map({k: v.mean() for k, v in preds.items() if k in cv})
perf["actual_rate"] = test.y.mean()
# top-30% capture on the hold-out for every ranking
def capture(p, share=.3):
    o_ = np.argsort(-p); k_ = int(round(share * len(p)))
    return test.y.values[o_[:k_]].sum() / test.y.sum()
perf["top30_buyer_capture"] = perf.model.map({k: capture(v) for k, v in preds.items()} | {"LEAKY: + delivered/opened/clicked (not usable)": capture(p_leak)})
perf.round(4).to_csv(TAB / "phase4_model_performance.csv", index=False)
res["model_performance"] = perf.round(4)
res["test_set"] = pd.Series({"train_exposures": len(train), "test_exposures": len(test), "test_buyers": int(test.y.sum()), "test_campaigns": "C011-C014"})

best = "Logistic regression" if perf.set_index("model").auc_test["Logistic regression"] >= perf.set_index("model").auc_test["Gradient boosting"] - 0.005 else "Gradient boosting"
res["chosen_model"] = best
t = test.assign(score=preds[best]).sort_values("score", ascending=False).reset_index(drop=True)
t["decile"] = (np.arange(len(t)) * 10 // len(t)) + 1
gains = t.groupby("decile").agg(exposures=("y", "size"), buyers=("y", "sum"), conv_rate=("y", "mean"), cost=("contact_cost_inr", "sum"),
                               net_rev=("net_rev", "sum"), gmv=("gross_merchandise_value_inr", "sum"), disc=("discount_inr", "sum"))
gains["lift"] = gains.conv_rate / t.y.mean()
gains["cum_share_buyers"] = gains.buyers.cumsum() / gains.buyers.sum()
gains["cum_share_cost"] = gains.cost.cumsum() / gains.cost.sum()
gains["cum_share_exposures"] = gains.exposures.cumsum() / gains.exposures.sum()
gains["contribution"] = MARGIN * gains.gmv - gains.disc - gains.cost
gains["cum_contribution"] = gains.contribution.cumsum()
gains.round(4).to_csv(TAB / "phase4_gains_deciles.csv")
res["gains"] = gains.round(3)
coef_names = models["Logistic regression"][0].get_feature_names_out()
coefs = pd.Series(models["Logistic regression"][-1].coef_[0], index=coef_names).sort_values()
coefs.round(4).to_csv(TAB / "phase4_logit_model_coefficients.csv")

# what if: contact only top-k deciles, and suppress non-consented
sim = []
for k in range(1, 11):
    top = t[t.decile <= k]
    sim.append(dict(deciles_contacted=k, share_contacts=len(top) / len(t), share_buyers=top.y.sum() / t.y.sum(),
                    share_cost=top.contact_cost_inr.sum() / t.contact_cost_inr.sum(),
                    conv_rate=top.y.mean(), attributed_contribution=(MARGIN * top.gross_merchandise_value_inr - top.discount_inr - top.contact_cost_inr).sum()))
sim = pd.DataFrame(sim); sim.round(4).to_csv(TAB / "phase4_targeting_simulation.csv", index=False)
res["targeting_sim"] = sim.round(3)

# ------------------------------------------------------------------ 4b. discount economics: does extra conversion pay for the extra discount?
# Expected contribution per exposure at discount d (fraction):  p(d) * (m - d) * GMV - contact cost
#   p(d) = p10 * OR^(100*(d - 0.10))  (odds-ratio approximation is fine at p ~ 5%)
# Every buyer gets the discount - including those who would have bought anyway - so the cost of a deeper
# discount is charged on ALL buyers; only the extra buyers bring margin. Two OR estimates: cross-sectional and within-customer.
gmv_avg = e.loc[e.y == 1, "gross_merchandise_value_inr"].mean()
p10 = e.loc[e.discount_pct == 10, "y"].mean()
disc_rows = []
for src, orr in [("cross-sectional logit", tab.loc["discount_pct", "odds_ratio"]), ("within-customer logit", cl_tab.loc["discount_pct", "odds_ratio"]),
                 ("within-customer lower 95% CI", cl_tab.loc["discount_pct", "ci_low"])]:
    for m in MARGINS:
        for d in range(10, 31, 5):
            odds = p10 / (1 - p10) * orr ** (d - 10); p = odds / (1 + odds)
            disc_rows.append(dict(estimate=src, margin=m, discount_pct=d, conv=p, contribution_per_exposure=p * (m - d / 100) * gmv_avg))
disc_econ = pd.DataFrame(disc_rows)
disc_econ.round(4).to_csv(TAB / "phase4_discount_economics.csv", index=False)
best_d = disc_econ.loc[disc_econ.groupby(["estimate", "margin"]).contribution_per_exposure.idxmax()][["estimate", "margin", "discount_pct", "contribution_per_exposure"]]
res["discount_best_in_grid"] = best_d.round(2)
res["discount_econ_60"] = disc_econ[disc_econ.margin == .6].pivot(index="discount_pct", columns="estimate", values="contribution_per_exposure").round(2)

# break-even incrementality by score decile (how 'sure' must we be that a contact causes the sale?)
gains["breakeven_incrementality"] = (gains.disc + gains.cost) / (MARGIN * gains.gmv)
gains.round(4).to_csv(TAB / "phase4_gains_deciles.csv")
res["gains_breakeven"] = gains[["conv_rate", "cost", "breakeven_incrementality"]].round(3)

# ------------------------------------------------------------------ 5. consent & frequency
cons = e.groupby("marketing_consent").apply(lambda g: pd.Series({"exposures": len(g), "conv_rate": g.y.mean(), "buyers": g.y.sum(),
                                                                 "contact_cost": g.contact_cost_inr.sum(), "net_rev": g.net_rev.sum()}))
cons["share_of_buyers"] = cons.buyers / cons.buyers.sum()
res["consent"] = cons.round(3)
cons.round(4).to_csv(TAB / "phase4_consent.csv")
freq = e.groupby(pd.cut(e.prior_contacts_30d, [-1, 0, 1, 2, 10], labels=["0", "1", "2", "3+"]), observed=True).y.agg(["mean", "size"])
res["frequency_30d"] = freq.round(4)

# ------------------------------------------------------------------ 6. segment x channel conversion (for targeting rules)
sxc = e.pivot_table(index="segment", columns="channel", values="y", aggfunc="mean", observed=True)[CH]
sxc_n = e.pivot_table(index="segment", columns="channel", values="y", aggfunc="size", observed=True)[CH]
sxc.round(4).to_csv(TAB / "phase4_segment_channel_conversion.csv")
res["segment_x_channel"] = (sxc * 100).round(1)
# segment-level economics on cheapest channel vs all
seg_econ = e.groupby("segment", observed=True).apply(econ)[["exposures", "conv_rate", "cost_per_order", "net_rev_per_exposure", "contribution_per_exposure", "breakeven_incrementality"]]
seg_econ.round(4).to_csv(TAB / "phase4_segment_economics.csv")
res["segment_economics"] = seg_econ.round(3)

# ================================================================== figures
# p4_01 funnel by campaign (heat-table style)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
c2 = camp.sort_values("campaign_id")
labels = [f"{r.campaign_id} {r.campaign_name} ({r.channel}, {r.discount_pct}%)" for r in c2.itertuples()]
ax = axs[0]
ax.barh(labels, c2.conv_rate * 100, color=[CHANNEL_COLORS[c] for c in c2.channel], height=.65)
for i_, v_ in enumerate(c2.conv_rate * 100): ax.text(v_, i_, f" {v_:.1f}%", va="center", fontsize=7, color=INK2)
ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=7); ax.set_xlim(0, 8.5)
ax.set_title("14-day purchase rate per exposure", fontsize=9)
ax = axs[1]
ax.barh(labels, c2.cost_per_order, color=[CHANNEL_COLORS[c] for c in c2.channel], height=.65); ax.set_xscale("log")
for i_, v_ in enumerate(c2.cost_per_order): ax.text(v_, i_, f" ₹{v_:,.0f}", va="center", fontsize=7, color=INK2)
ax.invert_yaxis(); ax.set_yticklabels([]); ax.grid(axis="y", visible=False); ax.set_title("Contact cost per attributed order (₹, log scale)", fontsize=9)
ax = axs[2]
ax.barh(labels, c2.breakeven_incrementality * 100, color=[CHANNEL_COLORS[c] for c in c2.channel], height=.65)
for i_, v_ in enumerate(c2.breakeven_incrementality * 100): ax.text(v_, i_, f" {v_:.0f}%", va="center", fontsize=7, color=INK2)
ax.invert_yaxis(); ax.set_yticklabels([]); ax.grid(axis="y", visible=False); ax.set_xlim(0, 100)
ax.set_title(f"Break-even incrementality (margin {MARGIN:.0%})\nshare of attributed orders that must be truly caused", fontsize=9)
save(fig, "p4_01_campaign_economics")

# p4_02 raw vs adjusted (dumbbell, sorted by adjusted)
fig, ax = plt.subplots(figsize=(8, 4.6))
mm_ = mix.reset_index().sort_values("adjusted_conv_rate")
y_ = np.arange(len(mm_))
ax.hlines(y_, mm_.conv_rate * 100, mm_.adjusted_conv_rate * 100, color=AXIS, lw=2)
ax.scatter(mm_.conv_rate * 100, y_, s=90, facecolor="none", edgecolor=INK2, lw=1.2, zorder=5, label="raw")
ax.scatter(mm_.adjusted_conv_rate * 100, y_, s=36, color=[CHANNEL_COLORS[c] for c in mm_.channel], zorder=4, label="adjusted to common audience")
ax.scatter(mm_.expected_from_audience * 100, y_, marker="|", s=80, color=MUTED, zorder=2, label="expected from audience alone")
ax.set_yticks(y_, [f"{r.campaign_id} {r.campaign_name} ({r.channel}, {r.discount_pct}%)" for r in mm_.itertuples()], fontsize=7.5)
ax.grid(axis="y", visible=False); ax.set_xlabel("14-day purchase rate (%)"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Campaign conversion: raw vs adjusted for who was targeted")
save(fig, "p4_02_mix_adjusted")

# p4_03 drivers forest (selected)
keep = [i for i in tab.index if any(k in i for k in ["cat_match", "channel", "discount_pct", "featured_category", "prior_contacts", "log_recency",
                                                      "log_orders", "buyer_before", "email_engagement", "website_visits", "marketing_consent"])]
ft = tab.loc[keep].copy()
ft.index = (ft.index.str.replace(r"C\(channel, Treatment\('Email'\)\)\[T\.", "channel: ", regex=True).str.replace(r"C\(featured_category\)\[T\.", "featured: ", regex=True)
            .str.replace("]", "").str.replace("cat_match", "featured = preferred category").str.replace("discount_pct", "discount (per pp)")
            .str.replace("prior_contacts_30d", "contacts in prior 30 days").str.replace("log_recency", "log days since last order")
            .str.replace("log_orders", "log orders before 2026").str.replace("buyer_before", "bought before 2026")
            .str.replace("email_engagement_dec_2025", "email engagement Dec-25").str.replace("website_visits_dec_2025", "web visits Dec-25")
            .str.replace("marketing_consent", "marketing consent"))
ft = ft.sort_values("odds_ratio")
fig, ax = plt.subplots(figsize=(7.5, 5))
y_ = np.arange(len(ft)); sig = (ft.p_value < .05).values
ax.errorbar(ft.odds_ratio, y_, xerr=[ft.odds_ratio - ft.ci_low, ft.ci_high - ft.odds_ratio], fmt="none", ecolor=AXIS, lw=1.5)
ax.scatter(ft.odds_ratio[~sig], y_[~sig], color=MUTED, s=30, zorder=3, label="p ≥ 0.05"); ax.scatter(ft.odds_ratio[sig], y_[sig], color=SERIES[0], s=30, zorder=3, label="p < 0.05")
ax.axvline(1, color=INK2, lw=1); ax.set_xscale("log")
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
ax.xaxis.set_major_locator(FixedLocator([.5, .75, 1, 1.5, 2, 3, 4])); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.set_yticks(y_, ft.index, fontsize=7.5); ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
ax.set_xlabel("odds ratio of 14-day purchase (95% CI, log scale)")
ax.set_title(f"Purchase drivers, pre-contact information only (n = {int(m_full.nobs):,})")
save(fig, "p4_03_drivers")

# p4_04 gains chart + targeting
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ax = axs[0]
x_ = np.r_[0, gains.cum_share_exposures.values] * 100
ax.plot(x_, np.r_[0, gains.cum_share_buyers.values] * 100, marker="o", ms=5, color=SERIES[0], label=f"{best} (AUC {perf.set_index('model').auc_test[best]:.2f})")
ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1, label="random targeting")
ax.plot(x_, np.r_[0, gains.cum_share_cost.values] * 100, color=SERIES[1], lw=1.4, label="share of contact cost")
ax.set_xlabel("% of exposures contacted (ranked by score)"); ax.set_ylabel("% captured"); ax.legend(fontsize=7.5, loc="lower right")
ax.set_title("Cumulative gains, hold-out campaigns C011–C014", fontsize=9)
ax = axs[1]
ax.bar(gains.index, gains.conv_rate * 100, color=SERIES[0], width=.7); ax.axhline(t.y.mean() * 100, color=INK2, ls="--", lw=1)
for i_, v_ in zip(gains.index, gains.conv_rate * 100): ax.text(i_, v_, f"{v_:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
ax.set_xticks(range(1, 11)); ax.set_xlabel("score decile (1 = highest)"); ax.set_ylabel("purchase rate %"); ax.grid(axis="x", visible=False)
ax.set_title("Purchase rate by score decile (dashed = average)", fontsize=9)
save(fig, "p4_04_targeting_gains")

# p4_07 model comparison: cumulative gains (hold-out) + calibration
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
ax = axs[0]
order_m = ["Logistic regression", "Gradient boosting", "Gaussian Naive Bayes", "Naive Bayes (categorical, binned)", "RFM score only"]
calib_m = ["Logistic regression", "Gradient boosting", "Gaussian Naive Bayes", "Naive Bayes (categorical, binned)", "Gaussian NB, redundant features removed"]
pf = perf.set_index("model")
for i_, nm in enumerate(order_m):
    o_ = np.argsort(-preds[nm]); cy = np.r_[0, np.cumsum(test.y.values[o_])] / test.y.sum()
    cx = np.arange(len(cy)) / len(test)
    ax.plot(cx * 100, cy * 100, color=SERIES[i_] if i_ < 4 else MUTED, lw=1.6, ls="-" if i_ < 4 else "--",
            label=f"{nm} (AUC {pf.auc_test[nm]:.3f})")
ax.plot([0, 100], [0, 100], color=AXIS, lw=1, ls=":"); ax.axvline(30, color=AXIS, lw=.8)
ax.set_xlabel("% of hold-out exposures contacted (ranked by model)"); ax.set_ylabel("% of buyers captured")
ax.legend(fontsize=7, loc="lower right"); ax.set_title("Cumulative gains on hold-out campaigns C011–C014", fontsize=9)
ax = axs[1]
for i_, nm in enumerate(calib_m):
    fp, mp = calibration_curve(test.y, preds[nm], n_bins=8, strategy="quantile")
    ax.plot(mp * 100, fp * 100, marker="o", ms=5, lw=1.4, color=SERIES[i_], ls="-" if i_ < 4 else "--",
            label=f"{nm} (mean {preds[nm].mean():.1%} vs actual {test.y.mean():.1%})")
ax.plot([0, 20], [0, 20], color=INK2, lw=1, ls=":", label="perfect calibration")
ax.set_xlim(0, 36); ax.set_ylim(0, 20); ax.set_xlabel("mean predicted purchase probability % (bin)"); ax.set_ylabel("actual purchase rate % (bin)")
ax.legend(fontsize=7, loc="upper left"); ax.set_title("Calibration: do predicted probabilities match reality?", fontsize=9)
save(fig, "p4_07_model_comparison")

# p4_05 segment x channel heat table
fig, ax = plt.subplots(figsize=(8, 3.2))
from common import SEQ_CMAP
im = ax.imshow(sxc.values * 100, cmap=SEQ_CMAP, aspect="auto")
ax.set_xticks(range(len(CH)), CH); ax.set_yticks(range(len(sxc)), sxc.index); ax.grid(False)
for i in range(sxc.shape[0]):
    for j in range(sxc.shape[1]):
        v = sxc.values[i, j] * 100
        ax.text(j, i, f"{v:.1f}%\nn={sxc_n.values[i, j]:,}", ha="center", va="center", fontsize=7, color="white" if v > 8 else "#0b0b0b")
fig.colorbar(im, ax=ax, shrink=.8, label="purchase rate %"); ax.set_title("14-day purchase rate by segment × channel")
save(fig, "p4_05_segment_channel")

# p4_06 discount economics
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for i_, m in enumerate(MARGINS):
    dd = disc_econ[(disc_econ.margin == m) & (disc_econ.estimate == "within-customer logit")]
    ax.plot(dd.discount_pct, dd.contribution_per_exposure, marker="o", ms=5, color=SERIES[i_], label=f"margin {m:.0%}")
ax.axvspan(10, 25, color=AXIS, alpha=.25, lw=0); ax.text(10.3, ax.get_ylim()[1] * .95, "tested range 10-25%", fontsize=7, color=INK2, va="top")
ax.set_xlabel("campaign discount %"); ax.set_ylabel("attributed contribution per exposure (₹)"); ax.legend(fontsize=7.5)
ax.set_title("Discount depth vs attributed contribution (point estimate, within-customer;\nat the lower 95% CI, 10% is best at every margin)", fontsize=9)
save(fig, "p4_06_discount_economics")

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 50)
for k, v in res.items():
    print(f"\n### {k}\n{v}")
