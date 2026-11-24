"""Phase 1: customer RFM + engagement scoring and k-means segmentation.

Inputs : customers.csv (pre-campaign snapshot at 31 Dec 2025)
Outputs: outputs/tables/customer_segments.csv  (one row per customer, used by Phases 2-7)
         outputs/tables/phase1_*.csv, outputs/figures/p1_*.png
Design :
  * 844 customers with orders_before_2026 = 0 have no meaningful recency/AOV -> a rule-based
    "Registered, never purchased" group, kept out of k-means.
  * Buyers are clustered on log recency, log frequency, log AOV, website visits and email
    engagement. Spend is not used directly because it is ~ orders x AOV (Spearman 0.93 with orders),
    which would double-weight frequency.
  * 2026 campaign outcomes are NOT used to build segments; they are reported afterwards as an
    out-of-sample check that the segments matter.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.decomposition import PCA
from common import load, save, plt, TAB, SERIES, INK2, MUTED, SEQ_CMAP, DIV_CMAP, AXIS as AXIS_C

SEED = 42
cu = load("customers")
ex = load("campaign_exposures")
cc = load("campaign_catalog")
INC = ["Under 4L", "4L-8L", "8L-15L", "15L-25L", "25L+"]

cu["aov_before_2026"] = np.where(cu.orders_before_2026 > 0, cu.spend_before_2026_inr / cu.orders_before_2026.replace(0, np.nan), 0)
cu["tenure_days"] = (pd.Timestamp("2025-12-31") - cu.signup_date).dt.days
cu["buyer"] = cu.orders_before_2026 > 0

# ------------------------------------------------------------------ RFM + engagement scores (all buyers)
b = cu[cu.buyer].copy()
def q5(s, ascending=True):
    """quintile score 1-5 on percentile rank (rank handles ties in discrete fields)"""
    r = s.rank(method="average", pct=True, ascending=ascending)
    return np.ceil(r * 5).clip(1, 5).astype(int)
b["R"] = q5(b.days_since_last_order_at_2025_12_31, ascending=False)  # fewer days = higher score
# orders are integers with heavy ties (1-3 orders = 70% of buyers), so quintiles collapse; use explicit bands
b["F"] = pd.cut(b.orders_before_2026, [0, 1, 2, 3, 5, 100], labels=[1, 2, 3, 4, 5]).astype(int)
b["M"] = q5(b.spend_before_2026_inr)
b["E"] = q5(b.website_visits_dec_2025 + b.email_engagement_dec_2025)
b["RFM"] = b.R.astype(str) + b.F.astype(str) + b.M.astype(str)
b["RFM_total"] = b[["R", "F", "M"]].sum(axis=1)
cu = cu.merge(b[["customer_id", "R", "F", "M", "E", "RFM", "RFM_total"]], on="customer_id", how="left")

# ------------------------------------------------------------------ k-means on buyers
feat_names = ["log_recency", "log_orders", "log_aov", "website_visits_dec_2025", "email_engagement_dec_2025"]
X = pd.DataFrame({
    "log_recency": np.log1p(b.days_since_last_order_at_2025_12_31),
    "log_orders": np.log1p(b.orders_before_2026),
    "log_aov": np.log(b.spend_before_2026_inr / b.orders_before_2026),
    "website_visits_dec_2025": b.website_visits_dec_2025,
    "email_engagement_dec_2025": b.email_engagement_dec_2025,
}, index=b.index)
Z = StandardScaler().fit_transform(X)

# ------------------------------------------------------------------ choosing k: five lines of evidence
# (1) internal fit: inertia (elbow), silhouette, Calinski-Harabasz, Davies-Bouldin
# (2) silhouette vs a NULL baseline: same k-means on data whose columns are permuted independently
#     (keeps every marginal distribution, destroys joint structure). Real - null = structure beyond noise.
# (3) gap statistic (Tibshirani et al. 2001) with a uniform reference box, B = 10
# (4) bootstrap stability: ARI between full-data labels and refits on 80% subsamples
# (5) external (predictive) validity: how much segment membership explains the 2026 purchase flag,
#     which is NOT used in clustering -> McFadden pseudo-R2 of purchase ~ segment, per k
from sklearn.metrics import calinski_harabasz_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

y26 = ex.groupby("customer_id").purchased_within_14_days.agg(["sum", "size"])  # buyer-level outcomes
yb = b[["customer_id"]].merge(y26, left_on="customer_id", right_index=True, how="left").fillna(0)

def pseudo_r2(labels):
    """McFadden R2 of exposure-level purchase on segment dummies (grouped binomial)."""
    ok = yb["size"].values > 0
    lab, succ, n = labels[ok], yb["sum"].values[ok], yb["size"].values[ok]
    p0 = succ.sum() / n.sum()
    ll0 = (succ * np.log(p0) + (n - succ) * np.log(1 - p0)).sum()
    df_ = pd.DataFrame(dict(l=lab, s=succ, n=n)).groupby("l").sum()
    pk = (df_.s / df_.n).clip(1e-9, 1 - 1e-9)
    ll1 = (df_.s * np.log(pk) + (df_.n - df_.s) * np.log(1 - pk)).sum()
    return 1 - ll1 / ll0

def cluster_jaccard(k, B=20):
    """Hennig (2007) clusterboot: mean Jaccard of each original cluster with its best match after bootstrap refits.
    > 0.85 highly stable, 0.75-0.85 stable, 0.6-0.75 doubtful pattern, < 0.6 dissolved."""
    base = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z); J = np.zeros((B, k))
    for bi in range(B):
        idx = np.random.default_rng(bi).choice(len(Z), len(Z), replace=True); u = np.unique(idx)
        lab = KMeans(k, n_init=10, random_state=bi).fit(Z[idx]).predict(Z[u])
        for c in range(k):
            A = set(u[base[u] == c])
            J[bi, c] = max(len(A & set(u[lab == d_])) / len(A | set(u[lab == d_])) for d_ in range(k))
    return J.mean(0)

ex_b = ex[ex.customer_id.isin(b.customer_id)].reset_index(drop=True)
cust_pos = pd.Series(np.arange(len(b)), index=b.customer_id.values)
def cv_logloss_gain(labels, folds=5):
    """customer-grouped 5-fold CV: log-loss of purchase predicted by segment rate vs overall rate (1 - LL/LL0)."""
    from sklearn.model_selection import GroupKFold
    c = labels[cust_pos[ex_b.customer_id].values]; y = ex_b.purchased_within_14_days.values; ll = ll0 = 0
    for tr, te in GroupKFold(folds).split(y, groups=ex_b.customer_id):
        rate = pd.Series(y[tr]).groupby(c[tr]).mean(); p = np.clip(pd.Series(c[te]).map(rate).values, 1e-6, 1 - 1e-6)
        p0 = y[tr].mean()
        ll += -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum(); ll0 += -(y[te] * np.log(p0) + (1 - y[te]) * np.log(1 - p0)).sum()
    return 1 - ll / ll0

def wk(Zm, lab):
    return sum(((Zm[lab == c] - Zm[lab == c].mean(0)) ** 2).sum() for c in np.unique(lab))

rows = []
rng = np.random.default_rng(SEED)
def permuted(r):
    g = np.random.default_rng(1000 + r)  # one generator per draw -> each column gets a DIFFERENT permutation
    return np.column_stack([g.permutation(Z[:, j]) for j in range(Z.shape[1])])
Zperms = [permuted(r) for r in range(10)]
lo, hi = Z.min(0), Z.max(0)
refs = [rng.uniform(lo, hi, size=Z.shape) for _ in range(10)]
for k in range(2, 10):
    km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
    aris = []
    for s_ in range(10):
        idx = np.random.default_rng(s_).choice(len(Z), int(.8 * len(Z)), replace=False)
        km2 = KMeans(k, n_init=10, random_state=100 + s_).fit(Z[idx])
        aris.append(adjusted_rand_score(km.labels_[idx], km2.labels_))
    nulls = [silhouette_score(P, KMeans(k, n_init=10, random_state=r).fit_predict(P)) for r, P in enumerate(Zperms)]
    logw = np.log(wk(Z, km.labels_))
    logw_ref = [np.log(wk(R, KMeans(k, n_init=5, random_state=SEED).fit_predict(R))) for R in refs]
    sizes = np.bincount(km.labels_)
    rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(Z, km.labels_),
                     silhouette_null=np.mean(nulls), silhouette_null_sd=np.std(nulls),
                     calinski_harabasz=calinski_harabasz_score(Z, km.labels_),
                     davies_bouldin=davies_bouldin_score(Z, km.labels_),
                     gap=np.mean(logw_ref) - logw, gap_se=np.std(logw_ref) * np.sqrt(1 + 1 / len(refs)),
                     stability_ari=np.mean(aris), stability_ari_min=np.min(aris),
                     smallest_cluster_share=sizes.min() / len(Z),
                     pseudo_r2_2026=pseudo_r2(km.labels_), cv_logloss_gain_2026=cv_logloss_gain(km.labels_),
                     min_cluster_jaccard=cluster_jaccard(k).min()))
ksel = pd.DataFrame(rows)
ksel["silhouette_lift_vs_null"] = ksel.silhouette - ksel.silhouette_null
ksel["silhouette_z_vs_null"] = ksel.silhouette_lift_vs_null / ksel.silhouette_null_sd
ksel["inertia_drop_pct"] = -ksel.inertia.pct_change() * 100
# Tibshirani rule: smallest k with gap(k) >= gap(k+1) - se(k+1)
ksel["gap_rule_ok"] = ksel.gap >= ksel.gap.shift(-1) - ksel.gap_se.shift(-1)
ksel = ksel.round(4)
ksel.to_csv(TAB / "phase1_k_selection.csv", index=False)
print(ksel.to_string(index=False))

# Selection rule. Silhouette never exceeds 0.27 (below the 0.25-0.50 'weak structure' band for k >= 3), so k
# cannot be read off the silhouette maximum (trivially k = 2). Two steps:
#   1. validity gates - a k is admissible only if
#      (a) it beats structureless data on silhouette   (z vs 10 column-permuted nulls >= 3)
#      (b) every cluster is highly stable              (min per-cluster bootstrap Jaccard >= 0.85, Hennig 2007)
#      (c) no cluster is below 10% of buyers           (big enough to run a campaign against)
#   2. usefulness - among admissible k, take the one that best predicts the held-out 2026 purchase flag
#      (customer-grouped 5-fold CV log-loss gain). Ties (< 0.001) go to the smaller k.
ksel["passes_rule"] = (ksel.silhouette_z_vs_null >= 3) & (ksel.min_cluster_jaccard >= .85) & (ksel.smallest_cluster_share >= .10)
ksel.to_csv(TAB / "phase1_k_selection.csv", index=False)
print(ksel[["k", "silhouette", "silhouette_null", "silhouette_lift_vs_null", "silhouette_z_vs_null", "min_cluster_jaccard", "smallest_cluster_share", "cv_logloss_gain_2026", "passes_rule"]].to_string(index=False))
adm = ksel[ksel.passes_rule]
best = adm.cv_logloss_gain_2026.max()
K = int(adm.loc[adm.cv_logloss_gain_2026 >= best - 0.001, "k"].min())
print("chosen K =", K)
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
panels = [("inertia", "Elbow: within-cluster SS", None), ("silhouette", "Silhouette: real vs permuted-null data", "silhouette_null"),
          ("calinski_harabasz", "Calinski-Harabasz (higher = better)", None), ("gap", "Gap statistic (± 1 s.e.)", None),
          ("min_cluster_jaccard", "Weakest cluster's bootstrap Jaccard", None), ("cv_logloss_gain_2026", "External validity: CV log-loss gain, 2026 purchase", None)]
for ax, (c, t, c2) in zip(axs.flat, panels):
    if c == "gap":
        ax.errorbar(ksel.k, ksel.gap, yerr=ksel.gap_se, marker="o", ms=6, color=SERIES[0], capsize=3)
    else:
        ax.plot(ksel.k, ksel[c], marker="o", ms=6, color=SERIES[0], label="customer data")
    if c2:
        ax.plot(ksel.k, ksel[c2], marker="o", ms=6, color=MUTED, ls="--", label="permuted null (mean of 10)")
        for y_, lab_ in [(.25, "0.25 'weak'"), (.5, "0.50 'reasonable'")]:
            ax.axhline(y_, color=SERIES[7], lw=.8, ls=":"); ax.text(2, y_, lab_, fontsize=7, color=INK2, ha="left", va="bottom")
        ax.set_ylim(0, .6); ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1, .93))
    if c == "min_cluster_jaccard":
        for y_, lab_ in [(.85, "0.85 highly stable"), (.75, "0.75 stable")]:
            ax.axhline(y_, color=SERIES[7], lw=.8, ls=":"); ax.text(9, y_, lab_, fontsize=7, color=INK2, ha="right", va="bottom")
    ax.axvline(K, color=MUTED, ls="--", lw=1); ax.set_title(t, fontsize=9); ax.set_xlabel("k"); ax.set_xticks(range(2, 10))
fig.suptitle(f"Choosing k for buyer segmentation (dashed line = chosen k = {K})", x=.01, ha="left", fontweight="bold")
save(fig, "p1_01_k_selection")

# per-cluster silhouette values for k = 3..6 (figure drawn after segments are named)
from sklearn.metrics import silhouette_samples
silsum, sil_runs = [], {}
for k in [3, 4, 5, 6]:
    lab = KMeans(k, n_init=50, random_state=SEED).fit_predict(Z)
    sv = silhouette_samples(Z, lab); sil_runs[k] = (lab, sv)
    for c in range(k):
        v = sv[lab == c]
        silsum.append(dict(k=k, cluster=c, n=len(v), mean_silhouette=v.mean(), share_negative=(v < 0).mean()))
silsum = pd.DataFrame(silsum).round(3)
print(silsum.to_string(index=False))

# how adjacent solutions nest: cross-tab of k=4 and k=6 against k=5
lab4, lab5, lab6 = (KMeans(k, n_init=50, random_state=SEED).fit_predict(Z) for k in (4, 5, 6))
print("k4 x k5\n", pd.crosstab(lab4, lab5)); print("k6 x k5\n", pd.crosstab(lab6, lab5))
pd.crosstab(lab4, lab5).to_csv(TAB / "phase1_k4_vs_k5.csv"); pd.crosstab(lab6, lab5).to_csv(TAB / "phase1_k6_vs_k5.csv")
lab3 = KMeans(3, n_init=50, random_state=SEED).fit_predict(Z)
print("k3 x k4\n", pd.crosstab(lab3, lab4)); pd.crosstab(lab3, lab4).to_csv(TAB / "phase1_k3_vs_k4.csv")

km = KMeans(K, n_init=50, random_state=SEED).fit(Z)
b["cluster"] = km.labels_
assert adjusted_rand_score(km.labels_, {4: lab4, 5: lab5, 6: lab6}[K]) > .99
cent = pd.DataFrame(km.cluster_centers_, columns=feat_names)
cent["engagement"] = cent.website_visits_dec_2025 + cent.email_engagement_dec_2025

# ---- deterministic naming from centroids (robust to label permutation)
names = {}
left = list(cent.index)
pick = lambda col, f: (lambda c: (names.__setitem__(c, f), left.remove(c)))(cent.loc[left, col].idxmax() if col[0] != "-" else cent.loc[left, col[1:]].idxmin())
if K >= 5:
    pick("engagement", "Loyal champions")
pick("-log_recency", "Recent actives")
pick("-log_aov", "Low-basket buyers")
hi = cent.loc[left, "log_orders"].idxmax(); names[hi] = "Lapsing regulars"; left.remove(hi)
names[left[0]] = "Occasional buyers"
b["segment"] = b.cluster.map(names)
silsum["segment"] = np.where(silsum.k == K, silsum.cluster.map(names), "")
silsum.to_csv(TAB / "phase1_silhouette_by_cluster.csv", index=False)
b["silhouette"] = silhouette_samples(Z, km.labels_)

ORDER = [n_ for n_ in ["Loyal champions", "Recent actives", "Lapsing regulars", "Low-basket buyers", "Occasional buyers"] if n_ in names.values()] + ["Registered, never purchased"]
SEG_COL = dict(zip(ORDER, SERIES[:len(ORDER) - 1] + ["#898781"]))

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, k in zip(axs, [K - 1, K, K + 1]):
    lab, sv = sil_runs[k]; y0 = 0
    order_c = sorted(range(k), key=lambda c: -np.mean(sv[lab == c]))
    for i_, c in enumerate(order_c):
        v = np.sort(sv[lab == c])
        col = SEG_COL[names[c]] if k == K else SERIES[i_]
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=col, lw=0)
        tag = names[c] if k == K else f"cluster {c}"
        ax.text(.42, y0 + len(v) / 2, f"{tag}\nmean {v.mean():.2f}, {(v<0).mean():.0%} < 0", va="center", fontsize=7, color=INK2)
        y0 += len(v) + 60
    ax.axvline(sv.mean(), color=INK2, ls="--", lw=1); ax.axvline(0, color=AXIS_C, lw=.8)
    ax.set_xlim(-.15, .75)
    ax.set_title(f"k = {k}{' (chosen)' if k == K else ''}: mean {sv.mean():.3f}, {(sv<0).mean():.1%} negative", fontsize=9)
    ax.set_yticks([]); ax.set_xlabel("silhouette value per buyer"); ax.grid(axis="y", visible=False)
save(fig, "p1_01b_silhouette_profiles")

cu = cu.merge(b[["customer_id", "cluster", "segment"]], on="customer_id", how="left")
cu["segment"] = cu.segment.fillna("Registered, never purchased")
cu["segment"] = pd.Categorical(cu.segment, ORDER, ordered=True)

# ------------------------------------------------------------------ 2026 outcomes (not used for clustering)
att = load("orders").dropna(subset=["attributed_exposure_id"])[["attributed_exposure_id", "order_total_inr"]]
e = ex.merge(att, left_on="exposure_id", right_on="attributed_exposure_id", how="left").merge(
    cu[["customer_id", "segment", "preferred_category", "marketing_consent"]], on="customer_id").merge(
    cc[["campaign_id", "featured_category"]], on="campaign_id")
e["order_total_inr"] = e.order_total_inr.fillna(0)
out26 = e.groupby("segment", observed=True).agg(exposures=("exposure_id", "size"), conv_rate=("purchased_within_14_days", "mean"),
                                               rev_per_exposure=("order_total_inr", "mean"), cost_per_exposure=("contact_cost_inr", "mean"),
                                               attributed_revenue=("order_total_inr", "sum"))
out26["exposure_share"] = out26.exposures / out26.exposures.sum()
out26["revenue_share_2026"] = out26.attributed_revenue / out26.attributed_revenue.sum()
match = e.assign(m=e.preferred_category == e.featured_category).groupby(["segment", "m"], observed=True).purchased_within_14_days.mean().unstack()
out26["conv_cat_match"] = match[True]; out26["conv_no_match"] = match[False]

# ------------------------------------------------------------------ profile table
g = cu.groupby("segment", observed=True)
prof = pd.DataFrame({
    "customers": g.size(),
    "share_customers": g.size() / len(cu),
    "share_pre2026_spend": g.spend_before_2026_inr.sum() / cu.spend_before_2026_inr.sum(),
    "median_recency_days": g.days_since_last_order_at_2025_12_31.median(),
    "mean_orders": g.orders_before_2026.mean(),
    "median_aov": g.aov_before_2026.median(),
    "mean_spend": g.spend_before_2026_inr.mean(),
    "mean_visits_dec": g.website_visits_dec_2025.mean(),
    "mean_email_eng_dec": g.email_engagement_dec_2025.mean(),
    "mean_RFM_total": g.RFM_total.mean(),
    "median_tenure_days": g.tenure_days.median(),
    "mean_age": g.age.mean(),
    "female_share": g.gender.apply(lambda s: (s == "Female").mean()),
    "consent_rate": g.marketing_consent.mean(),
    "mean_silhouette": g.customer_id.apply(lambda s_: b.set_index("customer_id").silhouette.reindex(s_).mean()),
    "survey_respondents": g.customer_id.apply(lambda s: s.isin(load("brand_perception_survey").customer_id).sum()),
}).join(out26[["exposures", "conv_rate", "rev_per_exposure", "cost_per_exposure", "revenue_share_2026", "conv_cat_match", "conv_no_match"]])
prof.round(4).to_csv(TAB / "phase1_segment_profile.csv")
pd.set_option("display.width", 250, "display.max_columns", 40)
print(prof.round(3).T)

# ------------------------------------------------------------------ demographics vs segment: chi-square / Kruskal
tests = []
for c in ["gender", "region", "city", "annual_income_band_inr", "preferred_category"]:
    ct = pd.crosstab(cu.segment, cu[c])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    tests.append(dict(variable=c, test="chi-square", stat=chi2, dof=dof, p_value=p, effect=f"Cramer's V {v:.3f}"))
    (pd.crosstab(cu.segment, cu[c], normalize="index") / cu[c].value_counts(normalize=True)).round(3).to_csv(TAB / f"phase1_index_{c}.csv")
h, p = kruskal(*[gg.age for _, gg in cu.groupby("segment", observed=True)])
tests.append(dict(variable="age", test="Kruskal-Wallis", stat=h, dof=len(ORDER) - 1, p_value=p, effect=""))
ct = pd.crosstab(cu.segment, cu.marketing_consent); chi2, p, dof, _ = chi2_contingency(ct)
tests.append(dict(variable="marketing_consent", test="chi-square", stat=chi2, dof=dof, p_value=p, effect=f"Cramer's V {np.sqrt(chi2/len(cu)):.3f}"))
ct = pd.crosstab(e.segment, e.purchased_within_14_days); chi2, p, dof, _ = chi2_contingency(ct)
tests.append(dict(variable="2026 purchase (exposure level)", test="chi-square", stat=chi2, dof=dof, p_value=p, effect=f"Cramer's V {np.sqrt(chi2/len(e)):.3f}"))
tests = pd.DataFrame(tests); tests.to_csv(TAB / "phase1_segment_tests.csv", index=False)
print(tests.round(4).to_string(index=False))

# ------------------------------------------------------------------ outputs for later phases
keep = ["customer_id", "segment", "cluster", "R", "F", "M", "E", "RFM", "RFM_total", "aov_before_2026", "tenure_days"]
cu[keep].to_csv(TAB / "customer_segments.csv", index=False)
cent_named = cent.assign(segment=cent.index.map(names)).set_index("segment")[feat_names]
cent_named.round(3).to_csv(TAB / "phase1_centroids_z.csv")

# ------------------------------------------------------------------ figures
segs = [s for s in ORDER if s in prof.index]
# 1. size vs value
fig, ax = plt.subplots(figsize=(9, 3.6))
y = np.arange(len(segs)); h_ = .38
ax.barh(y - h_ / 2, prof.loc[segs, "share_customers"] * 100, height=h_, color=SERIES[0], label="% of customers")
ax.barh(y + h_ / 2, prof.loc[segs, "share_pre2026_spend"] * 100, height=h_, color=SERIES[1], label="% of pre-2026 spend")
for i, s in enumerate(segs):
    ax.text(prof.loc[s, "share_customers"] * 100, i - h_ / 2, f" {prof.loc[s,'share_customers']:.0%}", va="center", fontsize=7.5, color=INK2)
    ax.text(prof.loc[s, "share_pre2026_spend"] * 100, i + h_ / 2, f" {prof.loc[s,'share_pre2026_spend']:.0%}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y, [f"{s} (n={prof.loc[s,'customers']:,})" for s in segs]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.legend(loc="lower right"); ax.set_xlabel("%"); ax.set_title("Segment size vs share of historical spend")
save(fig, "p1_02_segment_size_value")

# 2. snake plot of standardised centroid features
fig, ax = plt.subplots(figsize=(9, 3.6))
lab = ["Recency (log days)", "Orders (log)", "AOV (log)", "Web visits", "Email engagement"]
for s in segs[:-1]:
    ax.plot(lab, cent_named.loc[s], marker="o", ms=7, color=SEG_COL[s], label=s)
ax.axhline(0, color=MUTED, lw=1); ax.set_ylabel("z-score (buyer mean = 0)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
ax.set_title("What defines each segment (cluster centroids, standardised)"); ax.grid(axis="x", visible=False)
save(fig, "p1_03_snake_plot")

# 3. PCA scatter of buyers
pcs = PCA(2, random_state=SEED).fit(Z); P = pcs.transform(Z)
fig, axs = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw=dict(width_ratios=[1, 1, 1]))
ax = axs[0]
for s in segs[:-1]:
    m = (b.segment == s).values
    ax.scatter(P[m, 0], P[m, 1], s=4, alpha=.45, color=SEG_COL[s], label=s)
ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
ax.set_title(f"Buyers in PCA space (k-means, k = {K})", fontsize=9); ax.legend(markerscale=3, fontsize=7)
# 4. 2026 conversion by segment
ax = axs[1]
cv = prof.loc[segs, "conv_rate"] * 100
ax.barh(segs, cv, color=[SEG_COL[s] for s in segs], height=.6); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
for i, (s, v) in enumerate(cv.items()): ax.text(v, i, f" {v:.1f}%  (n={prof.loc[s,'exposures']:,})", va="center", fontsize=7.5, color=INK2)
ax.set_xlim(0, cv.max() * 1.45); ax.set_title("2026 14-day purchase rate per exposure", fontsize=9); ax.tick_params(axis="y", labelsize=8)
# 5. exposure share vs revenue share 2026
ax = axs[2]
x = np.arange(len(segs))
ax.bar(x - .2, prof.loc[segs, "exposures"] / prof.exposures.sum() * 100, width=.4, color=SERIES[0], label="% of 2026 exposures")
ax.bar(x + .2, prof.loc[segs, "revenue_share_2026"] * 100, width=.4, color=SERIES[1], label="% of attributed revenue")
ax.set_xticks(x, segs, fontsize=7, rotation=35, ha="right"); ax.legend(fontsize=7); ax.grid(axis="x", visible=False)
ax.set_title("Where 2026 contacts went vs where revenue came from", fontsize=9)
save(fig, "p1_04_pca_and_2026_validation")

# 6. demographic index heatmap (segment share / base share x 100)
blocks = []
for c, order in [("gender", None), ("region", None), ("annual_income_band_inr", INC), ("preferred_category", ["Skincare", "Haircare", "Bodycare", "Bundles"])]:
    idx = pd.crosstab(cu.segment, cu[c], normalize="index") / cu[c].value_counts(normalize=True) * 100
    if order: idx = idx[order]
    blocks.append(idx)
ix = pd.concat(blocks, axis=1).loc[segs]
fig, ax = plt.subplots(figsize=(12, 3.6))
im = ax.imshow(ix, cmap=DIV_CMAP, vmin=70, vmax=130, aspect="auto")
ax.set_xticks(range(ix.shape[1]), ix.columns, rotation=40, ha="right", fontsize=8); ax.set_yticks(range(len(segs)), segs); ax.grid(False)
for i in range(ix.shape[0]):
    for j in range(ix.shape[1]):
        ax.text(j, i, f"{ix.iloc[i,j]:.0f}", ha="center", va="center", fontsize=7, color=INK2)
fig.colorbar(im, ax=ax, shrink=.8, label="index (100 = base)")
ax.set_title("Demographic index by segment (100 = base). All differences n.s. or Cramér's V < 0.06; 'Non-disclosed' n = 178 only")
save(fig, "p1_05_demographic_index")

# 7. R x F grid: share of customers and 2026 conversion (classic RFM view)
eb = e.merge(cu[["customer_id", "R", "F"]], on="customer_id").dropna(subset=["R"])
grid_c = eb.pivot_table(index="R", columns="F", values="purchased_within_14_days", aggfunc="mean") * 100
grid_n = cu.dropna(subset=["R"]).pivot_table(index="R", columns="F", values="customer_id", aggfunc="count")
grid_c = grid_c.where(grid_n.reindex_like(grid_c) >= 50)  # suppress rates from cells with < 50 buyers
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, gr, t, fmt in [(axs[0], grid_n, "Buyers per R x F cell", "{:.0f}"), (axs[1], grid_c, "2026 purchase rate % (blank: < 50 buyers)", "{:.1f}")]:
    im = ax.imshow(gr.sort_index(ascending=False), cmap=SEQ_CMAP, aspect="auto")
    gs = gr.sort_index(ascending=False); gv = gs.values
    ax.set_xticks(range(gs.shape[1]), [f"F{int(i)}\n{lbl}" for i, lbl in zip(gs.columns, ["1 order", "2", "3", "4-5", "6+"])]); ax.set_yticks(range(gs.shape[0]), [f"R{int(i)}" for i in gs.index]); ax.grid(False)
    for i in range(gv.shape[0]):
        for j in range(gv.shape[1]):
            if not np.isnan(gv[i, j]):
                ax.text(j, i, fmt.format(gv[i, j]), ha="center", va="center", fontsize=8, color="white" if gv[i, j] > np.nanpercentile(gv, 60) else "#0b0b0b")
    ax.set_title(t, fontsize=9); fig.colorbar(im, ax=ax, shrink=.8)
save(fig, "p1_06_rfm_grid")
grid_c.round(2).to_csv(TAB / "phase1_rfm_grid_conversion.csv")

# ------------------------------------------------------------------ robustness of the chosen solution
from sklearn.mixture import GaussianMixture
rob = []
heap = (b.days_since_last_order_at_2025_12_31 == 2).values
rob.append(dict(check="share of 'Recent actives' at the recency = 2 heap", value=round(heap[(b.segment == "Recent actives").values].mean(), 3)))
km_nh = KMeans(K, n_init=50, random_state=SEED).fit(Z[~heap])        # fit without heaped customers ...
rob.append(dict(check="ARI: refit excluding recency=2 heap, heap assigned to nearest centroid", value=round(adjusted_rand_score(km.labels_, km_nh.predict(Z)), 3)))
X2 = X.copy(); X2["log_aov"] = np.log1p(b.spend_before_2026_inr)  # spend instead of AOV
lab2 = KMeans(K, n_init=50, random_state=SEED).fit_predict(StandardScaler().fit_transform(X2))
rob.append(dict(check="ARI: log spend instead of log AOV", value=round(adjusted_rand_score(km.labels_, lab2), 3)))
lab_rfm = KMeans(K, n_init=50, random_state=SEED).fit_predict(Z[:, [0, 1, 2]])
rob.append(dict(check="ARI: RFM-only features (drop engagement)", value=round(adjusted_rand_score(km.labels_, lab_rfm), 3)))
gm = GaussianMixture(K, covariance_type="full", n_init=5, random_state=SEED).fit(Z)
rob.append(dict(check="ARI: Gaussian mixture (full covariance), same k", value=round(adjusted_rand_score(km.labels_, gm.predict(Z)), 3)))
bic = {k: GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(Z).bic(Z) for k in range(2, 10)}
rob.append(dict(check="GMM BIC-optimal k (2-9)", value=min(bic, key=bic.get)))
rob = pd.DataFrame(rob); rob.to_csv(TAB / "phase1_robustness.csv", index=False)
print(rob.to_string(index=False)); print("GMM BIC by k:", {k: round(v) for k, v in bic.items()})
