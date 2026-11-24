"""Phase 6: brand perception (800 respondents x 8 brands x 5 attributes, 1-5 scale).

Outputs: outputs/tables/phase6_*.csv, outputs/figures/p6_*.png
Design : * every respondent rates every brand -> repeated measures. NexaGlow vs competitor comparisons use
           WITHIN-respondent differences (paired), with Holm correction across the 35 comparisons, and standard
           errors clustered by customer (38 customers answered more than once, Phase 0).
         * scale-use bias: respondents differ in how generously they rate (SD of personal means 0.28). Robustness:
           repeat on respondent-centred ratings (rating minus that respondent's mean over all 40 ratings).
         * perceptual map: PCA on the brand x attribute mean matrix (attributes standardised), biplot.
         * link to behaviour: survey customer_ids join to customers / Phase 1 segments / 2026 campaign outcomes.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import load, save, plt, TAB, SERIES, INK2, MUTED, AXIS, SEQ_CMAP, DIV_CMAP

bs, cu, ex = load("brand_perception_survey"), load("customers"), load("campaign_exposures")
seg = pd.read_csv(TAB / "customer_segments.csv")
ATTR = [c for c in bs.columns if c.endswith("_1_to_5")]
AN = [a.replace("perceived_", "").replace("_1_to_5", "") for a in ATTR]
bs = bs.rename(columns=dict(zip(ATTR, AN)))
BR = ["NexaGlow"] + sorted(b for b in bs.brand.unique() if b != "NexaGlow")
res = {}

long = bs.melt(id_vars=["respondent_id", "customer_id", "brand"], value_vars=AN, var_name="attribute", value_name="rating")
long["centred"] = long.rating - long.groupby("respondent_id").rating.transform("mean")

# ------------------------------------------------------------------ 1. means, ranks, and overall brand differences
means = bs.groupby("brand")[AN].mean().reindex(BR)
ses = bs.groupby("brand")[AN].sem().reindex(BR)
ranks = means.rank(ascending=False).astype(int)
means.round(3).to_csv(TAB / "phase6_brand_means.csv"); ranks.to_csv(TAB / "phase6_brand_ranks.csv")
res["means"] = means.round(2); res["nexaglow_rank"] = ranks.loc["NexaGlow"]
res["overall_mean"] = means.mean(axis=1).sort_values(ascending=False).round(3)
fr = []
for a in AN:
    w = bs.pivot(index="respondent_id", columns="brand", values=a)[BR]
    st, p = friedmanchisquare(*[w[b] for b in BR])
    fr.append(dict(attribute=a, friedman_chi2=st, p_value=p, kendalls_W=st / (len(w) * (len(BR) - 1))))
res["friedman"] = pd.DataFrame(fr).round(4)

# ------------------------------------------------------------------ 2. NexaGlow vs each competitor, paired within respondent
rows = []
for kind in ["raw", "centred"]:
    val = "rating" if kind == "raw" else "centred"
    wide = long.pivot_table(index=["respondent_id", "customer_id", "attribute"], columns="brand", values=val).reset_index()
    for a in AN:
        wa = wide[wide.attribute == a]
        for b in BR[1:]:
            d = pd.DataFrame({"diff": wa["NexaGlow"] - wa[b], "cust": wa.customer_id})
            m = smf.ols("diff ~ 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.cust.astype("category").cat.codes})
            rows.append(dict(scale=kind, attribute=a, competitor=b, mean_diff=m.params.Intercept, ci_low=m.conf_int().iloc[0, 0],
                             ci_high=m.conf_int().iloc[0, 1], p_value=m.pvalues.Intercept,
                             share_rate_nexaglow_higher=(d["diff"] > 0).mean(), share_rate_lower=(d["diff"] < 0).mean()))
pw = pd.DataFrame(rows)
pw["p_holm"] = pw.groupby("scale").p_value.transform(lambda p: multipletests(p, method="holm")[1])
pw["verdict"] = np.where(pw.p_holm >= .05, "no difference", np.where(pw.mean_diff > 0, "NexaGlow ahead", "NexaGlow behind"))
pw.round(4).to_csv(TAB / "phase6_nexaglow_vs_competitors.csv", index=False)
res["pairwise_raw"] = pw[pw.scale == "raw"].pivot(index="competitor", columns="attribute", values="mean_diff").round(2)[AN]
res["pairwise_verdicts"] = pd.crosstab(pw.scale, pw.verdict)
res["scale_robustness"] = (pw.pivot_table(index=["attribute", "competitor"], columns="scale", values="verdict", aggfunc="first")
                           .assign(same=lambda d: d.raw == d.centred).same.mean())

# ------------------------------------------------------------------ 3. perceptual map (PCA on brand means)
Zb = StandardScaler().fit_transform(means.values)
pca = PCA(2).fit(Zb)
scores = pd.DataFrame(pca.transform(Zb), index=BR, columns=["PC1", "PC2"])
load_ = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=AN, columns=["PC1", "PC2"])
res["pca_var"] = pd.Series(pca.explained_variance_ratio_, index=["PC1", "PC2"]).round(3)
res["pca_loadings"] = load_.round(3); res["pca_scores"] = scores.round(3)
scores.round(4).to_csv(TAB / "phase6_pca_scores.csv"); load_.round(4).to_csv(TAB / "phase6_pca_loadings.csv")
# nearest competitors in attribute space (Euclidean on standardised means)
dist = pd.Series({b: np.linalg.norm(Zb[0] - Zb[BR.index(b)]) for b in BR[1:]}).sort_values()
res["closest_competitors"] = dist.round(3)
# respondent-level attribute correlation (trade-offs perceived by consumers)
rc = bs[AN].corr(); res["attribute_corr_all_brands"] = rc.round(2)
rc_c = long.pivot_table(index=["respondent_id", "brand"], columns="attribute", values="centred")[AN].corr()
res["attribute_corr_centred"] = rc_c.round(2)

# ------------------------------------------------------------------ 4. who rates NexaGlow how? (segments, demographics)
nx = bs[bs.brand == "NexaGlow"].merge(cu, on="customer_id").merge(seg[["customer_id", "segment"]], on="customer_id")
comp_mean = bs[bs.brand != "NexaGlow"].groupby("respondent_id")[AN].mean().add_suffix("_comp")
nx = nx.merge(comp_mean, left_on="respondent_id", right_index=True)
for a in AN: nx[f"gap_{a}"] = nx[a] - nx[f"{a}_comp"]
nx["nx_overall"] = nx[AN].mean(axis=1); nx["gap_overall"] = nx[[f"gap_{a}" for a in AN]].mean(axis=1)
by_seg = nx.groupby("segment")[AN + ["nx_overall", "gap_overall"]].mean()
by_seg["n"] = nx.groupby("segment").size()
by_seg.round(3).to_csv(TAB / "phase6_nexaglow_by_segment.csv")
res["by_segment"] = by_seg.round(2)
tests = []
for var in ["segment", "gender", "region", "annual_income_band_inr", "preferred_category"]:
    for y in ["nx_overall", "gap_overall"]:
        m = smf.ols(f"{y} ~ C({var})", data=nx).fit(cov_type="cluster", cov_kwds={"groups": nx.customer_id.astype("category").cat.codes})
        tests.append(dict(outcome=y, variable=var, F_p_value=m.f_pvalue, r2=m.rsquared))
for y in ["nx_overall", "gap_overall"]:
    m = smf.ols(f"{y} ~ age + np.log1p(spend_before_2026_inr) + np.log1p(days_since_last_order_at_2025_12_31) + email_engagement_dec_2025", data=nx).fit(
        cov_type="cluster", cov_kwds={"groups": nx.customer_id.astype("category").cat.codes})
    tests.append(dict(outcome=y, variable="age + spend + recency + email engagement", F_p_value=m.f_pvalue, r2=m.rsquared))
tests = pd.DataFrame(tests); tests.round(4).to_csv(TAB / "phase6_who_rates_tests.csv", index=False)
res["who_rates_tests"] = tests.round(4)

# ------------------------------------------------------------------ 5. does perception predict 2026 purchase? (exposure level)
e = ex.merge(nx[["customer_id"] + AN + [f"gap_{a}" for a in AN] + ["nx_overall", "gap_overall", "days_since_last_order_at_2025_12_31",
                                        "orders_before_2026", "email_engagement_dec_2025"]].drop_duplicates("customer_id"), on="customer_id")
e["log_rec"] = np.log1p(e.days_since_last_order_at_2025_12_31); e["log_ord"] = np.log1p(e.orders_before_2026)
g = e.customer_id.astype("category").cat.codes
pp = []
MODEL_NAMES = ["NexaGlow overall rating", "gap vs competitors, overall", "NexaGlow attribute ratings", "gap vs competitors, per attribute"]
for mi, f in enumerate(["purchased_within_14_days ~ nx_overall + log_rec + log_ord + email_engagement_dec_2025",
          "purchased_within_14_days ~ gap_overall + log_rec + log_ord + email_engagement_dec_2025",
          "purchased_within_14_days ~ " + " + ".join(AN) + " + log_rec + log_ord + email_engagement_dec_2025",
          "purchased_within_14_days ~ " + " + ".join(f"gap_{a}" for a in AN) + " + log_rec + log_ord + email_engagement_dec_2025"]):
    m = smf.logit(f, data=e).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    for v in m.params.index:
        if v in AN + [f"gap_{a}" for a in AN] + ["nx_overall", "gap_overall"]:
            pp.append(dict(model=MODEL_NAMES[mi], term=v, odds_ratio=np.exp(m.params[v]),
                           ci_low=np.exp(m.conf_int().loc[v, 0]), ci_high=np.exp(m.conf_int().loc[v, 1]), p_value=m.pvalues[v]))
pp = pd.DataFrame(pp); pp.round(4).to_csv(TAB / "phase6_perception_predicts_purchase.csv", index=False)
res["perception_predicts_purchase"] = pp.round(3)
res["perception_sample"] = pd.Series({"respondent_customers_with_exposures": e.customer_id.nunique(), "exposures": len(e), "buyers": int(e.purchased_within_14_days.sum())})

# ================================================================== figures
# p6_01 heatmap of means + NexaGlow rank
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
mo = means.loc[means.mean(axis=1).sort_values(ascending=False).index]
im = ax.imshow(mo.values, cmap=SEQ_CMAP, aspect="auto", vmin=2.9, vmax=4.6)
ax.set_xticks(range(5), AN); ax.set_yticks(range(len(mo)), mo.index); ax.grid(False)
for i in range(mo.shape[0]):
    for j in range(5):
        v = mo.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 3.9 else "#0b0b0b",
                fontweight="bold" if mo.index[i] == "NexaGlow" else "normal")
fig.colorbar(im, ax=ax, shrink=.8); ax.set_title("Mean rating (1–5), brands sorted by overall mean", fontsize=9)
ax = axs[1]
pr = pw[pw.scale == "raw"].pivot(index="competitor", columns="attribute", values="mean_diff")[AN]
ph = pw[pw.scale == "raw"].pivot(index="competitor", columns="attribute", values="p_holm")[AN]
im = ax.imshow(pr.values, cmap=DIV_CMAP.reversed(), vmin=-1.2, vmax=1.2, aspect="auto")
ax.set_xticks(range(5), AN); ax.set_yticks(range(len(pr)), [f"vs {b}" for b in pr.index]); ax.grid(False)
for i in range(pr.shape[0]):
    for j in range(5):
        v, p = pr.values[i, j], ph.values[i, j]
        ax.text(j, i, f"{v:+.2f}" + ("" if p < .05 else "\nn.s."), ha="center", va="center", fontsize=7.5, color="white" if abs(v) > .7 else "#0b0b0b")
fig.colorbar(im, ax=ax, shrink=.8, label="NexaGlow minus competitor")
ax.set_title("NexaGlow vs each competitor (paired, Holm-corrected)", fontsize=9)
save(fig, "p6_01_brand_ratings")

# p6_02 perceptual map
fig, ax = plt.subplots(figsize=(7.5, 6))
for b in BR:
    nx_ = b == "NexaGlow"
    ax.scatter(scores.loc[b, "PC1"], scores.loc[b, "PC2"], s=120 if nx_ else 60, color=SERIES[1] if nx_ else SERIES[0], zorder=3, edgecolor="#fcfcfb")
    ax.annotate(b, (scores.loc[b, "PC1"], scores.loc[b, "PC2"]), xytext=(7, 4), textcoords="offset points", fontsize=9 if nx_ else 8,
                fontweight="bold" if nx_ else "normal", color="#0b0b0b" if nx_ else INK2)
sc = np.abs(scores.values).max() / np.abs(load_.values).max() * .85
for a in AN:
    ax.annotate("", xy=(load_.loc[a, "PC1"] * sc, load_.loc[a, "PC2"] * sc), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.text(load_.loc[a, "PC1"] * sc * 1.1, load_.loc[a, "PC2"] * sc * 1.1, a, color=MUTED, fontsize=8.5, ha="center", va="center", style="italic")
ax.axhline(0, color=AXIS, lw=.8); ax.axvline(0, color=AXIS, lw=.8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("Perceptual map: 8 brands on 5 attributes (PCA biplot of brand means)")
save(fig, "p6_02_perceptual_map")

# p6_03 NexaGlow by segment
fig, ax = plt.subplots(figsize=(9, 3.4))
segs = ["Recent actives", "Lapsing regulars", "Low-basket buyers", "Occasional buyers", "Registered, never purchased"]
x = np.arange(len(AN)); w = .16
for i_, s in enumerate(segs):
    ax.bar(x + (i_ - 2) * w, by_seg.loc[s, AN].values - means.drop("NexaGlow").mean()[AN].values, width=w * .92,
           color=SERIES[i_] if i_ < 4 else MUTED, label=f"{s} (n={int(by_seg.loc[s, 'n'])})")
ax.axhline(0, color=INK2, lw=1); ax.set_xticks(x, AN); ax.grid(axis="x", visible=False)
ax.set_ylabel("NexaGlow rating minus\ncompetitor average"); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
ax.set_title("NexaGlow's advantage over the competitor average, by segment (segment differences n.s., p = 0.86)")
save(fig, "p6_03_by_segment")

pd.set_option("display.width", 250, "display.max_columns", 30)
for k, v in res.items():
    print(f"\n### {k}\n{v}")
