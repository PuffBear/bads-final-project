"""Phase 6b: multidimensional scaling (MDS).
  A. Brand perceptual map from the survey (complements the Phase 6 PCA map).
  B. Product map from basket co-occurrence (visual companion to the Phase 3 association rules).

Inputs : brand_perception_survey, order_items, product_catalog, outputs/tables/phase6_pca_scores.csv
Outputs: outputs/tables/phase6b_*.csv, outputs/figures/p6b_*.png, outputs/tables/phase6b_log.txt
Depends: numpy, pandas, scipy, scikit-learn, matplotlib.

Method notes
  * Classical (Torgerson) MDS: double-centre the squared distances, take the top eigenvectors. With Euclidean
    distances on standardised means it reproduces PCA exactly, which is the check that both maps agree.
  * Metric / non-metric MDS: SMACOF (scikit-learn), 50 random starts. Non-metric MDS only keeps the RANK order of
    dissimilarities, the right choice when distances are not on an interval scale (e.g. Jaccard).
  * Fit: Kruskal stress-1 = sqrt( sum (d_ij - dhat_ij)^2 / sum d_ij^2 ).  Kruskal's guide: <0.05 excellent,
    0.05-0.10 good, 0.10-0.20 fair, >0.20 poor.
  * Attribute vectors ("property fitting"): regress each attribute's brand means on the 2 MDS coordinates; the arrow
    points in the direction the attribute increases, its length is the R^2.
  * Stability: bootstrap the 800 respondents (200 resamples), rebuild the map, Procrustes-align it to the original.
"""
import io
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from common import load, save, plt, TAB, SERIES, INK, INK2, MUTED, CATEGORY_COLORS

SEED = 42
log = io.StringIO()
def say(*a):
    print(*a); print(*a, file=log)

def classical_mds(D, k=2):
    n = len(D); J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    return V[:, :k] * np.sqrt(np.maximum(w[:k], 0)), w
def stress1(D, X, metric=True):
    d = pdist(X); delta = squareform(D, checks=False)
    if metric == "interval":  # allow scale + additive constant: delta ~ a + b*d
        A = np.c_[np.ones_like(d), d]; dhat = A @ np.linalg.lstsq(A, delta, rcond=None)[0]
        return np.sqrt(((delta - dhat) ** 2).sum() / ((delta - delta.mean()) ** 2).sum())
    if metric:  # ratio: allow a scale factor only
        b = (d @ delta) / (d @ d); return np.sqrt(((delta - b * d) ** 2).sum() / (delta ** 2).sum())
    # non-metric: monotone (isotonic) regression of map distances on dissimilarity ranks
    from sklearn.isotonic import IsotonicRegression
    dhat = IsotonicRegression().fit_transform(delta, d)
    return np.sqrt(((d - dhat) ** 2).sum() / (d ** 2).sum())
def smacof(D, k=2, metric=True):
    m = MDS(n_components=k, metric_mds=metric, init="random", dissimilarity="precomputed", n_init=50, max_iter=1000,
            random_state=SEED, normalized_stress="auto")
    return m.fit_transform(D)

# ================================================================== A. BRANDS
sv = load("brand_perception_survey")
ATTR = ["quality", "affordability", "innovation", "naturalness", "trendiness"]
sv = sv.rename(columns={f"perceived_{a}_1_to_5": a for a in ATTR})
means = sv.groupby("brand")[ATTR].mean()
brands = list(means.index)
Z = (means - means.mean()) / means.std(ddof=0)

# A1. aggregate dissimilarity: Euclidean distance between standardised mean profiles
D_agg = squareform(pdist(Z.values))
X_cl, eig = classical_mds(D_agg)
pos = eig[eig > 1e-9]
say(f"Classical MDS on brand means: dims 1-2 explain {pos[:2].sum() / pos.sum():.1%} of the squared-distance structure "
    f"(dim 1 {pos[0] / pos.sum():.1%}, dim 2 {pos[1] / pos.sum():.1%})")
pca = pd.read_csv(TAB / "phase6_pca_scores.csv", index_col=0).reindex(brands)
_, _, disp = procrustes(pca.values, X_cl)
say(f"Procrustes disparity, classical MDS vs Phase 6 PCA map: {disp:.6f} (0 = identical) -> the two maps are the same picture")

# A2. respondent-level dissimilarity: for each pair of brands, average over respondents of the distance between that
#     respondent's two rating vectors. Uses individual perceptions, not just the means.
R = sv.set_index(["respondent_id", "brand"])[ATTR].unstack("brand")      # respondents x (attr, brand)
arr = np.stack([R.xs(b, axis=1, level="brand")[ATTR].values for b in brands], axis=1)  # resp x brand x attr
def resp_dissim(a):
    diff = a[:, :, None, :] - a[:, None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).mean(0)
D_resp = resp_dissim(arr)
pd.DataFrame(D_resp, brands, brands).round(3).to_csv(TAB / "phase6b_brand_dissimilarity_respondent.csv")
stress_tab = []
for k in [1, 2, 3, 4]:
    stress_tab.append(dict(dims=k, classical_stress1=stress1(D_agg, classical_mds(D_agg, k)[0], True),
                           smacof_metric_stress1=stress1(D_agg, smacof(D_agg, k, True), True),
                           nonmetric_stress1=stress1(D_agg, smacof(D_agg, k, False), False)))
stress_tab = pd.DataFrame(stress_tab)
stress_tab.round(4).to_csv(TAB / "phase6b_brand_stress_by_dims.csv", index=False)
say("\nBrand MDS (standardised brand means), Kruskal stress-1 by number of dimensions:\n" + stress_tab.round(4).to_string(index=False))
say("Non-metric stress near 0 is not a better fit: with 8 brands (28 distances) a rank-only fit is nearly always perfect.")
# robustness: respondent-level map (distances averaged over respondents) vs the means map
X_r, _ = classical_mds(D_resp, 2)
_, _, disp2 = procrustes(X_cl, X_r)
say(f"Procrustes disparity, respondent-level map vs brand-means map: {disp2:.4f} (same configuration). Respondent-level distances "
    f"sit on a noise floor (all pairs {squareform(D_resp, checks=False).min():.2f}-{squareform(D_resp, checks=False).max():.2f}), "
    "which is why they are a robustness check, not the main map.")
_, X_m, _ = procrustes(pca.values, X_cl)          # orient like the Phase 6 PCA map
X_m = X_m * (np.abs(pca.values).max() / np.abs(X_m).max())

# property fitting
vec = {}
for a in ATTR:
    y = means[a].values; A = np.c_[np.ones(len(y)), X_m]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None); yhat = A @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    vec[a] = dict(dir_x=beta[1], dir_y=beta[2], r2=r2)
vec = pd.DataFrame(vec).T
vec.round(4).to_csv(TAB / "phase6b_brand_attribute_vectors.csv")
say("\nAttribute vectors fitted into the MDS map (R^2 = how well the map explains each attribute):\n" + vec.round(3).to_string())

coords = pd.DataFrame(X_m, brands, ["dim1", "dim2"])
dist_nx = pd.Series(squareform(pdist(X_m))[brands.index("NexaGlow")], brands).drop("NexaGlow").sort_values()
coords.round(4).to_csv(TAB / "phase6b_brand_mds_coords.csv")
say("\nNexaGlow's distance to each brand on the MDS map:\n" + dist_nx.round(3).to_string())
say("Raw respondent-level dissimilarity from NexaGlow:\n" + pd.Series(D_resp[brands.index("NexaGlow")], brands).drop("NexaGlow").sort_values().round(3).to_string())

# A3. bootstrap stability: resample respondents, recompute brand means, redo the map, Procrustes-align
rng = np.random.default_rng(SEED)
mu, sdv = means.mean(), means.std(ddof=0)
boot = []
for b in range(200):
    s_ = rng.integers(0, arr.shape[0], arr.shape[0])
    mb = pd.DataFrame(arr[s_].mean(0), brands, ATTR); Xb, _ = classical_mds(squareform(pdist(((mb - mu) / sdv).values)))
    _, Xb_al, _ = procrustes(X_m, Xb)
    boot.append(Xb_al * (np.abs(X_m).max() / np.abs(Xb_al).max()))
boot = np.array(boot)  # B x brand x 2
spread = pd.DataFrame({"sd_dim1": boot[:, :, 0].std(0), "sd_dim2": boot[:, :, 1].std(0)}, brands)
# how often is UrbanLeaf NexaGlow's nearest neighbour?
nn = [brands[np.argsort(squareform(pdist(bb))[brands.index("NexaGlow")])[1]] for bb in boot]
nn_share = pd.Series(nn).value_counts(normalize=True)
spread.round(4).to_csv(TAB / "phase6b_brand_bootstrap_spread.csv")
say(f"\nBootstrap (200 resamples of respondents): mean position SD {spread.values.mean():.3f} map units "
    f"(map spans ~{np.ptp(X_m[:, 0]):.1f}). NexaGlow's nearest neighbour: " + ", ".join(f"{k} {v:.0%}" for k, v in nn_share.items()))

# ================================================================== B. PRODUCTS (basket co-occurrence)
oi, pc = load("order_items"), load("product_catalog")
basket = pd.crosstab(oi.order_id, oi.product_id).clip(upper=1)
prods = list(basket.columns)
D_jac = squareform(pdist(basket.T.values.astype(bool), metric="jaccard"))
X_p = smacof(D_jac, 2, metric=False)
s_p = stress1(D_jac, X_p, metric=False)
p_info = pc.set_index("product_id").reindex(prods)
cats = np.asarray(p_info.category, dtype=object)
same = cats[:, None] == cats[None, :]; off = ~np.eye(len(prods), dtype=bool)
within, across = D_jac[same & off].mean(), D_jac[~same].mean()
# silhouette of categories on the MDS map
from sklearn.metrics import silhouette_score
sil = silhouette_score(X_p, cats)
say(f"\nProduct MDS (non-metric, Jaccard distance over {len(basket):,} orders): stress-1 {s_p:.3f}. "
    f"Mean Jaccard distance within category {within:.3f} vs across {across:.3f}; category silhouette on the map {sil:.2f}")
pd.DataFrame(X_p, prods, ["dim1", "dim2"]).assign(product=p_info.product_name.values, category=cats).round(4).to_csv(TAB / "phase6b_product_mds_coords.csv")
pd.Series({"stress1_nonmetric": s_p, "mean_jaccard_within_category": within, "mean_jaccard_across_category": across,
           "category_silhouette_on_map": sil}).round(4).to_csv(TAB / "phase6b_product_mds_fit.csv")

# ================================================================== figures
BCOL = {b: (SERIES[1] if b == "NexaGlow" else INK2) for b in brands}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
for i, b in enumerate(brands):
    ax[0].scatter(boot[:, i, 0], boot[:, i, 1], s=4, alpha=0.12, color=BCOL[b])
    ax[0].scatter(*X_m[i], s=90 if b == "NexaGlow" else 50, color=BCOL[b], zorder=3)
    ax[0].annotate(b, X_m[i], xytext=(6, 5), textcoords="offset points", fontsize=9,
                   fontweight="bold" if b == "NexaGlow" else "normal", color=BCOL[b])
scale = 0.85 * np.abs(X_m).max()
for a, r in vec.iterrows():
    v = np.array([r.dir_x, r.dir_y]); v = v / np.linalg.norm(v) * scale * r.r2
    ax[0].annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=1.4))
    ax[0].text(*(v * 1.1), f"{a}\n(R² {r.r2:.2f})", color=SERIES[0], fontsize=8, ha="center")
ax[0].axhline(0, color=MUTED, lw=0.6); ax[0].axvline(0, color=MUTED, lw=0.6); ax[0].set_aspect("equal", "datalim")
ax[0].set_title(f"Brand MDS (stress-1 {stress_tab.classical_stress1[1]:.3f}); clouds = 200 bootstrap maps")
ax[0].set_xlabel("dimension 1"); ax[0].set_ylabel("dimension 2")
ax[1].plot(stress_tab.dims, stress_tab.classical_stress1, "o-", label="classical (Torgerson) MDS", color=SERIES[0])
ax[1].plot(stress_tab.dims, stress_tab.smacof_metric_stress1, "o-", label="metric MDS (SMACOF)", color=SERIES[2])
for y, t in [(0.05, "excellent"), (0.10, "good"), (0.20, "fair")]:
    ax[1].axhline(y, color=MUTED, ls=":", lw=1); ax[1].text(4.05, y, t, color=MUTED, fontsize=8, va="center")
ax[1].set_xticks([1, 2, 3, 4]); ax[1].set_xlabel("number of dimensions"); ax[1].set_ylabel("Kruskal stress-1"); ax[1].legend()
ax[1].set_title("Scree: 2 dimensions are enough")
save(fig, "p6b_01_brand_mds")

fig, ax = plt.subplots(figsize=(8, 6.2))
for c in ["Skincare", "Haircare", "Bodycare", "Bundles"]:
    m = cats == c; ax.scatter(X_p[m, 0], X_p[m, 1], s=70, color=CATEGORY_COLORS[c], label=c, zorder=3)
for c in ["Skincare", "Haircare", "Bodycare", "Bundles"]:   # stagger labels inside each tight cluster
    ids = [i for i in range(len(prods)) if cats[i] == c]; ids.sort(key=lambda i: X_p[i, 1])
    for r, i in enumerate(ids):
        side = 1 if r % 2 == 0 else -1
        ax.annotate(p_info.product_name.iloc[i], X_p[i], xytext=(side * 14, (r - len(ids) / 2) * 9), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha="left" if side > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
ax.legend(); ax.set_aspect("equal", "datalim"); ax.set_xlabel("dimension 1"); ax.set_ylabel("dimension 2")
ax.set_title(f"Products placed by how often they share a basket (non-metric MDS, Jaccard; stress-1 {s_p:.2f})")
save(fig, "p6b_02_product_mds")
(TAB / "phase6b_log.txt").write_text(log.getvalue())
