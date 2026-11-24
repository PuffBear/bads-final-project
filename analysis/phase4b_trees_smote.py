"""Phase 4b: class methods applied to the campaign-response problem.
Decision trees and Gini impurity, SMOTE (KNN oversampling) + logistic regression, confusion matrices.

Inputs : campaign_exposures, campaign_catalog, customers, outputs/tables/customer_segments.csv (Phase 1)
Outputs: outputs/tables/phase4b_*.csv, outputs/figures/p4b_*.png, outputs/tables/phase4b_log.txt
Depends: numpy, pandas, scikit-learn, matplotlib only (no statsmodels / imblearn needed).

Three different "Gini" ideas appear here; they are NOT the same thing:
  1. Gini IMPURITY of a node          G = 1 - sum_k p_k^2  (0 = pure, 0.5 = worst for two classes).
                                       Used by decision trees to choose splits.
  2. Gini COEFFICIENT of a model       Gini = 2*AUC - 1     (0 = random ranking, 1 = perfect).
                                       Standard in credit scoring / response modelling.
  3. Gini COEFFICIENT of concentration (Lorenz curve)        (0 = everyone equal, 1 = one customer has it all).
                                       Used here for how concentrated customer value is.

Design (same as Phase 4 so results are comparable):
  * Unit = exposure (21,000); outcome = purchased_within_14_days (5.57%).
  * Pre-contact features only. delivered / opened / clicked are post-contact and leak the answer.
  * Train = campaigns C001-C010, test = later campaigns C011-C014 (time-based hold-out).
  * Hyper-parameters chosen by 5-fold CV GROUPED BY CUSTOMER inside the training campaigns only.
  * SMOTE is applied INSIDE each training fold only. Applying it before splitting leaks synthetic copies
    of test-fold buyers into training; we show how much that inflates the CV score.
"""
import io
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score, roc_curve)
from common import load, save, plt, TAB, SERIES, INK, INK2, MUTED, SEQ_CMAP

SEED = 42
rng = np.random.default_rng(SEED)
log = io.StringIO()
def say(*a):
    print(*a); print(*a, file=log)

# ------------------------------------------------------------------ data (identical feature build to Phase 4)
ex, cc, cu = (load(n) for n in ["campaign_exposures", "campaign_catalog", "customers"])
seg = pd.read_csv(TAB / "customer_segments.csv")
SEGS = ["Recent actives", "Lapsing regulars", "Low-basket buyers", "Occasional buyers", "Registered, never purchased"]
e = (ex.merge(cc.drop(columns="channel"), on="campaign_id").merge(cu, on="customer_id")
       .merge(seg[["customer_id", "segment", "R", "F", "M", "E"]], on="customer_id"))
e["y"] = e.purchased_within_14_days
e = e.sort_values(["customer_id", "exposure_date", "exposure_id"]).reset_index(drop=True)
e["prior_contacts"] = e.groupby("customer_id").cumcount()
e["prior_contacts_30d"] = e.groupby("customer_id").exposure_date.transform(
    lambda s: [((s < d) & (s >= d - pd.Timedelta(days=30))).sum() for d in s])
e["cat_match"] = (e.preferred_category == e.featured_category).astype(int)
e["buyer_before"] = (e.orders_before_2026 > 0).astype(int)
e["log_recency"] = np.log1p(e.days_since_last_order_at_2025_12_31)
e["log_orders"] = np.log1p(e.orders_before_2026)
e["log_spend"] = np.log1p(e.spend_before_2026_inr)
e["tenure_days"] = (pd.Timestamp("2025-12-31") - e.signup_date).dt.days
NUM = ["log_recency", "log_orders", "log_spend", "buyer_before", "website_visits_dec_2025", "email_engagement_dec_2025",
       "marketing_consent", "age", "tenure_days", "discount_pct", "cat_match", "prior_contacts", "prior_contacts_30d"]
CAT = ["gender", "region", "annual_income_band_inr", "preferred_category", "channel", "featured_category"]
train, test = e[e.campaign_id <= "C010"].copy(), e[e.campaign_id > "C010"].copy()
Xtr, ytr, gtr = train[NUM + CAT], train.y.values, train.customer_id.values
Xte, yte = test[NUM + CAT], test.y.values
say(f"Exposures {len(e):,}; purchase rate {e.y.mean():.4f}. Train {len(train):,} ({ytr.mean():.4f}), test {len(test):,} ({yte.mean():.4f})")

# ================================================================== 1. GINI IMPURITY: how mixed is the outcome, and which single split purifies it most?
def gini_imp(y):
    p = np.mean(y); return 1 - p**2 - (1 - p)**2
def split_gini(df, col):
    """Weighted Gini impurity after splitting on every level of `col`, and the decrease from the root."""
    root = gini_imp(df.y)
    g = df.groupby(col, observed=True).y.agg(["size", "mean"])
    g["gini"] = 1 - g["mean"]**2 - (1 - g["mean"])**2
    w = (g["size"] / g["size"].sum() * g["gini"]).sum()
    return root, w, g
root = gini_imp(e.y)
say(f"\nRoot-node Gini impurity = 1 - p^2 - (1-p)^2 with p = {e.y.mean():.4f}: {root:.4f}  (max for 2 classes = 0.5)")
say(f"A 'nobody buys' rule is {1 - e.y.mean():.1%} accurate: accuracy is useless here (accuracy paradox).")

e["recency_band"] = pd.cut(e.days_since_last_order_at_2025_12_31, [-1, 30, 90, 180, 365, 1e9],
                           labels=["0-30d", "31-90d", "91-180d", "181-365d", "365d+"])
e["orders_band"] = pd.cut(e.orders_before_2026, [-1, 0, 2, 5, 1e9], labels=["0", "1-2", "3-5", "6+"])
e["segment"] = pd.Categorical(e.segment, SEGS)
e["R"] = e.R.fillna(0)  # never-buyers have no recency score: their own group 0
rows, seg_detail = [], None
for col, label in [("segment", "K-means segment (Phase 1, 5 groups)"), ("recency_band", "Recency band (5)"),
                   ("orders_band", "Order-count band (4)"), ("R", "RFM recency quintile R"), ("cat_match", "Category match (2)"),
                   ("channel", "Channel (6)"), ("campaign_id", "Campaign (14)"), ("discount_pct", "Discount % (8 levels)"),
                   ("region", "Region"), ("annual_income_band_inr", "Income band"), ("gender", "Gender"), ("marketing_consent", "Consent")]:
    r0, w, g = split_gini(e, col)
    rows.append(dict(split=label, n_groups=len(g), weighted_gini_after=w, gini_decrease=r0 - w,
                     pct_impurity_removed=(r0 - w) / r0, highest_rate_group=f"{g['mean'].idxmax()} ({g['mean'].max():.1%} buy)", lowest_rate_group=f"{g['mean'].idxmin()} ({g['mean'].min():.1%} buy)"))
    if col == "segment": seg_detail = g.rename(columns={"size": "exposures", "mean": "purchase_rate"})
gsplit = pd.DataFrame(rows).sort_values("gini_decrease", ascending=False)
gsplit.round(6).to_csv(TAB / "phase4b_gini_single_splits.csv", index=False)
seg_detail.round(4).to_csv(TAB / "phase4b_gini_by_segment.csv")
say("\nGini impurity per K-means segment:\n" + seg_detail.round(4).to_string())
say("\nSingle-variable splits ranked by Gini decrease:\n" + gsplit.round(5).to_string(index=False))

# ================================================================== 2. DECISION TREE (criterion = gini), tuned by grouped CV
pre_tree = ColumnTransformer([("num", "passthrough", NUM), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
def cv_auc(model, X, y, groups, k=5, transform=None):
    out = []
    for tr, va in GroupKFold(k).split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y[tr])
        out.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
    return np.mean(out), np.std(out, ddof=1)
grid = []
for depth in [2, 3, 4, 5, 6, 8, None]:
    for leaf in [50, 100, 200, 400]:
        for cw in [None, "balanced"]:
            t = make_pipeline(pre_tree, DecisionTreeClassifier(criterion="gini", max_depth=depth, min_samples_leaf=leaf,
                                                               class_weight=cw, random_state=SEED))
            m, s = cv_auc(t, Xtr, ytr, gtr)
            grid.append(dict(max_depth=depth if depth else "none", min_samples_leaf=leaf, class_weight=cw or "none", cv_auc=m, cv_sd=s))
grid = pd.DataFrame(grid).sort_values("cv_auc", ascending=False)
grid.round(4).to_csv(TAB / "phase4b_tree_cv_grid.csv", index=False)
best = grid.iloc[0]
# one-standard-error rule: the simplest tree whose CV AUC is within 1 SE of the best
se = best.cv_sd / np.sqrt(5)
cand = grid[grid.cv_auc >= best.cv_auc - se].copy()
cand["d"] = cand.max_depth.replace("none", 99).astype(int)
pick = cand.sort_values(["d", "min_samples_leaf"], ascending=[True, False]).iloc[0]
say(f"\nTree CV grid ({len(grid)} settings). Best: depth {best.max_depth}, leaf {best.min_samples_leaf}, weights {best.class_weight}, "
    f"CV AUC {best.cv_auc:.4f} (sd {best.cv_sd:.4f}).\n1-SE rule picks: depth {pick.max_depth}, leaf {pick.min_samples_leaf}, "
    f"weights {pick.class_weight}, CV AUC {pick.cv_auc:.4f}")
tree_pipe = make_pipeline(pre_tree, DecisionTreeClassifier(criterion="gini", max_depth=int(pick.d) if pick.d < 99 else None,
                                                           min_samples_leaf=int(pick.min_samples_leaf),
                                                           class_weight=None if pick.class_weight == "none" else pick.class_weight,
                                                           random_state=SEED)).fit(Xtr, ytr)
tree = tree_pipe[-1]
fnames = NUM + list(tree_pipe[0].named_transformers_["cat"].get_feature_names_out(CAT))
# unpruned tree for the overfitting contrast
deep = make_pipeline(pre_tree, DecisionTreeClassifier(criterion="gini", min_samples_leaf=1, random_state=SEED)).fit(Xtr, ytr)
say(f"Unpruned tree: depth {deep[-1].get_depth()}, {deep[-1].get_n_leaves()} leaves, train AUC "
    f"{roc_auc_score(ytr, deep.predict_proba(Xtr)[:, 1]):.3f} vs test AUC {roc_auc_score(yte, deep.predict_proba(Xte)[:, 1]):.3f}  <- overfitting")

# node table: Gini impurity at every node, impurity decrease of every split
T = tree.tree_
nodes = []
for i in range(T.node_count):
    vals = T.value[i][0]; p1 = vals[1] / vals.sum()
    leaf = T.children_left[i] == -1
    row = dict(node=i, depth=None, is_leaf=leaf, samples=int(T.n_node_samples[i]), share_buyers=p1, gini=T.impurity[i],
               split=None if leaf else f"{fnames[T.feature[i]]} <= {T.threshold[i]:.3f}")
    if not leaf:
        l, r = T.children_left[i], T.children_right[i]
        nl, nr, n = T.weighted_n_node_samples[l], T.weighted_n_node_samples[r], T.weighted_n_node_samples[i]
        row["weighted_child_gini"] = (nl * T.impurity[l] + nr * T.impurity[r]) / n
        row["gini_decrease"] = T.impurity[i] - row["weighted_child_gini"]
    nodes.append(row)
# depth of each node
depth = np.zeros(T.node_count, int); stack = [(0, 0)]
while stack:
    n, d = stack.pop(); depth[n] = d
    if T.children_left[n] != -1: stack += [(T.children_left[n], d + 1), (T.children_right[n], d + 1)]
nodes = pd.DataFrame(nodes); nodes["depth"] = depth
nodes.round(5).to_csv(TAB / "phase4b_tree_nodes.csv", index=False)
leaves = nodes[nodes.is_leaf]
w_leaf_gini = (leaves.samples * leaves.gini).sum() / leaves.samples.sum()
say(f"\nChosen tree: depth {tree.get_depth()}, {tree.get_n_leaves()} leaves. Root Gini {T.impurity[0]:.4f} -> "
    f"sample-weighted leaf Gini {w_leaf_gini:.4f} (impurity removed {(T.impurity[0] - w_leaf_gini) / T.impurity[0]:.1%})")
say("Leaves (purest = most informative):\n" + leaves.sort_values("share_buyers", ascending=False)[["node", "samples", "share_buyers", "gini"]].round(4).to_string(index=False))
say("\nTree rules:\n" + export_text(tree, feature_names=fnames, show_weights=False, decimals=2))

# importance: Gini (mean decrease in impurity) vs permutation on the hold-out
mdi = pd.Series(tree.feature_importances_, index=fnames)
perm = permutation_importance(tree_pipe, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=SEED)
perm = pd.Series(perm.importances_mean, index=NUM + CAT)
imp = pd.DataFrame({"gini_importance_MDI": mdi}).query("gini_importance_MDI > 0").sort_values("gini_importance_MDI", ascending=False)
imp.round(4).to_csv(TAB / "phase4b_tree_gini_importance.csv")
perm.sort_values(ascending=False).round(4).to_frame("permutation_auc_drop_test").to_csv(TAB / "phase4b_tree_permutation_importance.csv")
say("\nGini importance (MDI, training):\n" + imp.round(4).to_string())
say("Permutation importance (AUC drop on hold-out):\n" + perm.sort_values(ascending=False).head(8).round(4).to_string())

# ================================================================== 3. SMOTE (KNN interpolation) + logistic regression
class SMOTEClassifier(BaseEstimator, ClassifierMixin):
    """SMOTE-NC style oversampling done only on the data passed to fit(), then the wrapped estimator.
    Numeric features: new = x + u * (neighbour - x), u ~ U(0,1), neighbour one of the k nearest minority points.
    Categorical (one-hot) block: copied from the seed point so synthetic customers stay valid (no '0.4 of a region').
    Distances use standardised numeric features plus one-hot dummies."""
    def __init__(self, estimator, k=5, ratio=1.0, n_num=len(NUM), random_state=SEED):
        self.estimator, self.k, self.ratio, self.n_num, self.random_state = estimator, k, ratio, n_num, random_state
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        r = np.random.default_rng(self.random_state)
        Xm = X[y == 1]; n_new = int(self.ratio * (y == 0).sum()) - len(Xm)
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Xm)
        idx = nn.kneighbors(Xm, return_distance=False)[:, 1:]
        seeds = r.integers(0, len(Xm), n_new); nbrs = idx[seeds, r.integers(0, self.k, n_new)]
        u = r.random((n_new, 1))
        syn = Xm[seeds].copy()
        syn[:, :self.n_num] = Xm[seeds, :self.n_num] + u * (Xm[nbrs, :self.n_num] - Xm[seeds, :self.n_num])
        self.n_synthetic_ = n_new
        self.estimator_ = clone(self.estimator).fit(np.vstack([X, syn]), np.r_[y, np.ones(n_new, int)])
        self.classes_ = self.estimator_.classes_
        return self
    def predict_proba(self, X): return self.estimator_.predict_proba(np.asarray(X, float))
    def predict(self, X): return self.estimator_.predict(np.asarray(X, float))

class RandomOver(SMOTEClassifier):
    """Plain random oversampling (duplicates buyers) for comparison."""
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y); r = np.random.default_rng(self.random_state)
        Xm = X[y == 1]; n_new = int(self.ratio * (y == 0).sum()) - len(Xm)
        syn = Xm[r.integers(0, len(Xm), n_new)]
        self.estimator_ = clone(self.estimator).fit(np.vstack([X, syn]), np.r_[y, np.ones(n_new, int)])
        self.classes_ = self.estimator_.classes_; return self

pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
LR = LogisticRegression(max_iter=3000)
models = {
    "Logistic regression (Phase 4 model)": make_pipeline(pre, LR),
    "Logistic, class_weight = balanced": make_pipeline(pre, LogisticRegression(max_iter=3000, class_weight="balanced")),
    "Random oversampling + logistic": make_pipeline(pre, RandomOver(LR)),
    "SMOTE (k=5) + logistic": make_pipeline(pre, SMOTEClassifier(LR, k=5)),
    "KNN classifier (k=101) alone": make_pipeline(pre, KNeighborsClassifier(n_neighbors=101, weights="distance")),
    "Decision tree (Gini, pruned)": tree_pipe,
}
# choose k for KNN by grouped CV (odd values, large k because events are rare)
knn_cv = {k: cv_auc(make_pipeline(pre, KNeighborsClassifier(n_neighbors=k, weights="distance")), Xtr, ytr, gtr)[0] for k in [25, 51, 101, 201, 401]}
kbest = max(knn_cv, key=knn_cv.get)
models["KNN classifier (k=101) alone"] = make_pipeline(pre, KNeighborsClassifier(n_neighbors=kbest, weights="distance"))
models = {(f"KNN classifier (k={kbest}) alone" if k.startswith("KNN") else k): v for k, v in models.items()}
say(f"\nKNN k chosen by grouped CV: {kbest}  ({ {k: round(v, 4) for k, v in knn_cv.items()} })")

# ---- correct vs wrong SMOTE inside cross-validation
def smote_outside_cv():
    """WRONG: oversample the whole training set first, then cross-validate. Synthetic buyers built from a
    validation-fold buyer's neighbours land in the training folds -> optimistic score."""
    Xt = pre.fit_transform(Xtr); r = np.random.default_rng(SEED)
    Xm = Xt[ytr == 1]; n_new = (ytr == 0).sum() - len(Xm)
    idx = NearestNeighbors(n_neighbors=6).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    s = r.integers(0, len(Xm), n_new); nb = idx[s, r.integers(0, 5, n_new)]; u = r.random((n_new, 1))
    syn = Xm[s].copy(); syn[:, :len(NUM)] += u * (Xm[nb, :len(NUM)] - Xm[s, :len(NUM)])
    X2, y2 = np.vstack([Xt, syn]), np.r_[ytr, np.ones(n_new, int)]
    out = []
    # plain stratified K-fold on the oversampled table: what is usually done after SMOTE
    from sklearn.model_selection import StratifiedKFold
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X2, y2):
        m = clone(LR).fit(X2[tr], y2[tr]); out.append(roc_auc_score(y2[va], m.predict_proba(X2[va])[:, 1]))
    return np.mean(out)
cv_rows = []
for nm, mdl in models.items():
    m, s = cv_auc(mdl, Xtr, ytr, gtr)
    cv_rows.append(dict(model=nm, cv_auc_grouped=m, cv_sd=s))
wrong = smote_outside_cv()
say(f"SMOTE applied BEFORE the CV split (wrong): CV AUC {wrong:.3f} vs {[r for r in cv_rows if r['model'].startswith('SMOTE')][0]['cv_auc_grouped']:.3f} "
    "when SMOTE runs inside each training fold (right). The gap is leakage, not skill.")

# ---- hold-out evaluation + confusion matrices
def cm_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), specificity=tn / (tn + fp), f1=f1_score(y, pred, zero_division=0),
                contacted_share=pred.mean())
def f1_threshold_cv(mdl):
    """F1-optimal cut-off chosen on out-of-fold TRAINING predictions (never on the test set)."""
    oof = np.zeros(len(ytr))
    for tr, va in GroupKFold(5).split(Xtr, ytr, gtr):
        oof[va] = clone(mdl).fit(Xtr.iloc[tr], ytr[tr]).predict_proba(Xtr.iloc[va])[:, 1]
    ts = np.unique(np.quantile(oof, np.linspace(0.5, 0.995, 150)))
    return ts[np.argmax([f1_score(ytr, (oof >= t).astype(int)) for t in ts])]
perf, cms, probs = [], [], {}
for nm, mdl in models.items():
    mdl.fit(Xtr, ytr); p = mdl.predict_proba(Xte)[:, 1]; probs[nm] = p
    auc = roc_auc_score(yte, p)
    perf.append(dict(model=nm, test_auc=auc, gini_coefficient=2 * auc - 1, avg_precision=average_precision_score(yte, p),
                     brier=brier_score_loss(yte, p), mean_predicted_prob=p.mean(), actual_rate=yte.mean()))
    t_f1 = f1_threshold_cv(mdl)
    for rule, thr in [("default 0.5 cut-off", 0.5), ("top 30% by score (capacity)", np.quantile(p, 0.70)),
                      ("F1-optimal cut-off (chosen on training CV)", t_f1)]:
        cms.append(dict(model=nm, rule=rule, threshold=thr, **cm_metrics(yte, (p >= thr).astype(int))))
perf = pd.DataFrame(perf).merge(pd.DataFrame(cv_rows), on="model")
perf.loc[len(perf)] = dict(model="WRONG: SMOTE before CV split", test_auc=np.nan, gini_coefficient=np.nan, avg_precision=np.nan,
                           brier=np.nan, mean_predicted_prob=np.nan, actual_rate=np.nan, cv_auc_grouped=wrong, cv_sd=np.nan)
cms = pd.DataFrame(cms)
perf.round(4).to_csv(TAB / "phase4b_model_comparison.csv", index=False)
cms.round(4).to_csv(TAB / "phase4b_confusion_matrices.csv", index=False)
say("\nHold-out (C011-C014) comparison:\n" + perf.round(4).to_string(index=False))
say("\nConfusion-matrix metrics:\n" + cms.round(3).to_string(index=False))

# ================================================================== 4. GINI COEFFICIENT of concentration (Lorenz)
def lorenz_gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n, np.r_[0, cum / cum[-1]]
od = load("orders")
att_rev = (od.dropna(subset=["attributed_exposure_id"]).assign(net=lambda d: d.gross_merchandise_value_inr - d.discount_inr)
             .groupby("customer_id").net.sum().reindex(cu.customer_id, fill_value=0))
conc = {}
for nm, x in [("Pre-2026 spend, all 6,000 customers", cu.spend_before_2026_inr),
              ("Pre-2026 spend, 5,156 buyers only", cu.spend_before_2026_inr[cu.orders_before_2026 > 0]),
              ("2026 campaign-attributed revenue, all customers", att_rev)]:
    g, L = lorenz_gini(x); conc[nm] = (g, L)
xs = np.sort(cu.spend_before_2026_inr.values)[::-1]
top20 = xs[: int(0.2 * len(xs))].sum() / xs.sum()
conc_tab = pd.DataFrame({"gini_coefficient": {k: v[0] for k, v in conc.items()}})
conc_tab.loc["Top 20% of customers' share of pre-2026 spend", "gini_coefficient"] = top20
conc_tab.round(4).to_csv(TAB / "phase4b_gini_concentration.csv")
say("\nGini coefficient of concentration:\n" + conc_tab.round(3).to_string())

# ================================================================== figures
# p4b_01: impurity — segment purity + single-split ranking
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
sd = seg_detail.reindex(SEGS)
ax[0].barh(sd.index[::-1], sd.gini[::-1], color=SERIES[0])
for i, (gv, pr) in enumerate(zip(sd.gini[::-1], sd.purchase_rate[::-1])):
    ax[0].text(gv + 0.002, i, f"G = {gv:.3f}  ({pr:.1%} buy)", va="center", fontsize=8, color=INK2)
ax[0].axvline(root, color=MUTED, ls="--", lw=1); ax[0].text(root, 4.45, f" root G = {root:.3f}", color=MUTED, fontsize=8)
ax[0].set_xlim(0, 0.25); ax[0].set_xlabel("Gini impurity of the purchase outcome (0 = pure)")
ax[0].set_title("Gini impurity inside each K-means segment")
gs = gsplit.sort_values("pct_impurity_removed")
ax[1].barh(gs.split, gs.pct_impurity_removed * 100, color=[SERIES[1] if "K-means" in s else SERIES[0] for s in gs.split])
ax[1].set_xlabel("% of root Gini impurity removed by the split"); ax[1].set_title("Which single variable purifies purchase most?")
save(fig, "p4b_01_gini_impurity")

# p4b_02: the pruned tree (top 3 levels)
fig, ax = plt.subplots(figsize=(24, 8))
plot_tree(tree, feature_names=fnames, class_names=["no buy", "buy"], filled=True, impurity=True, proportion=True,
          rounded=True, precision=3, max_depth=3, fontsize=7.5, ax=ax)
ax.set_title(f"Decision tree (criterion = Gini, depth {tree.get_depth()}, {tree.get_n_leaves()} leaves) — top 3 levels; "
             "'gini' in each box is the node impurity")
save(fig, "p4b_02_decision_tree")

# p4b_03: depth vs train/test AUC (overfitting) + Gini importance
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
dd = []
for d in range(1, 16):
    m = make_pipeline(pre_tree, DecisionTreeClassifier(criterion="gini", max_depth=d, min_samples_leaf=1, random_state=SEED)).fit(Xtr, ytr)
    dd.append((d, roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]), roc_auc_score(yte, m.predict_proba(Xte)[:, 1])))
dd = np.array(dd)
ax[0].plot(dd[:, 0], dd[:, 1], label="training AUC", color=SERIES[1]); ax[0].plot(dd[:, 0], dd[:, 2], label="hold-out AUC", color=SERIES[0])
ax[0].set_xlabel("max depth (no leaf-size limit)"); ax[0].set_ylabel("AUC"); ax[0].legend(); ax[0].set_title("Deeper trees memorise, they don't generalise")
top = imp.head(10)[::-1]
ax[1].barh(top.index, top.gini_importance_MDI, color=SERIES[0]); ax[1].set_xlabel("share of total Gini decrease (MDI)")
ax[1].set_title("Gini importance in the pruned tree")
save(fig, "p4b_03_tree_depth_importance")

# p4b_04: confusion matrices, logistic vs SMOTE at 0.5 and at top-30%
show = [("Logistic regression (Phase 4 model)", "default 0.5 cut-off"), ("SMOTE (k=5) + logistic", "default 0.5 cut-off"),
        ("Logistic regression (Phase 4 model)", "top 30% by score (capacity)"), ("SMOTE (k=5) + logistic", "top 30% by score (capacity)")]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for a, (mn, rl) in zip(axs, show):
    r = cms[(cms.model == mn) & (cms.rule == rl)].iloc[0]
    M = np.array([[r.TN, r.FP], [r.FN, r.TP]])
    a.imshow(M / M.sum(axis=1, keepdims=True), cmap=SEQ_CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = M[i, j] / M[i].sum()
            a.text(j, i, f"{M[i, j]:,}\n({frac:.0%} of row)", ha="center", va="center", fontsize=8, color="white" if frac > 0.6 else INK)
    a.set_xticks([0, 1], ["pred: no", "pred: buy"]); a.set_yticks([0, 1], ["actual: no", "actual: buy"]); a.grid(False)
    a.set_title(f"{'Logistic' if mn.startswith('Log') else 'SMOTE + logistic'}\n{rl}\nrecall {r.recall:.0%} · precision {r.precision:.1%}", fontsize=8.5)
save(fig, "p4b_04_confusion_matrices")

# p4b_05: ROC with Gini + calibration of SMOTE
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for (nm, p), c in zip(probs.items(), SERIES):
    fpr, tpr, _ = roc_curve(yte, p); a = roc_auc_score(yte, p)
    ax[0].plot(fpr, tpr, color=c, lw=1.5, label=f"{nm}: AUC {a:.3f}, Gini {2 * a - 1:.3f}")
ax[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1); ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate (recall)")
ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].set_title("ROC on hold-out; model Gini = 2·AUC − 1")
for nm, c in [("Logistic regression (Phase 4 model)", SERIES[0]), ("Logistic, class_weight = balanced", SERIES[2]), ("SMOTE (k=5) + logistic", SERIES[1])]:
    p = probs[nm]; q = pd.qcut(p, 10, labels=False, duplicates="drop")
    d = pd.DataFrame({"p": p, "y": yte, "q": q}).groupby("q").mean()
    ax[1].plot(d.p, d.y, "o-", color=c, label=nm, lw=1.5)
ax[1].plot([0, 0.6], [0, 0.6], color=MUTED, ls="--", lw=1); ax[1].set_xlim(0, 0.8); ax[1].set_ylim(0, 0.2)
ax[1].set_xlabel("mean predicted probability (decile)"); ax[1].set_ylabel("actual purchase rate"); ax[1].legend(fontsize=7)
ax[1].set_title("SMOTE / reweighting inflate probabilities ~8×")
save(fig, "p4b_05_roc_calibration")

# p4b_06: Lorenz curves
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for (nm, (g, L)), c in zip(conc.items(), SERIES):
    ax.plot(np.linspace(0, 1, len(L)), L, color=c, label=f"{nm}: Gini {g:.2f}")
ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="perfect equality")
ax.set_xlabel("cumulative share of customers (lowest value first)"); ax.set_ylabel("cumulative share of value"); ax.legend(fontsize=7)
ax.set_title("Lorenz curves: how concentrated is customer value?")
save(fig, "p4b_06_lorenz")

(TAB / "phase4b_log.txt").write_text(log.getvalue())
