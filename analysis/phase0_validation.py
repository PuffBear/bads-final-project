"""Phase 0: data validation. Writes outputs/tables/phase0_checks.csv."""
import numpy as np
import pandas as pd
from common import load_all, TAB

d = load_all()
cu, cc, ex, od, oi, pc, wk, bs = (d[k] for k in
    ["customers", "campaign_catalog", "campaign_exposures", "orders",
     "order_items", "product_catalog", "weekly_channel_performance",
     "brand_perception_survey"])
checks = []
def chk(area, name, value, ok=None, note=""):
    checks.append(dict(area=area, check=name, value=value,
                       status={True: "PASS", False: "FLAG", None: "INFO"}[ok], note=note))

# --- 1. Shape, nulls, duplicates ---
for k, df in d.items():
    chk("structure", f"{k}: rows x cols", f"{df.shape[0]} x {df.shape[1]}")
    nulls = df.isna().sum(); nulls = nulls[nulls > 0]
    chk("structure", f"{k}: null cells", int(nulls.sum()),
        ok=(nulls.sum() == 0) or (list(nulls.index) == ["attributed_exposure_id"]),
        note="; ".join(f"{c}={v}" for c, v in nulls.items()))
    chk("structure", f"{k}: duplicate rows", int(df.duplicated().sum()), ok=df.duplicated().sum() == 0)

pk = {"customers": ["customer_id"], "campaign_catalog": ["campaign_id"],
      "campaign_exposures": ["exposure_id"], "orders": ["order_id"],
      "order_items": ["order_item_id"], "product_catalog": ["product_id"],
      "weekly_channel_performance": ["week_start_date", "channel"],
      "brand_perception_survey": ["respondent_id", "brand"]}
for k, cols in pk.items():
    dup = d[k].duplicated(cols).sum()
    chk("keys", f"{k}: primary key {cols} unique", int(dup), ok=dup == 0)

# --- 2. Referential integrity ---
def orphan(child, ccol, parent, pcol, label):
    s = child[ccol].dropna()
    n = (~s.isin(parent[pcol])).sum()
    chk("joins", label, int(n), ok=n == 0, note=f"{len(s)} non-blank keys checked")
orphan(ex, "customer_id", cu, "customer_id", "exposures.customer_id not in customers")
orphan(ex, "campaign_id", cc, "campaign_id", "exposures.campaign_id not in catalog")
orphan(od, "customer_id", cu, "customer_id", "orders.customer_id not in customers")
orphan(od, "attributed_exposure_id", ex, "exposure_id", "orders.attributed_exposure_id not in exposures")
orphan(oi, "order_id", od, "order_id", "order_items.order_id not in orders")
orphan(oi, "product_id", pc, "product_id", "order_items.product_id not in products")
orphan(bs, "customer_id", cu, "customer_id", "survey.customer_id not in customers")
chk("joins", "orders with no line items", int((~od.order_id.isin(oi.order_id)).sum()),
    ok=(~od.order_id.isin(oi.order_id)).sum() == 0)
chk("joins", "weekly channels match catalog channels",
    sorted(set(wk.channel) ^ set(cc.channel)) or "identical",
    ok=set(wk.channel) == set(cc.channel))

# --- 3. Order arithmetic ---
tol = 0.01
diff = (od.gross_merchandise_value_inr - od.discount_inr + od.shipping_fee_inr - od.order_total_inr).abs()
chk("arithmetic", "GMV - discount + shipping = total (orders off)", int((diff > tol).sum()), ok=(diff > tol).sum() == 0)
line = (oi.quantity * oi.unit_price_inr - oi.line_gross_value_inr).abs()
chk("arithmetic", "qty x unit price = line value (lines off)", int((line > tol).sum()), ok=(line > tol).sum() == 0)
gm = oi.groupby("order_id").line_gross_value_inr.sum()
g = od.set_index("order_id").gross_merchandise_value_inr
gd = (g - gm.reindex(g.index)).abs()
chk("arithmetic", "order GMV = sum of line values (orders off)", int((gd > tol).sum()), ok=(gd > tol).sum() == 0)
m = oi.merge(pc, on="product_id")
chk("arithmetic", "unit price = list price (lines off)", int((m.unit_price_inr != m.list_price_inr).sum()),
    ok=(m.unit_price_inr != m.list_price_inr).sum() == 0, note="all discounting is order-level")
chk("arithmetic", "negative money values in orders", int((od.select_dtypes("number") < 0).sum().sum()),
    ok=(od.select_dtypes("number") < 0).sum().sum() == 0)

# --- 4. Attribution logic ---
att = od.dropna(subset=["attributed_exposure_id"]).merge(
    ex, left_on="attributed_exposure_id", right_on="exposure_id", suffixes=("", "_ex"))
chk("attribution", "attributed orders", len(att), note=f"{len(att)/len(od):.1%} of all orders")
chk("attribution", "attributed exposure id reused by >1 order", int(att.attributed_exposure_id.duplicated().sum()),
    ok=att.attributed_exposure_id.duplicated().sum() == 0)
chk("attribution", "attributed order customer != exposure customer", int((att.customer_id != att.customer_id_ex).sum()),
    ok=(att.customer_id == att.customer_id_ex).all())
lag = (att.order_date - att.exposure_date).dt.days
chk("attribution", "attributed order lag outside 1-14 days", int((~lag.between(1, 14)).sum()),
    ok=lag.between(1, 14).all(), note=f"lag range {lag.min()}-{lag.max()} days")
chk("attribution", "attributed to exposure with purchased_within_14_days=0", int((att.purchased_within_14_days == 0).sum()),
    ok=(att.purchased_within_14_days == 1).all())
buyers = ex.purchased_within_14_days.sum()
chk("attribution", "exposures flagged purchase=1 vs attributed orders", f"{buyers} vs {len(att)}",
    ok=None, note="flag=1 exposures without an attributed order: " + str(int(buyers - ex.exposure_id.isin(att.exposure_id).sum())))

# --- 5. Campaign/exposure consistency ---
e = ex.merge(cc, on="campaign_id", suffixes=("", "_cat"))
out = (~e.exposure_date.between(e.start_date, e.end_date)).sum()
chk("campaigns", "exposure date outside campaign window", int(out), ok=out == 0)
chk("campaigns", "exposure channel != catalog channel", int((e.channel != e.channel_cat).sum()), ok=(e.channel == e.channel_cat).all())
chk("campaigns", "duplicate customer x campaign exposures", int(ex.duplicated(["customer_id", "campaign_id"]).sum()),
    ok=ex.duplicated(["customer_id", "campaign_id"]).sum() == 0)

# --- 6. Funnel logic (observed event fields) ---
chk("funnel", "opened but not delivered", int(((ex.opened == 1) & (ex.delivered == 0)).sum()), ok=((ex.opened == 1) & (ex.delivered == 0)).sum() == 0)
chk("funnel", "clicked but not delivered", int(((ex.clicked == 1) & (ex.delivered == 0)).sum()), ok=((ex.clicked == 1) & (ex.delivered == 0)).sum() == 0)
cno = ex[(ex.clicked == 1) & (ex.opened == 0)]
chk("funnel", "clicked without opened", len(cno), ok=len(cno) == 0,
    note="by channel: " + ", ".join(f"{k}={v}" for k, v in cno.channel.value_counts().items()))
chk("funnel", "purchased but not delivered", int(((ex.purchased_within_14_days == 1) & (ex.delivered == 0)).sum()),
    ok=None, note="purchase may happen without contact; consider in interpretation")

# --- 7. Consent ---
ec = ex.merge(cu[["customer_id", "marketing_consent"]], on="customer_id")
nc = (ec.marketing_consent == 0).sum()
chk("compliance", "exposures to customers without marketing consent", int(nc), ok=nc == 0,
    note=f"{nc/len(ec):.1%} of exposures; cost INR {ec.loc[ec.marketing_consent==0,'contact_cost_inr'].sum():,.0f}")

# --- 8. Coverage, sample, control group ---
yr = od.order_date.dt.year.value_counts().sort_index()
chk("coverage", "orders by year", ", ".join(f"{k}: {v}" for k, v in yr.items()), ok=None,
    note="orders are a sample; do not read the 2026 drop as falling demand")
chk("coverage", "orders date range", f"{od.order_date.min().date()} to {od.order_date.max().date()}")
chk("coverage", "exposure date range", f"{ex.exposure_date.min().date()} to {ex.exposure_date.max().date()}")
chk("coverage", "weekly data range", f"{wk.week_start_date.min().date()} to {wk.week_start_date.max().date()}")
nweeks = wk.week_start_date.nunique()
expected = len(pd.date_range(wk.week_start_date.min(), wk.week_start_date.max(), freq="7D"))
chk("coverage", "weekly: weeks present vs expected", f"{nweeks} vs {expected}", ok=nweeks == expected and len(wk) == nweeks * wk.channel.nunique())
never = (~cu.customer_id.isin(ex.customer_id)).sum()
chk("design", "customers never exposed (potential control)", int(never), ok=None,
    note="not randomised; no holdout -> campaign results are associations")
chk("coverage", "survey respondents / brands", f"{bs.respondent_id.nunique()} / {bs.brand.nunique()}",
    ok=(bs.groupby("respondent_id").brand.nunique() == bs.brand.nunique()).all(), note="every respondent rates every brand")
rat = bs.filter(like="_1_to_5")
chk("range", "survey ratings outside 1-5", int(((rat < 1) | (rat > 5)).sum().sum()), ok=((rat >= 1) & (rat <= 5)).all().all())
chk("range", "customers signup after 2025-12-31", int((cu.signup_date > "2025-12-31").sum()), ok=(cu.signup_date <= "2025-12-31").all())
chk("range", "customers with orders_before_2026=0 but spend>0",
    int(((cu.orders_before_2026 == 0) & (cu.spend_before_2026_inr > 0)).sum()),
    ok=((cu.orders_before_2026 == 0) & (cu.spend_before_2026_inr > 0)).sum() == 0)
# recency should not exceed tenure
ten = (pd.Timestamp("2025-12-31") - cu.signup_date).dt.days
chk("range", "recency longer than tenure (days since last order > days since signup)",
    int((cu.days_since_last_order_at_2025_12_31 > ten).sum()), ok=(cu.days_since_last_order_at_2025_12_31 <= ten).all())

# --- 9. Label oddities in catalog ---
odd = cc[cc.campaign_name.str.contains("Valentines|Winter|Summer|Monsoon|New Year|Festive", case=False)]
chk("labels", "seasonal campaign names vs dates/category",
    "; ".join(f"{r.campaign_name}: {r.start_date:%d %b}, {r.featured_category}" for r in odd.itertuples()), ok=None,
    note="check names against dates and featured category")

dl = oi.duplicated(["order_id", "product_id"]).sum()
chk("structure", "order lines repeating a product already in the same order", int(dl), ok=None,
    note=f"{oi[oi.duplicated(['order_id','product_id'], keep=False)].order_id.nunique()} orders; arithmetic still holds - de-duplicate to product presence for basket analysis")

# --- 10. Cross-file consistency found during EDA ---
a = att.merge(cc, on="campaign_id")
gap = (a.discount_inr / a.gross_merchandise_value_inr * 100 - a.discount_pct).abs()
chk("arithmetic", "attributed order discount % = campaign discount_pct (orders off)", int((gap > 0.05).sum()), ok=(gap <= 0.05).all())
chk("arithmetic", "non-attributed orders carrying a discount",
    int((od.attributed_exposure_id.isna() & (od.discount_inr > 0)).sum()),
    ok=None, note="discounts only exist on campaign orders")
net = od.gross_merchandise_value_inr - od.discount_inr
bad_ship = (od.shipping_fee_inr != np.where(net >= 999, 0, 49)).sum()
chk("arithmetic", "shipping rule: free if (GMV - discount) >= 999 else INR 49 (orders off)", int(bad_ship), ok=bad_ship == 0,
    note=f"{(od.shipping_fee_inr == 49).mean():.1%} of orders pay shipping")
o25 = od[od.order_date.dt.year == 2025].groupby("customer_id").size()
z = cu[(cu.orders_before_2026 == 0) & cu.customer_id.isin(o25.index)]
chk("consistency", "customers with orders_before_2026=0 but 2025 orders in order table", len(z), ok=len(z) == 0,
    note="snapshot fields are not derived from the order sample (per brief); treat snapshot as authoritative")
from scipy.stats import spearmanr
smp = od[od.order_date.dt.year == 2025].groupby("customer_id").agg(n=("order_id", "size"), aov=("gross_merchandise_value_inr", "mean"), spend=("gross_merchandise_value_inr", "sum"))
smp = smp.join(cu.set_index("customer_id")[["orders_before_2026", "spend_before_2026_inr"]]).query("orders_before_2026 > 0")
smp["snap_aov"] = smp.spend_before_2026_inr / smp.orders_before_2026
rho = {k: spearmanr(smp[a_], smp[b_])[0] for k, (a_, b_) in {"orders": ("n", "orders_before_2026"), "spend": ("spend", "spend_before_2026_inr"), "AOV": ("aov", "snap_aov")}.items()}
chk("consistency", "snapshot vs 2025 order-sample agreement (Spearman, buyers)", ", ".join(f"{k} {v:.2f}" for k, v in rho.items()),
    ok=None, note="AOV agreement ~0: order-level baskets cannot validate the snapshot value fields (and vice versa)")
chk("consistency", "customers with orders_before_2026=0 but a days_since_last_order value",
    int((cu.orders_before_2026 == 0).sum()), ok=None, note="recency for never-buyers is undefined; handled in Phase 1")
r2 = (cu.days_since_last_order_at_2025_12_31 == 2).sum()
chk("consistency", "heap at days_since_last_order = 2 (minimum value)", int(r2), ok=None,
    note=f"next value (3 days) has {(cu.days_since_last_order_at_2025_12_31 == 3).sum()}; likely a floor - treat as 'bought in final days of 2025'")
a18 = (cu.age == 18).sum()
chk("consistency", "heap at age = 18 (minimum value)", int(a18), ok=None,
    note=f"age 19 has {(cu.age == 19).sum()}; under-18s likely recoded to 18 - avoid fine age cuts at the bottom")
rc = bs.drop_duplicates("respondent_id").customer_id.value_counts()
chk("consistency", "survey: customers answering more than once", int((rc > 1).sum()), ok=(rc == 1).all(),
    note=f"{bs.respondent_id.nunique()} respondents map to {rc.size} customers")
rp = ex.groupby(["customer_id", "campaign_id"]).size()
checks[[c["check"] for c in checks].index("duplicate customer x campaign exposures")].update(
    check="customer x campaign pairs with 2+ sends", value=int((rp > 1).sum()), status="INFO",
    note=f"{int(rp[rp>1].sum())} exposures sit in repeated pairs (max {rp.max()} sends); treat as frequency, not duplicates")

res = pd.DataFrame(checks)
res.to_csv(TAB / "phase0_checks.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 110, "display.max_rows", 200)
print(res.to_string(index=False))
