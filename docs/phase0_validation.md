# Phase 0 – Data validation

Script: `analysis/phase0_validation.py` → `outputs/tables/phase0_checks.csv` (82 checks: 53 PASS, 24 INFO, 5 FLAG).

## What passed (evidence of data integrity)

| Area | Check | Result |
|---|---|---|
| Structure | Duplicate rows, primary-key uniqueness (all 8 files) | 0 duplicates; every key unique |
| Structure | Nulls | Only `orders.attributed_exposure_id` (13,105 blanks = "not campaign-credited", as the brief says) |
| Joins | All 9 foreign-key joins (customer, campaign, exposure, order, product, survey) | 0 orphans; every order has line items |
| Arithmetic | GMV − discount + shipping = order total | 0 of 14,274 orders off |
| Arithmetic | Quantity × unit price = line value; Σ lines = order GMV | 0 off |
| Arithmetic | Unit price = catalogue list price | 0 off → **all discounting is at order level** |
| Arithmetic | Discount % on attributed orders = campaign `discount_pct` | 0 off (exact) |
| Arithmetic | Shipping = 0 if (GMV − discount) ≥ ₹999, else ₹49 | 0 off → a clean free-shipping threshold |
| Attribution | 1,169 attributed orders ↔ 1,169 exposures with `purchased_within_14_days = 1` | One-to-one, same customer, 1–14 days after exposure |
| Campaigns | Exposure date inside campaign window; exposure channel = catalogue channel | 0 off |
| Funnel | Opened/clicked without delivery | 0 |
| Coverage | Weekly table: 148 consecutive weeks × 6 channels, no gaps | Complete |
| Survey | 800 respondents × 8 brands; ratings in 1–5 | Complete, in range |

## Issues flagged (feed the Limitations section)

| # | Issue | Size | How we handle it |
|---|---|---|---|
| 1 | **Clicked without opened** | 1,446 exposures (6.9%), every channel (SMS 347 … Search 161) | Tracking artefact, not behaviour. Treat delivered / opened / clicked as separately measured flags, not a strict funnel. |
| 2 | **Contacts without marketing consent** | 3,957 exposures (18.8%), ₹16.2k contact cost; consent rate 81% | Compliance risk → headline recommendation (suppress at send time). |
| 3 | **Order table is a sample** | 11,005 orders in 2025 vs 3,269 in Jan–Oct 2026 | Do not read as falling demand. Use weekly channel data for trends. |
| 4 | **Snapshot fields ≠ order sample** | 595 customers have `orders_before_2026 = 0` yet 2025 orders in the table. For buyers, snapshot vs 2025 sample agreement (Spearman) is orders 0.25, spend 0.18, **AOV 0.02** | The brief says snapshot fields are not computed from the sample. Snapshot is authoritative for Phase 1; order table used only for basket/price analysis. |
| 5 | **Recency longer than tenure** | 794 customers (incl. 146 with 0 orders) | Recency is not internally consistent with `signup_date`. Use recency as a relative ranking, not an exact date. |
| 6 | **Recency for never-buyers** | 844 customers with 0 orders still carry a recency value | Undefined → they form a separate rule-based segment in Phase 1. |
| 7 | **Heaps at minimum values** | 764 customers at recency = 2 days (only 20 at 3 days); 520 at age 18 (only 91 at age 19) | Likely floors/recodes. Noted in Phase 1 robustness; avoid fine age cuts at the bottom. |
| 8 | **Survey respondents repeat** | 800 respondents = 760 customers (38 answered 2+ times) | Brand-level means unaffected. For segment cross-cuts, cluster standard errors by customer. |
| 9 | **No control group** | Only 169 customers never exposed, not randomised | Campaign effects are associations, not lift. Recommend a randomised holdout. |
| 10 | **Repeat sends** | 2,174 customer×campaign pairs got 2–5 sends (4,541 exposures) | Not duplicates (different IDs/dates). Treat as contact frequency. |
| 11 | **Platform-reported weekly data** | Email ROAS 32.7× vs Display 2.8× | Never join to customer data. Email ROAS is very likely over-credited. |
| 12 | **Repeated product lines** | 1,567 lines repeat a product already in the same order (1,469 orders) | Arithmetic still holds. Collapsed to product presence for basket analysis (Phase 3). |
| 13 | **Label/content mismatches** | "Valentines Self Care" ran 21 Mar–1 Apr; "Winter Skin Rescue" featured Haircare; "Festive Preview" in July | Use `featured_category` and dates, not names. |
