# NexaGlow Marketing Intelligence (BADS final project)

Management report for NexaGlow, a DTC beauty brand: customers, buying behaviour, product links, campaign and channel effectiveness, brand perception and demand over time. The brief is in `data/MARKETING_INTELLIGENCE_PROJECT.pdf`.

## Layout
```
data/                 raw CSVs + data dictionary + brief (unchanged inputs)
analysis/common.py    paths, loaders, chart style (fixed colour per channel / category)
analysis/phase0_validation.py   82 data-quality checks  -> outputs/tables/phase0_checks.csv
analysis/eda.py                 file-by-file EDA       -> outputs/figures/eda_*.png, outputs/tables/eda_*.csv
analysis/phase1_customers.py    RFM + k-means segments -> outputs/tables/customer_segments.csv, p1_* outputs
analysis/phase2_buying.py       buying behaviour       -> outputs/tables/order_level.csv, p2_* outputs
analysis/phase3_products.py     basket analysis/bundles-> p3_* outputs
analysis/phase4_campaigns.py    campaign economics, drivers, targeting model -> p4_* outputs
analysis/phase4b_trees_smote.py  Gini impurity, decision tree, SMOTE+logit, KNN, confusion matrices -> p4b_* outputs
analysis/phase5_channels.py     weekly channel KPIs, elasticities, budget scenarios -> p5_* outputs
analysis/phase6_brand.py        brand perception, paired tests, perceptual map -> p6_* outputs
analysis/phase6b_mds.py         MDS brand map (vs PCA, bootstrap) + product map from baskets -> p6b_* outputs
analysis/phase7_demand.py       trend/seasonality decomposition, forecast evaluation, 2027 forecast -> p7_* outputs
docs/                 written findings per phase (feed the final report)
outputs/              generated figures and tables (re-created by the scripts)
```

## Run
```bash
pip install -r requirements.txt
cd analysis
python3 phase0_validation.py
python3 eda.py
python3 phase1_customers.py   # ~3-5 min (bootstrap stability + gap statistic)
python3 phase2_buying.py      # needs phase 1 output
python3 phase3_products.py    # needs phase 2 output
python3 phase4_campaigns.py   # needs phase 1 output
python3 phase4b_trees_smote.py # needs phase 1 output; ~1 min
python3 phase5_channels.py    # needs phase 4 output (cross-check)
python3 phase6_brand.py       # needs phase 1 output
python3 phase6b_mds.py         # needs phase 6 output
python3 phase7_demand.py      # ~1 min
```

## Status
- [x] Phase 0: validation (`docs/phase0_validation.md`)
- [x] Detailed EDA (`docs/eda_findings.md`)
- [x] Phase 1: customer segmentation, k = 4 + never-purchased (`docs/phase1_customers.md`)
- [x] Phase 2: buying behaviour (`docs/phase2_buying.md`)
- [x] Phase 3: product relationships and bundles (`docs/phase3_products.md`)
- [x] Phase 4: campaign effectiveness and targeting (`docs/phase4_campaigns.md`)
- [x] Phase 4b: decision tree / Gini / SMOTE / confusion matrices (`docs/phase4b_trees_smote.md`)
- [x] Phase 5: channel performance and budget (`docs/phase5_channels.md`)
- [x] Phase 6: brand perception (`docs/phase6_brand.md`)
- [x] Phase 6b: multidimensional scaling (`docs/phase6b_mds.md`)
- [x] Phase 7: demand over time and 2027 forecast (`docs/phase7_demand.md`)
- [ ] Phase 8: management report (see `docs/REVIEW_gap_analysis.md`)
