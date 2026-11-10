# Paper → files

Section and table numbers refer to the IEEE Access version.

| Paper | What | Files |
|---|---|---|
| III-A, III-G | pools, anchors, matching, estimands, registration | `code/pairs.py`, `code/run_pipeline.py`, `code/analysis.py`, `registration/prereg.json` |
| III-B | linear CKA | `code/cka.py` |
| III-C | per-fold logit protocol, probe | `code/folds.py`, `tests/test_core.py` |
| III-D | fusion rules (Eqs. 4–5) | `code/fusion.py` |
| III-H | verification protocol | `code/verify_v3.py`, `code/prereg.py`, `registration/timeline.json` |
| IV-A, Table 1 | pools and datasets | `manifests/`, `code/imgcache.py`, `code/text_arm.py` |
| IV-B, Table 2 | encoders, extraction | `code/encoders.py`, `code/extract.py`, `code/ctranspath_worker.py`, `code/hf_worker.py`, `features_provenance/`, `logs/` |
| IV-C | encoder validation | `code/reviewer2.py` (S8), `results/reviewer/reviewer2_analyses.md` |
| IV-C, Table 3 | CTransPath correction | `pilot/` (pilot features), `code/ctranspath_worker.py` |
| IV-D | pilot feature continuity, ResNet-50 provenance | `results/v2_consistency.csv`, `results/resnet50_provenance_mhist.csv` |
| IV-E | probe solver; secondary-rule iteration cap | `tests/test_core.py`, `results/fusion_fit_log.csv` |
| V, Tables 4–12 | pilot and general-vision extension | `pilot/` (own README and `verify_claims.py`) |
| VI-A, Table 13 | estimands A and C | `results/battery.json`, `results/paper_tables.md` |
| VI-A, Table 14 | every pool, primary pairs | `results/pools/*.json`, `results/reviewer/reviewer2_analyses.md` (S7) |
| VI-B, Table 15 | estimand D (selection regret) | `results/battery.json` |
| VI-C, Table 16 | estimands B and E | `results/battery.json` |
| VI-D | text pools | `results/paper_tables.md` (leave-one-pool-out) |
| VI-E, Table 17 | caliper sensitivity | `results/reviewer/reviewer_analyses.md` (R2) |
| VI-E, Table 18 | other similarity measures | `results/reviewer/reviewer_analyses.md` (R7) |
| VI-E, Table 19 | cross-fitted regret, hybrid rules | `results/reviewer/reviewer_analyses.md` (R5) |
| VI-E, Table 20 | without the pilot's datasets | `results/reviewer/reviewer2_analyses.md` (S1) |
| VI-E, Table 21 | fully nested selection | `results/reviewer/reviewer2_analyses.md` (S3) |
| VI-E | gated rule, probe C, multiplicity, consistency | `results/reviewer/reviewer2_analyses.md` (S2, S4, S5, S6) |
| VII, Fig. 2, Table 22 | controlled study | `pilot/code/synthetic_study.py`, `pilot/results/` |
| Fig. 1 | overview | `results/pools/*.json`, `results/reviewer/` (values) |
