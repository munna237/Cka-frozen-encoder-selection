# Reproducing the study

## 1. Registered results, from this repository alone (CPU, about a minute)

```bash
pip install -r requirements.txt
python code/verify_v3.py --root .
```

The gate recomputes estimands A–E (paper Sections VI-A to VI-C; Tables 7, 8 and 9) from `results/pools/*.json`
and compares every field with `results/battery.json`. It also confirms that the analysis code matches the
registration. Exit code 0 means all 55 checks passed.

The primary-pair random-intercept model is degenerate by design (one observation per pool). The gate checks it
for the documented degeneracy instead of comparing optimiser noise.

## 2. Exploratory analyses, from the published features (CPU; about an hour)

```bash
pip install -r requirements-extract.txt   # only if you will refit probes
python code/fetch_features.py --record <zenodo record id> --root ~/cka_study
export V3_DRIVE=~/cka_study
```

`fetch_features.py` downloads and verifies every archive, then builds `~/cka_study/TMLR_v3/` with the features,
held-out logits, registration, manifests, per-pool records and a copy of `code/`. Then run:

- `notebooks/07_reviewer_analyses.ipynb`: competence gaps, calipers, strong candidates, dependence, cross-fitted
  regret, hybrid rules, Algorithm 1, other similarity measures, other outcome metrics (Section VI-E; Table 10, Supplementary Tables S2–S3).
- `notebooks/08_reviewer_round2.ipynb`: pilot-dataset exclusion, gated rule tests, fully nested selection, probe
  C, multiplicity, encoder validation (Sections VI-E and IV-C; Supplementary Tables S4–S5).

**Gated encoders.** The Zenodo record leaves out the features and logits of the six gated pathology encoders
(UNI, UNI2-h, Virchow2, H-optimus-0, Prov-GigaPath, Hibou-B). Notebooks 07 and 08 need every encoder's logits in a
pool, so after downloading they run on the vision and text pools as they are; for the pathology pools, first obtain
these models from their authors and regenerate their features (notebook 01) and logits (notebook 04).

The notebooks were written for Colab with Google Drive. Outside Colab, skip the `drive.mount` cell; the code reads
`V3_DRIVE`. Both notebooks first assert that the saved logits reproduce every registered utility and regret.
Fully nested selection (notebook 08, S3) refits probes and is the slow part; a GPU helps.

To re-run the registered analysis itself on the downloaded features, run `notebooks/04_analysis.ipynb`. It checks
the code against the registration before running.

## 3. Everything from raw public data (GPU)

1. **Notebook 00:** set up the project folder and run the tests. It would also write a *new* registration; keep
   `registration/prereg.json` from this repository if you want to check against ours.
2. **Notebooks 01–03:** extract features. The datasets download automatically, except BRACS, SICAPv2 and MHIST,
   which must be obtained from their creators; the expected paths are in `code/config.py`. Gated encoders need a
   Hugging Face token with each licence accepted. CTransPath needs its released checkpoint.
3. **Notebook 04:** the registered analysis. **Notebook 05:** the paper tables.

Every feature file is checked against the dataset manifest's row hash, so the rows align with ours. Exact
floating-point equality with our features is not guaranteed across GPU and library versions. Regenerated
features should reproduce our numbers to the precision reported in the paper; we have checked this only for the
pilot encoders (Section IV-D).
