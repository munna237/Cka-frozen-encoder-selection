# Reproducing the study

## 1. Registered results, from this repository alone (CPU, about a minute)

```bash
pip install -r requirements.txt
python code/verify_v3.py --root .
```

The gate recomputes estimands A–E (paper Section VI-A to VI-C; Tables 13, 15 and 16) from `results/pools/*.json`
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
  regret, hybrid rules, Algorithm 1, other similarity measures, other outcome metrics (Section VI-E; Tables 17–19).
- `notebooks/08_reviewer_round2.ipynb`: pilot-dataset exclusion, gated rule tests, fully nested selection, probe
  C, multiplicity, encoder validation (Section VI-E, Section IV-C; Tables 20–21).

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
