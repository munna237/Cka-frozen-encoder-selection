# Evaluating Centered Kernel Alignment for Frozen-Encoder Selection in Logit-Level Fusion

Code, registration, per-pool records and verification gate for:

> M. M. H. Munna, A. H. Kabir, R. F. Tulona, N. H. Antu, and M. M. Ul Mulk,
> *Evaluating Centered Kernel Alignment for Frozen-Encoder Selection in Logit-Level Fusion:
> A Pre-Specified Multi-Domain Study*, submitted to IEEE Access, 2026.

The study tests whether representational similarity, measured by linear CKA on frozen features, predicts which
pretrained encoders are worth combining. It uses 20 pools (pathology, general vision, text), 31 frozen encoders and
166 candidate additions, with an analysis plan hash-locked before any feature of the 20 pools was extracted.

**Frozen features and held-out logits:** Zenodo, DOI `10.5281/zenodo.<record>` (≈10 GB; see [`docs/DATA.md`](docs/DATA.md)).

---

## Verify the reported results in one minute

```bash
pip install -r requirements.txt        # numpy, scipy, statsmodels, ...; no GPU or torch needed
python code/verify_v3.py --root .        # exit code 0 = all 55 checks passed
```

This needs no GPU and no features. It checks five things:
1. **Code:** the analysis code here is the registered code (SHA-256 against `registration/prereg.json`).
2. **Records:** every per-pool record carries the registration and manifest hashes.
3. **Estimands:** recomputing estimands A–E from `results/pools/*.json` reproduces `results/battery.json`.
4. **Tables:** the paper tables rebuild.
5. **Paper numbers:** every registered number quoted in the paper matches.

## Reproduce further

| Goal | What you need | How |
|---|---|---|
| Registered results (Tables 13, 15, 16) | this repository | `python code/verify_v3.py --root .` |
| Exploratory analyses (Section VI-E, Tables 17–21) | features + logits from Zenodo | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) §2 |
| Everything from raw public data | GPU, the datasets, licences for gated models | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) §3 |
| The pilot and the controlled study (Sections V, VII) | `pilot/` | [`pilot/README.md`](pilot/README.md) |

[`docs/PAPER_MAP.md`](docs/PAPER_MAP.md) maps every section and table of the paper to the files that produce it.

## Repository layout

```
code/                  analysis, extraction, verification and export code
  config.py ... run_pipeline.py, analyse_all.py   registered analysis (hashes in registration/prereg.json;
                                                  do not edit, or verify_v3.py will report the change)
  extract.py, encoders.py, imgcache.py, text_arm.py, ctranspath_worker.py, hf_worker.py   feature extraction
  verify_v3.py         verification gate
  reviewer.py, reviewer2.py   exploratory analyses (Section VI-E)
  export_zenodo.py, fetch_features.py   features to and from Zenodo
notebooks/             00 setup and registration · 01–03 feature extraction · 04 analysis · 05 tables ·
                       06 supplementary build · 07–08 exploratory analyses · 09 export to Zenodo
tests/                 unit and end-to-end tests
registration/          prereg.json (the registration), timeline.json (registration -> extraction -> analysis),
                       drive_metadata.json (storage-service record of prereg.json)
results/               battery.json (estimands A–E), pools/*.json (per-pool records), paper tables,
                       reviewer/ (exploratory analyses), fusion fit log, continuity checks, figures
manifests/             row order, labels, groups and hashes of every dataset (dataset text not included)
features_provenance/   per (dataset, encoder): source, checksum, preprocessing, load report, extraction time
logs/                  extraction run logs
pilot/                 the four-encoder pilot and the controlled study, with its own verification harness
docs/                  REPRODUCE.md, PAPER_MAP.md, DATA.md
```

## Registration

The analysis plan, its parameters and the SHA-256 of every analysis file were fixed in
`registration/prereg.json` on 2026-10-25 at 13:57 UTC. That was before any feature of the 20 pools was extracted.
`registration/timeline.json` records the sequence:
- the first dataset manifest was built at 14:00;
- the first feature was saved at 14:16;
- all 206 feature files postdate the registration;
- the analysis ran on 29 October, with no declared deviation.

`registration/drive_metadata.json` holds the storage service's own creation time and checksum for the registration
file. No independent registry was used. The pilot's results were known at registration and are reported in Section V.

## Licence

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Records, results and manifests:** CC BY 4.0.
- **Datasets and pretrained encoders:** governed by their creators' licences. Several encoders are gated and must
  be obtained from their authors under their terms.

## Citation

See [`CITATION.cff`](CITATION.cff). Corresponding author: Acramul Haque Kabir, acramul.kabir@northsouth.edu.
