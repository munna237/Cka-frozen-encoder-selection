# Data

## In this repository
- **`results/`:** every per-pool record, from which all registered estimands are computed.
- **`manifests/`:** the row order, labels, slide or patient groups and row hashes of every dataset. The text of the
  text datasets is not redistributed; the row ids index the named public split.
- **`features_provenance/`:** for every (dataset, encoder) pair, the encoder source, weight checksum,
  preprocessing, pooling, load report, feature statistics and extraction time.

## On Zenodo (DOI [`10.5281/zenodo.24377082`](https://doi.org/10.5281/zenodo.24377082); CC BY-NC 4.0)
- **`features_<dataset>.zip`:** `<dataset>/<encoder>.npy` (float32, rows in manifest order) and its provenance
  `.json`.
- **`heldout_logits.zip`:** `pools/<dataset>_logits.npz`, the per-fold held-out probe logits of every encoder, with
  labels and fold masks.
- **`cka-frozen-encoder-selection-code.zip`:** a snapshot of this repository at the time of deposit.
- **`zenodo_manifest.json`:** the SHA-256 of every archive and of every file inside it, and the list of excluded encoders.

**Not on Zenodo:** the features and held-out logits of the six gated pathology encoders (UNI, UNI2-h, Virchow2,
H-optimus-0, Prov-GigaPath, Hibou-B), 36 of the 206 dataset–encoder pairs. Their licences may restrict redistributing
outputs, so they are left out as a precaution. The per-pool records here still cover them, so every registered number
is verifiable; regenerating their features needs each model from its authors.

`code/fetch_features.py` downloads, verifies and unpacks the record:

```bash
python code/fetch_features.py --record 24377082 --root ~/cka_study            # everything
python code/fetch_features.py --record 24377082 --root ~/cka_study --only dtd # one dataset + the logits
```

`notebooks/09_export_features_to_zenodo.ipynb`
created it from the project folder on Google Drive.

## Sources
Datasets and encoders, with their exact versions and sources, are listed in the paper (Tables 1–2) and in
`manifests/*.json` (`source` field) and `features_provenance/`. Gated encoders (UNI, UNI2-h, Virchow2, H-optimus-0,
Prov-GigaPath, Hibou-B) must be obtained from their authors under their licences. One dataset was taken from a
mirror: GTSRB, from `tanganke/gtsrb` on Hugging Face (same 26,640 training images and 43 classes), because the
official host was unavailable.
