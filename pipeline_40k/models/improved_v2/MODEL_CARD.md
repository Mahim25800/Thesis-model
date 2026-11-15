# Improved cached-feature ensemble v2

## Intended use

Binary real/synthetic classification and selective abstention on caches matching `data/cache_scaled/*_scaled_features.pt`. This artifact is for thesis experimentation on the audited historical dataset.

Raw-image inference is not supported. The historical cache contains normal-head outputs whose original random state was not saved, plus two highlight calculations that no longer match the current extractor.

## Model

- Histogram gradient boosting over all 14 cached features and four confidence values.
- Legacy transformer probability blended at weight 0.4.
- Gradient-boosting probability blended at weight 0.6.
- Platt calibration and decision threshold fitted only on the calibration split.
- Indeterminate band: calibrated threshold ±0.10 with observability ≥1.0.

## Audited result

| Metric | Improved v2 | Legacy transformer |
|---|---:|---:|
| Rows | 7,980 | 7,980 |
| Accuracy | 72.81% | 71.80% |
| ROC-AUC | 0.8069 | 0.7881 |
| EER | 27.08% | 28.28% |

Paired accuracy difference: +1.00 percentage point; paired bootstrap 95% interval: +0.24 to +1.78 points.

Selective result: 77.46% decided accuracy at 78.82% coverage.

## Limitations

- The fake test source is another DiffusionDB subset, not an unseen generator family.
- The historical test set had been inspected before this run and is not a fresh confirmatory benchmark.
- SHA-256 deduplication does not detect resized, recompressed, or semantic near-duplicates.
- Performance depends on the exact versioned caches and legacy checkpoint recorded in `metadata.json`.
- Loading `model.joblib` executes Python serialization. Load only this trusted local artifact.

See `evaluation.json`, `selection.json`, `policy.json`, and `metadata.json` for complete machine-readable evidence.
