# Multi-entity cached-feature model v3

## Intended use

Binary real/synthetic classification and selective abstention for the audited historical 14-feature, four-confidence cache. This is a research artifact for the thesis benchmark, not a raw-image detector.

## Architecture

V3 contains a three-seed `MultiEntityConsistencyHead` ensemble. It encodes four cached entity groups—illumination, specular optics, surface-normal residuals, and chromatic shadows—then learns all six pairwise interaction tokens. Entity confidences gate missing cues and downweight uncertain pair interactions.

The selected score is 20% multi-entity ensemble and 80% v2. Platt calibration and the decision threshold were fitted only on the calibration partition.

## Audited result

| Metric | V3 | V2 |
|---|---:|---:|
| Rows | 7,980 | 7,980 |
| Accuracy | 73.61% | 72.81% |
| ROC-AUC | 0.8134 | 0.8069 |
| EER | 26.50% | 27.08% |

Paired accuracy difference: +0.80 percentage points; paired bootstrap 95% interval: +0.29 to +1.27 points.

Selective result: 77.72% decided accuracy at 78.63% coverage.

## Entity diagnostic

The entity branch alone reaches 73.23% accuracy and 0.8135 ROC-AUC. In a post-freeze diagnostic, removing illumination, specular optics, surface normals, or chromatic shadows reduced its historical-test accuracy to 67.49%, 72.01%, 67.89%, and 69.16%, respectively. The largest average learned pair attention was illumination–surface normals (27.01%).

These diagnostics show that the branch uses multiple cached entities; they do not validate the physical correctness of the historical normal feature.

## Limitations

- The fake test source is another DiffusionDB subset, not an unseen generator family.
- The historical test set had been inspected before this run and is not a fresh confirmatory benchmark.
- Historical normal residual columns came from a random normal head whose state was not saved.
- SHA-256 deduplication does not detect resized, recompressed, or semantic near-duplicates.
- Load `model.pt` only from a trusted local run.
