# Scaled physics-inspired image detector

This folder contains the 40,000-image research pipeline and its external cross-generator evaluation. The current candidate is a frozen regional multi-physics ensemble built from official DSINE surface normals, scene-wide specular optics, illumination consistency, and chromatic-shadow consistency.

## Current regional multi-physics result

The reproducible DSINE v4 model replaced the historical random normal head with the official uncertainty-aware DSINE checkpoint. The regional v5 model then added global-plus-quadrant physics reasoning, and v6 added source-generator adversarial invariance. The final frozen v14 artifact applies spatial test-time averaging to v5 and combines 70% v5 with 30% v6. Its weights were selected using only internal development data and GenImage Wukong.

V14 reaches **80.31% accuracy and 0.8895 ROC-AUC** on 7,992 non-overlapping internal unseen rows. On nine Synthbuster generators it reaches **63.51% mean accuracy and 0.7024 mean ROC-AUC**. It improves over the global DSINE v4 model on every Synthbuster generator, although Firefly remains below the predeclared per-generator gate at 0.6410 AUC.

The separate frozen GenImage confirmation used 7,998 retained images from GLIDE, Midjourney, Stable Diffusion v1.4, and v1.5, sampled from archive positions 4,001-6,000. V14 reached **58.28% mean accuracy and 0.6239 mean ROC-AUC**. This failed the breakthrough gate and shows that the Synthbuster gain does not transfer uniformly across datasets. The generator-invariant v6 branch was stronger on this confirmation at 0.6330 mean AUC, but it was not promoted after the result was observed.

The implemented experiment stack is:

- `scripts/cache_regional_dsine_v2.py` extracts all four physics entities globally and over four image quadrants. DSINE runs once per image and its aligned normal/uncertainty maps are summarized per region.
- `scripts/train_regional_multi_physics_v5.py` measures the value of localized physics alone.
- `scripts/prepare_genimage_development.py` selects ADM, BigGAN, VQDM, and Wukong from GenImage under its academic/non-commercial license.
- `scripts/train_generator_invariant_v6.py` trains on DiffusionDB/COCO plus ADM, BigGAN, and VQDM with a generator-domain adversary. Wukong is held out for model selection.
- `scripts/evaluate_external_synthbuster_regional.py` evaluates the fully frozen artifact on Synthbuster only after training and selection are complete.
- `scripts/run_breakthrough_pipeline.py` records and resumes the long-running sequence through `experiments/breakthrough_pipeline_status.json`.
- `scripts/select_physics_ensemble_v9.py` selects a three-branch probability ensemble using only internal development and GenImage Wukong.
- `scripts/select_region_tta_v13.py` selects fixed spatial test-time transformations without reading a final test benchmark.
- `scripts/evaluate_genimage_confirmatory.py` performs a second frozen evaluation on disjoint GenImage GLIDE, Midjourney, SD v1.4, and SD v1.5 data and rejects any perceptual-hash overlap.
- `JOURNAL_READINESS_REPORT.md` records the contribution, ablations, confidence intervals, negative results, and claim boundary.

The supported conclusion is a **statistically validated regional multi-physics improvement on Synthbuster with a documented cross-dataset generalization limit**. The evidence does not support a universal breakthrough or state-of-the-art claim.

## Promoted result

The comparison uses the same audited test rows for both models. Candidate selection used a separate development partition; probability calibration and the operating threshold used a separate calibration partition.

| Model | Test rows | Accuracy | ROC-AUC | EER |
|---|---:|---:|---:|---:|
| Existing transformer, threshold 0.50 | 7,980 | 71.80% | 0.7881 | 28.28% |
| Improved v2, frozen calibrated policy | 7,980 | 72.81% | 0.8069 | 27.08% |
| Multi-entity v3, frozen calibrated policy | 7,980 | **73.61%** | **0.8134** | **26.50%** |

V3 improves on v2 by **+0.80 percentage points**. Its 2,000-sample paired bootstrap interval is **+0.29 to +1.27 points**. The fixed raw threshold 0.50 result is 73.52%; the table reports the policy selected only from calibration data.

The v3 selective policy reaches **77.72% decided accuracy at 78.63% coverage**. Accuracy and coverage must always be reported together.

Artifacts:

- `models/multi_entity_v3/model.pt`: v3 entity-interaction ensemble and calibration policy.
- `models/multi_entity_v3/evaluation.json`: complete frozen test metrics.
- `models/multi_entity_v3/selection.json`: development-only architecture and blend selection.
- `models/multi_entity_v3/metadata.json`: cache, split, checkpoint, environment, and limitation records.
- `models/multi_entity_v3/ablation.json`: post-freeze entity ablations and pair-attention diagnostic.
- `models/improved_v1/integrity/`: image manifest and deduplicated split indices.

## Important scope

The v3 artifact supports the audited historical feature caches. It does not claim reliable raw-image inference because the historical cache includes a randomly initialized normal head whose state was not saved, and two historical highlight calculations differ from the current extractor. `scripts/predict_multi_entity_v3.py` enforces that scope.

The current test fake images are another DiffusionDB subset. They do not demonstrate generalization to an unseen generator family. The test set was inspected during earlier project development, so an independent external generator benchmark remains necessary for a confirmatory thesis result.

The system should be described as **physics-inspired compact consistency features**, not as enforcing physical laws or as “100% pure physics.” V3 learns pairwise relationships among illumination, specular optics, surface-normal residuals, and chromatic shadows; these descriptors must be regenerated before claiming validated physical reasoning.

## Reproduce the multi-entity model

From `pipeline_40k`:

```powershell
..\.venv\Scripts\python.exe -B scripts\train_multi_entity_v3.py `
  --output-dir models\multi_entity_v3_reproduction
```

This command verifies the audited cache hashes, trains three multi-entity interaction models on deduplicated rows, selects the v2/entity blend on development data, fits the policy on calibration data, and evaluates the frozen artifact on the historical test partition.

Apply the promoted artifact to a compatible cache:

```powershell
..\.venv\Scripts\python.exe -B scripts\predict_multi_entity_v3.py `
  data\cache_scaled\test_scaled_features.pt `
  --row 0
```

Run tests:

```powershell
..\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

## Data integrity

The audit found 83 exact duplicate hash groups. The frozen split removes 67 duplicate training rows, one train/test overlap, and 19 duplicate test rows.

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Training | 13,600 | 13,535 | 27,135 |
| Development | 1,200 | 1,199 | 2,399 |
| Calibration | 1,200 | 1,199 | 2,399 |
| Test | 4,000 | 3,980 | 7,980 |

SHA-256 only finds byte-identical images. Perceptual and semantic near-duplicate checks are still required for a future external benchmark.

## Previous baselines

V3 uses the legacy transformer and the v2 histogram model as complementary branches. The transformer can still be trained directly:

```powershell
..\.venv\Scripts\python.exe -B scripts\train_scaled.py
```

Legacy evaluation now freezes calibration and decision thresholds on validation data before reading test labels:

```powershell
..\.venv\Scripts\python.exe -B scripts\evaluate_scaled.py
```

Run the v3 entity diagnostic after a frozen run:

```powershell
..\.venv\Scripts\python.exe -B scripts\diagnose_multi_entity_v3.py
```

The default legacy output names no longer overwrite the historical report.
