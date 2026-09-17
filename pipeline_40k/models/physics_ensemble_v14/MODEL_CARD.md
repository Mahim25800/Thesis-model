# Regional Multi-Physics Ensemble v14

## Purpose

This research artifact detects AI-generated images from compact, interpretable physical-consistency measurements. It is intended for thesis experiments on transfer across image generators.

## Inputs and physics entities

Each image is represented by a global region and four aligned quadrants. Every region contains 14 measurements and four confidence values covering:

1. illumination consistency;
2. scene-wide specular optics;
3. surface-normal consistency from the official pretrained DSINE estimator;
4. chromatic-shadow consistency.

## Frozen architecture

V14 is a calibrated probability ensemble selected on group-disjoint internal development data and held-out GenImage Wukong:

- regional multi-physics v5 with horizontal and vertical spatial test-time averaging: weight 0.70;
- generator-invariant regional v6: weight 0.30;
- illumination-disabled v10: weight 0.00 after development selection.

The branch file hashes, preprocessing statistics, calibration, operating threshold, and spatial transformations are stored in the artifact.

## Performance

| Evaluation | Retained samples | Accuracy | ROC-AUC | EER |
|---|---:|---:|---:|---:|
| Internal unseen | 7,992 | 80.31% | 0.8895 | 19.16% |
| Synthbuster, mean over nine generators | 18,000 | 63.51% | 0.7024 | 34.66% |
| GenImage confirmation, mean over four generators | 7,998 | 58.28% | 0.6239 | 41.57% |

Synthbuster per-generator AUC ranges from 0.6410 on Firefly to 0.8310 on Stable Diffusion 2. Confirmatory GenImage AUCs are 0.7085 on GLIDE, 0.5709 on Midjourney, 0.6376 on SD v1.4, and 0.5785 on SD v1.5.

On the confirmation set, the frozen v6 branch reached 0.6330 mean AUC and significantly exceeded v14 by 0.0091 AUC, with a paired 95% bootstrap interval of [0.0016, 0.0169]. V6 was not promoted after observing confirmatory labels.

## Intended use

- academic evaluation of physics-inspired synthetic-image detection;
- cross-generator and cross-dataset transfer analysis;
- regional, entity, and domain-invariance ablations.

## Limitations and claim boundary

- The predeclared breakthrough gate was not met.
- External performance changes materially with the benchmark and generator family.
- The calibrated threshold is benchmark sensitive; ROC-AUC is the primary transfer metric.
- DSINE is pretrained for surface-normal estimation, so the pipeline is not training-data independent.
- Illumination is dataset-sensitive and can conflict with the other physical cues.
- The artifact is not validated for forensic, legal, moderation, or autonomous decisions.

The supported claim is a statistically validated improvement over this project's global physics baseline on Synthbuster, together with a documented cross-dataset generalization limit. It is not evidence of universal state-of-the-art performance.

## Integrity

- Model selection did not read Synthbuster or confirmatory GenImage labels.
- The GenImage confirmation used archive positions 4,001-6,000, outside the prior sample.
- Two conservative 64-bit difference-hash matches with protected data were excluded and recorded, despite distinct SHA-256 hashes.
- Per-image predictions and 5,000-repeat paired bootstrap results are stored beside the evaluation reports.
