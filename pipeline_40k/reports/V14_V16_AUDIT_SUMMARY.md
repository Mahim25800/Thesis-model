# V14 Audit and V16 Raw-Augmentation Result

**Date:** 2026-09-24  
**Scope:** Frozen v14 descriptive audit and a separate rotated GenImage development comparison.  
**Selection integrity:** Neither the v16 control nor the raw-augmentation candidate read Synthbuster or the known GenImage confirmation labels.

## 1. Frozen v14 performance context

| Evaluation | Samples | Accuracy | ROC-AUC |
|---|---:|---:|---:|
| Group-disjoint internal unseen | 7,992 | 80.31% | 0.8895 |
| Synthbuster, mean of nine generators | 18,000 | 63.51% | 0.7024 |
| Disjoint GenImage confirmation, mean of four generators | 7,998 | 58.28% | 0.6239 |

The Synthbuster generator AUC range is 0.6410 on Firefly to 0.8310 on Stable Diffusion 2. The confirmation result remains the central transfer limitation: v14 is not a universal detector and is not a state-of-the-art claim.

## 2. Frozen v14 mechanism audit

The v14 ensemble contains regional v5-TTA at 70% and generator-invariant v6-TTA at 30%. The figures below are the change in ROC-AUC when a confidence-gated entity or region is removed. Positive values mean the full model performed better.

### Entity and regional reliance

| Ablation | Internal unseen AUC change | Wukong development AUC change | Interpretation |
|---|---:|---:|---|
| Remove illumination | -0.0293 | -0.0435 | Useful on both audited domains. |
| Remove specular optics | -0.0060 | -0.0357 | Small internally, useful on Wukong. |
| Remove surface normals | -0.1100 | -0.0153 | Dominant internal cue; Wukong effect is uncertain. |
| Remove chromatic shadows | -0.0535 | -0.0757 | Strongest Wukong cue. |
| Global region only | -0.0839 | -0.0421 | The four spatial quadrants add material signal. |
| Remove top-left quadrant | -0.0109 | -0.0110 | Small but reliable contribution. |
| Remove top-right quadrant | -0.0105 | -0.0129 | Small but reliable contribution. |
| Remove bottom-left quadrant | -0.0081 | +0.0042 | No reliable Wukong contribution. |
| Remove bottom-right quadrant | -0.0045 | +0.0038 | No reliable Wukong contribution. |

The 95% paired bootstrap interval excludes zero for every listed internal entity ablation. On Wukong, the surface-normal and bottom-quadrant changes have intervals containing zero; this warns against treating a single physical cue or spatial location as universally reliable.

The active v5/v6 branch-probability correlation is 0.7924 internally and 0.6349 on Wukong. The branches are related but not identical; the ensemble is not merely duplicating one score.

## 3. Rotated unseen-generator evaluation

Each row trains on the internal fitting split plus three GenImage domains, then holds the named generator out entirely for development. Candidate ranking is: maximize minimum held-generator ROC-AUC, then mean held-generator ROC-AUC.

| Held-out generator | V16 control AUC | V16 raw-augmentation AUC | Change |
|---|---:|---:|---:|
| ADM | 0.7356 | 0.7224 | -0.0131 |
| BigGAN | 0.8219 | 0.8230 | +0.0011 |
| VQDM | 0.7842 | 0.7830 | -0.0011 |
| Wukong | 0.6322 | 0.6566 | +0.0244 |
| **Minimum** | **0.6322** | **0.6566** | **+0.0244** |
| **Mean** | **0.7434** | **0.7463** | **+0.0028** |

The raw-augmentation recipe appended one deterministic non-identity copy of each fitting image. Each copy used exactly one seeded transform: JPEG recompression (quality 70–100), Gaussian blur (sigma 0.05–1.0 px), or downscale-upscale jitter (factor 0.65–1.0). Original images remained in the fitting set. Calibration, development, internal test, Synthbuster, and confirmation rows were never augmented.

## 4. Result and decision

The raw-augmentation candidate passes the prespecified primary gate because it improves the **worst** held-generator AUC by 0.0244. The improvement is concentrated in the limiting Wukong domain; ADM falls by 0.0131, while BigGAN and VQDM are effectively unchanged. This is a credible robustness improvement, not evidence of universal performance.

The candidate must still complete the planned illumination comparison, v6 refinement, restrained ensemble selection, and a fresh protected external holdout. Do not evaluate or tune this candidate on the already-read Synthbuster or GenImage confirmation labels to choose the next model.

## Reproducible artifacts

- `models/physics_ensemble_v14/robustness_audit.json`
- `models/multi_generator_v16_control/selection_summary.json`
- `models/multi_generator_v16_raw_augmentation/selection_summary.json`
- `scripts/diagnose_regional_ensemble.py`
- `scripts/cache_training_augmentations_v16.py`
- `scripts/select_multi_generator_robust.py`
