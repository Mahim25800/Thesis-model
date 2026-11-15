# Multi-physics generalization experiment

## Research question

Can localized interactions among illumination, specular highlights, DSINE surface-normal residuals, and chromatic shadows detect generated images from generator families that were absent from training?

The experiment tests transferable detection. Internal accuracy alone is insufficient because the v4 model reached 75.86% internally but only about 0.599 mean ROC-AUC across the distinct DALL-E, Firefly, GLIDE, and Midjourney families in Synthbuster.

## Frozen data roles

| Data | Role | Permitted use |
|---|---|---|
| DiffusionDB fake + COCO real training partition | Source domain | Fit model and train-only feature standardization |
| Internal development partition | Development | Early stopping and architecture comparison |
| Internal calibration partition | Calibration | Platt scaling, binary threshold, and abstention band |
| Internal unseen partition | Internal test | Report after model selection |
| GenImage ADM, BigGAN, VQDM | Additional source domains | Fit model and domain adversary |
| GenImage Wukong | Held-out external development generator | Model selection only; no gradient updates |
| Synthbuster + RAISE | Final external test | One frozen evaluation after all selection and calibration |
| GenImage GLIDE, Midjourney, SD v1.4, SD v1.5, archive positions 4,001–6,000 | Confirmatory external test | Frozen evaluation only; reject any perceptual-hash overlap with all earlier splits |

Difference-hash groups remain disjoint where the data permit it. Wukong development rows that overlap source-training difference-hash groups are excluded. Synthbuster is never read by either training script.

## Model changes

1. **Localized physics:** each image yields one global token and four quadrant tokens. Every token contains the same four physics entities and four confidence values.
2. **One DSINE pass:** DSINE predicts the complete normal and concentration fields once. Region summaries use aligned crops from those fields, preserving scene context and GPU efficiency.
3. **General specular regions:** the new cache finds scene-wide high-frequency highlight components. It does not assume fixed eye coordinates on arbitrary images.
4. **Entity and pair fusion:** each entity has a shared encoder. Six pair tokens describe within-region entity agreement. A region transformer then compares global and local consistency.
5. **Generator invariance:** a gradient-reversal domain classifier tries to recover the source generator from the physics embedding while the physics encoder learns to prevent that recovery. The classification loss remains the primary objective.
6. **Robustness augmentation:** training randomly flips spatial tokens, removes regions and entities, and adds small standardized feature noise.

## Required comparisons

| Experiment | Purpose |
|---|---|
| v4 global DSINE model | Frozen baseline |
| v5 regional model on original 40k data | Isolate the value of localization |
| v6 multi-source regional model without domain loss | Isolate the value of generator diversity |
| v6 full model with domain loss | Test generator-invariant learning |
| Remove each physics entity from the frozen v6 model | Verify that the claimed entities contribute |
| Global token only versus global + quadrants | Verify that local contradictions contribute |

The initial automated run produces v4, v5, and full-v6 comparisons. The no-domain-loss and frozen entity-removal ablations must be completed before a thesis claim is finalized.

## Metrics and success gate

Report ROC-AUC, equal-error rate, accuracy at the calibration-selected threshold, per-generator results, and selective accuracy with coverage. Accuracy is secondary because it depends on a threshold and class prevalence.

A breakthrough claim requires all of the following:

- Mean Synthbuster ROC-AUC of at least 0.70 on generator families absent from training.
- No major unseen generator family below 0.65 ROC-AUC.
- Improvement over v4 with a paired bootstrap confidence interval that excludes zero.
- No material collapse on the internal unseen split.
- Entity and spatial ablations showing that performance is not carried by one shortcut feature.
- Complete checkpoint, dataset, cache, split, and environment provenance.

If these conditions are not met, report the result as a stronger multi-physics baseline and use the per-generator failures to define the next experiment. Do not describe it as state of the art or a breakthrough.

## Frozen outcome recorded 2026-09-17

The final Wukong-selected v14 ensemble uses 70% spatially averaged v5 and 30% v6. It reached 0.7024 mean ROC-AUC and 63.51% mean accuracy across nine Synthbuster generators. Internal unseen performance remained strong at 0.8895 AUC and 80.31% accuracy. The pre-TTA v7 ensemble had a +0.0783 mean AUC gain over v4, with a 5,000-repeat paired 95% bootstrap interval of [+0.0720, +0.0843].

The worst Synthbuster generator was Firefly at 0.6410 AUC, so the predefined 0.65 minimum-generator gate was missed. Learned reliability attention, illumination-disabled retraining, and fixed one/two-entity masks did not improve the Wukong selection score and received zero final weight.

The frozen GenImage confirmation retained 7,998 images across GLIDE, Midjourney, SD v1.4, and SD v1.5 after two conservative perceptual-hash exclusions. V14 reached 0.6239 mean AUC and 58.28% mean accuracy. The previously frozen v6 branch reached 0.6330 mean AUC; v14 minus v6 was -0.0091 with paired 95% CI [-0.0169, -0.0016]. The confirmation therefore failed, and the experiment does not meet the breakthrough definition above.
