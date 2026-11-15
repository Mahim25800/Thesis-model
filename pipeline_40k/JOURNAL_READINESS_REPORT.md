# Journal Readiness Report: Regional Multi-Physics Deepfake Detection

## Research contribution

This project studies whether physically interpretable image cues can generalize across unseen image generators without using semantic foundation-model embeddings or a pretrained deepfake classifier. The final detector combines four physics-inspired entities:

1. illumination consistency;
2. scene-wide specular optics;
3. surface-normal consistency from the official pretrained DSINE geometry estimator;
4. chromatic-shadow consistency.

Each entity is measured globally and in four aligned image quadrants. A regional transformer models spatial disagreement, and pair modules model contradictions between physical entities. The final frozen v14 candidate is a probability ensemble of:

- 70% regional multi-physics v5 with development-selected spatial test-time averaging;
- 30% generator-invariant v6, trained with a gradient-reversal generator-domain adversary.

The ensemble weight was selected on a group-disjoint internal development split and GenImage Wukong. Synthbuster was not used by the selection script.

## Main results

| Model | Internal accuracy | Internal AUC | Synthbuster mean accuracy | Synthbuster mean AUC | Worst generator AUC |
|---|---:|---:|---:|---:|---:|
| DSINE global multi-entity v4 | 75.86% | 0.8404 | 58.44% | 0.6232 | 0.5518 |
| Generator-invariant regional v6 | 74.54% | 0.8249 | 58.55% | 0.6297 | 0.5065 |
| Frozen physics ensemble v14 | **80.31%** | **0.8895** | **63.51%** | **0.7024** | **0.6410** |

Across nine Synthbuster generators, the final ensemble improved over v4 on every generator. A 5,000-repeat paired, generator-stratified bootstrap estimated:

- mean AUC improvement over v4: **+0.0783**, 95% CI **[+0.0720, +0.0843]**;
- mean accuracy improvement over v4: **+5.56 percentage points**, 95% CI **[+4.89, +6.24]**;
- mean AUC improvement over v6: **+0.0718**, 95% CI **[+0.0674, +0.0762]**;
- mean accuracy improvement over v6: **+5.45 percentage points**, 95% CI **[+4.86, +6.05]**.

These confidence intervals measure sampling uncertainty within the evaluated benchmark. They do not establish superiority over every published detector. They apply to v7, the pre-TTA version of the same 70/30 branch ensemble; v14 adds only fixed spatial averaging selected before final evaluation.

## Independent confirmatory result

After v14 was frozen, it was evaluated on a separately sampled GenImage set. The four generator domains use archive positions 4,001-6,000, outside the earlier development sample. Two conservative 64-bit difference-hash matches with protected data were excluded even though their SHA-256 hashes differed, leaving 7,998 observations.

| Frozen model | GLIDE AUC | Midjourney AUC | SD v1.4 AUC | SD v1.5 AUC | Mean AUC | Mean accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Generator-invariant v6 | **0.7698** | **0.6177** | 0.6070 | 0.5376 | **0.6330** | **59.57%** |
| Physics ensemble v14 | 0.7085 | 0.5709 | **0.6376** | **0.5785** | 0.6239 | 58.28% |

V14 was significantly weaker than v6 on this confirmation: mean AUC difference **-0.0091**, paired 95% bootstrap CI **[-0.0169, -0.0016]**. This result invalidates a universal breakthrough claim. V6 is reported as a diagnostic comparison and was not substituted as the final candidate after observing confirmatory labels.

## Physics ablation evidence

The frozen v6 ablation on Synthbuster produced the following mean AUC values:

| Input | Mean AUC | Change from full v6 |
|---|---:|---:|
| Full regional physics | 0.6297 | — |
| Global region only | 0.5710 | -0.0587 |
| Without illumination | 0.6615 | +0.0318 |
| Without specular optics | 0.6175 | -0.0122 |
| Without surface normals | 0.5506 | -0.0791 |
| Without chromatic shadows | 0.6106 | -0.0190 |

The strongest evidence is therefore:

- regional comparison is essential;
- DSINE surface-normal consistency is the most important entity;
- specular and chromatic evidence add complementary signal;
- illumination is dataset-sensitive and should be treated as a reliability-limited cue.

## Negative results retained for the thesis

Three predeclared follow-up experiments did not pass development selection:

- v8 learned per-image entity attention: internal AUC 0.8218, Wukong AUC 0.6277;
- v10 illumination-disabled adversarial training: internal AUC 0.8032, Wukong AUC 0.6213;
- v12 exhaustive fixed masks removing one or two entities: the unmasked v5 branch remained best.

These results show that added gating capacity and hard removal can both weaken transfer. The useful effect comes from diverse frozen branches and regional physics, rather than a larger fusion head.

## Evaluation protocol

- Near-duplicate control uses 64-bit difference hashes and group-disjoint splitting.
- Normal features use the official DSINE checkpoint with its source commit and SHA-256 recorded.
- Train-only standardization is stored in every model artifact.
- Calibration and decision thresholds are fit on a calibration split.
- Wukong is a development generator and is not reported as a final test.
- Synthbuster evaluation stores per-image predictions for paired bootstrap analysis.
- The completed confirmatory GenImage test uses GLIDE, Midjourney, SD v1.4, and SD v1.5 sampled from archive positions 4,001-6,000. Its evaluator excludes perceptual-hash matches with training, calibration, development, or prior internal-test caches and records every exclusion.

## Claim boundary

The current supported claim is:

> Regional reasoning, official DSINE geometry, and generator-invariant fusion improve the project’s prior global physics detector across nine Synthbuster generators, reaching 0.7024 mean AUC; a separate GenImage confirmation reaches 0.6239 mean AUC and identifies a material cross-dataset generalization limit.

The project did not satisfy its predeclared breakthrough gate. A “state of the art” claim would additionally require protocol-matched comparisons with published baselines. The work remains suitable for a thesis or journal submission framed around an interpretable multi-physics method, regional and entity ablations, strict leakage controls, statistically supported Synthbuster gains, and the measured confirmation failure.

## Reproducibility artifacts

- `BREAKTHROUGH_EXPERIMENT_PROTOCOL.md`: frozen experiment plan and claim gates.
- `models/physics_ensemble_v14/model.pt`: final development-selected candidate.
- `models/physics_ensemble_v14/MODEL_CARD.md`: final model scope, performance, and limitations.
- `models/physics_ensemble_v14/external_synthbuster/evaluation.json`: nine-generator metrics.
- `models/physics_ensemble_v14/genimage_confirmatory/evaluation.json`: disjoint four-generator confirmation and paired bootstrap comparisons.
- `models/physics_ensemble_v7/external_synthbuster/comparison_vs_v4_v6_bootstrap.json`: paired confidence intervals.
- `models/generator_invariant_v6/external_synthbuster/ablation.json`: physics ablation.
- `scripts/evaluate_genimage_confirmatory.py`: overlap-rejecting confirmatory evaluator.
- `reports/figures/regional_multi_physics_results.png` and `.pdf`: publication-ready Synthbuster and ablation figure generated from frozen JSON artifacts.
- `reports/figures/cross_dataset_confirmation.png` and `.pdf`: publication-ready visualization of the independent confirmation result.
