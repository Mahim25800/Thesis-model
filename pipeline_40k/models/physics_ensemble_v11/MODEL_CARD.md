# Regional Multi-Physics Ensemble v11

## Purpose

This artifact detects AI-generated images using compact, interpretable consistency measurements rather than semantic image embeddings. It is a research model for cross-generator evaluation and thesis experiments.

## Inputs

The model consumes a cached tensor with five regions per image: global, top-left, top-right, bottom-left, and bottom-right. Each region contains 14 measurements and four confidence values covering:

- illumination consistency;
- scene-wide specular optics;
- DSINE surface-normal consistency;
- chromatic-shadow consistency.

The raw-image extractor and its official DSINE checkpoint provenance are recorded in the cache metadata.

## Architecture

The artifact is a calibrated probability ensemble:

- regional multi-physics v5: weight 0.70;
- generator-invariant regional v6: weight 0.30;
- illumination-disabled v10: weight 0.00 after development selection.

The weights were selected using group-disjoint internal development data and held-out GenImage Wukong. The networks were trained with seeds 71, 73, and 79. The final decision threshold and abstention band were fit on the internal calibration split.

## Frozen performance

| Evaluation | Samples | Accuracy | ROC-AUC | EER |
|---|---:|---:|---:|---:|
| Internal unseen | 7,992 | 80.54% | 0.8892 | 19.23% |
| Synthbuster mean across nine generators | 18,000 | 64.00% | 0.7015 | 34.74% |

The weakest Synthbuster generator is Firefly at 0.6394 AUC. Stable Diffusion 2 and SDXL are strongest at 0.8293 and 0.7977 AUC.

A 5,000-repeat paired bootstrap gives a mean Synthbuster AUC gain over the global DSINE v4 model of +0.0783, 95% CI [+0.0720, +0.0843].

## Intended use

- academic evaluation of physics-inspired synthetic-image detection;
- generator-transfer and ablation experiments;
- analysis with an abstention option when physics evidence is ambiguous.

This artifact is not validated for forensic, legal, moderation, or fully autonomous decisions. Results depend on image source, resizing, compression, generator family, and the availability of physical cues.

## Known limitations

- Mean external performance is materially lower than internal performance.
- Firefly and Midjourney remain below the original 0.65 per-generator AUC gate.
- The DSINE model is pretrained for surface normal estimation, so the full pipeline is not training-data independent.
- Illumination is dataset-sensitive: masking it helped a frozen v6 ablation, while hard removal during retraining hurt development transfer.
- Synthbuster was observed during the research sequence. The separately frozen GenImage confirmation is required for a stronger confirmatory claim.

## Integrity

- Branch file hashes are embedded in `model.pt` and checked during ensemble evaluation.
- Standardization, calibration, and thresholds are stored with the branch artifacts.
- Difference-hash groups are separated across internal splits.
- The confirmatory evaluator aborts on overlap with training, calibration, development, or earlier internal-test caches.
