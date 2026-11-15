# Physics-plus-geometry experiment protocol

## Question

Does adding confidence-gated image-space projective geometry improve transfer beyond the frozen four-entity regional physics model?

## Added entity

The fifth entity consists of eight bounded descriptors for the global image and four aligned quadrants:

1. line density;
2. mean normalized line length;
3. line coverage;
4. line-orientation entropy;
5. dominant orientation-family support;
6. dominant vanishing-point-hypothesis support;
7. vanishing-point-hypothesis compactness;
8. cross-scale orientation agreement.

Its confidence depends on observable line structure and projective-hypothesis support. A low confidence replaces the geometry entity with the model's learned missing token. These values are evidence summaries; they do not prove that one image violates a physical law.

## Frozen roles

| Data | Role |
|---|---|
| Internal DiffusionDB/COCO cache | Primary source, development, calibration, and internal unseen test |
| GenImage ADM, BigGAN, VQDM | Additional source domains for adversarial-invariant training |
| GenImage Wukong | Held-out external development domain for early stopping and selection |
| Synthbuster + RAISE | Post-selection external evaluation only |
| Disjoint GenImage GLIDE, Midjourney, SD v1.4, SD v1.5 | Confirmatory evaluation only after the candidate is frozen |

## Decision rule

The new five-entity branch is trained without Synthbuster or confirmatory data. It is retained only if it improves the Wukong selection score without materially reducing group-disjoint internal development AUC. Synthbuster is then evaluated once with a geometry-masked post-freeze ablation. If it is promising, the disjoint GenImage confirmation is evaluated without further model changes.

## Claim boundary

A positive Synthbuster result alone does not establish universal generalization. The final claim must report the geometry ablation, the new confirmation result, all per-generator metrics, and failures on low-geometry images.
