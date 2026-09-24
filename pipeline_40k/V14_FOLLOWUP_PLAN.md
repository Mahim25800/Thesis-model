# V14 Follow-up Plan: Robust Cross-Generator Physics Detection

**Basis:** external review feedback received 2026-09-24, reconciled with frozen v14/v15 artifacts.  
**Scope:** `pipeline_40k` only.  
**Objective:** improve out-of-distribution transfer without weakening the thesis’s leakage controls or turning frozen benchmarks into tuning data.

## Decision first: what the review gets right and what is already done

| Review item | Assessment against current artifacts | Decision |
|---|---|---|
| Raw-image JPEG, blur, and resize robustness augmentation | Absent from the current DSINE/physics cache build. Existing `augment_regions` only flips regional tokens, simulates missing entities, and adds feature-space noise after extraction. | **Implement after validation protocol is fixed.** |
| Train v5 without illumination | The post-freeze v6 external ablation improved by +0.0318 mean AUC when illumination was masked, but the v10 illumination-disabled adversarial model lost Wukong development performance. No direct v5-without-illumination training exists. | **Run as a prespecified candidate, not an assumed improvement.** |
| One model with v5 regional design and v6 domain adversary | Already implemented: v6 uses the same regional transformer/pair-fusion design as v5 and adds gradient reversal with a generator-domain classifier. | **Do not duplicate. Improve the existing v6 training family instead.** |
| Shortcut diagnostic for v5/v6/v14 | Outstanding. The current `diagnose_shortcuts.py` loads the older `TransformerPhysicsCrossGenHead`, so it cannot audit v5, v6, or the v14 ensemble. | **Highest immediate engineering task.** |
| Multiple held-out development generators | Current v14 selection uses only Wukong, and the confirmation result showed that this is not sufficiently robust for cross-dataset selection. | **Highest immediate experimental-design task.** |
| Physics plus CLIP/DINO-style semantic fusion | A legitimate future hybrid baseline, but it changes the thesis’s central pure-physics claim and adds a large pretrained semantic model. | **Defer until the pure-physics track has a stronger protocol and result.** |

## Priority order

### 1. Build an ensemble-aware shortcut and robustness audit

This is the first task because a new cache or training run cannot be interpreted if v14’s performance is carried by a source shortcut, a single entity, or one spatial region.

Create `scripts/diagnose_regional_ensemble.py`. It must load `physics_ensemble_v14/model.pt`, verify branch hashes, and evaluate the actual v14 score. It will report, without fitting or threshold selection:

- internal unseen, each GenImage development generator, and per-generator Synthbuster descriptive metrics;
- each entity removed by setting its confidence to zero;
- global-only versus full five-region input;
- each regional quadrant removed;
- score distributions by source, resolution, JPEG quality when available, and observability;
- branch-wise correlation and disagreement, so the contribution of v5-TTA and v6 is visible;
- bootstrap confidence intervals for each ablation delta.

**Acceptance gate:** no candidate can be promoted if one entity, source metadata field, or spatial region alone explains the apparent transfer gain, or if the audit cannot reproduce the frozen v14 predictions from the saved artifact.

### 2. Replace one-generator selection with multi-generator leave-one-domain-out selection

This is the highest-impact design correction. The model should no longer be chosen by Wukong alone.

Use GenImage ADM, BigGAN, VQDM, and Wukong in rotated roles. For every candidate architecture and augmentation setting:

1. train on the permitted source domains and the internal training partition;
2. hold out each GenImage generator in turn for development;
3. calculate per-held-generator ROC-AUC using a common score direction;
4. rank candidates by the **minimum** held-generator AUC, then the mean held-generator AUC;
5. fit calibration only after the candidate is selected.

Keep every known Synthbuster and first confirmation label out of this selection. The repository evidence confirms that `data/cache_genimage_confirmatory_disjoint_regional_v2` was already evaluated by v14, so it is not an untouched final test for this follow-up. Create and freeze a fresh documented external holdout before making a new transfer claim.

**Acceptance gate:** the selected candidate must improve the minimum rotated-held-generator AUC relative to v14’s development selection, without a material internal-unseen collapse.

### 3. Add training-only raw-image robustness augmentation and rebuild feature caches

This is the review’s highest-leverage model-change proposal, but it must be introduced only after the selection protocol above exists.

Implement an auditable augmentation stage in the DSINE regional cache builder:

- apply the same label-preserving transformation family to real and fake training images only;
- use seeded JPEG recompression, mild Gaussian blur, and resize/downsample-upsample jitter before DSINE and all physics extractors;
- keep originals plus one deterministic augmented variant per training image in the first experiment;
- never augment internal development, calibration, internal test, held-generator development, Synthbuster, or confirmation images;
- store origin image hash, transform type, parameters, seed, and parent sample group in the manifest;
- keep all variants of an image inside the fitting partition, never across a split boundary.

Initial conservative ranges:

| Transform | Training distribution |
|---|---|
| JPEG recompression | quality uniformly sampled from 70–100 |
| Gaussian blur | sigma uniformly sampled from 0–1.0 px |
| Resize jitter | downscale factor uniformly sampled from 0.65–1.0, then restore with Lanczos/area interpolation |

Start with one transform or the identity per augmented copy, rather than composing all degradations. This makes the ablation interpretable and avoids destroying the physics signal itself.

**Acceptance gate:** multi-generator development minimum AUC must improve or stay within 0.005 AUC of the no-augmentation control while the new model improves its paired robustness test on recompressed/resized validation copies. Otherwise discard the change.

### 4. Run direct v5 illumination ablations under the new multi-generator protocol

Train three prespecified regional v5 candidates from identical data and seeds:

1. full four-entity v5 control;
2. illumination confidence masked to zero during all train and inference passes;
3. illumination dropout, where the illumination confidence is randomly zeroed during fitting but remains available at inference.

The third candidate is preferable to simply deleting illumination because the current evidence says illumination can help on some domains while harming others. It lets the model learn not to depend on it excessively.

**Acceptance gate:** retain an illumination variant only if it improves the minimum held-generator development AUC and does not rely on a threshold change alone. Record entity ablations again after the candidate freezes.

### 5. Train the improved regional domain-adversarial v6 family, then select a simple ensemble only if justified

Do not create a separate “merged v5/v6” architecture because v6 already is that merger. Instead, train the current v6 architecture with:

- multi-generator rotated development selection;
- the accepted raw-image augmentation recipe, if any;
- the accepted illumination strategy, if any;
- the existing gradient-reversal source-domain loss;
- fixed seed set and early-stopping policy.

Compare:

- regional v5 control;
- improved regional v5;
- existing v6 control;
- improved v6;
- a small set of convex v5/v6 mixtures.

Only choose an ensemble if it beats both individual branches on the multi-generator minimum-AUC criterion. Do not search a large unconstrained weight grid after external tests have been opened.

### 6. Create a new final holdout, then freeze, audit, and evaluate once

Create a fresh external holdout with no shared source positions or perceptual-hash groups, and keep its labels closed throughout selection. Freeze the winning candidate before opening that holdout.

For the winning candidate:

1. save model/checkpoint hashes, extractor revisions, cache provenance, split manifests, calibration parameters, and selection table;
2. run the new ensemble-aware shortcut audit;
3. evaluate exactly once on the protected external final holdout;
4. run paired bootstrap comparisons against v14, stratified by generator;
5. update the thesis report with both gains and failures.

A new breakthrough claim requires at least:

- mean final-holdout ROC-AUC at least 0.70;
- no final generator below 0.65 ROC-AUC;
- a paired confidence interval for the v14 comparison excluding zero in the positive direction;
- a robust multi-generator development result, not a one-generator win;
- entity/spatial ablations and shortcut audit that support the claimed mechanism.

### 7. Optional later research track: hybrid semantic-plus-physics model

Keep this separate from the pure-physics thesis headline. A frozen CLIP or DINOv2 embedding can be fused late with the physics score, using the same multi-generator selection and untouched final test. Report it as a hybrid baseline and quantify whether it adds complementary transfer performance over v14. Do not call any improvement “physics-based” unless the physics-only branch is reported separately.

## Concrete next deliverables

1. `scripts/diagnose_regional_ensemble.py` and a frozen v14 diagnostic report.
2. `scripts/select_multi_generator_robust.py` with rotated-held-generator selection and a saved candidate table.
3. A training-only raw-augmentation cache builder with explicit provenance.
4. v5 full/masked/dropout illumination comparison.
5. Improved v6 and a restrained ensemble selection.
6. A freshly created protected external holdout, one final evaluation, and an updated thesis report.

## Why this order is important

Raw augmentation is likely the most promising **model improvement**, but the diagnostic and multi-generator selection steps are more important **first**. The v14 confirmation already showed that strong internal performance and one held generator do not guarantee transfer. Fixing the test and selection method first prevents the project from spending compute optimizing another score that will not generalize.




