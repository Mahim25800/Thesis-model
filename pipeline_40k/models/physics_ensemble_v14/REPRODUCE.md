# Physics Ensemble v14 release

This directory is the frozen, strongest cross-generator artifact in this repository.

## Included model graph

`model.pt` is an ensemble manifest rather than a single neural-network state. It validates
SHA-256 hashes before loading these exact branch artifacts:

- `models/region_tta_v13/regional_v5_tta/model.pt` — weight 0.70;
- `models/region_tta_v13/generator_invariant_v6_tta/model.pt` — weight 0.30;
- `models/physics_masked_v10/model.pt` — retained in the manifest with zero selected weight.

Each branch contains its network weights, train-only standardizer, calibration policy and
region test-time-augmentation configuration. The ensemble manifest contains the selected
weights and final calibration policy.

## Install

Use Python 3.12 and install the pinned CUDA runtime:

```powershell
python -m pip install -r requirements-lock.txt
```

## Re-evaluate the released model

The original image datasets and caches are intentionally not distributed with this release.
To reproduce the recorded Synthbuster evaluation, build a cache with schema
`regional_physics_dsine_v2_5x14_features_5x4_confidences` at
`data/cache_synthbuster_regional_v2`, then run:

```powershell
python -B scripts/evaluate_physics_ensemble_v7.py `
  --model models/physics_ensemble_v14/model.pt `
  --cache-dir data/cache_synthbuster_regional_v2 `
  --output-dir models/physics_ensemble_v14/re_evaluation
```

The evaluator verifies every referenced branch file hash before inference. The frozen
published reports are `evaluation_internal.json`, `external_synthbuster/evaluation.json`,
and `genimage_confirmatory/evaluation.json`; paired per-image probabilities are retained
under their corresponding `predictions` folders.

## Claim boundary

The artifact achieved 0.7024 mean ROC-AUC over nine Synthbuster generators, but 0.6239
mean ROC-AUC on the disjoint GenImage confirmation. It is a research artifact, not a
universal or forensic deployment detector. See `MODEL_CARD.md` for the complete protocol
and limitations.
