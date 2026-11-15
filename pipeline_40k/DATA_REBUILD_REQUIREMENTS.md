# Data rebuild requirements for a validated multi-entity detector

The v3 result is valid only for the audited historical cache. A raw-image, externally generalizing detector requires a fresh cache build with these gates:

1. Train or adopt a licensed surface-normal model, save its exact weights, and record its source, license, hash, and preprocessing recipe.
2. Version the illumination, specular, shadow, and perspective extractors. Save feature names, code revision, image size, and numerical settings with every cache shard.
3. Recompute all feature columns from raw images. Never mix old and regenerated descriptor versions in one experiment.
4. Deduplicate by perceptual hash and source/identity group, in addition to SHA-256. Split groups before model fitting.
5. Build generator-held-out test sets and keep their labels inaccessible until model and calibration choices are frozen.
6. Report each entity-only score, all pairwise ablations, calibration quality, coverage, and cross-generator results.

Until these gates are complete, describe v3 as a multi-entity cached-feature research result rather than a validated physical-consistency detector.

## Implemented DSINE cache command

The official uncertainty-aware DSINE checkpoint is stored under `models/normal_estimators/dsine/exp002_kappa/`, with its provenance and SHA-256 hash beside it. Build a new cache without overwriting the historical benchmark:

```powershell
..\.venv\Scripts\python.exe -B scripts\cache_dsine_v1.py
```

The command expects raw `data/scaled/train/{real,fake}` and `data/scaled/test_unseen/{real,fake}` folders. It writes a distinct `data/cache_regenerated_dsine_v1/` directory with feature tensors, image manifests, hashes, and DSINE provenance.
