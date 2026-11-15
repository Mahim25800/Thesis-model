# Thesis project assessment and improvement plan

Reviewed: 16 September 2026

## Overall assessment

This is a working research prototype for classifying real and AI-generated still images using compact, physics-inspired features. Its strongest assets are the modular extractor design, inexpensive cached-feature experiments, confidence-aware fusion, and an explicit abstention mechanism.

The highest-value improvements are **experimental validity, physical validity of the features, and reproducibility**. Increasing the transformer size or training longer should come after these foundations. The current evidence does not establish a “100% pure physics” detector, physical law enforcement, or generalization to unseen generator families.

Recommended working thesis title:

**Physics-Inspired Multimodal Consistency and Selective Classification for AI-Generated Image Detection**

This assessment covers the repository, raw-image file hashes, cached tensors, checkpoint inference, and small diagnostic experiments. No thesis manuscript was present at the project root. Existing models, images, caches, and source code were not changed. The added audit script writes only its own report.

## 1. What was verified

Reproduction command, from `G:\Thesis`:

```powershell
.\.venv\Scripts\python.exe -B reports/audit_project.py
```

Evidence: [audit results](G:/Thesis/reports/project_audit.json), [audit script](G:/Thesis/reports/audit_project.py).

| Check | Observed result | Interpretation |
|---|---|---|
| Raw scaled images | 40,000: 16,000 real + 16,000 fake training; 4,000 + 4,000 test | The claimed file count exists |
| Training cache | 32,000 × 14 features; 32,000 × 4 confidences | Four measured modalities |
| Validation cache | 4,800 rows, all present in the full training cache | Consistent with the intended 15% split; it is not another 4,800 independent images |
| Test cache | 8,000 × 14 features; 8,000 × 4 confidences | Perspective is absent from the actual cached inputs |
| Cache numerical values | All finite | Basic numerical integrity passes |
| Current transformer | 75,457 parameters | Small enough for inexpensive cached-feature experiments |
| Current checkpoint, test AUC | 0.788028 | Reproduces the scaled-directory report |
| Current checkpoint, accuracy at 0.50 | 71.7875% | Reproduces the scaled-directory report |
| Threshold selected using saved validation | 0.521536; test accuracy 70.8875% | Diagnostic comparison; not a fresh benchmark |
| SHA-256 image audit | 39,913 unique files; 83 duplicate groups | Duplicate handling is needed |
| Exact train/test overlap | One test fake is byte-identical to one training fake | Confirmed leakage, but too small alone to explain the overall result |
| Cross-label exact duplicates | None found | Does not verify dataset provenance or detect near-duplicates |
| Single-example focal loss | Raises shape mismatch | Unrestricted `squeeze()` breaks batch size one |
| Pilot training loss call | Raises `AttributeError` because model returns a tuple | Root pilot trainer is incompatible with the current classifier alias |

The image audit detects identical bytes. It does not exclude recompressed duplicates, related prompts, repeated scenes, or semantic overlap. The cache lacks image IDs, so correspondence between every feature row and image cannot be independently certified.

### Small baseline experiment

Fixed logistic regression with training-only standardization, without test tuning:

| Inputs | Test AUC | Accuracy at 0.50 |
|---|---:|---:|
| Confidence scores only | 0.5420 | 51.43% |
| Fourteen features only | 0.7135 | 65.36% |
| Features + confidences | 0.7551 | 68.76% |
| Existing transformer checkpoint | 0.7880 | 71.79% |

The baseline excludes every training row matching a saved validation row, leaving 27,182 rows; this is slightly smaller than the intended 27,200 because duplicate feature rows are also excluded. These exploratory results are not a controlled architecture comparison or a significance test. They indicate that simple baselines are useful and confidence-only linear separation is weak on this split. They do not rule out nonlinear shortcuts or dataset bias.

## 2. Critical findings and required changes

### P0 — Make the evaluation independent of test labels

**Evidence:** [evaluate_scaled.py:164](G:/Thesis/pipeline_40k/scripts/evaluate_scaled.py:164) selects Youden's threshold from test labels. The calibrated threshold and observability-subset thresholds are also selected on test labels. Abstention bands are centered on this test-selected threshold. [train_scaled.py:276](G:/Thesis/pipeline_40k/scripts/train_scaled.py:276) repeats test-based threshold optimization. Setting `val_split=0` makes test data the model-selection set.

**Why it matters:** This makes the reported tuned decision accuracy optimistic as an estimate of a policy fixed before deployment. Computing test AUC and descriptive EER is legitimate; choosing an operational threshold using test outcomes is the problem. The amount of bias cannot be inferred from code alone.

**Plan:**

1. Use separate training, model-selection validation, calibration, and final test partitions, grouped by duplicate/scene/prompt where possible.
2. Fit feature normalization only on training data. Select model/hyperparameters on validation data.
3. Fit the probability calibrator and choose thresholds and abstention settings using held-out development data. Use additional splitting or cross-validation within development data when selecting among policies.
4. Serialize the chosen policy with the checkpoint. Test evaluation must only apply it.
5. Remove the test-as-validation fallback and training-cache calibration fallback.
6. Treat the currently inspected test set as development evidence for future work. Reserve a new untouched generator/source holdout for the final thesis claims.

**Acceptance:** Changing test labels cannot change the saved threshold, calibration coefficients, checkpoint choice, or per-image predictions. The evaluated and deployed policies are identical. See the official guidance on [threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html) and [disjoint calibration data](https://scikit-learn.org/stable/modules/calibration.html).

### P0 — Repair and document dataset provenance

**Evidence:** [download_scaled.py:193](G:/Thesis/pipeline_40k/scripts/download_scaled.py:193) uses `diffusiondb_2m_random_50k` for training fakes and `diffusiondb_random_10k` for test fakes. Its comments mention FLUX/Midjourney, but this code does not fetch those generators. DiffusionDB is based on Stable Diffusion; different subsets alone do not establish a new generator family. [DiffusionDB paper](https://arxiv.org/abs/2210.14896).

Other ingestion defects:

- The generic exception handler can substitute DiffusionDB even when downloading the real partition, retaining the real label. This is a code-path risk, not evidence it occurred in the existing corpus.
- Resume skips output filenames but restarts the source iterator, allowing earlier samples to be saved under later filenames.
- Stream exhaustion restarts the source, permitting repetition.
- Existence/size checks are weaker than the documented atomic, verified resume behavior.
- Confirmed exact overlap: `test_unseen/fake/fake_001703.jpg` and `train/fake/fake_006434.jpg`.

**Plan:** Create a manifest with stable image ID, source dataset and revision, source row, generator and version when available, prompt/scene group, label, original dimensions, encoding, SHA-256, perceptual hash, and split. Never substitute a different source silently. Resume by source ID. Write atomically and record failed items explicitly. Deduplicate before grouped splitting.

Use multiple real-photo sources and genuinely different generator families. Match or stratify content, resolution, and encoding across classes. Re-encoding all images once as JPEG does not erase their previous compression histories. Published work documents JPEG and size biases in these benchmarks. [Fake or JPEG?](https://arxiv.org/abs/2403.17608).

**Acceptance:** Verified source labels, no exact cross-split duplicates, reviewed near-duplicate groups, reproducible manifests, and a train-generator × test-generator matrix. GenImage provides a relevant cross-generator and degraded-image protocol, but its biases still need controls. [GenImage](https://arxiv.org/abs/2306.08571).

### P0 — Replace or remove the untrained normal estimator

**Evidence:** [surface_normals.py:57](G:/Thesis/pipeline_40k/src/extractors/surface_normals.py:57) loads a pretrained classification backbone, then constructs `FoundationNormalHead` with random weights. There is no normal-head checkpoint loading or supervised normal training in this path. Calling `.eval()` does not train the head. Two fresh heads produced substantially different outputs on the same feature tensor in the audit.

**Why it matters:** These outputs are not validated surface normals. A cached feature vector may be repeatable when read from disk while extraction changes between fresh runs. The classifier checkpoint does not preserve the randomly created extractor state. A silent download/load failure also switches the feature definition to Sobel gradients.

**Plan:** First disable this cue and establish an honest reduced-cue baseline. Then either use a frozen, task-trained normal model with versioned weights or limit geometry estimation to a domain where it can be validated. DSINE is a candidate to evaluate, not a guaranteed fix. [Surface normal estimation paper](https://arxiv.org/abs/2403.00712).

Validate angular normal error on known geometry and shading residual behavior on controlled scenes. Bilateral filtering alone should be described as a reflectance approximation, not verified recovery of intrinsic albedo. Regenerate affected caches and retrain after changing the extractor.

**Acceptance:** Extractor identity and weight hashes are recorded; repeated extraction agrees within stated tolerance; no silent model substitution; normal accuracy is measured independently of real/fake classification. If learned estimators are used, describe the method as physics-inspired with learned perception.

### P1 — Align feature claims with measurements

| Cue or claim | Current implementation | Improvement |
|---|---|---|
| Corneal optics | Fixed rectangular crops, not eye detection | Detect faces and eye landmarks; check pose, occlusion, and resolution; abstain when unavailable |
| General specular branch | At 256 × 256, both fixed crops are 51 × 71, so they pass the 24-pixel gate even in non-face scenes | Gate on detected anatomy or materials, not crop size |
| General highlight matching | Arbitrary reflective surfaces need not share highlight placement or profile | Establish comparable geometry/material and illumination assumptions, or call it a brightness descriptor |
| Chromatic shadows | Global bright/dark percentiles can select different objects/materials | Segment actual shadow/lit pairs on comparable surfaces; handle colored illumination and clipping |
| SH illumination | Fits lighting using normals derived from the same intensity gradients | Validate against known geometry and lighting; report fit residuals and conditioning; treat current outputs as proxies |
| Perspective | Extractor exists but is absent from the pipeline and caches | Keep a truthful four-cue baseline; integrate perspective only after geometric validation |
| Vanishing-point consistency | Intersects many unrelated lines, splits by median y, fits a horizon to intersections | Cluster line families and robustly estimate vanishing directions; support points at infinity and distinguish horizontal from vertical directions |
| Physical coupling | Attention mixes scalar summaries; inputs contain no shadow direction or aligned surface-plane measurements | Retain spatial/directional measurements and compute explicit pairwise residuals, or describe this as statistical fusion |

References in code: [eye crops](G:/Thesis/pipeline_40k/src/data/preprocessor.py:111), [shadow masks](G:/Thesis/pipeline_40k/src/extractors/chromatic_shadow.py:46), [gradient-derived normals](G:/Thesis/pipeline_40k/src/extractors/illumination_sh.py:42), [four-cue pipeline](G:/Thesis/pipeline_40k/src/models/pipeline.py:72), [missing fifth token](G:/Thesis/pipeline_40k/src/models/cross_gen_gated.py:173), [perspective heuristic](G:/Thesis/pipeline_40k/src/extractors/perspective_vp.py:116).

The fifth mask token is learnable but constant across images with the current schema; it cannot supply measured perspective evidence. Self-attention does not by itself impose a physical equation.

**Acceptance:** Every included cue has documented assumptions, a validity mask, positive and negative controls, and tests showing sensitivity to the intended violation rather than only image content or contrast. Missing cues should be allowed to have zero confidence.

### P1 — Make selective predictions and reporting consistent

The main README's 92.74% decided accuracy refers to only 15% coverage, with 85% abstention, in an older run. Report both quantities together. The current checkpoint and updated caches produce different results.

[pipeline.py:110](G:/Thesis/pipeline_40k/src/models/pipeline.py:110) uses a fixed 0.5 threshold plus observability gating, while the evaluator uses probability bands. These are different decision rules. The exported “authentic accuracy” is precision among predictions called authentic (negative predictive value under fake=positive), not recall across all real images. Some insufficient-subset branches copy overall metrics into the subset report, and the text formatter always prints threshold 0.50.

**Plan:** One shared, saved decision policy; accurate metric names; explicit undefined values for empty/single-class subsets; risk–coverage curves; class/source-specific coverage; false-positive rates; Brier score, reliability diagrams, and ECE with specified bins. Do not label an uncalibrated score a trustworthy probability. Measure inference time and extraction failure rates as well as accuracy.

### P1 — Repair experiment provenance and software consistency

- Consolidate the duplicated `src/` trees into one importable package. Keep pilot/scaled differences in consumed configuration files.
- Fix pilot tuple-output handling, replace dimension-unsafe `squeeze()` calls, and validate feature dimensions, confidence ranges, labels, and finite values.
- Make preprocessing identical for caching and inference. Initial caching resizes to 256 × 256; the SH-update script and raw inference use native resolution. This changes gradient scale and region measurements.
- Add cache schema version, ordered image IDs, preprocessing specification, extractor commit/weight hashes, and split hashes. Refuse stale/incompatible shards.
- Resume caches by image ID, not total completed row count. Quarantine corrupt images instead of substituting black images with valid labels.
- Persist configuration, epoch, seed, optimizer/scheduler state, normalizer, calibrator, policy, and dependency versions with each run. Retain legacy artifacts under explicit run IDs.
- Pin a tested environment and publish a clean-environment reproduction command. The current environment is CPU-only, so prioritize cached experiments and pilot extraction benchmarks before scaling costly perception models.
- Add an actual license document and dataset attribution/usage records; the README's “Academic Research License” is not a complete license specification.

**Acceptance:** A clean checkout reproduces the documented run; caches fail clearly on schema changes; interrupted extraction resumes without duplicates or reordered rows; single-image inference matches cached inference after the same preprocessing.

## 3. Research questions and experiment matrix

Recommended research questions:

1. Do validated physical consistency cues generalize across held-out generator families and real-image sources?
2. Does observability-aware fusion improve performance at a fixed coverage compared with simple fusion?
3. Does abstention reduce errors under corruptions and missing cues without disproportionately rejecting certain sources or scene types?
4. Do explicit cross-cue residuals contribute beyond independent per-cue statistics?

| Experiment | Required comparison | Purpose |
|---|---|---|
| Simple classifiers | Standardized logistic regression, small MLP, tree baseline, transformer | Justify model complexity |
| Loss functions | BCE; focal; BCE + SupCon; focal + SupCon | Attribute improvements to individual choices |
| Cue ablations | Each cue alone, leave-one-out retraining, incremental combinations | Establish feature utility |
| Gating | No gating, fixed confidence, learned/validated confidence, modality dropout | Isolate confidence value |
| Shortcut controls | Confidence-only; color/contrast/size/encoding controls; matched content | Identify dataset-specific alternatives |
| Generalization | Held-out generator families and held-out real-photo sources | Support cross-domain claims |
| Robustness | Shared JPEG qualities, downsampling, blur, crops, exposure and white balance changes | Measure realistic degradation behavior |
| Physical validity | Known-light/geometry scenes; controlled violations; no-cue scenes | Check whether residuals measure their intended mechanism |
| Selective prediction | Matched coverage, fixed development-selected thresholds | Compare risk fairly |
| External comparator | Reproduce an established image-based detector on the same splits | Put the compact-feature tradeoff in context |

UniversalFakeDetect is one appropriate established comparator; keep it separate from the proposed compact-feature method. [Paper](https://arxiv.org/abs/2302.10174).

Start with three seeds and use five for final shortlisted comparisons if feasible. Report mean and standard deviation across training seeds, and bootstrap confidence intervals over independent image groups. Use paired comparisons on the same test images. Current inference-time token masking is a sensitivity analysis; complement it with retrained ablations before claiming a cue's causal contribution.

Predefine the primary metric and protocol before final testing. Suggested primary metric: macro-average ROC-AUC across held-out generators, supplemented by worst-generator AUC, real-photo false-positive rate, and selective risk at fixed coverage. Report PR-AUC and prevalence assumptions where relevant. An arbitrary accuracy target should not determine whether the thesis is successful.

## 4. Prioritized delivery plan

Illustrative schedule for one researcher: approximately 4–6 weeks, depending on dataset access and extraction hardware. These are planning estimates, not measured runtimes.

| Phase | Work | Deliverable and exit condition |
|---|---|---|
| 1: First 2–3 days | Preserve current run; correct claims; remove test tuning; unify metric/policy logic; fix critical runtime bugs | Traceable baseline and tests proving policy independence from test labels |
| 2: Remainder of week 1 | Source manifest, safe downloader/resume, exact and perceptual deduplication, grouped splits, fresh generator holdout | Auditable data protocol with explicit overlap checks |
| 3: Week 2 | Remove random normal head; validate existing cues on a small controlled set; unify preprocessing; define cache schema | Small, repeatable, defensible feature pipeline |
| 4: Week 3 | Re-extract versioned data; run simple baselines, loss and retrained cue ablations | Results identifying which cues and architecture choices help |
| 5: Week 4 | Cross-generator/source evaluation, corruption tests, calibration and risk–coverage study | Locked final benchmark with uncertainty and error analysis |
| 6: Weeks 5–6 if available | Add one validated cue/coupling contribution; finalize figures, limitations, reproducibility and thesis writing | Clearly supported contribution and complete experiment package |

Dependencies: data integrity and extractor validation precede feature regeneration; stable features precede model selection; model selection and calibration precede final evaluation. If time is short, remove unsupported cues and complete a smaller controlled study. A narrower, rigorously evaluated thesis is preferable to unverified five-modality claims.

### First implementation batch

1. Introduce run/config/cache metadata and freeze the current baseline.
2. Refactor calibration and decision policy out of test evaluation; add policy-independence checks.
3. Repair data-source fallback and resume behavior; create the manifest and deduplication report.
4. Disable the random normal branch and fix eye-cue validity; regenerate a small pilot cache.
5. Run standardized logistic/MLP/transformer comparisons on the repaired pilot before reprocessing all images.

### Thesis deliverables

- A claim-to-evidence table connecting each contribution to an experiment.
- Dataset/source/generator table and split diagram.
- Equations with assumptions and an explicit distinction between physical measurements and learned estimates.
- Main results with per-source/generator performance and uncertainty.
- Ablation, calibration, risk–coverage, and robustness figures.
- Qualitative examples showing cue regions, missing cues, and failure cases; attention plots alone are insufficient explanations.
- Limitations covering ambiguous image formation, learned perception errors, source/content bias, edited photographs, CGI/art, and unseen generators.
- Reproduction commands, dependency lock, dataset manifests, cache/model hashes, and versioned experiment reports.

## Recommendation

Make the central contribution **validated compact consistency cues with observability-aware selective classification**. Establish that contribution through clean splits, controlled physics tests, and fair baselines. The current prototype is useful groundwork, but its strongest present claims exceed what the implementation and benchmark demonstrate.

