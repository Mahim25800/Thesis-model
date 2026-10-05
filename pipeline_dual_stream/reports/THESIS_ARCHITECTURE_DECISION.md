# Thesis Architecture & System Unification Decision
**Date:** October 6, 2026  
**Document Purpose:** Resolves Review Item 3 by declaring the official primary system for the thesis, defining the relationship between the two repository pipelines, and locking in a coherent, single-narrative structure for the final dissertation and journal submission.

---

## 1. Executive Decision: Which Model is "The Thesis"?

**The Primary System of the Thesis is the Dual-Stream Physical-Semantic Detector (`pipeline_dual_stream/models/universal_v5_disagreement_gate/best_model.pt`).**

The earlier pure-physics line (`pipeline_40k/models/physics_ensemble_v14`) will **not** be presented as an independent or competing system. Instead, it serves as the **motivational precursor and investigative baseline (Part I / Chapter 3)** of the thesis.

---

## 2. The Unified Three-Part Narrative Structure

Presenting two disconnected architectures confuses reviewers and creates the impression of two half-finished projects. By structuring them chronologically as a problem-and-solution investigation, the thesis tells a rigorous scientific story:

```
[Part I: Investigative Study]
"Can Physical Invariants Alone Solve Deepfake Detection?"
├── Models: physics_ensemble_v7 -> v11 -> v14 (pipeline_40k)
├── Key Finding: Strong in-distribution (0.8895 AUC), but hits an empirical ceiling
│   in the wild (0.7024 on Synthbuster, 0.6239 on GenImage, ~0.52 on Chameleon).
└── Mechanistic Diagnosis: Camera sensor noise, resolution kurtosis explosion,
    and artistic split-toning cause pure-physics false alarms.
        │
        ▼ (Motivates Foundation Model Fusion)
[Part II: Core Methodological Contribution]
"Fusing Vision Foundation Models with Physical Consistency"
├── Models: Dual-Stream Hybrid Detector (pipeline_dual_stream)
├── Backbone: Frozen DINOv2 ViT-Base + 4 Regional Physics Tokens (DSINE, SH, Shadows, Glints)
├── Discovery of Disagreement Starvation: Under standard ERM, gate unconditionally
│   trusts DINOv2 (99% accurate on training) and ignores physics during conflicts.
└── The Solution: Disagreement-Exposed Gate (v4_disagreement)
    ├── Synthetic counterfactual semantic error injection (35%)
    ├── Competence-guided directional gate supervision (L_gate)
    └── Dynamic Evidential Temperature Scaling T(x) on cross-modal discrepancy.
        │
        ▼ (Empirical Validation & Explainability)
[Part III: Multi-Benchmark Evaluation & Forensic Explainability]
"Cross-Generator Generalization & Auditable Forensics"
├── 24,000 images across 8 unseen generator families (Strict p < 0.00001 superiority, +5.78% mean AUC).
├── 13,500 images on Synthbuster (Positive synergy on 9/9 commercial generators).
├── 26,033 images on Chameleon in-the-wild (0.7655 AUC, +1.01% Acc, +0.0134 AUC, p < 0.00001).
├── 1,449 authentic photographs rescued from semantic false alarms (27.8% rescue rate).
└── Multi-Modal Forensic Certificates: Reconstructed 3D surface geometry (DSINE),
    spherical harmonic light consistency, and sensor noise residuals (PRNU).
```

---

## 3. Chapter-by-Chapter Dissertation Mapping

### Chapter 1: Introduction & Problem Statement
- The rapid advancement of text-to-image generative models (Diffusion, GANs, Autoregressive).
- The vulnerability of existing detectors to domain shifts and out-of-distribution generators.
- Research Questions:
  1. *Can optical and physical laws (lighting, geometry, shadows) provide generator-invariant detection?*
  2. *How can foundation vision models be fused with physical cues without suffering from semantic overconfidence?*

### Chapter 2: Related Work
- Artifact-based and frequency-based deepfake detection (Wang et al., Frank et al.).
- Foundation vision-language detectors (Ojha et al. UnivFD, CLIP-based probes).
- Physics-based digital image forensics (Kee & Farid, Johnson & Farid, normal consistency).
- Multimodal fusion and evidential deep learning.

### Chapter 3 (Part I): The Limits of Standalone Physical Consistency
- Formulation of the 4 physical entities across 5 spatial regions (Global + 4 Quadrants).
- The `pipeline_40k` experiments (`physics_ensemble_v14`).
- Results: 0.8895 AUC internal unseen, 0.7024 on Synthbuster, 0.6239 on confirmatory GenImage.
- **The Empirical Wall:** Forensic analysis of why standalone physics degrades in the wild (resolution-dependent normal kurtosis, split-toning color grading).
- Conclusion: Physical features alone cannot serve as an autonomous detector; they require contextual semantic anchoring.

### Chapter 4 (Part II): Dual-Stream Physical-Semantic Architecture & Disagreement-Exposed Gating
- Architecture of the Dual-Stream Hybrid Detector.
- Spatial token alignment between DINOv2 ViT-Base patch tokens and physical regional tokens.
- Cross-attention mechanism (Semantic Queries $\times$ Physical Keys/Values).
- **The "Disagreement Starvation" Bottleneck:** Mathematical proof that standard ERM drives gate trust $\alpha \to 1.0$, rendering the fusion head blind to physical invariants during semantic failure.
- **The Proposed Interventions:**
  1. Synthetic counterfactual error exposure during training.
  2. Stream competence-guided gate supervision ($\mathcal{L}_{\text{gate}}$).
  3. Dynamic Evidential Temperature Scaling ($T(x)$) conditioned on cross-modal discrepancy.
  4. Robust joint routing (`v4_disagreement`).

### Chapter 5 (Part III): Empirical Experiments, Statistical Validation & Explainability
- Benchmark 1: Multi-Generator Generalization (8 unseen families, 24,000 samples, $p < 0.00001$).
- Benchmark 2: Commercial Generators (Synthbuster, 9 generators, 13,500 samples).
- Benchmark 3: Extreme In-The-Wild (Chameleon, 26,033 samples).
- **Statistical Rigor:** 2,000-iteration paired class-stratified bootstrap results ($p < 0.00001$, 95% CI: `[+0.0116, +0.0152]`).
- Comparative baseline analysis (DINOv2, UnivFD/CLIP, CNN-Synth).
- Qualitative Case Studies: Walkthrough of authentic photos rescued by evidential temperature scaling.
- Web-based explainable forensic tool (`demo_app.py`).

### Chapter 6: Conclusion, Limitations & Future Directions
- Summary of verified contributions.
- Honest limitations: computational overhead of 3D normal estimation, remaining failure cases.
- Roadmap for future work: end-to-end differentiable physical estimators.

---

## 4. Status of the Action Items

| Action Item | Reviewer Critique | Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Item 2: Paired Bootstrap Test** | Gate delta (+0.0134 AUC) needs rigorous bootstrap CI before claiming significance. | **COMPLETED** | 2,000 iterations: $\Delta \text{AUC} = +0.0134$, 95% CI `[+0.0116, +0.0152]`, $p < 0.00001$. Artifact: `reports/paired_bootstrap_gate_significance.json`. |
| **Item 3: Unified Thesis Model** | Two active lines (`pipeline_40k` vs. `pipeline_dual_stream`) create disjoint narratives. | **COMPLETED** | Formalized: Dual-Stream v5 is the flagship thesis model; `pipeline_40k` is the foundational Part I study. |
| **Item 1: UnivFD/CLIP Baseline** | Missing competitive baseline comparison on Chameleon. | **IN PROGRESS** | Official weights (`fc_weights.pth`) downloaded; benchmarking script prepared. |
