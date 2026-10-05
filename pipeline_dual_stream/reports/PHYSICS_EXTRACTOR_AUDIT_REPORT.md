# Physical Feature Extractor Audit & Modernization Report
**Date:** October 5, 2026  
**Repository Scope:** `G:\Thesis\pipeline_dual_stream`  
**Evaluation Scripts:** 
- [`scripts/evaluate_modernized_extractors_on_chameleon.py`](file:///G:/Thesis/pipeline_dual_stream/scripts/evaluate_modernized_extractors_on_chameleon.py) (Raw extractor distribution benchmark)
- [`scripts/evaluate_modernized_end_to_end.py`](file:///G:/Thesis/pipeline_dual_stream/scripts/evaluate_modernized_end_to_end.py) (End-to-end model swap evaluation)
**Raw Benchmark Artifacts:** 
- [`reports/modernized_extractor_chameleon_eval.json`](file:///G:/Thesis/pipeline_dual_stream/reports/modernized_extractor_chameleon_eval.json)
- [`reports/modernized_extractor_end_to_end_chameleon.json`](file:///G:/Thesis/pipeline_dual_stream/reports/modernized_extractor_end_to_end_chameleon.json)

---

## 1. Executive Summary

This report documents the diagnosis, implementation overhaul, and empirical end-to-end verification of the physical feature extractors in `pipeline_dual_stream`.

Previously, the standalone physics-based model exhibited near-random performance on the in-the-wild Chameleon benchmark (**51.85% Accuracy**, **0.5289 AUC**). Through per-entity physical ablation ([`reports/chameleon_physical_entity_ablation.json`](file:///G:/Thesis/pipeline_dual_stream/reports/chameleon_physical_entity_ablation.json)), we diagnosed two primary structural flaws:
1. **Unnormalized normal error metrics:** causing high-resolution camera textures to be miscategorized as synthetic.
2. **Color-grading split-toning artifacts:** triggering false alarms in the chromatic shadow consistency checks.

We modernized all physical extractors with scale/resolution-invariant formulas and ran two empirical evaluations on a held-out test split of 200 raw Chameleon images (100 authentic, 100 synthetic):
1. **Direct feature separation audit:** measuring raw physical signal discriminability directly from pixels.
2. **End-to-end model swap evaluation:** swapping the modernized extractors directly under frozen `universal_v5_calibrated` and `universal_v4` checkpoints to determine whether feature changes alone propagate through the gate.

---

## 2. Root Cause Analysis (Verified Empirical Findings)

From the verified per-entity ablation report (`46fb332`), zeroing out individual physics cues on Chameleon false alarms revealed:

### A. Chromatic Shadow Consistency (Entity 3)
* **Observed Mechanism:** The legacy extractor computed raw log chromaticity divergence between lit and cast-shadow regions without checking scene-wide color dispersion.
* **Empirical Impact:** In artistic photography with warm highlights and cool shadows (split-toning), the divergence score reached $1.5 - 3.0$ with maximum confidence ($1.0$). Ablating Entity 3 reduced false alarm scores across 90.7% of physical false positives.

### B. Surface Normal Kurtosis & Resolution Sensitivity (Entity 2)
* **Observed Mechanism:** The legacy Lambertian error calculation used raw, unnormalized pixel variance and high-order kurtosis.
* **Empirical Impact:** In high-resolution real photography (Flickr), camera sensors naturally capture sharp micro-textures, creating high normal kurtosis that previously acted as an inverted shortcut.

---

## 3. Extractor Architecture & Code Modernizations

All modernized extractors reside in `src/extractors/` and maintain drop-in schema compatibility (`[5, 14]` feature matrix and `[5, 4]` confidence matrix across 5 spatial regions):

1. **[`src/extractors/surface_normals.py`](file:///G:/Thesis/pipeline_dual_stream/src/extractors/surface_normals.py)** & **[`src/extractors/dsine_normals.py`](file:///G:/Thesis/pipeline_dual_stream/src/extractors/dsine_normals.py)**:
   - **Contrast-Normalized Residuals:** Replaced raw error variance with local dynamic-range normalized residuals:
     $$\text{norm\_error} = \frac{\text{raw\_error}}{\max(0.05, p_{95} - p_{5})}$$
   - **Bounded Kurtosis:** Centered and clamped normal residual kurtosis around the standard normal baseline ($[-3.0, 5.0]$) to eliminate macro-texture explosions.

2. **[`src/extractors/chromatic_shadow.py`](file:///G:/Thesis/pipeline_dual_stream/src/extractors/chromatic_shadow.py)**:
   - **Global Dispersion Awareness:** Computes scene-wide chromaticity dispersion ($\sigma_{\text{chroma}}$).
   - **Split-Toning Discounting:** Applies an exponential dampening factor to confidence when global split-toning is detected:
     $$\text{penalty} = \exp\left(-\max(0, \sigma_{\text{chroma}} - 0.05) \cdot 15.0\right)$$

3. **[`src/extractors/illumination_sh.py`](file:///G:/Thesis/pipeline_dual_stream/src/extractors/illumination_sh.py)** & **[`src/extractors/corneal_optics.py`](file:///G:/Thesis/pipeline_dual_stream/src/extractors/corneal_optics.py)**:
   - Standardized dynamic range normalization across regional patch Spherical Harmonics ($K=2$ directional clustering).
   - Generalized highlight candidate tracking beyond eye pupils to all reflective scene materials.

4. **[`src/extractors/regional_physics.py`](file:///G:/Thesis/pipeline_dual_stream/src/extractors/regional_physics.py)**:
   - Coordinates all 4 physical entities across Global and 4 Quadrants into unified $[5, 14]$ features and $[5, 4]$ confidences.

---

## 4. Empirical Evaluation 1: Raw Feature Separation

Executed via [`scripts/evaluate_modernized_extractors_on_chameleon.py`](file:///G:/Thesis/pipeline_dual_stream/scripts/evaluate_modernized_extractors_on_chameleon.py) on 200 raw held-out test images (100 authentic, 100 synthetic) using CUDA-accelerated DSINE.

### A. Raw Global Feature Separation

| Physical Feature | Authentic (Real) Mean | Synthetic (Fake) Mean | Diff (Fake - Real) | Raw Single-Feature AUC |
| :--- | :---: | :---: | :---: | :---: |
| **`Chroma_RG`** (Shadow RG Ratio) | 0.2722 | 0.2739 | +0.0017 | **0.6416** |
| **`SH_0`** (SH Order-0 Mean Res) | 0.2137 | 0.2602 | +0.0464 | **0.5987** |
| **`Chroma_BG`** (Shadow BG Ratio) | 0.2224 | 0.2484 | +0.0260 | **0.5883** |
| **`SH_2`** (SH Mode Variance) | 0.0506 | 0.0603 | +0.0098 | **0.5877** |
| **`SH_1`** (SH Order-1 Peak Res) | 0.6842 | 0.7695 | +0.0853 | **0.5760** |
| **`Spec_prof`** (Profile Discrepancy)| 0.0777 | 0.0898 | +0.0121 | **0.5484** |
| **`DSINE_var`** (Norm. Error Var) | 0.0291 | 0.0254 | -0.0037 | **0.5431** |
| **`Spec_theta`** (Angular Glint Delta)| 0.4171 | 0.4610 | +0.0439 | **0.5416** |
| **`Spec_dy`** (Vertical Glint Shift)| 0.2061 | 0.2299 | +0.0237 | 0.5198 |
| **`Spec_dx`** (Horizontal Glint Shift)| 0.2215 | 0.2342 | +0.0127 | 0.5161 |
| **`DSINE_skew`** (Norm. Error Skew)| 2.1953 | 2.1568 | -0.0385 | 0.4705 |
| **`SH_4`** (Spatial Mode Coherence)| 0.5496 | 0.5325 | -0.0171 | 0.4734 |
| **`DSINE_kurt`** (Norm. Error Kurt)| 3.6898 | 3.4720 | -0.2178 | 0.4672 |
| **`SH_3`** (Mode Angular Sep) | 1.9287 | 1.9218 | -0.0069 | 0.4323 |

### B. Impact of the Split-Toning Confidence Discounting

| Physical Entity | Authentic (Real) Mean Conf | Synthetic (Fake) Mean Conf | Real Samples Discounted ($c < 0.2$) |
| :--- | :---: | :---: | :---: |
| **Chromatic Shadow** | **0.4256** | **0.5831** | **27.0%** of real photos discounted |
| **Illumination (SH)**| **0.4876** | **0.6655** | **15.0%** of real photos discounted |
| **Surface Normals** | 0.4997 | 0.5735 | 5.0% |
| **Specular Optics** | 0.9664 | 0.9954 | 1.0% |

**Key Finding:** In **27.0%** of authentic photographs, the new split-toning aware extractor identified scene-wide color grading and dampened confidence to $< 0.20$ (compared to the legacy extractor assigning 1.0 confidence), mitigating the false-alarm cascade documented in the ablation report.

---

## 5. Empirical Evaluation 2: End-to-End Swapped Extractor Results

Executed via [`scripts/evaluate_modernized_end_to_end.py`](file:///G:/Thesis/pipeline_dual_stream/scripts/evaluate_modernized_end_to_end.py) comparing frozen `universal_v5_calibrated` and `universal_v4` models with the legacy vs. modernized extractors on the exact same 200 held-out Chameleon images:

| Model & Extractor Configuration | Standalone Physics Acc | Standalone Physics AUC | Fused Final Acc | Fused Final AUC | Gate $\bar{\alpha}$ (Semantic Weight) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DINOv2 Semantic Stream Alone** | — | — | 72.50% | 0.8075 | — |
| **Universal v5 (Legacy Extractors)** | 51.00% | 0.5282 | **73.00%** | **0.8107** | 0.6675 |
| **Universal v5 (Modernized Swapped In)** | 52.00% | 0.5263 | **71.50%** | **0.8086** | 0.6606 |
| $\Delta$ (v5 Impact) | *+1.00%* | *-0.0019* | **-1.50%** | **-0.0021** | *-0.0069* |
| **Universal v4 (Legacy Extractors)** | 51.00% | 0.5282 | **74.00%** | **0.8108** | 0.6737 |
| **Universal v4 (Modernized Swapped In)** | 52.00% | 0.5263 | **72.00%** | **0.8083** | 0.6669 |
| $\Delta$ (v4 Impact) | *+1.00%* | *-0.0019* | **-2.00%** | **-0.0025** | *-0.0068* |

---

## 6. Core Scientific Finding & Mechanistic Diagnosis

The end-to-end evaluation reveals a crucial architectural finding:

1. **The Gate & Neural Head Are the Primary Bottleneck:**
   - Swapping the modernized extractors under frozen neural network weights leaves the fused performance essentially flat (moving only $\pm 0.002$ in AUC and within standard error).
   - The gate's trust allocation remains anchored to the semantic stream ($\bar{\alpha} \approx 0.66 - 0.67$) because the gating network (`v3_calibrated` and `v2_confidence_adaptive`) was trained on legacy feature standardizers.
2. **Feature-Level Normalization Alone Is Insufficient Without Joint Recalibration:**
   - While the raw physical features possess genuine individual discriminative power on pixels (e.g. `Chroma_RG` 0.6416 AUC, `SH_0` 0.5987 AUC), these signals cannot improve the fused end-to-end model unless the downstream fusion gate is retrained or recalibrated to allocate trust based on the updated confidence distributions.
3. **Next Direction & Resolution:**
   - As hypothesized, subsequent optimization targeted **gate trust allocation and decision-level fusion**. This has now been implemented and empirically validated via disagreement-exposed gate training: see [`reports/DISAGREEMENT_GATE_AUDIT_REPORT.md`](file:///G:/Thesis/pipeline_dual_stream/reports/DISAGREEMENT_GATE_AUDIT_REPORT.md) where resolving the optimizer generator exhaustion and training on counterfactual semantic failures lifted full Chameleon AUC from 0.7521 to **0.7655** (+0.0134) and rescued **1,449 semantic false alarms**.

---

## 7. Verification Artifacts & Reproducibility

All numbers in this report are backed by committed, runnable scripts and raw JSON logs:
- Raw extractor benchmark:
  ```bash
  G:\Thesis\.venv\Scripts\python.exe scripts/evaluate_modernized_extractors_on_chameleon.py --num_real 100 --num_fake 100
  ```
  Artifact: [`reports/modernized_extractor_chameleon_eval.json`](file:///G:/Thesis/pipeline_dual_stream/reports/modernized_extractor_chameleon_eval.json)
- End-to-end model swap benchmark:
  ```bash
  G:\Thesis\.venv\Scripts\python.exe scripts/evaluate_modernized_end_to_end.py
  ```
  Artifact: [`reports/modernized_extractor_end_to_end_chameleon.json`](file:///G:/Thesis/pipeline_dual_stream/reports/modernized_extractor_end_to_end_chameleon.json)
