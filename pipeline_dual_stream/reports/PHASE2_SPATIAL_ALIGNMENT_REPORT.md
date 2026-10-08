# Phase 2: Dense Spatial Geometry & Multimodal Alignment Report

**Date:** October 8, 2026  
**Repository Scope:** `pipeline_dual_stream`  
**Primary Architectures:**
- [`src/models/phase2_fusion.py`](file:///G:/Thesis/pipeline_dual_stream/src/models/phase2_fusion.py)
- [`src/models/hybrid_detector.py`](file:///G:/Thesis/pipeline_dual_stream/src/models/hybrid_detector.py) (`gate_mode="phase2"`)  
**Training Script:** [`scripts/train_phase2_spatial_alignment.py`](file:///G:/Thesis/pipeline_dual_stream/scripts/train_phase2_spatial_alignment.py)  
**Evaluation Scripts:**
- [`scripts/evaluate_universal_v4.py`](file:///G:/Thesis/pipeline_dual_stream/scripts/evaluate_universal_v4.py)
- [`scripts/evaluate_disagreement_gate_on_chameleon.py`](file:///G:/Thesis/pipeline_dual_stream/scripts/evaluate_disagreement_gate_on_chameleon.py)  

---

## 1. Executive Summary & Architectural Motivation

In Phase 1, our dual-stream system diagnosed the multimodal fusion bottleneck: when a dominant foundation model (DINOv2) is combined with a compact 41-dimensional scalar physical vector, standard empirical risk minimization (ERM) starves the weaker physical stream. While our decision-level disagreement gate repaired this bottleneck on Chameleon (yielding +0.0134 AUC and rescuing 1,449 false alarms), the physical stream remained constrained by the severe feature bandwidth mismatch.

**Phase 2 breaks this representational bottleneck** by introducing **Dense Spatial Alignment and Bidirectional Cross-Attention**:
1. **Bidirectional Multi-Head Cross-Attention (256-dim space):**
   - **Path A ($\text{Sem} \rightarrow \text{Phys}$):** DINOv2 semantic patch queries attend to regional 3D normals and illumination vectors.
   - **Path B ($\text{Phys} \rightarrow \text{Sem}$):** Physical normal vectors query semantic visual embeddings to corroborate geometric boundaries.
2. **Multi-Instance Spatial Anomaly Pooling (MIL):**
   - Replaces naive quadrant averaging with a dynamic **Max-Anomaly Regional Pooling** network, preventing localized generative artifacts (e.g., warped hands, melting edges) from being diluted by authentic background regions.
3. **Explicit Physical-Semantic Cosine Alignment Metric ($M[i, i]$):**
   - Evaluates region-wise cosine similarity across all 5 spatial regions (Global + 4 Quadrants), feeding mean alignment, maximum spatial misalignment, and alignment variance directly into the decision head.
4. **Contrastive Geometric Alignment Loss:**
   - Explicitly rewards geometric-semantic coherence on authentic camera photos and penalizes generative inconsistencies.

---

## 2. Empirical Benchmark 1: Unseen Cross-Generator Evaluation (9 Generators)

*Artifact:* [`reports/phase2_unseen_generators_evaluation.json`](file:///G:/Thesis/pipeline_dual_stream/reports/phase2_unseen_generators_evaluation.json)  
Evaluated across 7,500 unseen commercial generator images, 6,000 in-family diffusion images, 500 held-out RAISE DSLR photos, and 400 CelebA portraits:

### A. Truly Unseen Commercial Generators (0% Training Exposure)
| Generator | Standalone Physics AUC | Standalone DINOv2 AUC | Phase 2 Dual-Stream AUC | Net Synergy ($\Delta$) | Accuracy Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OpenAI GLIDE** | 0.6998 | 0.9078 | **0.9251** | **+0.0173** | 76.27% $\rightarrow$ 75.87% |
| **DALL-E 2** | 0.6870 | 0.7601 | **0.7896** | **+0.0295** | 53.60% $\rightarrow$ **54.00%** |
| **DALL-E 3** | 0.6909 | 0.9360 | **0.9585** | **+0.0225** | 77.27% $\rightarrow$ **81.60% (+4.33%)** |
| **Adobe Firefly** | 0.6692 | 0.7311 | **0.7612** | **+0.0302** | 47.40% $\rightarrow$ **47.80%** |
| **Midjourney v5** | 0.6578 | 0.7578 | **0.7825** | **+0.0247** | 51.47% $\rightarrow$ 49.73% |
| **Unseen Average** | **0.6809** | **0.8186** | **0.8434** | **+0.0248** | — |

### B. In-Family Diffusion Generators (Stable Diffusion Family)
| Generator | Standalone Physics AUC | Standalone DINOv2 AUC | Phase 2 Dual-Stream AUC | Net Synergy ($\Delta$) | Accuracy Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SD 1.3** | 0.6902 | 0.8595 | **0.8996** | **+0.0401** | 64.53% $\rightarrow$ **67.40% (+2.87%)** |
| **SD 1.4** | 0.6884 | 0.8576 | **0.8965** | **+0.0389** | 64.40% $\rightarrow$ **68.47% (+4.07%)** |
| **SD 2.0** | 0.8254 | 0.9416 | **0.9646** | **+0.0229** | 82.60% $\rightarrow$ **86.67% (+4.07%)** |
| **SDXL** | 0.7992 | 0.9116 | **0.9390** | **+0.0274** | 74.87% $\rightarrow$ **77.87% (+3.00%)** |
| **Diffusion Average** | **0.7508** | **0.8926** | **0.9249** | **+0.0323** | — |

> **Milestone:** Positive synergy across **9 out of 9 generators (100% win rate)**. Overall mean AUC increased from **0.8515 (DINOv2) $\rightarrow$ 0.8796 (Phase 2 Dual-Stream)** (+0.0282 AUC gain).

### C. Portrait False Alarm & Blur Robustness
- **Held-Out Real Portraits (400 CelebA DSLR Photos):** Standalone physics produced 48.75% FPR due to shallow depth-of-field; Phase 2 Dual-Stream suppressed false alarms to **0.25% FPR (only 1 single error in 400 photos)**, yielding **99.75% real accuracy**.
- **Optical Blur Perturbations:** Under Gaussian smoothing ($\sigma = 1.0, 2.0, 3.0$), FPR remained consistently between 2.00% and 2.40% (vs. clean 2.20%).

---

## 3. Empirical Benchmark 2: Chameleon In-The-Wild Benchmark (26,033 Images)

*Artifact:* [`reports/phase2_chameleon_eval.json`](file:///G:/Thesis/pipeline_dual_stream/reports/phase2_chameleon_eval.json)  

| Model / Stream Configuration | Accuracy | ROC-AUC | Comparison to Semantic Alone |
| :--- | :---: | :---: | :---: |
| **Standalone Multi-Physics Alone** | 52.16% | 0.5223 | — |
| **Standalone DINOv2 Semantic Alone** | 67.60% | 0.7322 | *baseline* |
| **Phase 1 Fused Model (v5 Disagreement Gate)** | 70.76% | 0.7655 | +3.16% Acc / +0.0333 AUC |
| **⭐ Phase 2 Joint Bidirectional Stream** | **70.81%** | **0.7690** | **+3.21% Acc / +0.0368 AUC** |

### Key Scientific Insight:
The **Phase 2 Bidirectional Joint Cross-Attention stream** achieves the highest standalone joint performance recorded in the thesis (**70.81% Accuracy and 0.7690 ROC-AUC**), beating standalone DINOv2 by **+3.21% in accuracy and +0.0368 in AUC**.

---

## 4. Checkpoints & Reproducibility Commands

1. **Phase 2 Training:**
   ```bash
   G:\Thesis\.venv\Scripts\python.exe scripts/train_phase2_spatial_alignment.py --epochs 8 --batch_size 256 --lr 5e-4 --device cuda
   ```
   *Checkpoint:* `models/universal_v6_phase2_alignment/best_model.pt`  
   *Log:* [`reports/phase2_alignment_training_log.json`](file:///G:/Thesis/pipeline_dual_stream/reports/phase2_alignment_training_log.json)

2. **Unseen 9-Generator Evaluation:**
   ```bash
   G:\Thesis\.venv\Scripts\python.exe scripts/evaluate_universal_v4.py --checkpoint models/universal_v6_phase2_alignment/best_model.pt --output reports/phase2_unseen_generators_evaluation.json --device cuda
   ```

3. **Chameleon Full Benchmark Evaluation:**
   ```bash
   G:\Thesis\.venv\Scripts\python.exe scripts/evaluate_disagreement_gate_on_chameleon.py --checkpoint models/universal_v6_phase2_alignment/best_model.pt --gate_mode phase2 --out_json reports/phase2_chameleon_eval.json
   ```
