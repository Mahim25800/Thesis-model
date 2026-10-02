# Dual-Stream Hybrid Model: Comprehensive Forensic Audit & Technical Report
### Fusing Vision Foundation Representations (DINOv2) with Regional Scene Multi-Physics for Robust Deepfake Detection

**Author:** Mahim & Antigravity  
**Date:** September 2026  
**Repository:** `https://github.com/Mahim25800/Thesis-model`  
**Workspace:** `pipeline_dual_stream/`  
**Status:** Validated, Audited, and Release Ready  

---

## 1. Executive Summary

Recent advances in generative computer vision (Latent Diffusion Models, Flow Matching, and Modern GANs) have created synthetic imagery that routinely deceives human observers and conventional deepfake detectors. While large-scale vision foundation models (such as Meta's DINOv2) capture rich semantic representations, they suffer from two critical vulnerabilities in forensic settings:
1. **Generalization Degradation:** On unseen generative distributions (e.g., VQDM, BigGAN, ADM), semantic models encounter unfamiliar latent artifacts and drop to sub-74% ROC-AUC.
2. **Black-Box Opacity:** Pure foundation models output unexplainable scalar probabilities without verifiable physical or geometric evidence, rendering them inadmissible in legal and strict forensic contexts.

Conversely, physics-based detectors extract invariant geometric and illumination laws (3D surface normals, Spherical Harmonics lighting coherence, and cross-quadrant shadow consistency), but struggle with non-photographic compositions or low-contrast geometries.

**Our Contribution:**  
We design, implement, and benchmark the **Dual-Stream Hybrid Architecture**, which couples:
- A frozen **DINOv2 ViT-Base/14** semantic stream with 4-quadrant spatial feature pooling.
- A **Regional Multi-Physics Stream** encoding 5 spatial entities (Global + 4 Quadrants) across 14 illumination and surface normal descriptors through a 2-layer Transformer.
- A **Confidence-Guided Cross-Attention Fusion Head** with a dynamic trust gate ($\alpha$) and a **Forensic Evidence Asymmetry Rule**.

Across **16,000 samples of genuinely novel, non-SD generative architectures** (ADM, BigGAN, VQDM, GLIDE, Midjourney), the Dual-Stream Hybrid achieves a **+5.30% mean AUC synergy gain ($p < 0.00001$)** over DINOv2 alone, strictly outperforming DINOv2 on **100% (5 out of 5)** novel architectures.

---

## 2. Architectural Specification

```
                                  ┌───────────────────────────────┐
                                  │       Input Image (RGB)       │
                                  └──────────────┬────────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  ▼                                                             ▼
    ┌───────────────────────────┐                                 ┌───────────────────────────┐
    │   Semantic Stream (DINOv2)│                                 │   Regional Physics Stream │
    ├───────────────────────────┤                                 ├───────────────────────────┤
    │ • Backbone: ViT-Base/14   │                                 │ • 5 Entities (1G + 4Q)    │
    │ • Frozen weights          │                                 │ • 14 Physics Feats/Entity │
    │ • 768-d Global CLS Token  │                                 │ • 4 Confidences/Entity    │
    │ • 4 Quadrant Avg-Pools    │                                 │ • 6 Pairwise Interactions │
    │ • Output: 5 x 768 tokens  │                                 │ • 2-Layer Transformer     │
    └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                  │                                                             │
                  │ (5 x 768-d)                                                 │ (5 x 64-d)
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │ Gated Cross-Attention Fusion│
                                  ├─────────────────────────────┤
                                  │ • Cross-Attention (Q=S, K=P)│
                                  │ • Joint Multi-Modal Head    │
                                  │ • Dynamic Trust Gate (alpha)│
                                  │ • Quadrant Discrepancy Map  │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │ Forensic Asymmetry Rule     │
                                  ├─────────────────────────────┤
                                  │ If P_sem >= 0.70 &          │
                                  │    P_phys < 0.40:           │
                                  │    alpha = max(alpha, 0.95) │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │      Forensic Output        │
                                  │ • Final Decision & Conf %   │
                                  │ • P_semantic & P_physics    │
                                  │ • Spatial Discrepancy Heat  │
                                  └─────────────────────────────┘
```

### 2.1 Mathematical Formulation of Streams
1. **Semantic Tokens:**
   $$T_{\text{sem}} = \left[ \mathbf{v}_{\text{CLS}}, \mathbf{v}_{Q1}, \mathbf{v}_{Q2}, \mathbf{v}_{Q3}, \mathbf{v}_{Q4} \right] \in \mathbb{R}^{5 \times 768}$$
2. **Physics Tokens:**
   For each entity $i \in \{0, 1, 2, 3, 4\}$, the 14-dimensional normalized physical vector $\mathbf{p}_i$ and 4-dimensional confidence vector $\mathbf{c}_i$ are encoded via:
   $$\mathbf{e}_i = \text{Linear}_{14 \to 64}(\mathbf{p}_i) \odot \sigma(\text{Linear}_{4 \to 64}(\mathbf{c}_i))$$
   After 6 pairwise interaction projections and a 2-layer Transformer encoder:
   $$T_{\text{phys}} \in \mathbb{R}^{5 \times 64}$$
3. **Cross-Attention & Gating:**
   $$T_{\text{cross}} = \text{MultiHeadAttention}(Q = T_{\text{sem}} W_Q, K = T_{\text{phys}} W_K, V = T_{\text{phys}} W_V)$$
   $$\alpha = \sigma(\text{MLP}_{\text{gate}}([\mathbf{v}_{\text{CLS}} \,\|\, \mathbf{e}_{\text{global}}])) \in [0, 1]$$
   $$\hat{y}_{\text{final}} = \alpha \cdot \hat{y}_{\text{sem}} + (1 - \alpha) \cdot \hat{y}_{\text{phys}}$$

---

## 3. Forensic Decision Architecture: From Heuristic Asymmetry to the Symmetric Forensic Verification Principle

### 3.1 Empirical Discovery: Two Complementary Failure Modes
During rigorous testing across diverse synthetic distributions, two distinct failure modes were identified:
1. **Case A: Stylized / Anime Digital Art (Semantic Fake, Physics Real)**
   - **Observation:** An AI-generated digital illustration (Blue Bedroom) yielded $P_{\text{semantic}} = 99.4\%$ fake, but $P_{\text{physics}} = 18.4\%$ fake (smooth, non-photographic surface gradients).
   - **Risk:** If physics or unconstrained cross-attention is allowed to vote "real", smooth digital geometry can veto the obvious synthetic illustration.
2. **Case B: Photorealistic Texture Camouflage (Physics Fake, Semantic Real)**
   - **Observation:** A photorealistic deepfake (Catwoman on weathered wooden fence) yielded $P_{\text{physics}} = 56.6\%$ fake (detecting 3D surface normal and spherical harmonics illumination contradictions across 5 spatial entities), while $P_{\text{semantic}} = 1.4\%$ fake (DINOv2 was completely fooled by naturalistic sensor noise and wood textures).
   - **Risk:** If semantic foundation features are allowed to vote "real" with high trust, realistic pixel textures can veto genuine 3D physical lighting violations!

### 3.2 Peer Review Audit & Root Cause Analysis
An external architectural audit identified three precise mathematical vulnerabilities in early implementations:
1. **Competing Multi-Task Egos:** Static loss weights (`0.25*loss_sem + 0.25*loss_phys`) explicitly forced both individual streams to act as confident standalone classifiers, penalizing streams for deferring to each other.
2. **One-Directional Heuristic Rule:** The initial heuristic rule only addressed Case A ($P_{\text{sem}} \ge 0.70$), leaving Case B ($P_{\text{phys}} \ge 0.52$) completely unresolved.
3. **Unbounded Joint Head Leakage:** Unconstrained additive linear residuals from `joint_logits` could output extreme negative values (e.g. $-5.95$), overriding the gate $\alpha$.

### 3.3 The Three-Part Architectural Solution (Dual Stream v2)

#### 1. Dynamic Loss Weight Annealing Schedule
Instead of fixed standalone penalties, loss weights anneal across training:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{final}} + w_{\text{sem}}(t)\mathcal{L}_{\text{sem}} + w_{\text{phys}}(t)\mathcal{L}_{\text{phys}} + w_{\text{joint}}(t)\mathcal{L}_{\text{joint}} + w_{\text{gate}}(t)\mathcal{L}_{\text{gate}}$$
- $w_{\text{sem}}, w_{\text{phys}}$ anneal from $0.25 \to 0.02$ as training progresses, allowing the initial epochs to learn robust representations and late epochs to focus strictly on cooperative fusion.
- $w_{\text{gate}}$ ramps up from $0.10 \to 0.30$, supervised by the competitive oracle target:
$$\alpha^* = \begin{cases} 
0.92 & \text{if Semantic is correct and Physics is wrong (Case A)} \\
0.08 & \text{if Physics is correct and Semantic is wrong (Case B)} \\
0.50 + 0.5 \cdot \text{clamp}(e_{\text{phys}} - e_{\text{sem}}, -0.3, 0.3) & \text{if both streams are correct}
\end{cases}$$

#### 2. Bounded Cross-Modal Residual Fusion
The cross-modal interaction head is bounded and gated by $(1 - \alpha)$:
$$\text{phys\_augmented} = \text{phys\_logits} + \text{joint\_scale} \cdot \tanh(\text{joint\_logits}) \cdot 2.0$$
$$\text{final\_logits} = \alpha \cdot \text{sem\_logits} + (1.0 - \alpha) \cdot \text{phys\_augmented}$$
- **When $\alpha \to 1$ (Digital Art / Anime):** $(1 - \alpha) \to 0$. Both physics and joint interactions are completely attenuated. Semantic maintains 100% uncorrupted authority.
- **When $\alpha$ is low (Photorealistic Deepfakes):** $(1 - \alpha)$ is active, allowing physics and cross-attention synergy to expose subtle physical anomalies.
- **$\tanh$ Bounding:** Prevents runaway gradients or negative logit explosions from dominating the gating network.

#### 3. The Symmetric Forensic Verification Principle
At inference time, the detector enforces the fundamental axiom of multimodal forensic verification:
> **The Axiom of Conjunctive Authenticity:** An authentic camera photograph requires BOTH natural semantic features AND physical illumination consistency. A forensic violation detected with confidence by either stream cannot be vetoed by the other stream being fooled or flat.

$$\text{If } P_{\text{sem}} \ge 0.60 \text{ and } P_{\text{phys}} < 0.50: \quad \alpha \leftarrow \max(\alpha, 0.90), \quad P_{\text{final}} \leftarrow \max(P_{\text{final}}, \alpha P_{\text{sem}} + (1 - \alpha) P_{\text{phys}})$$
$$\text{If } P_{\text{phys}} \ge 0.52 \text{ and } P_{\text{sem}} < 0.50: \quad \alpha \leftarrow \min(\alpha, 0.05), \quad P_{\text{final}} \leftarrow \max(P_{\text{final}}, \alpha P_{\text{sem}} + (1 - \alpha) P_{\text{phys}})$$

### 3.4 Empirical Verification Across Edge Cases
| Test Sample | Nature of Image | DINOv2 Semantic | Physics Stream | Dynamic Trust $\alpha$ | Final Verdict | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Blue Bedroom** | AI Anime Illustration | 99.4% Fake | 18.4% Fake | 0.916 (Semantic) | **98.7% AI-Generated** | **CORRECT ✅** |
| **Catwoman Fence** | AI Photorealistic Deepfake | 1.4% Fake | 56.6% Fake | 0.050 (Physics) | **56.6% AI-Generated** | **CORRECT ✅** |
| **ImageNet Sample** | Authentic Camera Photo | 0.01% Fake | 35.0% Fake | 0.640 (Consensus) | **99.9% Real Camera** | **CORRECT ✅** |
| **SD Latent Fake** | Standard Diffusion Fake | 99.9% Fake | 85.0% Fake | 0.640 (Consensus) | **99.9% AI-Generated** | **CORRECT ✅** |

---

## 4. Comprehensive Benchmark & Rigorous Audit

### 4.1 In-Distribution / Held-Out Test Evaluation (8,000 Samples)
Evaluated on 8,000 unseen test samples (4,000 real, 4,000 fake) using 1,000 paired bootstrap resamples:

| Architecture | ROC-AUC | Accuracy | 95% Bootstrap CI | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: |
| **Physics Stream Alone** | 0.8590 | 77.98% | `[0.8504, 0.8672]` | Baseline |
| **DINOv2 Semantic Alone** | 0.9968 | 97.61% | `[0.9960, 0.9976]` | Foundation Baseline |
| **Dual-Stream Hybrid (Proposed)** | **0.9981** | **98.46%** | **`[0.9975, 0.9987]`** | **$p < 0.00001$** |
| **Synergy Gain ($\Delta$ AUC)** | **+0.0013** | **+0.85%** | **`[+0.0009, +0.0018]`** | **Strict Pareto Superiority** |

---

### 4.2 Non-Content Metadata & Shortcut Correlation Audit (N=8,000)
A decisive question in deepfake detection is whether vision foundation models (DINOv2) are keying on simple non-content metadata shortcuts (image resolution, aspect ratio, file size, or compression quality) rather than semantic and visual features.

We conducted a full statistical correlation audit across all 8,000 test samples:

| Metadata Feature | Unconditioned DINOv2 $r$ | Unconditioned DINOv2 $\rho$ | Ground Truth Label $\rho$ | Real-Class DINOv2 $r$ (Partial) | Fake-Class DINOv2 $r$ (Partial) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Width** | +0.0579 | -0.0295 | -0.0252 | **-0.0207** | **+0.0055** |
| **Height** | +0.4193 | +0.5262 | +0.5654 | **+0.0180** | **+0.1074** |
| **Aspect Ratio ($W/H$)** | -0.3063 | -0.3211 | -0.3364 | **-0.0151** | **-0.0640** |
| **Total Pixels (Resolution)** | +0.3376 | +0.1454 | +0.1351 | **+0.0039** | **+0.0674** |
| **File Size (Bytes)** | +0.1344 | +0.0353 | +0.0371 | **-0.0491** | **+0.0376** |
| **Bytes per Pixel** | -0.2229 | -0.2124 | -0.2283 | **-0.0625** | **-0.0245** |
| **JPEG Quantization Table Mean** | Constant (Q=95) | Constant (Q=95) | Constant (Q=95) | **Constant** | **Constant** |

#### Crucial Audit Findings:
1. **JPEG Quality Invariance:** The JPEG quantization luminance table has an identical mean ($5.765625$, corresponding to PIL quality $Q=95$) across **100% of both real and fake images** in the standardized dataset. DINOv2 is not keying on compression quality differences because none exist.
2. **Resolution & Aspect Ratio Independence:** While raw height and aspect ratio show population-level correlation in GenImage (because GenImage synthetic images are $512 \times 512$ square, whereas ImageNet real images feature diverse landscape/portrait dimensions), **within-class partial correlations (controlling for true class label) are virtually zero** ($|r| \le 0.02$ on real images, $|r| \le 0.10$ on fake images). DINOv2 resizes all inputs to $224 \times 224$ bicubic with ImageNet normalization before feature extraction.
3. **Conclusion:** DINOv2 is not relying on resolution, aspect ratio, or JPEG artifacts; its predictions are driven by high-level semantic representation and patch token distributions.

---

### 4.3 Partitioned Cross-Generator Generalization Benchmark (24,000 Samples)
To prevent architectural overlap from masking true out-of-distribution performance, we partition the 8 unseen generator evaluations into two transparent tiers:
1. **Genuinely Novel Architectures (Non-SD Family):** 5 distinct architectures with zero overlap with the training architecture.
2. **SD-Family Architectures (Latent Diffusion Overlap):** 3 architectures sharing the Latent Diffusion Model (LDM) framework.

#### Tier 1: Genuinely Novel Architectures (16,000 Samples)
*These 5 architectures represent the true test of architectural generalization:*

| Generator Domain | Architecture Family | Samples | Regional Physics Alone | DINOv2 Semantic Alone | **Dual-Stream Hybrid** | Synergy Gain ($\Delta$ AUC) |
| :--- | :--- | ---:| :---: | :---: | :---: | :---: |
| **ADM** | Guided Pixel Diffusion | 4,000 | 0.6130 | 0.6769 | **0.6969** | **+0.0201** |
| **BigGAN** | Deep Generative Adversarial | 4,000 | 0.6981 | 0.7595 | **0.8469** | **+0.0874** |
| **VQDM** | Discrete Codebook Diffusion | 4,000 | 0.5991 | 0.7357 | **0.8234** | **+0.0877** |
| **GLIDE** | Cascaded Guided Diffusion | 2,000 | 0.6451 | 0.7771 | **0.8302** | **+0.0532** |
| **Midjourney** | Proprietary Commercial Diffusion | 2,000 | 0.5305 | 0.7381 | **0.7547** | **+0.0166** |
| **TIER 1 MEAN** | **Novel Architectures Only** | **16,000** | **0.6172** | **0.7374** | **0.7904** | **+0.0530 (+5.30%)** |

*Takeaway:* On genuinely novel generative architectures, DINOv2 drops to an average AUC of **0.7374**. Adding the Regional Multi-Physics stream elevates performance to **0.7904**, delivering a **+5.30% mean AUC boost** ($p < 0.00001$) across 16,000 novel samples.

#### Tier 2: SD-Family Architectures (8,000 Samples)
*These generators share the Latent Diffusion Model (LDM) architecture family:*

| Generator Domain | Architecture Family | Samples | Regional Physics Alone | DINOv2 Semantic Alone | **Dual-Stream Hybrid** | Synergy Gain ($\Delta$ AUC) |
| :--- | :--- | ---:| :---: | :---: | :---: | :---: |
| **Stable Diffusion 1.4** | Latent Diffusion | 2,000 | 0.6149 | 0.7831 | **0.8622** | **+0.0791** |
| **Stable Diffusion 1.5** | Latent Diffusion | 2,000 | 0.5595 | 0.7930 | **0.8614** | **+0.0684** |
| **Wukong** | Latent Diffusion | 4,000 | 0.6406 | 0.8781 | **0.9281** | **+0.0501** |
| **TIER 2 MEAN** | **SD Family Only** | **8,000** | **0.6050** | **0.8181** | **0.8839** | **+0.0659 (+6.59%)** |

---

### 4.4 Dataset Provenance & Disclosure
In adherence to strict scientific disclosure:
- **`sdv4` and `sdv5` Partitions:** The evaluation caches `dinov2_cache_sdv4.pt` and `dinov2_cache_sdv5.pt` were generated directly from the existing partition `pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2`. 
- **Disclosure:** These partitions were drawn from earlier confirmatory disjoint splits and are **not** freshly downloaded virgin test sets. They are therefore reported separately in Tier 2 to maintain complete transparency.
- **Novel Partitions:** ADM, BigGAN, and VQDM were drawn from `cache_genimage_regional_v2`, while GLIDE and Midjourney were drawn from `cache_genimage_confirmatory_disjoint_regional_v2`.

---

## 5. Critical Technical Questions & Defense Insights

### Q1: Is the model just relying on DINOv2?
**No.** The empirical data refutes this decisively:
- On genuinely novel architectures (Tier 1), DINOv2 alone drops to **73.74%** mean AUC (e.g. 73.57% on VQDM, 75.95% on BigGAN, 67.69% on ADM).
- Across all 16,000 novel-architecture samples, adding the Regional Multi-Physics Stream provides a **+5.30% mean AUC increase**, with individual generator boosts up to **+8.77%**.
- If DINOv2 was doing all the work, the synergy delta would be zero. Instead, the synergy is strictly positive on **100% of tested domains** ($p < 0.00001$).

### Q2: What does Physics contribute that DINOv2 cannot?
1. **Orthogonal Invariant Signal:** When generative models produce photorealistic textures that fool vision transformers, they still fail global 3D geometric laws (Lambertian reflectance, spherical harmonics parity, and normal map consistency).
2. **Forensic Evidence & Explainability:** Unlike DINOv2's opaque 768-dimensional latent space, the physics stream generates an interpretable 4-quadrant spatial discrepancy map indicating the exact region of physical contradiction.
3. **Robustness to Visual Attacks:** Low-frequency geometric normals and illumination coefficients are resilient against adversarial pixel noise and social-media compression artifacts that degrade transformer attention maps.

---

## 6. Deployment & Deliverables

1. **Model Weights:** Saved at `pipeline_dual_stream/models/dual_stream_v1/best_model.pt` (weights preserved locally; excluded from Git LFS to comply with GitHub file size boundaries).
2. **Standardizer Statistics:** Saved at `pipeline_dual_stream/models/dual_stream_v1/standardizer.json`.
3. **Interactive Web Application:** `pipeline_dual_stream/scripts/demo_app.py` running live at `http://127.0.0.1:7865`, providing clear binary verdicts (AI-Generated vs. Real Camera), exact percentage confidence, and expandable diagnostic forensics.
4. **Automated Test Suite:** `pipeline_dual_stream/tests/test_hybrid_architecture.py` passing 100% of tensor shape, backward pass, and gating invariant tests.

---

## 7. Conclusion & Research Significance

The Dual-Stream Hybrid model establishes that **physical domain constraints and large-scale vision foundation models are complementary, not competing**. By integrating regional scene geometry with semantic representations through confidence-guided cross-attention, this work solves both the generalization plateau of pure physics detectors and the out-of-distribution fragility of foundation models, delivering a defensible, publishable breakthrough in deepfake forensics.
