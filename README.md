# 100% Pure Physics Deepfake Detection System

A principled, cross-generator computer vision framework for detecting AI-generated imagery strictly through derived optical and 3D projective geometry invariants.

> **Academic Thesis Posture**: The downstream classification network receives **zero raw RGB pixels, texture patches, or generator-specific frequency fingerprints**. It evaluates strictly whether an image violates real-world optical rendering equations and classical projective camera geometry.

---

## Architecture Overview

Deepfake generators (GANs, Latent Diffusion Models, Flow Matching) learn statistical pixel priors and high-frequency texture correlations that mimic real imagery superficially. However, they consistently fail to enforce global, multi-physics optical invariants across independent physical domains.

```
                     Input Image (Pure Scene Optics)
                                    │
    ┌──────────────┬────────────────┼──────────────┬──────────────┐
    ▼              ▼                ▼              ▼              ▼
[Modality 1]   [Modality 2]    [Modality 3]   [Modality 4]   [Modality 5]
Illumination     Corneal/         Surface       Chromatic     3D Perspective
SH Clustering    Specular         Normals        Shadows      Vanishing Lines
(d=5, c_1)     (d=4, c_2)       (d=3, c_3)     (d=2, c_4)     (d=3, c_5)
    │              │                │              │              │
    └──────────────┴────────────────┼──────────────┴──────────────┘
                                    ▼
                     5 Pure Physics Tokens in R^64
                                    │
                                    ▼
                [2-Layer MHSA Transformer Cross-Attention]
            (Shadow-Light Alignment & Normal-Perspective Coupling)
                                    │
                                    ▼
             Selective Verdict (Authentic / Synthetic / Abstain)
```

### The 5 Pure Physics Modalities (17 Physical Dimensions)

1. **Multi-Source Spherical Harmonics Illumination** ($\Delta_{\text{light}} \in \mathbb{R}^5, c_{\text{light}}$):
   Decomposes regional $4 \times 4$ patch light vectors into $K=2$ dominant directional modes via spherical k-means:
   $$r_i = \min_{k \in \{1, 2\}} (1.0 - \hat{\mathbf{l}}_i \cdot \mathbf{m}_k)$$
   Accounts for multi-illuminant indoor environments (ceiling bulb + desk lamp + window) to eliminate false alarms on authentic photography.
2. **Corneal & Specular Optics** ($\Delta_{\text{spec}} \in \mathbb{R}^4, c_{\text{spec}}$):
   Measures highlight reflection congruence, bilateral displacement consistency, and reflection profile matching across eyes, metals, and reflective surfaces.
3. **3D Surface Normal Residuals** ($\Delta_{\text{norm}} \in \mathbb{R}^3, c_{\text{norm}}$):
   Decouples surface albedo via bilateral filtering and compares normal orientations $\mathbf{n}(x, y)$ against theoretical Lambertian shading $\max(0, \mathbf{n} \cdot \mathbf{l})$.
4. **Chromatic Shadow Rays** ($\Delta_{\text{chrom}} \in \mathbb{R}^2, c_{\text{chrom}}$):
   Evaluates illuminant color constancy ($R/G, B/G$) across lit and cast-shadow facial regions, measuring atmospheric penumbra chromatic drift.
5. **3D Perspective Vanishing Lines** ($\Delta_{\text{persp}} \in \mathbb{R}^3, c_{\text{persp}}$):
   Detects structural 3D line segments and measures vanishing point intersection dispersion variance and collinearity divergence ($\mathbf{l}_1 \times \mathbf{l}_2 = \mathbf{v}_\infty$).

---

## Physical Cross-Modal Coupling via MHSA Transformer

The model uses a 2-Layer Multi-Head Self-Attention (MHSA) Transformer ($d_{\text{model}} = 64$, 4 heads) where each physical cue is projected to a token and modulated by its physical observability confidence:
$$\mathbf{t}_i = c_i \cdot W_i \Delta_i + (1 - c_i) \cdot \mathbf{m}_{\text{mask}, i}$$

The self-attention mechanism verifies inter-law physical consistency:
- **Shadow-to-Light Alignment**: Does the cast shadow vector oppose the Spherical Harmonics illuminant?
- **Normal-to-Perspective Alignment**: Do planar surface normals align with the orthogonal vanishing planes computed by perspective geometry?
- **Specular Consistency**: Do highlight radiance angles align with dominant directional radiant vectors?

---

## 40,000-Image Benchmark Results (8,000 Unseen Test Set)

Evaluated on 4,000 authentic photos (MS-COCO Val 2017) and 4,000 unseen synthetic images (DiffusionDB):

| Benchmark Stage | Architecture | Objective | Decision Paradigm | Accuracy | AUC-ROC | EER |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Initial Scaled Baseline** | Concatenated MLP | Binary Cross-Entropy | Forced Binary ($\tau = 0.50$) | 64.75% | 0.7131 | 35.22% |
| **Stage 1 (Contrastive Head)** | Gated MLP | BCE + 0.3 SupCon | Forced Binary ($\tau = 0.50$) | 66.45% | 0.7349 | 33.56% |
| **Stage 2 (MHSA Transformer)** | 2-Layer MHSA | BCE + 0.3 SupCon | Youden's $J^*$ Optimal ($\tau^* = 0.457$) | 70.00% | 0.7758 | 30.14% |
| **100% Pure Physics Transformer** | **5-Token MHSA** | **Focal + 0.3 SupCon** | **Youden's $J^*$ Optimal ($\tau^* = 0.484$)** | **70.17%** | **0.7735** | **30.37%** |
| **Forensic Decided Band ($\pm 0.08$)** | **5-Token MHSA** | **Focal + 0.3 SupCon** | **Primary Forensic Operational Band** | **83.58%** | **0.7735** | **—** |
| **Forensic Decided Band ($\pm 0.10$)** | **5-Token MHSA** | **Focal + 0.3 SupCon** | **Primary Forensic Operational Band** | **86.61%** | **0.7735** | **—** |
| **Forensic Decided Band ($\pm 0.12$)** | **5-Token MHSA** | **Focal + 0.3 SupCon** | **Primary Forensic Operational Band** | **89.62%** | **0.7735** | **—** |
| **Forensic Decided Band ($\pm 0.15$)** | **5-Token MHSA** | **Focal + 0.3 SupCon** | **Primary Forensic Operational Band** | **92.74%** | **0.7735** | **—** |

### 3-Way Selective Forensic Policy

In legal and institutional verification, forcing binary decisions on images with near-zero physical cues ($O < 1.0$) or borderline probabilities ($0.45 < p < 0.55$) is methodologically invalid.

```
Band Margin   Tau Low   Tau High  Coverage    Abstain %   Decided Acc   Authentic Acc
------------------------------------------------------------------------------------------
+/- 0.03      0.454     0.514     71.0%       29.0%       75.26%        77.00%
+/- 0.05      0.434     0.534     55.5%       44.5%       78.69%        80.13%
+/- 0.08      0.404     0.564     37.9%       62.1%       83.58%        85.10%
+/- 0.10      0.384     0.584     28.6%       71.4%       86.61%        87.55%
+/- 0.12      0.364     0.604     22.0%       78.0%       89.62%        89.94%
+/- 0.15      0.334     0.634     15.0%       85.0%       92.74%        92.66%
```

- **Authentic (Real)**: $P(\text{Fake}) < 0.404$ and $O \ge 1.0 \implies$ **85.10% authentic accuracy**
- **Synthetic (Fake)**: $P(\text{Fake}) \ge 0.564$ and $O \ge 1.0 \implies$ **82.35% synthetic accuracy**
- **Indeterminate (Abstain)**: $O < 1.0$ or $0.404 \le P(\text{Fake}) < 0.564 \implies$ Prevents ungrounded guesses on low-evidence scenes.

---

## Directory Structure

```text
Thesis/
├── configs/
│   └── config.yaml                     # Pilot baseline configuration
├── data/
│   └── cache/                          # Cached pilot feature tensors
├── models/
│   ├── checkpoints/                    # Pilot model checkpoints
│   ├── evaluation_results.json         # Pilot evaluation metrics
│   └── evaluation_scaled_40k.json      # Scaled 40k evaluation metrics
├── pipeline_40k/                       # Production Scaled 40,000+ Image Pipeline
│   ├── configs/
│   │   └── config_scaled.yaml          # Scaled pipeline configuration
│   ├── data/
│   │   └── cache_scaled/               # Cached 40k feature tensors (32k train, 8k test)
│   ├── models/
│   │   ├── gated_cross_gen_40k_best.pt # Best 5-token pure physics transformer
│   │   └── evaluation_scaled_40k.json  # Comprehensive evaluation export
│   ├── scripts/
│   │   ├── download_scaled.py          # Resumable multi-threaded 40k image ingest
│   │   ├── cache_scaled_features.py    # Batched sharded discrepancy caching
│   │   ├── train_scaled.py             # 40-epoch Focal + SupCon transformer trainer
│   │   └── evaluate_scaled.py          # 3-way forensic selective evaluation engine
│   └── src/
│       ├── extractors/                 # Pure physics feature extractors
│       ├── models/                     # Cross-generator MHSA transformer head
│       └── utils/                      # Forensic metrics & threshold optimizers
├── src/                                # Core physics extractor suite
│   ├── extractors/
│   │   ├── illumination_sh.py          # Multi-source spherical harmonics (d=5)
│   │   ├── corneal_optics.py           # Specular highlight glints (d=4)
│   │   ├── surface_normals.py          # Lambertian normal shading (d=3)
│   │   ├── chromatic_shadow.py         # Penumbra chromatic drift (d=2)
│   │   └── perspective_vp.py           # 3D vanishing point geometry (d=3)
│   └── models/
│       └── cross_gen_gated.py          # 5-token transformer architecture
├── requirements.txt
└── README.md
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Immediate Training on Cached 40k Feature Manifold
Pre-extracted 14D/17D physical discrepancy tensors are already cached in `pipeline_40k/data/cache_scaled/`:
```bash
python pipeline_40k/scripts/train_scaled.py --epochs 40 --batch-size 256
```

### 3. Run Comprehensive Forensic Evaluation
```bash
python pipeline_40k/scripts/evaluate_scaled.py \
  --checkpoint pipeline_40k/models/gated_cross_gen_40k_best.pt \
  --test-cache pipeline_40k/data/cache_scaled/test_scaled_features.pt \
  --train-cache pipeline_40k/data/cache_scaled/train_scaled_features.pt
```

---

## License

Academic Research License.
