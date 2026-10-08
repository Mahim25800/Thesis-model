"""Phase 2: Dense Spatial Geometry & Multimodal Alignment Fusion.

Innovations over Phase 1:
1. Bidirectional Cross-Attention:
   - Sem -> Phys (Semantic queries Physics)
   - Phys -> Sem (Physics queries Semantic)
2. Multi-Instance Spatial Anomaly Pooling (MIL):
   - Rather than averaging quadrants, computes both mean pooling and MAX-ANOMALY pooling
     to detect localized generative glitches without spatial dilution.
3. Explicit Physical-Semantic Cosine Alignment Metric:
   - Computes region-wise alignment M[i, i] = cos(q_sem[i], k_phys[i]).
   - Feeds mean alignment, max misalignment, and spatial variance directly into the decision gate.
4. Anomaly-Routed Evidential Calibration:
   - When localized spatial misalignment is detected, evidential temperature T(x) scales up
     to dampen overconfident semantic predictions and route authority to the joint physical anomaly head.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase2SpatialAlignmentFusion(nn.Module):
    """Dense Spatial Multimodal Alignment and Dynamic Anomaly-Gated Fusion."""

    def __init__(
        self,
        sem_dim: int = 768,
        phys_dim: int = 64,
        proj_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.sem_dim = sem_dim
        self.phys_dim = phys_dim
        self.proj_dim = proj_dim

        # 1. High-Capacity Projections into Shared Alignment Space (dim: 256)
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # 2. Bidirectional Cross-Attention
        # Path A: Semantic queries Physics (Geometry corroborates semantics)
        self.sem_to_phys_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_sem_cross = nn.LayerNorm(proj_dim)

        # Path B: Physics queries Semantic (Semantics justifies geometric boundaries)
        self.phys_to_sem_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_phys_cross = nn.LayerNorm(proj_dim)

        # 3. Stream-Specific Classifiers
        self.sem_head = nn.Sequential(
            nn.Linear(sem_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 1),
        )

        # 4. Multi-Instance Spatial Anomaly Classifier (MIL Head)
        # Evaluates joint representation: [Global, Mean_Quads, Max_Anomaly_Quad] -> 3 * proj_dim
        self.joint_mil_head = nn.Sequential(
            nn.Linear(4 * proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
        )

        # 5. Local Quadrant Anomaly Scorer
        self.quad_anomaly_scorer = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
        )

        # 6. Evidential Temperature Calibration Network T(x)
        # Conditioned on 16 alignment & discrepancy features:
        # - mean diagonal alignment (1)
        # - max quadrant misalignment (1)
        # - alignment variance (1)
        # - per-quadrant misalignment (4)
        # - raw stream probability disparity |p_sem - p_phys| (1)
        # - raw stream logit disparity |z_sem - z_phys| (1)
        # - max quadrant anomaly score (1)
        # - regional physics confidences (5)
        # - confidence disparity (1)
        # Total: 16 features
        self.calib_net = nn.Sequential(
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.calib_net[-1].weight)
        nn.init.constant_(self.calib_net[-1].bias, -2.2) # Softplus(-2.2) ~ 0.105 -> T ~ 1.10 initially

        # 7. Adaptive Disagreement Gate alpha(x)
        gate_in_dim = (proj_dim // 4) + (proj_dim // 4) + 18  # 64 + 64 + 18 = 146
        self.gate_sem_proj = nn.Sequential(
            nn.Linear(sem_dim, proj_dim // 4),
            nn.LayerNorm(proj_dim // 4),
            nn.GELU(),
        )
        self.gate_phys_proj = nn.Sequential(
            nn.Linear(2 * phys_dim, proj_dim // 4),
            nn.LayerNorm(proj_dim // 4),
            nn.GELU(),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Blending parameter for residual physical anchor
        self.joint_scale = nn.Parameter(torch.tensor(0.65))

    def forward(
        self,
        sem_tokens: torch.Tensor,       # [B, 5, 768] (Global, TL, TR, BL, BR)
        sem_global: torch.Tensor,       # [B, 768]
        phys_tokens: torch.Tensor,      # [B, 5, phys_dim]
        phys_global: torch.Tensor,      # [B, 2 * phys_dim]
        phys_conf: torch.Tensor,        # [B, 5, 4]
        physics_logits: torch.Tensor,   # [B, 1]
    ) -> Dict[str, torch.Tensor]:
        B = sem_tokens.shape[0]

        # 1. Project into Shared Alignment Space
        q_sem = self.sem_proj(sem_tokens)     # [B, 5, 256]
        k_phys = self.phys_proj(phys_tokens)  # [B, 5, 256]

        # 2. Explicit Physical-Semantic Cosine Alignment Metric
        norm_q = F.normalize(q_sem, p=2, dim=-1)   # [B, 5, 256]
        norm_k = F.normalize(k_phys, p=2, dim=-1)  # [B, 5, 256]

        # Diagonal alignment per region [B, 5]
        diag_sim = (norm_q * norm_k).sum(dim=-1)   # [B, 5] in [-1, 1]
        mean_align = diag_sim.mean(dim=-1, keepdim=True)         # [B, 1]
        quad_misalign = 1.0 - diag_sim[:, 1:]                    # [B, 4] in [0, 2]
        max_misalign = quad_misalign.max(dim=-1, keepdim=True)[0]# [B, 1]
        var_misalign = quad_misalign.var(dim=-1, keepdim=True)   # [B, 1]

        # 3. Bidirectional Cross-Attention
        # Path A: Sem -> Phys
        attn_sem_phys, _ = self.sem_to_phys_attn(
            query=q_sem, key=k_phys, value=k_phys
        )
        fused_sem = self.norm_sem_cross(q_sem + attn_sem_phys)  # [B, 5, 256]

        # Path B: Phys -> Sem
        attn_phys_sem, _ = self.phys_to_sem_attn(
            query=k_phys, key=q_sem, value=q_sem
        )
        fused_phys = self.norm_phys_cross(k_phys + attn_phys_sem) # [B, 5, 256]

        # 4. Multi-Instance Spatial Anomaly Pooling (MIL)
        # Quadrant tokens: [B, 4, 256]
        quad_joint = torch.cat([fused_sem[:, 1:], fused_phys[:, 1:]], dim=-1) # [B, 4, 512]
        quad_anomaly_scores = self.quad_anomaly_scorer(quad_joint).squeeze(-1) # [B, 4]
        
        # Softmax weights for max-anomaly pooling
        mil_weights = F.softmax(quad_anomaly_scores * 2.0, dim=-1).unsqueeze(-1) # [B, 4, 1]
        max_anomaly_token = (fused_sem[:, 1:] * mil_weights).sum(dim=1) # [B, 256]
        mean_quad_token = fused_sem[:, 1:].mean(dim=1)                 # [B, 256]
        global_token = fused_sem[:, 0]                                  # [B, 256]
        phys_global_token = fused_phys[:, 0]                            # [B, 256]

        # Full joint spatial representation [B, 4 * 256 = 1024]
        fused_joint = torch.cat([
            global_token,
            mean_quad_token,
            max_anomaly_token,
            phys_global_token,
        ], dim=-1)

        # 5. Raw Stream Logits
        sem_logits = self.sem_head(sem_global)         # [B, 1]
        phys_logits = physics_logits                   # [B, 1]
        joint_logits = self.joint_mil_head(fused_joint)# [B, 1]

        prob_sem_raw = torch.sigmoid(sem_logits)
        prob_phys = torch.sigmoid(phys_logits)
        raw_discrepancy = torch.abs(prob_sem_raw - prob_phys)
        sem_conf_raw = 2.0 * torch.abs(prob_sem_raw - 0.5)
        phys_conf_val = 2.0 * torch.abs(prob_phys - 0.5)
        conf_disparity_raw = phys_conf_val - sem_conf_raw
        max_quad_anomaly = quad_anomaly_scores.max(dim=-1, keepdim=True)[0]
        region_obs = phys_conf.mean(dim=-1) # [B, 5]

        # 6. Evidential Temperature Calibration
        calib_feat = torch.cat([
            mean_align,
            max_misalign,
            var_misalign,
            quad_misalign,
            raw_discrepancy,
            torch.abs(sem_logits - phys_logits),
            max_quad_anomaly,
            region_obs,
            conf_disparity_raw,
        ], dim=-1) # [B, 16]

        temperature = 1.0 + F.softplus(self.calib_net(calib_feat)) # [B, 1] >= 1.0
        sem_logits_cal = sem_logits / temperature
        prob_sem = torch.sigmoid(sem_logits_cal)

        # 7. Disagreement-Aware Gate
        gate_sem_feat = self.gate_sem_proj(sem_global)     # [B, 64]
        gate_phys_feat = self.gate_phys_proj(phys_global)  # [B, 64]

        gate_in = torch.cat([
            gate_sem_feat,
            gate_phys_feat,
            prob_sem,
            prob_phys,
            torch.abs(prob_sem - prob_phys),
            mean_align,
            max_misalign,
            var_misalign,
            quad_misalign,
            max_quad_anomaly,
            region_obs,
            2.0 * torch.abs(prob_sem - 0.5),
            phys_conf_val,
        ], dim=-1) # [B, 64 + 64 + 20 = 148]

        alpha = self.gate_net(gate_in) # [B, 1]

        # Robust physical anchor: Joint MIL prediction + scaled physical signal
        phys_augmented = joint_logits + self.joint_scale * torch.tanh(phys_logits)

        # Fused verdict
        final_logits = (
            alpha * sem_logits_cal +
            (1.0 - alpha) * phys_augmented
        ).squeeze(-1)

        return {
            "logits": final_logits,
            "sem_logits": sem_logits_cal.squeeze(-1),
            "sem_logits_raw": sem_logits.squeeze(-1),
            "phys_logits": phys_logits.squeeze(-1),
            "joint_logits": joint_logits.squeeze(-1),
            "alpha": alpha.squeeze(-1),
            "temperature": temperature.squeeze(-1),
            "mean_align": mean_align.squeeze(-1),
            "max_misalign": max_misalign.squeeze(-1),
            "quad_anomaly_scores": quad_anomaly_scores,
            "region_observability": region_obs,
        }
