"""Gated Cross-Attention Fusion & Decision Gating Head.
Fuses DINOv2 semantic features with Regional Multi-Physics tokens.
Produces explainable multi-stream verdicts and per-quadrant physical-semantic discrepancy maps.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionFusion(nn.Module):
    """Confidence-guided cross-attention and dynamic trust gating."""

    def __init__(
        self,
        sem_dim: int = 768,
        phys_dim: int = 64,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.sem_dim = sem_dim
        self.phys_dim = phys_dim
        self.proj_dim = proj_dim

        # 1. Projections into shared latent space
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Cross-Attention: Semantic Queries attend to Physics Keys/Values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(proj_dim)

        # 3. Standalone Stream Classifiers (for strict ablation & multi-task loss)
        self.sem_head = nn.Sequential(
            nn.Linear(sem_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

        # 4. Joint Cross-Attended Classifier Head
        self.joint_head = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

        # 5. Dynamic Gating Network: computes trust balance alpha in [0, 1]
        # Input: [sem_global, phys_global, mean_confidences]
        gate_in_dim = sem_dim + (2 * phys_dim) + 5
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Learned blending parameter for joint residual
        self.joint_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        sem_tokens: torch.Tensor,       # [B, 5, 768] (Global, TL, TR, BL, BR)
        sem_global: torch.Tensor,       # [B, 768]
        phys_tokens: torch.Tensor,      # [B, 5, phys_dim]
        phys_global: torch.Tensor,      # [B, 2 * phys_dim]
        phys_conf: torch.Tensor,        # [B, 5, 4]
        physics_logits: torch.Tensor,   # [B, 1]
    ) -> Dict[str, torch.Tensor]:
        batch_size = sem_tokens.shape[0]

        # 1. Project to shared dimension
        q_sem = self.sem_proj(sem_tokens)     # [B, 5, proj_dim]
        k_phys = self.phys_proj(phys_tokens)  # [B, 5, proj_dim]
        v_phys = k_phys

        # 2. Regional observability bias for cross-attention
        region_obs = phys_conf.mean(dim=-1)   # [B, 5]
        # In multihead attention, key_padding_mask or soft bias can guide attention
        attn_out, attn_weights = self.cross_attn(
            query=q_sem,
            key=k_phys,
            value=v_phys,
        )  # [B, 5, proj_dim]

        # Residual connection + LayerNorm
        fused_tokens = self.norm_cross(q_sem + attn_out)  # [B, 5, proj_dim]

        # 3. Global summaries
        fused_global = fused_tokens[:, 0]  # [B, proj_dim]
        fused_quads = fused_tokens[:, 1:].mean(dim=1)  # [B, proj_dim]
        fused_joint = torch.cat([fused_global, fused_quads], dim=-1)  # [B, 2 * proj_dim]

        # 4. Stream Logits
        sem_logits = self.sem_head(sem_global)          # [B, 1]
        phys_logits = physics_logits                    # [B, 1]
        joint_logits = self.joint_head(fused_joint)     # [B, 1]

        # 5. Dynamic Trust Gate alpha
        gate_input = torch.cat([sem_global, phys_global, region_obs], dim=-1)
        alpha = self.gate_net(gate_input)  # [B, 1] in [0, 1] (1 = Semantic trust, 0 = Physics trust)

        # 6. Final Decision Fusion:
        # Blends semantic and physics according to dynamic trust, plus joint cross-attention residual
        final_logits = (
            alpha * sem_logits +
            (1.0 - alpha) * phys_logits +
            self.joint_scale * joint_logits
        ).squeeze(-1)

        # 7. Spatial Discrepancy Heatmap (Cosine discrepancy between semantic & physics quadrants)
        # Quadrants 1..4 (TL, TR, BL, BR)
        sem_quads = F.normalize(q_sem[:, 1:], p=2, dim=-1)   # [B, 4, proj_dim]
        phys_quads = F.normalize(k_phys[:, 1:], p=2, dim=-1) # [B, 4, proj_dim]
        # Discrepancy = 1.0 - cosine_similarity
        quad_discrepancy = 1.0 - (sem_quads * phys_quads).sum(dim=-1)  # [B, 4]

        return {
            "logits": final_logits,
            "sem_logits": sem_logits.squeeze(-1),
            "phys_logits": phys_logits.squeeze(-1),
            "joint_logits": joint_logits.squeeze(-1),
            "alpha": alpha.squeeze(-1),
            "quad_discrepancy": quad_discrepancy,
            "attn_weights": attn_weights,
            "region_observability": region_obs,
        }


def dual_stream_hybrid_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    sem_weight: float = 0.25,
    phys_weight: float = 0.25,
    joint_weight: float = 0.20,
) -> Dict[str, torch.Tensor]:
    """Multi-task loss ensuring both individual streams and joint fusion are supervised."""
    labels = labels.float().reshape(-1)

    loss_final = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
    loss_sem = F.binary_cross_entropy_with_logits(outputs["sem_logits"], labels)
    loss_phys = F.binary_cross_entropy_with_logits(outputs["phys_logits"], labels)
    loss_joint = F.binary_cross_entropy_with_logits(outputs["joint_logits"], labels)

    total_loss = (
        loss_final +
        sem_weight * loss_sem +
        phys_weight * loss_phys +
        joint_weight * loss_joint
    )

    return {
        "total_loss": total_loss,
        "loss_final": loss_final,
        "loss_sem": loss_sem,
        "loss_phys": loss_phys,
        "loss_joint": loss_joint,
    }
