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
        gate_mode: str = "v2",
    ):
        super().__init__()
        self.sem_dim = sem_dim
        self.phys_dim = phys_dim
        self.proj_dim = proj_dim
        self.gate_mode = gate_mode

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
        if self.gate_mode == "v1":
            gate_in_dim = sem_dim + (2 * phys_dim) + 5  # 901
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, proj_dim // 2),
                nn.LayerNorm(proj_dim // 2),
                nn.GELU(),
                nn.Linear(proj_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate_sem_proj = nn.Sequential(
                nn.Linear(sem_dim, proj_dim // 2),
                nn.LayerNorm(proj_dim // 2),
                nn.GELU(),
            )
            self.gate_phys_proj = nn.Sequential(
                nn.Linear(2 * phys_dim, proj_dim // 2),
                nn.LayerNorm(proj_dim // 2),
                nn.GELU(),
            )
            gate_in_dim = (proj_dim // 2) + (proj_dim // 2) + 17
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, proj_dim // 2),
                nn.LayerNorm(proj_dim // 2),
                nn.GELU(),
                nn.Linear(proj_dim // 2, 1),
                nn.Sigmoid(),
            )
            self.gate_net[-2].bias.data.fill_(0.0)

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

        # 5. Spatial Discrepancy Heatmap (Cosine discrepancy between semantic & physics quadrants)
        sem_quads = F.normalize(q_sem[:, 1:], p=2, dim=-1)   # [B, 4, proj_dim]
        phys_quads = F.normalize(k_phys[:, 1:], p=2, dim=-1) # [B, 4, proj_dim]
        quad_discrepancy = 1.0 - (sem_quads * phys_quads).sum(dim=-1)  # [B, 4]

        # 6. Trust Gate alpha and Decision Fusion
        if self.gate_mode == "v1":
            gate_input = torch.cat([sem_global, phys_global, region_obs], dim=-1)
            alpha = self.gate_net(gate_input)
            final_logits = (
                alpha * sem_logits +
                (1.0 - alpha) * phys_logits +
                self.joint_scale * joint_logits
            ).squeeze(-1)
        else:
            prob_sem = torch.sigmoid(sem_logits)
            prob_phys = torch.sigmoid(phys_logits)
            stream_discrepancy = torch.abs(prob_sem - prob_phys)
            sem_conf = 2.0 * torch.abs(prob_sem - 0.5)
            phys_conf = 2.0 * torch.abs(prob_phys - 0.5)
            conf_disparity = phys_conf - sem_conf
            phys_fake_signal = 2.0 * torch.relu(prob_phys - 0.5)
            sem_fake_signal = 2.0 * torch.relu(prob_sem - 0.5)

            gate_sem_feat = self.gate_sem_proj(sem_global)     # [B, 64]
            gate_phys_feat = self.gate_phys_proj(phys_global)  # [B, 64]

            gate_input = torch.cat([
                gate_sem_feat,
                gate_phys_feat,
                prob_sem,
                prob_phys,
                stream_discrepancy,
                quad_discrepancy,
                region_obs,
                sem_conf,
                phys_conf,
                conf_disparity,
                phys_fake_signal,
                sem_fake_signal,
            ], dim=-1) # [B, 145]

            alpha = self.gate_net(gate_input)  # [B, 1] in [0, 1]

            phys_augmented = phys_logits + self.joint_scale * torch.tanh(joint_logits) * 2.0
            final_logits = (
                alpha * sem_logits +
                (1.0 - alpha) * phys_augmented
            ).squeeze(-1)

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
    gate_weight: float = 0.20,
) -> Dict[str, torch.Tensor]:
    """Multi-task loss with dynamic annealing and Oracle-supervised gate training.
    
    1. Base classification losses:
       - loss_final: BCE on blended prediction (primary objective)
       - loss_sem: standalone supervision on semantic stream
       - loss_phys: standalone supervision on physics stream
       - loss_joint: cross-attended joint head loss
    
    2. Oracle Relative-Error Gate Supervision:
       - Measures individual stream prediction errors |P_sem - y| and |P_phys - y|
       - Computes optimal oracle trust target alpha* = err_phys^2 / (err_sem^2 + err_phys^2 + eps)
       - Directly supervises gate network to route towards whichever stream is genuinely closer to the truth!
    """
    labels = labels.float().reshape(-1)

    loss_final = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
    loss_sem = F.binary_cross_entropy_with_logits(outputs["sem_logits"], labels)
    loss_phys = F.binary_cross_entropy_with_logits(outputs["phys_logits"], labels)
    loss_joint = F.binary_cross_entropy_with_logits(outputs["joint_logits"], labels)

    # Balanced Competitive Oracle Target for Gate alpha:
    p_sem = torch.sigmoid(outputs["sem_logits"]).detach()
    p_phys = torch.sigmoid(outputs["phys_logits"]).detach()
    labels_bool = (labels >= 0.5)
    sem_correct = ((p_sem >= 0.5) == labels_bool).float()
    phys_correct = ((p_phys >= 0.5) == labels_bool).float()
    err_sem = torch.abs(p_sem - labels)
    err_phys = torch.abs(p_phys - labels)

    alpha_target = torch.full_like(p_sem, 0.50)
    # Direction 1: Semantic correct, Physics wrong (e.g. digital art/anime) -> trust Semantic
    alpha_target[(sem_correct == 1.0) & (phys_correct == 0.0)] = 0.92
    # Direction 2: Physics correct, Semantic wrong (e.g. photorealistic deepfakes) -> trust Physics
    alpha_target[(sem_correct == 0.0) & (phys_correct == 1.0)] = 0.08
    # Direction 3: Both correct -> gentle calibration around 0.50 based on error margin
    both_correct = (sem_correct == 1.0) & (phys_correct == 1.0)
    margin_diff = torch.clamp(err_phys - err_sem, -0.3, 0.3)
    alpha_target[both_correct] = 0.50 + margin_diff[both_correct] * 0.5

    pred_alpha = torch.clamp(outputs["alpha"], 1e-6, 1.0 - 1e-6)
    loss_gate = F.binary_cross_entropy(pred_alpha, alpha_target)

    total_loss = (
        loss_final +
        sem_weight * loss_sem +
        phys_weight * loss_phys +
        joint_weight * loss_joint +
        gate_weight * loss_gate
    )

    return {
        "total_loss": total_loss,
        "loss_final": loss_final,
        "loss_sem": loss_sem,
        "loss_phys": loss_phys,
        "loss_joint": loss_joint,
        "loss_gate": loss_gate,
        "alpha_target_mean": alpha_target.mean(),
    }
