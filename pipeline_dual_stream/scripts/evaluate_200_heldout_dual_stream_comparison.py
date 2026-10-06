"""Comprehensive 200-Image Held-Out Chameleon Benchmark: Dual-Stream vs UnivFD & Single Streams.

Evaluates on the exact held-out test split of 200 raw Chameleon images (100 real, 100 fake, seed 42):
1. UnivFD (Ojha et al., CVPR 2023 - CLIP ViT-L/14 linear probe)
2. Standalone DINOv2 Semantic Stream
3. Standalone Regional Multi-Physics Stream
4. Universal v5 Calibrated (Baseline Gate)
5. Universal v5 Disagreement Gate (Proposed Full Model)

Outputs exact, reproducible metrics to reports/chameleon_200_heldout_dual_stream_benchmark.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

from src.models.hybrid_detector import DualStreamHybridDetector


class UnivFD(nn.Module):
    def __init__(self, clip_path: Path, fc_path: Path, device: str = "cuda"):
        super().__init__()
        jit_model = torch.jit.load(str(clip_path), map_location="cpu")
        self.visual = jit_model.visual.to(device).eval()
        for p in self.visual.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(768, 1).to(device)
        self.fc.load_state_dict(torch.load(str(fc_path), map_location="cpu", weights_only=False))
        self.fc.eval()
        for p in self.fc.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x.type(self.visual.conv1.weight.dtype))
        return self.fc(features.float()).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 200 Held-Out Chameleon Samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out_json",
        type=str,
        default="reports/chameleon_200_heldout_dual_stream_benchmark.json",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("200-IMAGE HELD-OUT CHAMELEON BENCHMARK: FULL DUAL-STREAM & UNIVFD COMPARISON")
    print(f"Device: {args.device}")
    print("=" * 80)

    # 1. Deterministically sample the exact 200 images
    all_real = sorted(glob.glob(str(ROOT_DIR / "data/Chameleon/Chameleon/test/0_real/*.jpg")))
    all_fake = sorted(glob.glob(str(ROOT_DIR / "data/Chameleon/Chameleon/test/1_fake/*.jpg")))
    np.random.seed(42)
    np.random.shuffle(all_real)
    np.random.shuffle(all_fake)

    sel_real = all_real[:100]
    sel_fake = all_fake[:100]
    samples = [(p, 0) for p in sel_real] + [(p, 1) for p in sel_fake]
    np.random.seed(42)
    np.random.shuffle(samples)

    labels = np.array([s[1] for s in samples])
    filenames = [os.path.basename(s[0]) for s in samples]
    image_paths = [s[0] for s in samples]

    # 2. Load DINOv2 and Physics features from cache for exact matching
    cham = torch.load(ROOT_DIR / "data/universal_v4/chameleon/chameleon_cache.pt", map_location="cpu", weights_only=True)
    name_to_idx = {name: i for i, name in enumerate(cham["filenames"])}
    indices = [name_to_idx[f] for f in filenames]

    dinov2_cls = cham["dinov2_cls"][indices].to(args.device)
    dinov2_reg = cham["dinov2_regional"][indices].to(args.device)
    phys_f = cham["physics_features"][indices]
    phys_c = cham["physics_confidences"][indices].to(args.device)

    # 3. Evaluate UnivFD
    weights_dir = ROOT_DIR / "models" / "baselines" / "univfd"
    clip_path = weights_dir / "ViT-L-14.pt"
    fc_path = weights_dir / "fc_weights.pth"

    print("Evaluating UnivFD (Ojha et al., CVPR 2023)...")
    univfd_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    jit_clip = torch.jit.load(str(clip_path), map_location="cpu").visual.to(args.device).eval()
    univfd_fc = nn.Linear(768, 1).to(args.device)
    univfd_fc.load_state_dict(torch.load(str(fc_path), map_location="cpu", weights_only=False))
    univfd_fc.eval()

    univfd_probs = []
    with torch.no_grad():
        for i in range(0, len(image_paths), 64):
            batch_tensors = [univfd_transform(Image.open(p).convert("RGB")) for p in image_paths[i : i + 64]]
            x = torch.stack(batch_tensors).to(args.device)
            feats = jit_clip(x.type(jit_clip.conv1.weight.dtype))
            logits = univfd_fc(feats.float()).squeeze(-1)
            univfd_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    p_univfd = np.array(univfd_probs)

    # 4. Helper function to evaluate Dual-Stream checkpoints
    def run_dual_stream(ckpt_path: Path, gate_mode: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        std = ckpt["standardizer"]
        mean = torch.tensor(std["mean"])
        scale = torch.tensor(std["scale"])
        norm_pf = ((phys_f - mean.unsqueeze(0)) / scale.unsqueeze(0)).to(args.device)

        model = DualStreamHybridDetector(
            load_pretrained_dinov2=False,
            gate_mode=gate_mode,
            dropout=0.15,
        ).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            out = model(
                physics_features=norm_pf,
                physics_confidences=phys_c,
                dinov2_cls=dinov2_cls,
                dinov2_regional=dinov2_reg,
            )
            p_final = out["prob_final"].cpu().numpy()
            p_sem = out["prob_semantic"].cpu().numpy()
            p_phys = out["prob_physics"].cpu().numpy()
            alpha = out["alpha"].cpu().numpy()
            temp = out["temperature"].cpu().numpy() if "temperature" in out else np.ones_like(alpha)

        return p_final, p_sem, p_phys, alpha, temp

    print("Evaluating Universal v5 Calibrated (Baseline Gate)...")
    v5_path = ROOT_DIR / "models/universal_v5_calibrated/best_model.pt"
    p_final_v5, p_sem_v5, p_phys_v5, alpha_v5, temp_v5 = run_dual_stream(v5_path, "v3_calibrated")

    print("Evaluating Universal v5 Disagreement Gate (Proposed Full Model)...")
    disagree_path = ROOT_DIR / "models/universal_v5_disagreement_gate/best_model.pt"
    p_final_dis, p_sem_dis, p_phys_dis, alpha_dis, temp_dis = run_dual_stream(disagree_path, "v4_disagreement")

    # 5. Compute Metrics
    def calc_metrics(y, p):
        pred = (p >= 0.5).astype(int)
        acc = float(accuracy_score(y, pred))
        auc = float(roc_auc_score(y, p))
        real_mask = y == 0
        fake_mask = y == 1
        fp = int(((p >= 0.5) & real_mask).sum())
        fn = int(((p < 0.5) & fake_mask).sum())
        fpr = float(fp / max(1, real_mask.sum()))
        fnr = float(fn / max(1, fake_mask.sum()))
        return {
            "accuracy": acc,
            "roc_auc": auc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "false_positives": fp,
            "false_negatives": fn,
            "mean_fake_prob_real": float(p[real_mask].mean()),
            "mean_fake_prob_fake": float(p[fake_mask].mean()),
        }

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "Chameleon Held-Out Raw Images (200 Samples: 100 Real, 100 Fake)",
        "models": {
            "univfd_clip_baseline": {
                "description": "UnivFD (Ojha et al., CVPR 2023 - CLIP ViT-L/14 Linear Probe)",
                **calc_metrics(labels, p_univfd),
            },
            "physics_stream_alone": {
                "description": "Regional Multi-Physics Alone (DSINE Normals, SH Lighting, Shadows, Glints)",
                **calc_metrics(labels, p_phys_dis),
            },
            "dinov2_semantic_alone": {
                "description": "DINOv2 ViT-Base Semantic Stream Alone",
                **calc_metrics(labels, p_sem_dis),
            },
            "universal_v5_calibrated_baseline": {
                "description": "Universal v5 Calibrated (Standard Gate without Disagreement Exposure)",
                "mean_alpha": float(alpha_v5.mean()),
                "mean_temperature": float(temp_v5.mean()),
                **calc_metrics(labels, p_final_v5),
            },
            "universal_v5_disagreement_gate_proposed": {
                "description": "Proposed Dual-Stream Model (Disagreement-Exposed Gate + Evidential Calibration)",
                "mean_alpha": float(alpha_dis.mean()),
                "mean_temperature": float(temp_dis.mean()),
                **calc_metrics(labels, p_final_dis),
            },
        },
        "comparisons": {
            "dual_stream_vs_univfd_auc_delta": float(roc_auc_score(labels, p_final_dis) - roc_auc_score(labels, p_univfd)),
            "dual_stream_vs_univfd_acc_delta": float(accuracy_score(labels, (p_final_dis >= 0.5).astype(int)) - accuracy_score(labels, (p_univfd >= 0.5).astype(int))),
            "dual_stream_vs_dinov2_auc_delta": float(roc_auc_score(labels, p_final_dis) - roc_auc_score(labels, p_sem_dis)),
            "dual_stream_vs_dinov2_acc_delta": float(accuracy_score(labels, (p_final_dis >= 0.5).astype(int)) - accuracy_score(labels, (p_sem_dis >= 0.5).astype(int))),
            "dual_stream_vs_v5_baseline_auc_delta": float(roc_auc_score(labels, p_final_dis) - roc_auc_score(labels, p_final_v5)),
            "dual_stream_vs_v5_baseline_acc_delta": float(accuracy_score(labels, (p_final_dis >= 0.5).astype(int)) - accuracy_score(labels, (p_final_v5 >= 0.5).astype(int))),
        },
    }

    out_file = ROOT_DIR / args.out_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("200-IMAGE HELD-OUT CHAMELEON BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model / Architecture':<42} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'FPR':<8} | {'FNR':<8}")
    print("-" * 86)
    m = results["models"]
    for k in ["univfd_clip_baseline", "physics_stream_alone", "dinov2_semantic_alone", "universal_v5_calibrated_baseline", "universal_v5_disagreement_gate_proposed"]:
        row = m[k]
        print(f"{row['description'][:42]:<42} | {row['accuracy']*100:6.2f}%    | {row['roc_auc']:8.4f}   | {row['false_positive_rate']*100:5.1f}%  | {row['false_negative_rate']*100:5.1f}%")
    print("=" * 80)
    print(f"Dual-Stream vs UnivFD Delta AUC    : {results['comparisons']['dual_stream_vs_univfd_auc_delta']:+.4f} ({results['comparisons']['dual_stream_vs_univfd_acc_delta']*100:+.2f}% Acc)")
    print(f"Dual-Stream vs DINOv2 Delta AUC    : {results['comparisons']['dual_stream_vs_dinov2_auc_delta']:+.4f} ({results['comparisons']['dual_stream_vs_dinov2_acc_delta']*100:+.2f}% Acc)")
    print(f"Dual-Stream vs v5 Base Delta AUC   : {results['comparisons']['dual_stream_vs_v5_baseline_auc_delta']:+.4f} ({results['comparisons']['dual_stream_vs_v5_baseline_acc_delta']*100:+.2f}% Acc)")
    print(f"Saved results artifact to          : {out_file.relative_to(ROOT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
