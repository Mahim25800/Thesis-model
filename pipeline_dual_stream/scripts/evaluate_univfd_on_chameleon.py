"""Evaluate UnivFD (Ojha et al., CVPR 2023) Baseline on Chameleon Benchmark.

UnivFD Architecture:
- Backbone: Frozen CLIP ViT-L/14 visual encoder (768-dim projection).
- Classifier: Linear probe (fc_weights.pth) trained on ProGAN (20 classes, 720k images).

Downloads official weights directly from official sources:
- Backbone: OpenAI official ViT-L-14.pt
- Linear Probe: Ojha et al. official fc_weights.pth

Evaluates on:
1. Held-out 200 raw Chameleon test images (identical split to evaluate_modernized_end_to_end.py)
2. Full Chameleon benchmark (or subset)
Saves raw JSON to reports/univfd_chameleon_eval.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request
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


CLIP_VIT_L14_URL = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
UNIVFD_FC_URL = "https://huggingface.co/dkarageo/sidbench/resolve/main/univfd/fc_weights.pth"


def download_with_progress(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"File already exists: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"Downloading {output_path.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name)
        while True:
            chunk = resp.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
    print(f"Successfully downloaded to {output_path}")


class UnivFD(nn.Module):
    """Universal Fake Detector (Ojha et al., CVPR 2023)."""

    def __init__(self, clip_model_path: Path, fc_weights_path: Path, device: str = "cuda"):
        super().__init__()
        self.device = device
        print(f"Loading CLIP ViT-L/14 from {clip_model_path.name}...")
        jit_model = torch.jit.load(str(clip_model_path), map_location="cpu")
        self.visual = jit_model.visual.to(device).eval()
        for p in self.visual.parameters():
            p.requires_grad = False

        print(f"Loading UnivFD linear probe from {fc_weights_path.name}...")
        self.fc = nn.Linear(768, 1).to(device)
        state_dict = torch.load(str(fc_weights_path), map_location="cpu", weights_only=False)
        self.fc.load_state_dict(state_dict)
        self.fc.eval()
        for p in self.fc.parameters():
            p.requires_grad = False

        # CLIP ViT-L/14 standard bicubic normalization
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
        """x: [B, 3, 224, 224] -> logits: [B, 1]"""
        # visual encoder in CLIP JIT returns 768-dim projection
        features = self.visual(x.type(self.visual.conv1.weight.dtype))
        logits = self.fc(features.float())
        return logits.squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate UnivFD on Chameleon")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split", type=str, default="200_raw", choices=["200_raw", "full"])
    parser.add_argument("--out_json", type=str, default="reports/univfd_chameleon_eval.json")
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATING UNIVFD (OJHA ET AL., CVPR 2023) ON CHAMELEON")
    print(f"Split  : {args.split}")
    print(f"Device : {args.device}")
    print("=" * 80)

    # 1. Download official weights
    weights_dir = ROOT_DIR / "models" / "baselines" / "univfd"
    clip_path = weights_dir / "ViT-L-14.pt"
    fc_path = weights_dir / "fc_weights.pth"

    download_with_progress(UNIVFD_FC_URL, fc_path)
    download_with_progress(CLIP_VIT_L14_URL, clip_path)

    # 2. Build model
    univfd = UnivFD(clip_path, fc_path, device=args.device)

    # 3. Gather test samples
    if args.split == "200_raw":
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
    else:
        all_real = sorted(glob.glob(str(ROOT_DIR / "data/Chameleon/Chameleon/test/0_real/*.jpg")))
        all_fake = sorted(glob.glob(str(ROOT_DIR / "data/Chameleon/Chameleon/test/1_fake/*.jpg")))
        samples = [(p, 0) for p in all_real] + [(p, 1) for p in all_fake]

    print(f"Running inference on {len(samples):,} Chameleon images...")
    y_true = np.array([s[1] for s in samples])
    paths = [s[0] for s in samples]

    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), args.batch_size), desc="UnivFD Inference"):
            batch_paths = paths[i : i + args.batch_size]
            batch_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(univfd.transform(img))
            x = torch.stack(batch_tensors).to(args.device)
            logits = univfd(x)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
            all_probs.extend(probs)

    p_univfd = np.array(all_probs)
    pred_univfd = (p_univfd >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, pred_univfd))
    auc = float(roc_auc_score(y_true, p_univfd))

    # Real vs Fake breakdown
    real_mask = y_true == 0
    fake_mask = y_true == 1
    tn = int(((p_univfd < 0.5) & real_mask).sum())
    fp = int(((p_univfd >= 0.5) & real_mask).sum())
    fn = int(((p_univfd < 0.5) & fake_mask).sum())
    tp = int(((p_univfd >= 0.5) & fake_mask).sum())

    fpr = float(fp / max(1, real_mask.sum()))
    fnr = float(fn / max(1, fake_mask.sum()))

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "UnivFD (Ojha et al., CVPR 2023 - CLIP ViT-L/14 Linear Probe)",
        "split": args.split,
        "sample_count": len(samples),
        "num_real": int(real_mask.sum()),
        "num_fake": int(fake_mask.sum()),
        "metrics": {
            "accuracy": acc,
            "roc_auc": auc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "mean_fake_prob_real": float(p_univfd[real_mask].mean()),
            "mean_fake_prob_fake": float(p_univfd[fake_mask].mean()),
        },
    }

    out_file = ROOT_DIR / args.out_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("UNIVFD CHAMELEON EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy                 : {acc * 100:.2f}%")
    print(f"ROC-AUC                  : {auc:.4f}")
    print(f"False Positive Rate (FPR): {fpr * 100:.2f}% ({fp} / {real_mask.sum()})")
    print(f"False Negative Rate (FNR): {fnr * 100:.2f}% ({fn} / {fake_mask.sum()})")
    print(f"Mean Fake Prob on Real   : {p_univfd[real_mask].mean():.4f}")
    print(f"Mean Fake Prob on Fake   : {p_univfd[fake_mask].mean():.4f}")
    print(f"Saved results to         : {out_file.relative_to(ROOT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
