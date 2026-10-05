"""Empirical End-to-End Evaluation of Modernized vs Legacy Extractors on Chameleon.

Swaps modernized extractors into Universal v5 Calibrated and Universal v4
and evaluates on the exact same 200 held-out Chameleon images.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from src.extractors.regional_physics import RegionalPhysicsExtractor
from src.extractors.surface_normals import SurfaceNormalsExtractor
from src.models.hybrid_detector import DualStreamHybridDetector


def main():
    print("=" * 80)
    print("END-TO-END EVALUATION: UNIVERSAL V5 & V4 WITH SWAPPED EXTRACTORS")
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

    # 2. Load DINOv2 cached representations for these 200 samples
    cham = torch.load(ROOT_DIR / "data/universal_v4/chameleon/chameleon_cache.pt", map_location="cpu", weights_only=True)
    name_to_idx = {name: i for i, name in enumerate(cham["filenames"])}
    indices = [name_to_idx[f] for f in filenames]

    dinov2_cls = cham["dinov2_cls"][indices]
    dinov2_reg = cham["dinov2_regional"][indices]

    # Legacy physics features from cache for exact same 200 samples
    legacy_pf = cham["physics_features"][indices]
    legacy_pc = cham["physics_confidences"][indices]

    # 3. Extract modernized physics features
    normals = SurfaceNormalsExtractor(normal_backend="dsine", dsine_device="cuda")
    extractor = RegionalPhysicsExtractor(normals=normals)

    print("Extracting modernized physics features directly from raw image pixels...")
    modern_pf_list = []
    modern_pc_list = []
    for p, _ in tqdm(samples, desc="Extracting"):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        f, c = extractor.extract(img)
        modern_pf_list.append(f)
        modern_pc_list.append(c)

    modern_pf = torch.stack(modern_pf_list)
    modern_pc = torch.stack(modern_pc_list)

    # 4. Evaluation Function
    def evaluate_model(model_path, gate_mode, model_name):
        ckpt = torch.load(model_path, map_location="cpu")
        standardizer = ckpt["standardizer"]
        mean = torch.tensor(standardizer["mean"])
        scale = torch.tensor(standardizer["scale"])

        model = DualStreamHybridDetector(load_pretrained_dinov2=False, gate_mode=gate_mode, dropout=0.15).cuda()
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        # Run with legacy physics
        with torch.no_grad():
            b_pf = ((legacy_pf - mean.unsqueeze(0)) / scale.unsqueeze(0)).cuda()
            b_pc = legacy_pc.cuda()
            out_leg = model(physics_features=b_pf, physics_confidences=b_pc, dinov2_cls=dinov2_cls.cuda(), dinov2_regional=dinov2_reg.cuda())
            p_final_leg = out_leg["prob_final"].cpu().numpy()
            p_phys_leg = out_leg["prob_physics"].cpu().numpy()
            p_sem_leg = out_leg["prob_semantic"].cpu().numpy()
            alpha_leg = out_leg["alpha"].cpu().numpy()

        # Run with modern physics
        with torch.no_grad():
            b_pf_mod = ((modern_pf - mean.unsqueeze(0)) / scale.unsqueeze(0)).cuda()
            b_pc_mod = modern_pc.cuda()
            out_mod = model(physics_features=b_pf_mod, physics_confidences=b_pc_mod, dinov2_cls=dinov2_cls.cuda(), dinov2_regional=dinov2_reg.cuda())
            p_final_mod = out_mod["prob_final"].cpu().numpy()
            p_phys_mod = out_mod["prob_physics"].cpu().numpy()
            p_sem_mod = out_mod["prob_semantic"].cpu().numpy()
            alpha_mod = out_mod["alpha"].cpu().numpy()

        return {
            "model_name": model_name,
            "semantic_stream": {
                "acc": float(accuracy_score(labels, (p_sem_leg >= 0.5).astype(int))),
                "auc": float(roc_auc_score(labels, p_sem_leg)),
            },
            "legacy": {
                "final_acc": float(accuracy_score(labels, (p_final_leg >= 0.5).astype(int))),
                "final_auc": float(roc_auc_score(labels, p_final_leg)),
                "phys_acc": float(accuracy_score(labels, (p_phys_leg >= 0.5).astype(int))),
                "phys_auc": float(roc_auc_score(labels, p_phys_leg)),
                "mean_alpha": float(np.mean(alpha_leg)),
            },
            "modernized": {
                "final_acc": float(accuracy_score(labels, (p_final_mod >= 0.5).astype(int))),
                "final_auc": float(roc_auc_score(labels, p_final_mod)),
                "phys_acc": float(accuracy_score(labels, (p_phys_mod >= 0.5).astype(int))),
                "phys_auc": float(roc_auc_score(labels, p_phys_mod)),
                "mean_alpha": float(np.mean(alpha_mod)),
            },
        }

    v5_res = evaluate_model(ROOT_DIR / "models/universal_v5_calibrated/best_model.pt", "v3_calibrated", "Universal v5 Calibrated")
    v4_res = evaluate_model(ROOT_DIR / "models/universal_v4/best_model.pt", "v2_confidence_adaptive", "Universal v4")

    print("\n" + "=" * 80)
    print("EMPIRICAL END-TO-END COMPARISON ON 200 HELD-OUT CHAMELEON SAMPLES")
    print("=" * 80)
    for r in [v5_res, v4_res]:
        name = r["model_name"]
        print(f"\nModel: {name}")
        print(f"  Semantic Stream Alone: Acc = {r['semantic_stream']['acc']*100:.2f}%, AUC = {r['semantic_stream']['auc']:.4f}")
        print(f"  1. With Legacy Extractors:")
        print(f"     Physics Stream: Acc = {r['legacy']['phys_acc']*100:.2f}%, AUC = {r['legacy']['phys_auc']:.4f}")
        print(f"     Final Fused:    Acc = {r['legacy']['final_acc']*100:.2f}%, AUC = {r['legacy']['final_auc']:.4f}")
        print(f"     Mean Alpha:     {r['legacy']['mean_alpha']:.4f}")
        print(f"  2. With Modernized Extractors Swapped In:")
        print(f"     Physics Stream: Acc = {r['modernized']['phys_acc']*100:.2f}%, AUC = {r['modernized']['phys_auc']:.4f}")
        print(f"     Final Fused:    Acc = {r['modernized']['final_acc']*100:.2f}%, AUC = {r['modernized']['final_auc']:.4f}")
        print(f"     Mean Alpha:     {r['modernized']['mean_alpha']:.4f}")
        delta_auc = r["modernized"]["final_auc"] - r["legacy"]["final_auc"]
        delta_acc = r["modernized"]["final_acc"] - r["legacy"]["final_acc"]
        print(f"  --> Delta Fused AUC: {delta_auc:+.4f} | Delta Fused Acc: {delta_acc*100:+.2f}%")

    # Save to JSON
    out_json = ROOT_DIR / "reports/modernized_extractor_end_to_end_chameleon.json"
    with open(out_json, "w") as f:
        json.dump({"v5_calibrated": v5_res, "v4_baseline": v4_res}, f, indent=2)
    print(f"\nSaved raw results artifact to {out_json}")


if __name__ == "__main__":
    main()
