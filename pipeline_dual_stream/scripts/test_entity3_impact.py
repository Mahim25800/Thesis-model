import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.hybrid_detector import DualStreamHybridDetector

ckpt = torch.load("models/universal_v5_calibrated/best_model.pt", map_location="cpu", weights_only=False)
standardizer = ckpt["standardizer"]
model = DualStreamHybridDetector(load_pretrained_dinov2=False, gate_mode="v3_calibrated", dropout=0.15).to("cuda")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

cham = torch.load("data/universal_v4/chameleon/chameleon_cache.pt", map_location="cpu", weights_only=True)
pf = cham["physics_features"]
pc = cham["physics_confidences"]
dc = cham["dinov2_cls"]
dr = cham["dinov2_regional"]
labels = cham["labels"].numpy()

mean = torch.tensor(standardizer["mean"])
scale = torch.tensor(standardizer["scale"])


def eval_physics(pf_t, pc_t):
    phys_probs = []
    final_probs = []
    with torch.no_grad():
        for i in range(0, len(labels), 256):
            b_pf = ((pf_t[i : i + 256] - mean.unsqueeze(0)) / scale.unsqueeze(0)).to("cuda")
            b_pc = pc_t[i : i + 256].to("cuda")
            b_dc = dc[i : i + 256].to("cuda")
            b_dr = dr[i : i + 256].to("cuda")
            out = model(physics_features=b_pf, physics_confidences=b_pc, dinov2_cls=b_dc, dinov2_regional=b_dr)
            phys_probs.extend(out["prob_physics"].cpu().numpy().tolist())
            final_probs.extend(out["prob_final"].cpu().numpy().tolist())
    p_p = np.array(phys_probs)
    p_f = np.array(final_probs)
    return {
        "phys_acc": float(accuracy_score(labels, (p_p >= 0.5).astype(int))),
        "phys_auc": float(roc_auc_score(labels, p_p)),
        "final_acc": float(accuracy_score(labels, (p_f >= 0.5).astype(int))),
        "final_auc": float(roc_auc_score(labels, p_f)),
    }


print("Baseline:")
res_base = eval_physics(pf, pc)
print(f"  Physics Acc: {res_base['phys_acc']*100:.2f}% | Physics AUC: {res_base['phys_auc']:.4f}")
print(f"  Final Acc:   {res_base['final_acc']*100:.2f}% | Final AUC:   {res_base['final_auc']:.4f}")

# Neutralize Entity 3 (Chromatic Shadow)
pf_no_e3 = pf.clone()
pc_no_e3 = pc.clone()
pf_no_e3[:, :, 12:14] = 0.0
pc_no_e3[:, :, 3] = 0.0

print("\nNeutralizing Entity 3 (Chromatic Shadow):")
res_no_e3 = eval_physics(pf_no_e3, pc_no_e3)
print(f"  Physics Acc: {res_no_e3['phys_acc']*100:.2f}% | Physics AUC: {res_no_e3['phys_auc']:.4f}")
print(f"  Final Acc:   {res_no_e3['final_acc']*100:.2f}% | Final AUC:   {res_no_e3['final_auc']:.4f}")

# Also test neutralizing Entity 0 (Illumination)
pf_no_e0_e3 = pf_no_e3.clone()
pc_no_e0_e3 = pc_no_e3.clone()
pf_no_e0_e3[:, :, 0:5] = 0.0
pc_no_e0_e3[:, :, 0] = 0.0

print("\nNeutralizing BOTH Entity 3 (Shadow) and Entity 0 (Illumination):")
res_no_e0_e3 = eval_physics(pf_no_e0_e3, pc_no_e0_e3)
print(f"  Physics Acc: {res_no_e0_e3['phys_acc']*100:.2f}% | Physics AUC: {res_no_e0_e3['phys_auc']:.4f}")
print(f"  Final Acc:   {res_no_e0_e3['final_acc']*100:.2f}% | Final AUC:   {res_no_e0_e3['final_auc']:.4f}")
