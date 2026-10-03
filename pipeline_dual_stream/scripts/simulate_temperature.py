"""Fast temperature scaling simulation on Chameleon predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_parquet("reports/chameleon_per_image_predictions.parquet")
y_true = df["label"].values
eps = 1e-6
p_sem = np.clip(df["prob_semantic"].values, eps, 1.0 - eps)
p_phys = np.clip(df["prob_physics"].values, eps, 1.0 - eps)
z_sem = np.log(p_sem / (1.0 - p_sem))
z_phys = np.log(p_phys / (1.0 - p_phys))
base_alpha = df["alpha"].values

dis_mask = (df["prob_semantic"] >= 0.5) != (df["prob_physics"] >= 0.5)

print("Temperature Scaling Simulation:")
print(f"{'T':<8} {'Accuracy (%)':<15} {'AUC':<12} {'Disagreement Acc (%)':<22}")
for T in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]:
    z_sem_scaled = z_sem / T
    z_fused = base_alpha * z_sem_scaled + (1.0 - base_alpha) * z_phys
    p_fused = 1.0 / (1.0 + np.exp(-z_fused))
    acc = accuracy_score(y_true, (p_fused >= 0.5).astype(int))
    auc = roc_auc_score(y_true, p_fused)
    dis_acc = accuracy_score(y_true[dis_mask], (p_fused[dis_mask] >= 0.5).astype(int))
    print(f"{T:<8.1f} {acc * 100:<15.2f} {auc:<12.4f} {dis_acc * 100:<22.2f}")
