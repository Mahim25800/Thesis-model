"""Analyze feature direction alignment between Training Corpus and Chameleon Benchmark."""

from pathlib import Path
import numpy as np
import torch

train = torch.load("data/universal_v4/universal_train_corpus.pt", map_location="cpu", weights_only=True)
cham = torch.load("data/universal_v4/chameleon/chameleon_cache.pt", map_location="cpu", weights_only=True)

pf_tr = train["physics_features"][:, 0, :].numpy()
y_tr = train["labels"].numpy()

pf_ch = cham["physics_features"][:, 0, :].numpy()
y_ch = cham["labels"].numpy()

feat_names = [
    "SH_0", "SH_1", "SH_2", "SH_3", "SH_4",
    "Spec_dx", "Spec_dy", "Spec_theta", "Spec_prof",
    "DSINE_var", "DSINE_skew", "DSINE_kurt",
    "Chroma_RG", "Chroma_BG",
]

print(f"{'Feature':<14} {'Train Diff (Fake-Real)':<24} {'Cham Diff (Fake-Real)':<24} {'Alignment':<16}")
print("-" * 80)

invariant_indices = []
for i, name in enumerate(feat_names):
    tr_diff = pf_tr[y_tr == 1, i].mean() - pf_tr[y_tr == 0, i].mean()
    ch_diff = pf_ch[y_ch == 1, i].mean() - pf_ch[y_ch == 0, i].mean()
    same_sign = (tr_diff * ch_diff) > 0
    align_str = "YES (Consistent)" if same_sign else "INVERTED (Shortcut!)"
    if same_sign:
        invariant_indices.append(i)
    print(f"{name:<14} {tr_diff:<+24.4f} {ch_diff:<+24.4f} {align_str:<16}")

print("\nTruly Invariant Features across both datasets:")
for idx in invariant_indices:
    print(f"  - [{idx}] {feat_names[idx]}")
