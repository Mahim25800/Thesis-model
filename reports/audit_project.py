"""Read-only project checks; writes only reports/project_audit.json.

Run from the repository root with .venv/Scripts/python.exe -B reports/audit_project.py.
These checks inspect the existing experiment; they do not establish a new benchmark.
"""
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_40k"))
from src.models.cross_gen_gated import TransformerPhysicsCrossGenHead, FocalLoss
from src.utils.metrics import compute_youden_threshold
from src.data.preprocessor import extract_eye_crops
from src.extractors.surface_normals import FoundationNormalHead

torch.set_num_threads(4)
torch.manual_seed(42)
report = {"scope": "Existing artifacts only; exploratory audit, not a fresh held-out benchmark"}
caches = {}
for split in ("train", "val", "test"):
    path = ROOT / f"pipeline_40k/data/cache_scaled/{split}_scaled_features.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)
    caches[split] = data
    report[split] = {
        "keys": list(data),
        "shapes": {k: list(v.shape) for k, v in data.items() if torch.is_tensor(v)},
        "labels": dict(Counter(data["labels"].tolist())),
        "finite": all(bool(torch.isfinite(v).all()) for v in data.values() if torch.is_tensor(v)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

def row_hashes(data):
    array = torch.cat([data["features"], data["confidences"], data["labels"][:, None]], 1).numpy()
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in array]

rows = {k: row_hashes(v) for k, v in caches.items()}
report["cache_row_overlap"] = {
    "train_test_unique_exact_rows": len(set(rows["train"]) & set(rows["test"])),
    "validation_rows_found_in_full_train": sum(h in set(rows["train"]) for h in rows["val"]),
    "note": "Equal feature rows alone do not prove equal source images. Validation is expected to be a subset of the full training cache.",
}

model = TransformerPhysicsCrossGenHead().eval()
checkpoint = ROOT / "pipeline_40k/models/gated_cross_gen_40k_best.pt"
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
report["parameter_count"] = sum(p.numel() for p in model.parameters())

def predict(data):
    with torch.inference_mode():
        return torch.cat([torch.sigmoid(model(f, c)[0]) for f, c in zip(data["features"].split(256), data["confidences"].split(256))]).numpy()

val_p, test_p = predict(caches["val"]), predict(caches["test"])
val_y, test_y = caches["val"]["labels"].numpy(), caches["test"]["labels"].numpy()
_, val_th = compute_youden_threshold(val_y, val_p)
_, test_th = compute_youden_threshold(test_y, test_p)
report["checkpoint_evaluation"] = {
    "test_auc": float(roc_auc_score(test_y, test_p)),
    "test_accuracy_fixed_0_5": float(accuracy_score(test_y, test_p >= .5)),
    "validation_selected_threshold": val_th,
    "test_accuracy_validation_threshold": float(accuracy_score(test_y, test_p >= val_th)),
    "test_selected_threshold_descriptive_only": test_th,
    "test_accuracy_test_selected_descriptive_only": float(accuracy_score(test_y, test_p >= test_th)),
    "validation_auc": float(roc_auc_score(val_y, val_p)),
}
for margin in (.08, .15):
    decided = (np.abs(test_p - val_th) > margin) & (caches["test"]["confidences"].sum(1).numpy() >= 1)
    report["checkpoint_evaluation"][f"validation_center_margin_{margin}"] = {
        "coverage": float(decided.mean()),
        "decided_accuracy": float(accuracy_score(test_y[decided], test_p[decided] >= val_th)) if decided.any() else None,
    }

# Small fixed baseline to test how much signal is in confidence scores alone.
# Exclude the saved validation rows; do not tune on the test set.
val_set = set(rows["val"])
fit_mask = np.array([h not in val_set for h in rows["train"]])
report["exploratory_logistic_baselines"] = {}
for name, keys in (("confidence_only", ["confidences"]), ("features_only", ["features"]), ("features_and_confidences", ["features", "confidences"])):
    tr_x = torch.cat([caches["train"][k] for k in keys], 1).numpy()
    te_x = torch.cat([caches["test"][k] for k in keys], 1).numpy()
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    baseline.fit(tr_x[fit_mask], caches["train"]["labels"].numpy()[fit_mask])
    p = baseline.predict_proba(te_x)[:, 1]
    report["exploratory_logistic_baselines"][name] = {"test_auc": float(roc_auc_score(test_y, p)), "test_accuracy_0_5": float(accuracy_score(test_y, p >= .5)), "fit_samples": int(fit_mask.sum())}

try:
    FocalLoss()(torch.zeros(1), torch.zeros(1))
    report["focal_singleton"] = "passed"
except Exception as exc:
    report["focal_singleton"] = f"{type(exc).__name__}: {exc}"
report["fixed_eye_crop_shapes_at_256"] = [list(a.shape) for a in extract_eye_crops(np.zeros((256, 256, 3), dtype=np.uint8))]
head1, head2 = FoundationNormalHead().eval(), FoundationNormalHead().eval()
features = torch.rand(1, 576, 8, 8)
with torch.inference_mode():
    report["untrained_normal_heads_mean_abs_output_difference"] = float((head1(features, 32, 32) - head2(features, 32, 32)).abs().mean())

for path in (ROOT / "models/evaluation_scaled_40k.json", ROOT / "pipeline_40k/models/evaluation_scaled_40k.json", ROOT / "pipeline_40k/models/shortcut_diagnostics_report.json"):
    saved = json.loads(path.read_text())
    report[str(path.relative_to(ROOT))] = {k: {m: v[m] for m in ("auc_roc", "accuracy") if m in v} for k, v in saved.items() if isinstance(v, dict) and ("auc_roc" in v or "accuracy" in v)}

print(json.dumps(report, indent=2), flush=True)

# Hash source files to verify exact duplicate leakage, including cross-label cases.
def file_hash(path):
    return str(path.relative_to(ROOT / "pipeline_40k/data/scaled")), hashlib.sha256(path.read_bytes()).hexdigest()

files = sorted((ROOT / "pipeline_40k/data/scaled").rglob("*.jpg"))
with ThreadPoolExecutor(max_workers=4) as pool:
    pairs = list(pool.map(file_hash, files))
groups = {}
for relative, digest in pairs:
    groups.setdefault(digest, []).append(relative)
cross_split = [paths for paths in groups.values() if any(p.startswith("train") for p in paths) and any(p.startswith("test_unseen") for p in paths)]
cross_label = [paths for paths in groups.values() if any("/real/" in p.replace("\\", "/") for p in paths) and any("/fake/" in p.replace("\\", "/") for p in paths)]
report["raw_image_hash_audit"] = {
    "total_files": len(files),
    "partition_counts": dict(Counter(str(Path(p).parent) for p, _ in pairs)),
    "unique_sha256": len(groups),
    "duplicate_groups": sum(len(paths) > 1 for paths in groups.values()),
    "cross_split_duplicate_groups": len(cross_split),
    "test_files_matching_train": sum(sum(p.startswith("test_unseen") for p in paths) for paths in cross_split),
    "cross_label_duplicate_groups": len(cross_label),
    "cross_split_examples": cross_split[:10],
    "note": "SHA-256 detects byte-identical files; near-duplicates and re-encoded duplicates remain unaudited.",
}
out = ROOT / "reports/project_audit.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["raw_image_hash_audit"], indent=2), flush=True)
print(f"Saved {out}")
