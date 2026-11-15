"""Development-only screening for a higher-accuracy cached-feature head."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits


def score(labels, probabilities):
    return {
        "accuracy_0_5": float(accuracy_score(labels, probabilities >= 0.5)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
    }


def main():
    cache = torch.load(ROOT / "data/cache_scaled/train_scaled_features.pt", map_location="cpu", weights_only=True)
    indices = np.load(ROOT / "models/improved_v1/integrity/split_indices.npz")
    train, development = indices["train_indices"], indices["development_indices"]
    features, confidence = cache["features"].numpy(), cache["confidences"].numpy()
    matrices = {"all_18": np.column_stack([features, confidence])}
    labels = cache["labels"].numpy().astype(int)
    specifications = []
    for leaves in (15, 31, 63, 127):
        for rate, l2 in ((0.04, 1.0), (0.06, 5.0), (0.08, 10.0)):
            specifications.append(("hgb", {"max_leaf_nodes": leaves, "learning_rate": rate,
                "l2_regularization": l2, "max_iter": 500, "min_samples_leaf": 20,
                "early_stopping": True, "validation_fraction": 0.1,
                "n_iter_no_change": 35, "random_state": 42}))
    report = {"protocol": "Development-only screening; calibration and test labels not loaded.", "results": []}
    output = Path(__file__).with_name("round2_full_cache.json")
    with threadpool_limits(limits=1):
        for schema, matrix in matrices.items():
            for kind, params in specifications:
                started = time.perf_counter()
                model = HistGradientBoostingClassifier(**params)
                model.fit(matrix[train], labels[train])
                probabilities = model.predict_proba(matrix[development])[:, 1]
                row = {"schema": schema, "model": kind, "params": params,
                    "development": score(labels[development], probabilities),
                    "seconds": round(time.perf_counter() - started, 3)}
                report["results"].append(row)
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(row), flush=True)
    best = max(report["results"], key=lambda row: (row["development"]["accuracy_0_5"], row["development"]["auc"]))
    report["best"] = best
    report["status"] = "complete"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
