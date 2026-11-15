"""Auditable, duplicate-safe splits for the legacy 40k feature cache.

Legacy caches do not contain image paths. Mapping therefore explicitly assumes the
original cache builder's sorted real-files, then sorted fake-files order. Counts
and labels can validate that assumption, but cannot prove historical file identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cache(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    arrays = {key: data[key].detach().cpu().numpy() for key in ("features", "confidences", "labels")}
    size = len(arrays["labels"])
    if not size or any(len(value) != size for value in arrays.values()):
        raise ValueError(f"Empty or inconsistent cache: {path}")
    if arrays["labels"].ndim != 1 or not np.isin(arrays["labels"], [0, 1]).all():
        raise ValueError(f"Expected binary one-dimensional labels: {path}")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Non-finite data in cache: {path}")
    return arrays


def _row_fingerprints(cache):
    # Normalize dtype so identical float32 values stored as float64 still match.
    matrix = np.ascontiguousarray(np.column_stack([
        cache["features"], cache["confidences"], cache["labels"]
    ]), dtype="<f8")
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix]


def _legacy_paths(partition, labels):
    samples = []
    for name, label in (("real", 0), ("fake", 1)):
        directory = Path(partition) / name
        if not directory.is_dir():
            raise ValueError(f"Missing image directory: {directory}")
        samples.extend((path, label) for path in sorted(directory.iterdir())
                       if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                       and path.stat().st_size > 100)
    if len(samples) != len(labels):
        raise ValueError(f"Legacy mapping impossible: {partition} has {len(samples)} images; cache has {len(labels)} rows")
    if not np.array_equal(np.array([label for _, label in samples]), labels):
        raise ValueError(f"Legacy real-then-fake ordering does not match cache labels: {partition}")
    return [path for path, _ in samples]


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_splits(train_cache, val_cache, test_cache, raw_dir, output_dir, seed=42):
    """Return numpy row indices and persist a reproducible split/image manifest.

    ``train_indices``, ``development_indices``, and ``calibration_indices`` index
    the full train cache. ``test_indices`` indexes the test cache. Every train row
    matching a saved-validation row is reserved. Exact image and feature matches
    are kept together; one representative per image hash is retained. Test image
    hashes present anywhere in the original train cache are removed. No test
    labels influence partition allocation.
    """
    train = _load_cache(train_cache)
    val = _load_cache(val_cache)
    test = _load_cache(test_cache)
    train_paths = _legacy_paths(Path(raw_dir) / "train", train["labels"])
    test_paths = _legacy_paths(Path(raw_dir) / "test_unseen", test["labels"])
    train_fingerprints = _row_fingerprints(train)
    val_fingerprints = _row_fingerprints(val)
    by_feature = defaultdict(list)
    for index, fingerprint in enumerate(train_fingerprints):
        by_feature[fingerprint].append(index)
    missing = set(val_fingerprints) - set(by_feature)
    if missing:
        raise ValueError(f"Saved validation cache cannot be mapped: {len(missing)} feature fingerprints absent from train cache")
    with ThreadPoolExecutor(max_workers=8) as executor:
        all_hashes = list(executor.map(_sha256_file, train_paths + test_paths))
    train_hashes = all_hashes[:len(train_paths)]
    test_hashes = all_hashes[len(train_paths):]

    hash_labels = defaultdict(set)
    for sha, label in zip(all_hashes, np.concatenate([train["labels"], test["labels"]])):
        hash_labels[sha].add(int(label))
    conflicts = [sha for sha, labels in hash_labels.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(f"Conflicting labels for {len(conflicts)} identical image hashes; resolve before training")

    # Union both duplicate-image groups and ambiguous feature matches. This also
    # reserves all historical validation matches even when many rows are equal.
    parent = list(range(len(train_paths)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left, right):
        parent[find(right)] = find(left)
    first_hash = {}
    first_feature = {}
    for index, (sha, fingerprint) in enumerate(zip(train_hashes, train_fingerprints)):
        if sha in first_hash:
            union(first_hash[sha], index)
        else:
            first_hash[sha] = index
        if fingerprint in first_feature:
            union(first_feature[fingerprint], index)
        else:
            first_feature[fingerprint] = index
    groups = defaultdict(list)
    for index in range(len(train_paths)):
        groups[find(index)].append(index)
    reserved_roots = {find(index) for fingerprint in val_fingerprints for index in by_feature[fingerprint]}
    rng = np.random.default_rng(seed)
    allocated = {}
    for label in (0, 1):
        roots = sorted(root for root in reserved_roots if train["labels"][groups[root][0]] == label)
        if len(roots) < 2:
            raise ValueError(f"Need at least two independent validation groups for label {label}")
        rng.shuffle(roots)
        # Greedy balancing by image count preserves groups while avoiding a very
        # large duplicate group distorting the two halves.
        totals = {"development": 0, "calibration": 0}
        for root in roots:
            split = min(totals, key=totals.get)
            allocated[root] = split
            totals[split] += len({train_hashes[i] for i in groups[root]})

    indices = {name: [] for name in ("train", "development", "calibration", "test")}
    records = []
    retained_train_hashes = set()
    for index, (path, sha) in enumerate(zip(train_paths, train_hashes)):
        split = allocated.get(find(index), "train")
        retained = sha not in retained_train_hashes
        if retained:
            indices[split].append(index)
            retained_train_hashes.add(sha)
        records.append({"cache": "train", "row_index": index, "path": str(path.resolve()),
                        "label": int(train["labels"][index]), "sha256": sha,
                        "feature_sha256": train_fingerprints[index], "split": split,
                        "retained": retained, "excluded_reason": None if retained else "duplicate_image"})
    seen_test = set()
    overlap_count = 0
    duplicate_test_count = 0
    for index, (path, sha) in enumerate(zip(test_paths, test_hashes)):
        reason = None
        if sha in retained_train_hashes:
            reason = "image_present_in_original_train"
            overlap_count += 1
        elif sha in seen_test:
            reason = "duplicate_image"
            duplicate_test_count += 1
        else:
            indices["test"].append(index)
            seen_test.add(sha)
        records.append({"cache": "test", "row_index": index, "path": str(path.resolve()),
                        "label": int(test["labels"][index]), "sha256": sha,
                        "split": "test", "retained": reason is None, "excluded_reason": reason})

    for name, rows in indices.items():
        if not rows:
            raise ValueError(f"Integrity filtering left empty {name} split")
        if name != "test" and len(np.unique(train["labels"][rows])) != 2:
            raise ValueError(f"Both labels required in {name} split")
    distribution = {}
    for name, rows in indices.items():
        labels = test["labels"] if name == "test" else train["labels"]
        distribution[name] = {"total": len(rows), "real": int(np.sum(labels[rows] == 0)),
                              "fake": int(np.sum(labels[rows] == 1))}
    manifest = {
        "schema_version": 1, "seed": seed,
        "mapping": "ASSUMPTION: legacy sorted real files followed by sorted fake files; count and label sequence checked, historical identity unverified",
        "duplicate_policy": "SHA256 of image bytes; transitive image/feature groups reserved across train/development/calibration; one row per image retained",
        "validation_policy": "All full-train rows matching saved validation features+confidences+labels are reserved; groups split by class into development/calibration",
        "test_policy": "Remove image hashes present anywhere in original train and duplicate test hashes; labels used only for conflict checks and reporting",
        "test_labels_used_for_selection": False,
        "source_caches": {name: {"path": str(Path(path).resolve()), "sha256": _sha256_file(path)}
                          for name, path in (("train", train_cache), ("validation", val_cache), ("test", test_cache))},
        "original_counts": {"train": len(train_paths), "saved_validation": len(val_fingerprints), "test": len(test_paths)},
        "saved_validation_matching_train_rows": sum(len(by_feature[f]) for f in set(val_fingerprints)),
        "reserved_train_rows_including_groups": sum(len(groups[root]) for root in reserved_roots),
        "duplicate_hash_groups": sum(count > 1 for count in Counter(all_hashes).values()),
        "removed_train_duplicate_rows": len(train_paths) - sum(len(indices[name]) for name in ("train", "development", "calibration")),
        "removed_test_overlap_rows": overlap_count,
        "removed_test_duplicate_rows": duplicate_test_count,
        "label_conflict_count": 0, "splits": distribution,
        "limitations": ["Byte hashes do not detect resized, recompressed or near-duplicate images",
                        "Existing held-out DiffusionDB subset does not demonstrate unseen-generator generalization",
                        "This legacy test set was examined before this improved run; independent future external validation is still needed"],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {name + "_indices": np.array(rows, dtype=np.int64) for name, rows in indices.items()}
    with tempfile.NamedTemporaryFile(mode="wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **result)
    os.replace(temporary, output_dir / "split_indices.npz")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(temporary, output_dir / "image_manifest.jsonl")
    _atomic_json(output_dir / "split_manifest.json", manifest)
    result["manifest"] = manifest
    return result
