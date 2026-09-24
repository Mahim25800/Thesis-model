"""Run or summarize rotated held-generator selection for regional v6.

This script uses only internal development data plus the four GenImage domains.
It never reads Synthbuster, prior confirmatory labels, or a new final holdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENIMAGE_DOMAINS = ("adm", "biggan", "vqdm", "wukong")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(selection: dict) -> tuple[dict, dict]:
    external = selection.get("ensemble_held_out_generator_development")
    if external is None:  # compatibility with existing frozen v6 artifacts
        external = selection.get("ensemble_wukong_development")
    internal = selection.get("ensemble_internal_development")
    if not isinstance(internal, dict) or not isinstance(external, dict):
        raise ValueError("Training selection artifact lacks internal or held-generator metrics")
    return internal, external


def train_one(args, held: str, sources: tuple[str, ...], output: Path) -> None:
    completion = output / "model.pt"
    if completion.is_file():
        return
    command = [
        sys.executable, "-B", "scripts/train_generator_invariant_v6.py",
        "--base-cache", str(args.base_cache.relative_to(ROOT)),
        "--genimage-cache", str(args.genimage_cache.relative_to(ROOT)),
        "--output-dir", str(output.relative_to(ROOT)),
        "--source-domains", *sources,
        "--held-out-domain", held,
        "--seeds", *(str(value) for value in args.seeds),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--domain-loss-weight", str(args.domain_loss_weight),
    ]
    if args.base_augmentation_cache is not None:
        command.extend([
            "--base-augmentation-cache", str(args.base_augmentation_cache.relative_to(ROOT)),
            "--genimage-augmentation-cache", str(args.genimage_augmentation_cache.relative_to(ROOT)),
        ])
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0 or not completion.is_file():
        raise RuntimeError(f"Training failed for held domain {held}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=ROOT / "data/cache_regional_dsine_v2")
    parser.add_argument("--genimage-cache", type=Path, default=ROOT / "data/cache_genimage_regional_v2")
    parser.add_argument("--base-augmentation-cache", type=Path, default=None)
    parser.add_argument("--genimage-augmentation-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/multi_generator_v16_control")
    parser.add_argument("--run", action="store_true", help="Train missing held-generator folds sequentially on CUDA when available")
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 73, 79])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--domain-loss-weight", type=float, default=0.10)
    args = parser.parse_args()
    args.base_cache, args.genimage_cache, args.output_dir = args.base_cache.resolve(), args.genimage_cache.resolve(), args.output_dir.resolve()
    if (args.base_augmentation_cache is None) != (args.genimage_augmentation_cache is None):
        raise ValueError("Provide both augmentation caches or neither")
    if args.base_augmentation_cache is not None:
        args.base_augmentation_cache = args.base_augmentation_cache.resolve()
        args.genimage_augmentation_cache = args.genimage_augmentation_cache.resolve()
    paths = (args.base_cache, args.genimage_cache, args.output_dir) + (
        () if args.base_augmentation_cache is None else (args.base_augmentation_cache, args.genimage_augmentation_cache)
    )
    if not all(path.is_relative_to(ROOT) for path in paths):
        raise ValueError("All paths must remain inside pipeline_40k")
    if not 0.0 <= args.domain_loss_weight <= 1.0:
        raise ValueError("domain-loss-weight must lie in [0, 1]")
    if args.run:
        for held in GENIMAGE_DOMAINS:
            train_one(args, held, tuple(value for value in GENIMAGE_DOMAINS if value != held), args.output_dir / held)
    folds = {}
    missing = []
    for held in GENIMAGE_DOMAINS:
        selection_path = args.output_dir / held / "selection.json"
        if not selection_path.is_file():
            missing.append(held)
            continue
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        internal, external = metrics(selection)
        folds[held] = {
            "source_domains": selection.get("source_domains", [value for value in GENIMAGE_DOMAINS if value != held]),
            "held_out_domain": selection.get("held_out_development_domain", held),
            "internal_auc": internal["auc_roc"],
            "held_out_auc": external["auc_roc"],
            "internal_accuracy": internal["accuracy"],
            "held_out_accuracy": external["accuracy"],
            "selection_artifact": str(selection_path.relative_to(ROOT)).replace("\\", "/"),
        }
    report = {
        "protocol": "rotated held-generator development selection; no Synthbuster or confirmatory labels read",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.output_dir.name,
        "raw_image_augmentation": args.base_augmentation_cache is not None,
        "folds": folds,
        "missing_folds": missing,
    }
    if len(folds) == len(GENIMAGE_DOMAINS):
        held_aucs = [folds[name]["held_out_auc"] for name in GENIMAGE_DOMAINS]
        internal_aucs = [folds[name]["internal_auc"] for name in GENIMAGE_DOMAINS]
        report["selection_score"] = {
            "minimum_held_generator_auc": min(held_aucs),
            "mean_held_generator_auc": sum(held_aucs) / len(held_aucs),
            "mean_internal_development_auc": sum(internal_aucs) / len(internal_aucs),
            "ranking": "maximize minimum held-generator AUC, then mean held-generator AUC",
        }
    write_json(args.output_dir / "selection_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
