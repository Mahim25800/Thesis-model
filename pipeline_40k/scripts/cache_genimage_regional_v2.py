"""Cache selected GenImage development domains with regional DSINE physics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_regional_dsine_v2 import SCHEMA, atomic_text, build_partition
from src.extractors.dsine_normals import DEFAULT_CHECKPOINT
from src.extractors.regional_physics import REGION_NAMES, RegionalPhysicsExtractor
from src.extractors.surface_normals import SurfaceNormalsExtractor


GENERATORS = ("adm", "biggan", "vqdm", "wukong")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/external_genimage_dev/selected")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cache_genimage_regional_v2")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generators", nargs="+", default=list(GENERATORS))
    args = parser.parse_args()
    data_dir, output, checkpoint = args.data_dir.resolve(), args.output_dir.resolve(), args.checkpoint.resolve()
    if not all(path.is_relative_to(ROOT) for path in (data_dir, output, checkpoint)):
        raise ValueError("All paths must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose an empty cache directory or pass --resume")
    source_provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    normals = SurfaceNormalsExtractor(
        normal_backend="dsine", dsine_checkpoint=checkpoint, dsine_device=args.device
    )
    extractor = RegionalPhysicsExtractor(normals)
    preamble = {
        "schema_version": 2, "feature_schema": SCHEMA, "regions": list(REGION_NAMES),
        "extractor": extractor.provenance(), "shard_size": args.shard_size,
        "source_provenance": source_provenance, "limit_per_class": args.limit_per_class, "domains": {},
    }
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        old_hash = previous.get("extractor", {}).get("normal_extractor", {}).get("checkpoint_sha256")
        new_hash = preamble["extractor"]["normal_extractor"].get("checkpoint_sha256")
        if old_hash != new_hash:
            raise ValueError("Cannot resume with a different DSINE checkpoint")
    else:
        atomic_text(provenance_path, json.dumps(preamble, indent=2, sort_keys=True) + "\n")
    missing = [generator for generator in args.generators if not (data_dir / generator).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing generator directories: {missing}")
    for generator in args.generators:
        print(f"Caching GenImage development domain: {generator}", flush=True)
        preamble["domains"][generator] = build_partition(
            generator, data_dir / generator, output, extractor, args.shard_size, args.limit_per_class
        )
        atomic_text(provenance_path, json.dumps(preamble, indent=2, sort_keys=True) + "\n")
    print(json.dumps(preamble, indent=2), flush=True)


if __name__ == "__main__":
    main()
