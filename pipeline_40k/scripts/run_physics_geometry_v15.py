"""Run the frozen five-entity physics-plus-geometry experiment after cache generation.

This runner deliberately selects the model only on the internal development split and
GenImage Wukong development domain.  Synthbuster is evaluated exactly once after
selection and is never used to choose a model or a threshold.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CACHE = ROOT / "data/cache_regional_physics_geometry_v15"
GENIMAGE_CACHE = ROOT / "data/cache_genimage_physics_geometry_v15"
MODEL_DIR = ROOT / "models/generator_invariant_geometry_v15"
STATUS = ROOT / "experiments/physics_geometry_v15_status.json"
LOG_DIR = ROOT / "experiments/physics_geometry_v15_logs"
SCHEMA = "regional_physics_geometry_v15_5x22_features_5x5_confidences"
ENTITY_DIMS = [5, 4, 3, 2, 8]
ENTITY_NAMES = [
    "illumination", "specular_optics", "surface_normals", "chromatic_shadows", "projective_geometry",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(stage: str, state: str, **details: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"updated_at_utc": now(), "stage": stage, "state": state, **details}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS)


def cache_ready(path: Path, expected_partitions: list[str]) -> bool:
    required = [path / "provenance.json"]
    required += [path / f"{partition}_features.pt" for partition in expected_partitions]
    required += [path / f"{partition}_manifest.jsonl" for partition in expected_partitions]
    if not all(item.is_file() for item in required):
        return False
    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    return (
        provenance.get("feature_schema") == SCHEMA
        and provenance.get("entity_dims") == ENTITY_DIMS
        and provenance.get("entity_names") == ENTITY_NAMES
    )


def shard_count(path: Path, partition: str) -> int:
    return len(list((path / "shards" / partition).glob("*.pt")))


def wait_for_caches() -> None:
    while True:
        base_complete = cache_ready(BASE_CACHE, ["train", "test_unseen"])
        dev_complete = cache_ready(GENIMAGE_CACHE, ["adm", "biggan", "vqdm", "wukong"])
        write_status(
            "waiting_for_geometry_caches",
            "running",
            base_cache_ready=base_complete,
            genimage_development_cache_ready=dev_complete,
            base_train_shards=shard_count(BASE_CACHE, "train"),
            base_test_unseen_shards=shard_count(BASE_CACHE, "test_unseen"),
            genimage_shards={name: shard_count(GENIMAGE_CACHE, name) for name in ENTITY_NAMES[:0] + ["adm", "biggan", "vqdm", "wukong"]},
        )
        if base_complete and dev_complete:
            return
        time.sleep(30)


def run_stage(name: str, args: list[str], completion: Path) -> None:
    if completion.is_file():
        write_status(name, "already_complete", completion=str(completion.relative_to(ROOT)))
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{name}.stdout.log"
    stderr_path = LOG_DIR / f"{name}.stderr.log"
    write_status(name, "running", command=[sys.executable, *args])
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}; see {stderr_path.relative_to(ROOT)}")
    if not completion.is_file():
        raise RuntimeError(f"Stage {name} exited without its expected artifact: {completion.relative_to(ROOT)}")
    write_status(name, "complete", completion=str(completion.relative_to(ROOT)))


def main() -> None:
    try:
        wait_for_caches()
        run_stage(
            "train_five_entity_model",
            [
                "-B", "scripts/train_generator_invariant_v6.py",
                "--base-cache", "data/cache_regional_physics_geometry_v15",
                "--genimage-cache", "data/cache_genimage_physics_geometry_v15",
                "--output-dir", "models/generator_invariant_geometry_v15",
                "--feature-schema", SCHEMA,
                "--entity-dims", *(str(value) for value in ENTITY_DIMS),
                "--entity-names", *ENTITY_NAMES,
            ],
            MODEL_DIR / "model.pt",
        )
        run_stage(
            "evaluate_synthbuster_after_freeze",
            [
                "-B", "scripts/evaluate_external_geometry_v15.py",
                "--model", "models/generator_invariant_geometry_v15/model.pt",
                "--output-dir", "models/generator_invariant_geometry_v15/external_synthbuster",
            ],
            MODEL_DIR / "external_synthbuster/evaluation.json",
        )
        write_status(
            "complete", "complete",
            internal_report="models/generator_invariant_geometry_v15/evaluation_internal.json",
            external_report="models/generator_invariant_geometry_v15/external_synthbuster/evaluation.json",
        )
    except Exception as error:
        write_status("failed", "failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
