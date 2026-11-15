"""Resume the long-running regional cache, development, training, and final evaluation stages."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "experiments/breakthrough_pipeline_status.json"
LOG_DIR = ROOT / "experiments/breakthrough_pipeline_logs"
ARCHIVE = ROOT / "data/external_genimage_dev/genimage_test.zip"
EXPECTED_ARCHIVE_BYTES = 25_408_572_820
EXPECTED_ARCHIVE_SHA256 = "1ee98d0958a5905b4e1b2f7f44bb384069704cad8f70e741512e1911e29dba97"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(stage: str, state: str, **details) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "updated_at_utc": now(), "stage": stage, "state": state, **details,
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_inputs() -> None:
    base_cache = ROOT / "data/cache_regional_dsine_v2"
    while True:
        archive_bytes = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
        train_shards = len(list((base_cache / "shards/train").glob("*.pt")))
        test_shards = len(list((base_cache / "shards/test_unseen").glob("*.pt")))
        cached_images_estimate = min(32_000, train_shards * 32) + min(8_000, test_shards * 32)
        cache_ready = all((base_cache / name).is_file() for name in (
            "train_features.pt", "test_unseen_features.pt", "provenance.json"
        ))
        write_status(
            "waiting_for_inputs", "running", regional_cache_ready=cache_ready,
            regional_train_shards=train_shards, regional_test_shards=test_shards,
            regional_cached_images_estimate=cached_images_estimate,
            regional_cache_percent=round(100.0 * cached_images_estimate / 40_000, 2),
            genimage_download_bytes=archive_bytes,
            genimage_download_expected_bytes=EXPECTED_ARCHIVE_BYTES,
            genimage_download_percent=round(100.0 * archive_bytes / EXPECTED_ARCHIVE_BYTES, 2),
        )
        if cache_ready and archive_bytes == EXPECTED_ARCHIVE_BYTES:
            break
        if archive_bytes > EXPECTED_ARCHIVE_BYTES:
            raise RuntimeError("GenImage archive is larger than the recorded source artifact")
        time.sleep(30)
    write_status("verify_genimage_archive", "running")
    actual_hash = sha256(ARCHIVE)
    if actual_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"GenImage archive SHA-256 mismatch: {actual_hash}")


def run_stage(name: str, arguments: list[str], completion: Path) -> None:
    if completion.is_file():
        write_status(name, "already_complete", completion=str(completion.relative_to(ROOT)))
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = LOG_DIR / f"{name}.stdout.log", LOG_DIR / f"{name}.stderr.log"
    write_status(name, "running", command=[sys.executable, *arguments])
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}; see {stderr_path}")
    if not completion.is_file():
        raise RuntimeError(f"Stage {name} exited successfully without {completion}")
    write_status(name, "complete", completion=str(completion.relative_to(ROOT)))


def main() -> None:
    try:
        wait_for_inputs()
        run_stage(
            "train_regional_v5",
            ["scripts/train_regional_multi_physics_v5.py"],
            ROOT / "models/regional_multi_physics_v5/evaluation.json",
        )
        run_stage(
            "prepare_genimage_development",
            ["scripts/prepare_genimage_development.py"],
            ROOT / "data/external_genimage_dev/selected/provenance.json",
        )
        run_stage(
            "cache_genimage_regional",
            ["scripts/cache_genimage_regional_v2.py", "--resume", "--device", "cuda"],
            ROOT / "data/cache_genimage_regional_v2/wukong_features.pt",
        )
        run_stage(
            "train_generator_invariant_v6",
            ["scripts/train_generator_invariant_v6.py"],
            ROOT / "models/generator_invariant_v6/model.pt",
        )
        run_stage(
            "train_generator_diverse_no_adversary",
            [
                "scripts/train_generator_invariant_v6.py",
                "--output-dir", "models/generator_diverse_v6_no_adversary",
                "--domain-loss-weight", "0",
            ],
            ROOT / "models/generator_diverse_v6_no_adversary/model.pt",
        )
        run_stage(
            "evaluate_synthbuster_frozen",
            ["scripts/evaluate_external_synthbuster_regional.py"],
            ROOT / "models/generator_invariant_v6/external_synthbuster/evaluation.json",
        )
        run_stage(
            "evaluate_synthbuster_no_adversary",
            [
                "scripts/evaluate_external_synthbuster_regional.py",
                "--model", "models/generator_diverse_v6_no_adversary/model.pt",
                "--output-dir", "models/generator_diverse_v6_no_adversary/external_synthbuster",
            ],
            ROOT / "models/generator_diverse_v6_no_adversary/external_synthbuster/evaluation.json",
        )
        run_stage(
            "ablate_synthbuster_regional_physics",
            ["scripts/ablate_external_regional_physics.py"],
            ROOT / "models/generator_invariant_v6/external_synthbuster/ablation.json",
        )
        write_status(
            "complete", "complete",
            internal_report="models/generator_invariant_v6/evaluation_internal.json",
            external_report="models/generator_invariant_v6/external_synthbuster/evaluation.json",
            no_adversary_report="models/generator_diverse_v6_no_adversary/external_synthbuster/evaluation.json",
            ablation_report="models/generator_invariant_v6/external_synthbuster/ablation.json",
        )
    except Exception as error:
        write_status("failed", "failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
