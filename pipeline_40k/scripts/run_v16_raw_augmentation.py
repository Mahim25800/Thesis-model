"""Wait for complete augmentation caches, then run rotated v6 selection."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_AUGMENTATION = ROOT / "data/cache_regional_dsine_v16_augmented"
GENIMAGE_AUGMENTATION = ROOT / "data/cache_genimage_regional_v16_augmented"
OUTPUT = ROOT / "models/multi_generator_v16_raw_augmentation"
STATUS = ROOT / "experiments/multi_generator_v16_raw_augmentation_status.json"
LOG_DIR = ROOT / "experiments/multi_generator_v16_raw_augmentation_logs"
DOMAINS = ("adm", "biggan", "vqdm", "wukong")


def write_status(stage: str, state: str, **details) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "state": state, **details}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def caches_complete() -> bool:
    base_path, genimage_path = BASE_AUGMENTATION / "provenance.json", GENIMAGE_AUGMENTATION / "provenance.json"
    if not base_path.is_file() or not genimage_path.is_file():
        return False
    base, genimage = json.loads(base_path.read_text(encoding="utf-8")), json.loads(genimage_path.read_text(encoding="utf-8"))
    if not base.get("train", {}).get("complete"):
        return False
    return all(genimage.get("domains", {}).get(domain, {}).get("complete") for domain in DOMAINS)


def main() -> None:
    try:
        while not caches_complete():
            write_status("waiting_for_complete_augmentation_caches", "running")
            time.sleep(30)
        output = OUTPUT / "selection_summary.json"
        if output.is_file() and not json.loads(output.read_text(encoding="utf-8")).get("missing_folds"):
            write_status("complete", "complete", selection_summary=str(output.relative_to(ROOT)))
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-B", "scripts/select_multi_generator_robust.py", "--run",
            "--output-dir", str(OUTPUT.relative_to(ROOT)),
            "--base-augmentation-cache", str(BASE_AUGMENTATION.relative_to(ROOT)),
            "--genimage-augmentation-cache", str(GENIMAGE_AUGMENTATION.relative_to(ROOT)),
        ]
        write_status("rotated_held_generator_raw_augmentation", "running", command=command)
        with (LOG_DIR / "stdout.log").open("w", encoding="utf-8") as stdout, (LOG_DIR / "stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError("raw-image augmentation selection failed")
        summary = json.loads(output.read_text(encoding="utf-8"))
        if summary.get("missing_folds"):
            raise RuntimeError(f"raw-image augmentation selection incomplete: {summary['missing_folds']}")
        write_status("complete", "complete", selection_summary=str(output.relative_to(ROOT)))
    except Exception as error:
        write_status("failed", "failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
