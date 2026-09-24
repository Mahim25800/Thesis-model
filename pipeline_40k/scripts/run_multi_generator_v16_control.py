"""Wait for the v14 audit, then run the rotated held-generator control experiment."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "models/physics_ensemble_v14/robustness_audit.json"
STATUS = ROOT / "experiments/multi_generator_v16_control_status.json"
LOG_DIR = ROOT / "experiments/multi_generator_v16_control_logs"


def write_status(stage: str, state: str, **details):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "state": state, **details}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main():
    try:
        while not AUDIT.is_file():
            write_status("waiting_for_v14_audit", "running")
            time.sleep(30)
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        if "datasets" not in audit or "internal_unseen" not in audit["datasets"]:
            raise RuntimeError("v14 audit is incomplete or malformed")
        output = ROOT / "models/multi_generator_v16_control/selection_summary.json"
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if not existing.get("missing_folds"):
                write_status("complete", "complete", selection_summary=str(output.relative_to(ROOT)))
                return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", "scripts/select_multi_generator_robust.py", "--run"]
        write_status("rotated_held_generator_control", "running", command=command)
        with (LOG_DIR / "stdout.log").open("w", encoding="utf-8") as stdout, (LOG_DIR / "stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError("rotated held-generator control experiment failed")
        write_status("complete", "complete", selection_summary=str(output.relative_to(ROOT)))
    except Exception as error:
        write_status("failed", "failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
