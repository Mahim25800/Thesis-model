"""Wait for frozen evaluations, then run paired statistical comparisons."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "experiments/post_pipeline_status.json"
REQUIRED = (
    ROOT / "models/generator_invariant_v6/external_synthbuster/evaluation.json",
    ROOT / "models/generator_invariant_v6/external_synthbuster/ablation.json",
    ROOT / "models/generator_diverse_v6_no_adversary/external_synthbuster/evaluation.json",
)
OUTPUT = ROOT / "models/generator_invariant_v6/external_synthbuster/comparison_bootstrap.json"


def status(state: str, **details) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "state": state, **details,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    try:
        while not all(path.is_file() for path in REQUIRED):
            status("waiting", missing=[str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()])
            time.sleep(60)
        if not OUTPUT.is_file():
            status("running_bootstrap")
            log_dir = ROOT / "experiments/breakthrough_pipeline_logs"
            with (log_dir / "comparison_bootstrap.stdout.log").open("w", encoding="utf-8") as stdout, \
                    (log_dir / "comparison_bootstrap.stderr.log").open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    [sys.executable, "scripts/compare_external_models_bootstrap.py"],
                    cwd=ROOT, stdout=stdout, stderr=stderr,
                )
            if result.returncode != 0:
                raise RuntimeError(f"Bootstrap comparison failed with exit code {result.returncode}")
        status("complete", output=str(OUTPUT.relative_to(ROOT)))
    except Exception as error:
        status("failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
