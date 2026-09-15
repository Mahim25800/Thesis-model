"""Master End-to-End Orchestrator for Scaled 40k Physics Deepfake Pipeline.

Executes sequentially:
1. scripts/download_scaled.py  (Resumable multi-worker streaming)
2. scripts/cache_scaled_features.py  (Batched sharded extraction & auto-merge)
3. scripts/train_scaled.py  (40 epochs AdamW + CosineAnnealingLR)
4. scripts/evaluate_scaled.py  (8,000 unseen test evaluation & observability sweep)
"""

import sys
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PYTHON_EXE = sys.executable


def run_step(step_name: str, script_name: str, extra_args: list = None):
    print("\n" + "#" * 80)
    print(f"STEP: {step_name}")
    print("#" * 80)

    script_path = SCRIPT_DIR / script_name
    cmd = [PYTHON_EXE, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PIPELINE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=str(PIPELINE_ROOT), env=env)
    if result.returncode != 0:
        print(f"\nERROR: Step '{step_name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\nCOMPLETED: Step '{step_name}' finished successfully.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run 40k Pipeline End-to-End")
    parser.add_argument("--train-real", type=int, default=16000)
    parser.add_argument("--train-fake", type=int, default=16000)
    parser.add_argument("--test-real", type=int, default=4000)
    parser.add_argument("--test-fake", type=int, default=4000)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--skip-download", action="store_true", help="Skip download step if already downloaded")
    parser.add_argument("--skip-cache", action="store_true", help="Skip caching step if already cached")
    args = parser.parse_args()

    # Step 1: Download
    if not args.skip_download:
        run_step(
            step_name="1. Resumable Dataset Streaming Ingestion",
            script_name="download_scaled.py",
            extra_args=[
                "--train-real", str(args.train_real),
                "--train-fake", str(args.train_fake),
                "--test-real", str(args.test_real),
                "--test-fake", str(args.test_fake),
            ],
        )
    else:
        print("\nSkipping download step as requested.")

    # Step 2: Feature Caching & Shard Merging
    if not args.skip_cache:
        run_step(
            step_name="2. Batched Sharded Feature Extraction & Auto-Merging",
            script_name="cache_scaled_features.py",
            extra_args=[
                "--shard-size", str(args.shard_size),
            ],
        )
    else:
        print("\nSkipping caching step as requested.")

    # Step 3: 40k Training
    run_step(
        step_name="3. Scaled Classifier Training (40 Epochs, Cosine Annealing)",
        script_name="train_scaled.py",
        extra_args=[
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
        ],
    )

    # Step 4: Evaluation & Observability Sweep
    run_step(
        step_name="4. Unseen Test Evaluation & Observability Sweep",
        script_name="evaluate_scaled.py",
        extra_args=[
            "--obs-threshold", "1.0",
        ],
    )

    print("\n" + "=" * 80)
    print("ALL 40,000+ PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
