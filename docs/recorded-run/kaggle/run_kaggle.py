from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPOSITORY = "https://github.com/Doithoo/pytorch-panoptic-segmentation-lab.git"
REVISION = "f6fb554"
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "pytorch-panoptic-segmentation-lab"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print(json.dumps({"command": command, "cwd": str(cwd) if cwd else None}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_with_heartbeat(command: list[str], *, cwd: Path) -> None:
    process = subprocess.Popen(command, cwd=cwd)
    stop = threading.Event()

    def heartbeat() -> None:
        started = time.time()
        while not stop.wait(60):
            print(
                json.dumps(
                    {
                        "phase": "training",
                        "status": "running",
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    return_code = process.wait()
    stop.set()
    thread.join()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"working directory already exists: {PROJECT}")
    run(["git", "clone", "--filter=blob:none", REPOSITORY, str(PROJECT)])
    run(["git", "checkout", REVISION], cwd=PROJECT)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    print(json.dumps({"phase": "source", "revision": revision}), flush=True)
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=PROJECT)
    synthetic = WORKING / "synthetic-panoptic"
    run(
        [
            sys.executable,
            "scripts/create_synthetic_data.py",
            "--output",
            str(synthetic),
            "--count",
            "256",
            "--size",
            "128",
        ],
        cwd=PROJECT,
    )
    run_with_heartbeat(
        [
            sys.executable,
            "scripts/kaggle_train.py",
            "--input",
            str(synthetic),
            "--config",
            "configs/reference_kaggle.yaml",
        ],
        cwd=PROJECT,
    )
    summary = json.loads((WORKING / "kaggle-run-summary.json").read_text(encoding="utf-8"))
    summary["source_revision"] = revision
    (WORKING / "kaggle-run-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"phase": "kernel", "status": "complete", "source_revision": revision}), flush=True)


if __name__ == "__main__":
    main()
