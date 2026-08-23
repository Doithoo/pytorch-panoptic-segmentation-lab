from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPOSITORY = "https://github.com/Doithoo/pytorch-panoptic-segmentation-lab.git"
REVISION = os.environ.get("PANOPTIC_REVISION", "main")
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "pytorch-panoptic-segmentation-lab"
RAW_DATA = Path("/kaggle/input")
CONTRACT = WORKING / "soccer-contract"


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
    started = time.time()
    if PROJECT.exists():
        raise RuntimeError(f"working directory already exists: {PROJECT}")
    run(["git", "clone", "--filter=blob:none", REPOSITORY, str(PROJECT)])
    run(["git", "checkout", REVISION], cwd=PROJECT)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    print(json.dumps({"phase": "source", "revision": revision}), flush=True)
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=PROJECT)
    run([sys.executable, "-m", "pip", "install", "opencv-python-headless"], cwd=PROJECT)
    run(
        [
            sys.executable,
            "scripts/convert_kaggle_soccer.py",
            str(RAW_DATA),
            "--output",
            str(CONTRACT),
            "--max-frames",
            "240",
            "--frame-stride",
            "5",
            "--resize-width",
            "512",
        ],
        cwd=PROJECT,
    )
    print(json.dumps({"phase": "convert", "contract": str(CONTRACT)}), flush=True)
    run_with_heartbeat(
        [
            sys.executable,
            "scripts/kaggle_train.py",
            "--input",
            str(CONTRACT),
            "--schema",
            "configs/kaggle_soccer_schema.yaml",
            "--config",
            "configs/kaggle_soccer.yaml",
        ],
        cwd=PROJECT,
    )
    summary_path = WORKING / "kaggle-run-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "source_revision": revision,
            "dataset": "quantigoai/soccer-dataset",
            "dataset_license": "CC-BY-SA-4.0",
            "conversion": {"max_frames": 240, "frame_stride": 5, "resize_width": 512},
            "total_elapsed_seconds": time.time() - started,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "complete", "summary": str(summary_path), "revision": revision}), flush=True)


if __name__ == "__main__":
    main()
