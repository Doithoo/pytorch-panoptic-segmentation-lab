from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/Doithoo/pytorch-panoptic-segmentation-lab.git"
REVISION = "main"
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "pytorch-panoptic-segmentation-lab"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print(json.dumps({"command": command, "cwd": str(cwd) if cwd else None}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"working directory already exists: {PROJECT}")
    run(["git", "clone", "--filter=blob:none", REPOSITORY, str(PROJECT)])
    run(["git", "checkout", REVISION], cwd=PROJECT)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    print(json.dumps({"phase": "source", "revision": revision}), flush=True)
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=PROJECT)
    run([sys.executable, "-m", "pip", "install", "cityscapesscripts"], cwd=PROJECT)
    run([sys.executable, "scripts/kaggle_cityscapes.py"], cwd=PROJECT)
    summary = WORKING / "cityscapes-run-summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["source_revision"] = revision
    summary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"phase": "kernel", "status": "complete", "source_revision": revision}), flush=True)


if __name__ == "__main__":
    main()
