from __future__ import annotations

from pathlib import Path

import pytest
import torch

from panoptic_segmenter.training.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)


def _payload() -> dict[str, object]:
    return {
        "config": {},
        "schema": {},
        "model_state": {"weight": torch.ones(1)},
        "epoch": 1,
        "best_metric": 0.5,
        "run_metadata": {},
    }


def test_checkpoint_roundtrip_is_versioned_and_safe(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, _payload())
    loaded = load_checkpoint(path)
    assert loaded["schema_version"] == 1
    assert torch.equal(loaded["model_state"]["weight"], torch.ones(1))
    assert len(sha256_file(path)) == 64
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_rejects_wrong_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save({**_payload(), "schema_version": 99}, path)
    with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
        load_checkpoint(path)
