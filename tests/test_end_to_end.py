from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from panoptic_segmenter.cli import main
from panoptic_segmenter.config import ExperimentConfig
from panoptic_segmenter.evaluation.evaluate import evaluate_checkpoint
from panoptic_segmenter.inference import Predictor
from panoptic_segmenter.training.checkpoint import load_checkpoint
from panoptic_segmenter.training.train import train_from_config


def test_show_config_works_without_repository_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show-config", "--set", "run_name=wheel-smoke"]) == 0
    captured = capsys.readouterr()
    assert yaml.safe_load(captured.out)["run_name"] == "wheel-smoke"


def test_complete_training_evaluation_and_prediction(
    prepared_experiment: tuple[ExperimentConfig, Path],
) -> None:
    config, root = prepared_experiment
    run_dir = train_from_config(config)
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "run.yaml").is_file()
    assert (run_dir / "metrics.csv").is_file()
    checkpoint = load_checkpoint(run_dir / "best.pt")
    assert checkpoint["epoch"] == 1
    config.train.epochs = 2
    resumed = train_from_config(config, resume=run_dir / "last.pt")
    assert resumed == run_dir
    assert load_checkpoint(run_dir / "last.pt")["epoch"] == 2
    assert len((run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()) == 3
    scores = evaluate_checkpoint(run_dir / "best.pt", split="test", device="cpu")
    assert {"pq", "sq", "rq", "pq_thing", "pq_stuff"} <= scores.keys()
    image_path = root / "odd-size.png"
    Image.new("RGB", (35, 29), (80, 100, 120)).save(image_path)
    outputs = Predictor.from_checkpoint(run_dir / "best.pt", "cpu").predict_path(image_path, root / "prediction")
    assert set(outputs) == {"semantic", "instance", "semantic_color", "overlay"}
    assert all(path.is_file() for path in outputs.values())
    assert Image.open(outputs["semantic"]).size == (35, 29)
    assert Image.open(outputs["overlay"]).size == (35, 29)
