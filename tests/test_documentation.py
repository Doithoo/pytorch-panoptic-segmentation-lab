from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from panoptic_segmenter.config import ExperimentConfig

LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
URL = re.compile(r"https?://[^)\s]+")


def _anchor_slug(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-")


def test_local_markdown_links_resolve() -> None:
    root = Path(__file__).parents[1]
    missing: list[str] = []
    for document in root.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(root).parts):
            continue
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            clean, _, anchor = target.partition("#")
            if not clean or "://" in clean or clean.startswith("mailto:"):
                continue
            destination = (document.parent / clean).resolve()
            if not destination.exists():
                missing.append(f"{document.relative_to(root)} -> {target}")
                continue
            if anchor:
                headings = {
                    _anchor_slug(line.lstrip("#").strip())
                    for line in destination.read_text(encoding="utf-8").splitlines()
                    if line.startswith("#")
                }
                if anchor not in headings:
                    missing.append(f"{document.relative_to(root)} -> missing anchor {target}")
    assert not missing, "broken local documentation links:\n" + "\n".join(missing)


def test_english_documentation_has_chinese_counterparts() -> None:
    root = Path(__file__).parents[1]
    missing: list[str] = []
    for document in (root / "docs").rglob("*.md"):
        if document.name.endswith(".zh-CN.md") or "recorded-run/kaggle" in document.as_posix():
            continue
        counterpart = document.with_name(f"{document.stem}.zh-CN.md")
        if not counterpart.is_file():
            missing.append(str(document.relative_to(root)))
    assert not missing, "missing Chinese counterparts:\n" + "\n".join(missing)


def test_repository_templates_and_cli_reference_do_not_drift() -> None:
    root = Path(__file__).parents[1]
    issue_config = (root / ".github/ISSUE_TEMPLATE/config.yml").read_text(encoding="utf-8")
    learning_template = (root / ".github/ISSUE_TEMPLATE/learning-question.yml").read_text(encoding="utf-8")
    cli_reference = (root / "docs/reference/cli.md").read_text(encoding="utf-8")
    cli_source = (root / "src/panoptic_segmenter/cli.py").read_text(encoding="utf-8")
    assert "pytorch-panoptic-segmentation-lab" in issue_config
    assert "examples/02_model_contract.py" in learning_template
    commands = re.findall(r'sub\.add_parser\("([^"]+)"', cli_source)
    for command in commands:
        assert f"## `{command}`" in cli_reference


def test_configuration_reference_covers_nested_fields() -> None:
    root = Path(__file__).parents[1]
    reference = (root / "docs/reference/config-reference.md").read_text(encoding="utf-8")
    config = ExperimentConfig()
    for section in dataclasses.fields(config):
        value = getattr(config, section.name)
        if not dataclasses.is_dataclass(value):
            continue
        for field in dataclasses.fields(value):
            assert f"`{section.name}.{field.name}`" in reference


def test_external_github_links_stay_in_this_repository() -> None:
    root = Path(__file__).parents[1]
    wrong: list[str] = []
    for document in root.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(root).parts):
            continue
        for url in URL.findall(document.read_text(encoding="utf-8")):
            if "github.com/" in url and "Doithoo/pytorch-panoptic-segmentation-lab" not in url:
                wrong.append(f"{document.relative_to(root)} -> {url}")
    assert not wrong, "external GitHub links point outside this repository:\n" + "\n".join(wrong)
