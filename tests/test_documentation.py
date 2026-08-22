from __future__ import annotations

import re
from pathlib import Path

LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    root = Path(__file__).parents[1]
    missing: list[str] = []
    for document in root.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(root).parts):
            continue
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            clean = target.split("#", 1)[0]
            if not clean or "://" in clean or clean.startswith("mailto:"):
                continue
            destination = (document.parent / clean).resolve()
            if not destination.exists():
                missing.append(f"{document.relative_to(root)} -> {target}")
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
