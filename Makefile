.PHONY: lint format typecheck test coverage build check

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

coverage:
	uv run pytest --cov=panoptic_segmenter --cov-report=term-missing

build:
	uv run python -m build

check: lint format typecheck test coverage build
