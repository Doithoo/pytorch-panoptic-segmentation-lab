.PHONY: lint format typecheck test build check

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

build:
	uv run python -m build

check: lint format typecheck test build
