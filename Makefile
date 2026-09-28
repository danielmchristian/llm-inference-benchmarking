.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check

help:  ## Show available targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create venv and install pinned deps from uv.lock
	uv sync --locked

lint:  ## Ruff lint + formatting check (no changes)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply ruff formatting and safe autofixes
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## mypy strict over src and tests
	uv run mypy

test:  ## Run the pytest suite
	uv run pytest

check: lint typecheck test  ## Everything CI runs, locally
