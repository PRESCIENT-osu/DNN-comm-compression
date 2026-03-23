.PHONY: lint format lint-fix install install-dev validate

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV):
	python -m venv $(VENV)

install: $(VENV)
	$(PIP) install -e .

install-dev: $(VENV)
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/ruff format .

lint-fix:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

validate:
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "Usage: make validate EXPERIMENT=experiments/<name> [SHOW_RUNS=1]"; \
		exit 1; \
	fi
	python -m framework.config.validate $(EXPERIMENT) $(if $(SHOW_RUNS),--show-runs)
