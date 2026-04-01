.PHONY: lint format lint-fix install install-dev validate \
        generate generate-all generate-all-validate \
        generate-multi generate-multi-all generate-multi-all-validate \
        build build-resnet build-llama build-multi build-metrics build-orchestrator

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

IMAGE_TAG ?= latest
IMAGE_RESNET      := dnn-compute-resnet:$(IMAGE_TAG)
IMAGE_LLAMA       := dnn-compute-llama:$(IMAGE_TAG)
IMAGE_MULTI       := dnn-compute-multi:$(IMAGE_TAG)
IMAGE_METRICS     := dnn-metrics:$(IMAGE_TAG)
IMAGE_ORCHESTRATOR := dnn-orchestrator:$(IMAGE_TAG)

$(VENV):
	python -m venv $(VENV)

install: $(VENV)
	git submodule update --init --recursive
	$(PIP) install -e .

install-dev: $(VENV)
	git submodule update --init --recursive
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
	$(PYTHON) -m framework.validate $(EXPERIMENT) $(if $(SHOW_RUNS),--show-runs)

generate:
	@if [ -z "$(SPEC)" ] || [ -z "$(PROFILE)" ]; then \
		echo "Usage: make generate SPEC=specs/<path> PROFILE=profiles/<path>.yaml [SUB_EXPERIMENTS='a b']"; \
		exit 1; \
	fi
	$(PYTHON) tools/generate.py --spec $(SPEC) --profile $(PROFILE) \
		$(if $(SUB_EXPERIMENTS),--sub-experiments $(SUB_EXPERIMENTS))

generate-all:
	$(PYTHON) tools/generate.py --all

generate-all-validate:
	$(PYTHON) tools/generate.py --all --validate --show-runs

generate-multi:
	@if [ -z "$(SPEC)" ] || [ -z "$(PROFILE)" ]; then \
		echo "Usage: make generate-multi SPEC=multispecs/<name> PROFILE=profiles/<path>.yaml [SUB_EXPERIMENTS='a b']"; \
		exit 1; \
	fi
	$(PYTHON) tools/generate.py --multi --spec $(SPEC) --profile $(PROFILE) \
		$(if $(SUB_EXPERIMENTS),--sub-experiments $(SUB_EXPERIMENTS))

generate-multi-all:
	$(PYTHON) tools/generate.py --multi --all

generate-multi-all-validate:
	$(PYTHON) tools/generate.py --multi --all --validate --show-runs

# ---------------------------------------------------------------------------
# Docker image builds  (build context is always the repo root)
# Override IMAGE_TAG to tag a specific version, e.g. make build IMAGE_TAG=v0.2
# ---------------------------------------------------------------------------

build: build-resnet build-llama build-multi build-metrics build-orchestrator

build-resnet:
	docker build -f docker/Dockerfile.compute-resnet -t $(IMAGE_RESNET) .

build-llama:
	docker build -f docker/Dockerfile.compute-llama -t $(IMAGE_LLAMA) .

build-multi:
	docker build -f docker/Dockerfile.compute-multi -t $(IMAGE_MULTI) .

build-metrics:
	docker build -f docker/Dockerfile.metrics -t $(IMAGE_METRICS) .

build-orchestrator:
	docker build -f docker/Dockerfile.orchestrator -t $(IMAGE_ORCHESTRATOR) .
