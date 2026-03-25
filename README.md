# DNN-comm-compression

A research framework for distributed DNN inference with communication compression. Models are partitioned into sequential pipeline stages deployed across multiple nodes (containers). Intermediate activations transmitted between nodes are optionally compressed to reduce communication overhead. The framework supports sweeping over compression methods and rates, collecting metrics, and comparing results against baselines.

## Documentation

- [Configuration](docs/config.md) — experiment and infrastructure config schemas
- [Compression](docs/compression.md) — compression interface and implementations
- [Node Server](docs/node.md) — inference API, management API, link probing
- [Metrics](docs/metrics.md) — metric events, async emission, metrics server
- [Orchestrator](docs/orchestrator.md) — experiment runner, sweep loop, data client
- [Deployment](docs/deployment.md) — Docker Compose and Kubernetes deployment tool
- [Analysis](docs/analysis.md) — post-processing, plots, and cleanup

### Model Guides

- [ResNet-56](docs/resnet.md) — partitioning, experiment configs, running on Docker and Kubernetes
- [Llama-3.1-8B](docs/llama.md) — partitioning, experiment configs, running Llama experiments

## Setup

Install dependencies:

```bash
pip install -e ".[dev]"
```

Install pre-commit hooks:

```bash
pre-commit install
```

Or use the Makefile:

```bash
make install-dev
```

## Code Style

```bash
make lint        # check for linting issues
make format      # format code
make lint-fix    # fix linting issues and format
```

## Validating a Config

```bash
python -m framework.config.validate experiments/resnet56_topk_sweep
python -m framework.config.validate experiments/resnet56_topk_sweep --show-runs

# via Makefile
make validate EXPERIMENT=experiments/resnet56_topk_sweep
make validate EXPERIMENT=experiments/resnet56_topk_sweep SHOW_RUNS=1
```

Validation checks both configs, resolves infra inheritance, expands the sweep, validates node topology, and runs fairness checks against referenced baselines.

## Running an Experiment

See the model-specific guides for full step-by-step instructions:
- [ResNet-56 on Docker / Kubernetes](docs/resnet.md)
- [Llama-3.1-8B](docs/llama.md)

After running, analyze results:

```bash
python -m framework.analysis.analyze --experiment resnet56_topk_sweep
python -m framework.analysis.visualize --experiment resnet56_topk_sweep
```

## Model Partitioning

Partitioning scripts are in `models/` and are independent of the experiment framework. Run them once to produce partition artifacts consumed by node servers. More info in the model-specific instructions mentioned above.

