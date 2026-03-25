# DNN-comm-compression

A research framework for distributed DNN inference with communication compression. Models are partitioned into sequential pipeline stages deployed across multiple nodes (containers). Intermediate activations transmitted between nodes are optionally compressed to reduce communication overhead. The framework supports sweeping over compression methods and rates, collecting metrics, and comparing results against baselines.

## Documentation

- [Configuration](docs/config.md) — experiment and infrastructure config schemas
- [Compression](docs/compression.md) — compression interface and implementations
- [Node Server](docs/node.md) — inference API, management API, link probing
- [Metrics](docs/metrics.md) — metric events, async emission, metrics server
- [Orchestrator](docs/orchestrator.md) — experiment runner, sweep loop, data client
- [Deployment](docs/deployment.md) — Docker images, Docker Compose and Kubernetes deployment
- [Analysis](docs/analysis.md) — post-processing, plots, and cleanup

### Model Guides

- [ResNet-56](docs/resnet.md) — partitioning, experiment configs, running on Docker and Kubernetes
- [Llama-3.1-8B](docs/llama.md) — partitioning, experiment configs, running Llama experiments

## Project Structure

```
framework/
├── validate.py                     # python -m framework.validate
├── datamodels/                     # all Pydantic schemas and data containers
│   ├── experiment.py               # ExperimentConfig, NodeConfig, SweepEntry, …
│   ├── infra.py                    # InfraConfig, InfraNodeConfig, …
│   ├── events.py                   # BaseEvent and all metric event types
│   ├── api.py                      # InferRequest, ConfigUpdate, ResultPayload
│   └── results.py                  # RunRecord, LlamaRunRecord
├── utils/
│   └── loader.py                   # load_experiment_dir, load_infra_config, …
├── nodes/
│   ├── compute/
│   │   ├── common/
│   │   │   ├── server.py           # build_app() factory shared by all compute nodes
│   │   │   └── compressor.py       # compression logic
│   │   ├── resnet/server.py        # ResNet compute node (torch.jit.load)
│   │   └── llama/server.py         # Llama compute node (torch.load)
│   ├── metrics/
│   │   ├── server.py               # metrics ingestion server
│   │   └── emitter.py              # async MetricsEmitter
│   └── orchestrator/
│       ├── runner.py               # sweep loop and CLI entry point
│       ├── controller.py           # config push and drain helpers
│       ├── data_client.py          # callback server + batch sending (image models)
│       ├── llama_data_client.py    # data client for Llama
│       └── datasets.py             # image dataset loaders
├── deploy/
│   ├── __main__.py                 # python -m framework.deploy
│   ├── docker_backend.py
│   └── k8s_backend.py
└── analysis/
    ├── analyze.py
    └── visualize.py

docker/
├── Dockerfile.compute-resnet       # CUDA + PyTorch, torchvision
├── Dockerfile.compute-llama        # CUDA + PyTorch, transformers
├── Dockerfile.metrics              # python:slim, fastapi only
└── Dockerfile.orchestrator         # python:slim, CPU torch + transformers
```

## Setup

Install dependencies:

```bash
pip install -e ".[dev]"
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

## Building Docker Images

```bash
make build                        # all four images (tag: latest)
make build-resnet                 # ResNet compute node only
make build-llama                  # Llama compute node only
make build-metrics                # metrics server only
make build-orchestrator           # orchestrator only

make build IMAGE_TAG=v0.2         # all images with a specific tag
```

See [docs/deployment.md](docs/deployment.md) for details on each image.

## Validating a Config

```bash
python -m framework.validate experiments/resnet56_topk_sweep
python -m framework.validate experiments/resnet56_topk_sweep --show-runs

# via Makefile
make validate EXPERIMENT=experiments/resnet56_topk_sweep
make validate EXPERIMENT=experiments/resnet56_topk_sweep SHOW_RUNS=1
```

Validation checks both configs, resolves infra inheritance, expands the sweep, validates node topology, and runs fairness checks against referenced baselines.

## Deploying an Experiment

```bash
python -m framework.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply
```

See the model-specific guides for full step-by-step instructions:
- [ResNet-56 on Docker / Kubernetes](docs/resnet.md)
- [Llama-3.1-8B](docs/llama.md)

## Analyzing Results

```bash
python -m framework.analysis.analyze --experiment resnet56_topk_sweep
python -m framework.analysis.visualize --experiment resnet56_topk_sweep
```

## Model Partitioning

Partitioning scripts are in `models/` and are independent of the experiment framework. Run them once to produce partition artifacts consumed by node servers. More info in the model-specific instructions mentioned above.
