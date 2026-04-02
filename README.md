# DNN-comm-compression

A research framework for distributed DNN inference with communication compression. Models are partitioned into sequential pipeline stages deployed across multiple nodes (containers). Intermediate activations transmitted between nodes are optionally compressed to reduce communication overhead. The framework supports sweeping over compression methods and rates, collecting metrics, and comparing results against baselines.

## Documentation

- [Configuration](docs/config.md) — experiment and infrastructure config schemas
- [Compression](docs/compression.md) — compression interface and implementations
- [Node Server](docs/node.md) — inference API, management API, link probing
- [Metrics](docs/metrics.md) — metric events, async emission, metrics server
- [Orchestrator](docs/orchestrator.md) — experiment runner, sweep loop, data client
- [Multi-Model Experiments](docs/multi_task.md) — multi-pipeline node server, orchestrator, workload patterns
- [Optimizer Experiments](docs/optimizer.md) — adaptive compression optimizer, optspecs format, all sub-experiment types
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
│   ├── spec.py                     # GeneratedExperimentConfig, ResolvedSubExperiment
│   ├── multi_experiment.py         # MultiExperimentConfig, PipelineConfig, WorkloadConfig, …
│   ├── opt_experiment.py           # OptExperimentConfig and all sub-experiment types
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
│   │   ├── llama/server.py         # Llama compute node (torch.load)
│   │   └── multi/server.py         # Multi-pipeline node (FIFO queue, per-pipeline state)
│   ├── metrics/
│   │   ├── server.py               # metrics ingestion server
│   │   └── emitter.py              # async MetricsEmitter
│   └── orchestrator/
│       ├── runner.py               # single-model sweep loop and CLI entry point
│       ├── controller.py           # config push and drain helpers
│       ├── data_client.py          # callback server + batch sending (image models)
│       ├── llama_data_client.py    # data client for Llama
│       ├── datasets.py             # image dataset loaders
│       ├── multi_controller.py     # per-pipeline config push; multi-node health/idle polling
│       ├── multi_runner.py         # multi-model sweep loop, workload submission
│       └── opt_runner.py           # optimizer experiment runner (slot loop, adapter interface)
├── optimizer/
│   ├── inference_optimizer_adapter.py  # adapter: framework configs → external optimizer
│   ├── channel_estimators.py       # moving average, LCB, last-observation estimators
│   ├── accuracy_model.py           # surrogate accuracy model training + querying
│   ├── evaluators.py               # profiling and accuracy sweep evaluators
│   └── compression_mapper.py       # η → (method, rate) conversion
├── deploy/
│   ├── __main__.py                 # python -m framework.deploy
│   ├── docker_backend.py
│   └── k8s_backend.py
└── analysis/
    ├── analyze.py
    └── visualize.py

tools/
└── generate.py                     # experiment generation tool

specs/                              # single-model experiment specs (what)
├── resnet56/
│   ├── experiment.yaml             # model-level defaults
│   ├── sub_experiments.yaml        # compression sub-experiments
│   ├── equal-split/
│   ├── single-node/
│   └── ...
└── llama/
    └── ...

multispecs/                         # multi-model experiment specs (what)
└── resnet56_llama_mmlu/
    ├── experiment.yaml             # nodes, pipelines, datasets, workload
    └── sub_experiments.yaml        # per-pipeline compression sweep definitions

optspecs/                           # optimizer experiment specs (what)
└── resnet56_llama_mmlu/
    ├── experiment.yaml             # pipelines, links with eta_min, simulation config
    └── sub_experiments.yaml        # optimizer sub-experiment types (no_csi, csi_aware, baselines)

profiles/                           # infrastructure definitions (where)
├── single-node/
│   └── default.yaml
├── linear-3/
│   ├── default.yaml                # 1 Gbps, no constraints
│   ├── 100mbps.yaml
│   └── wan.yaml                    # WAN with delay/jitter/loss
└── linear-3-multi/
    ├── docker.yaml
    └── 100mbps.yaml

experiments/                        # generated output — gitignored
docker/
├── Dockerfile.compute-resnet       # CUDA + PyTorch, torchvision
├── Dockerfile.compute-llama        # CUDA + PyTorch, transformers
├── Dockerfile.compute-multi        # CUDA + PyTorch, torchvision + transformers (multi-pipeline)
├── Dockerfile.metrics              # python:slim, fastapi only
└── Dockerfile.orchestrator         # CUDA + PyTorch (Stein oracle needs GPU), transformers
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

## Generating an Experiment Config

Experiments are defined by composing a **spec** (what runs — model, partitioning, compression variants) with a **profile** (where it runs — topology, network conditions, resource limits). The `tools/generate.py` tool resolves the combination and writes a ready-to-deploy experiment directory under `experiments/`.

```bash
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml
```

This produces `experiments/resnet56_equal-split_linear-3_100mbps/` containing:
- `experiment.yaml` — merged spec with all sub-experiments resolved
- `infra.yaml` — the profile verbatim

To include only specific sub-experiments:

```bash
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml \
    --sub-experiments baseline topk_paired
```

Referenced baseline experiments are materialised automatically. See [docs/config.md](docs/config.md) for full documentation on specs, profiles, sub-experiments, and baseline references.

## Building Docker Images

```bash
make build                        # all images (tag: latest)
make build-resnet                 # ResNet compute node only
make build-llama                  # Llama compute node only
make build-multi                  # multi-pipeline compute node only
make build-metrics                # metrics server only
make build-orchestrator           # orchestrator only (CUDA image)

make build IMAGE_TAG=v0.2         # all images with a specific tag
```

See [docs/deployment.md](docs/deployment.md) for details on each image.

## Validating a Config

```bash
python -m framework.validate experiments/resnet56_equal-split_linear-3_100mbps
python -m framework.validate experiments/resnet56_equal-split_linear-3_100mbps --show-runs

# via Makefile
make validate EXPERIMENT=experiments/resnet56_equal-split_linear-3_100mbps
make validate EXPERIMENT=experiments/resnet56_equal-split_linear-3_100mbps SHOW_RUNS=1
```

Validation checks both configs, resolves infra inheritance, expands the sweep, validates node topology, and runs fairness checks against referenced baselines.

## Deploying an Experiment

```bash
python -m framework.deploy \
  --experiment experiments/resnet56_equal-split_linear-3_100mbps \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply
```

See the model-specific guides for full step-by-step instructions:
- [ResNet-56 on Docker / Kubernetes](docs/resnet.md)
- [Llama-3.1-8B](docs/llama.md)

## Optimizer Experiments

Optimizer experiments use adaptive compression — the orchestrator runs a slot-based control loop that queries the external `Inference_Optimizer` library to choose per-pipeline compression rates dynamically, rather than sweeping a fixed grid.

Optimizer specs live in `optspecs/` and are generated with the `--opt` flag:

```bash
python tools/generate.py --opt \
    --spec optspecs/resnet56_llama_mmlu \
    --profile profiles/linear-3-multi/100mbps.yaml
```

This produces `experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps/`. Run the optimizer loop:

```bash
python -m framework.nodes.orchestrator.opt_runner \
    experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --callback-host orchestrator \
    --callback-port 8080
```

Sub-experiment types include `no_csi` (Lyapunov-based, channel-free), `csi_aware` (closed-form η* with LCB), and a full set of single- and multi-task baselines. See [docs/optimizer.md](docs/optimizer.md) for the complete reference.

## Analyzing Results

```bash
python -m framework.analysis.analyze --experiment resnet56_equal-split_linear-3_100mbps
python -m framework.analysis.visualize --experiment resnet56_equal-split_linear-3_100mbps
```

## Model Partitioning

Partitioning scripts are in `models/` and are independent of the experiment framework. Run them once to produce partition artifacts consumed by node servers. More info in the model-specific instructions mentioned above.
