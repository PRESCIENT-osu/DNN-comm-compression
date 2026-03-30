# Multi-Model Multi-Pipeline Experiments

This document covers the multi-model scenario, in which a single set of physical nodes hosts partitions of **multiple models simultaneously**. Traffic from all models shares the same inter-node links, and compute tasks from all models compete for the same per-node inference worker. This exposes compute contention and link contention as first-class measurements, rather than isolating each model on dedicated hardware.

---

## Motivation

The single-model setup studies how activation compression affects one model's accuracy in isolation. Real deployments, however, co-locate many inference services. To study the *interaction* between models:

- Two models running concurrently on the same nodes compete for GPU compute time via a **shared FIFO queue**.
- Their activations travel over the **same physical links**, so bandwidth consumed by one pipeline reduces available bandwidth for the other.
- Compression applied to one pipeline changes its contribution to link utilisation and its queue occupancy, which affects the latency and throughput of the other pipeline.

Multi-model experiments are kept entirely separate from single-model experiments. They live under `multispecs/` (specs) and `experiments/multi/` (generated), use a distinct node server (`framework/nodes/compute/multi/server.py`), and a distinct orchestrator (`framework/nodes/orchestrator/multi_runner.py`).

---

## Key Concepts

### Pipelines

A **pipeline** is a named inference instance with a specific model, a partition-to-node assignment, and an execution flow order. Multiple pipelines can coexist on the same physical nodes:

```
Node A              Node B              Node C
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ resnet-a: p1 │──►│ resnet-a: p2 │──►│ resnet-a: p4 │
│ llama-mmlu-a:│   │ llama-mmlu-a:│   │ llama-mmlu-a:│
│   p1         │──►│   p2         │──►│   p3         │
└──────────────┘   └──────────────┘   └──────────────┘
```

Each pipeline has its own:
- Partition set loaded on each node
- Compression config per link (independent from other pipelines)
- Dataset and evaluation metric

Tasks from different pipelines are identified by `pipeline_id` throughout the system.

### FIFO Queue and Single Worker

Each node runs a single async worker that dequeues tasks in FIFO order. Tasks from any pipeline enter the same queue. This means:

- A burst of large llama tasks delays pending resnet tasks.
- Queue depth, per-pipeline breakdown, and wait times are all recorded.
- There is no priority scheduling — the queue is intentionally fair (FIFO) to expose natural contention.

### Per-Pipeline Compression

Compression on a link is configured **per (pipeline, link) pair**, not per link globally. On the A→B link, resnet-a might use TopK while llama-mmlu-a uses quantization, or one pipeline might be uncompressed while the other is not. The sender compresses each pipeline's activations according to that pipeline's config, and the receiver decompresses using the matching config.

---

## Directory Structure

Multi-model experiments use a separate source/output hierarchy:

```
multispecs/                         ← source (model-agnostic)
  {name}/
    experiment.yaml                 ← nodes, pipelines, datasets, workload
    sub_experiments.yaml            ← per-pipeline compression sweep definitions

profiles/                           ← shared with single-model
  linear-3-multi/
    docker.yaml
    100mbps.yaml

experiments/multi/                  ← generated output (gitignored)
  {name}/
    experiment.yaml                 ← fully resolved, consumed by orchestrator
    infra.yaml                      ← verbatim profile copy

framework/nodes/compute/multi/
  server.py                         ← multi-model node FastAPI server

framework/nodes/orchestrator/
  multi_controller.py               ← per-pipeline config push, node health/idle polling
  multi_runner.py                   ← sweep loop, workload submission, result collection
```

---

## Multispec Format

### `experiment.yaml`

Defines the static properties of a multi-model experiment (model-agnostic — no host/port):

```yaml
nodes:
  - A
  - B
  - C

pipelines:
  - name: resnet-a
    model: resnet56
    partitions:
      A: [p1]
      B: [p2, p3]
      C: [p4, p5]
    flow: [A, B, C]

  - name: llama-mmlu-a
    model: llama-3.1-8b
    partitions:
      A: [p1]
      B: [p2]
      C: [p3]
    flow: [A, B, C]

datasets:
  resnet56:
    name: cifar10
    path: .datasets/cifar10
    batch_size: 100
    max_in_flight: 10
    max_samples: 1000
    seed: 42
  llama-3.1-8b:
    name: mmlu
    path: .datasets/mmlu
    tokenizer_path: models/llama/.partitions/tokenizer
    batch_size: 4
    max_in_flight: 4
    subjects: [college_computer_science, high_school_mathematics]
    samples_per_subject: 20
    seed: 42

workload:
  pattern: fill
  window_per_pipeline: 4
  mix:
    resnet-a: 0.5
    llama-mmlu-a: 0.5

metrics_server:
  host: metrics
  port: 9100
```

**`nodes`** is a list of name strings only — host/port are filled in from the profile during generation.

**`pipelines`**: each pipeline specifies:
- `name` — unique pipeline identifier; used as `pipeline_id` in all events and run IDs
- `model` — model type string; must match a key in `datasets`
- `partitions` — dict from node name to list of partition file IDs to load
- `flow` — node names in execution order (first node receives tasks, last node sends results to callback)

**`datasets`** is keyed by model type string (not pipeline name). Multiple pipelines of the same model share the same dataset config.

**`workload`** — see [Workload Patterns](#workload-patterns).

### `sub_experiments.yaml`

Defines named compression sweep groups. Each sub-experiment expands into one or more resolved runs. The key difference from single-model sub-experiments is that each link's `compression` field is a **dict keyed by pipeline ID**, not a single method:

```yaml
sub_experiments:

  baseline:
    links:
      - from: A
        to: B
        compression:
          resnet-a:
            compression: none
          llama-mmlu-a:
            compression: none
      - from: B
        to: C
        compression:
          resnet-a:
            compression: none
          llama-mmlu-a:
            compression: none

  topk_both_paired:
    sweep_mode: paired
    sweep:
      - links:
          - from: A
            to: B
            compression:
              resnet-a:
                compression: topk
                rates: [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
              llama-mmlu-a:
                compression: topk
                rates: [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
          - from: B
            to: C
            compression:
              resnet-a:
                compression: topk
                rates: [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
              llama-mmlu-a:
                compression: topk
                rates: [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]

  topk_resnet_only:
    sweep_mode: product
    sweep:
      - links:
          - from: A
            to: B
            compression:
              resnet-a:
                compression: topk
                rates: [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
              llama-mmlu-a:
                compression: none
          - from: B
            to: C
            compression:
              resnet-a:
                compression: topk
                rates: [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
              llama-mmlu-a:
                compression: none
```

#### Sweep modes

The same `paired` / `product` modes apply, but the dimensions are now **all (pipeline, link) pairs** in the entry:

| Mode | Behaviour | Example |
|------|-----------|---------|
| `paired` | All dimensions advance together (same index). All lists must have the same length. | 4 pipelines × 6 rates → 6 runs |
| `product` | Cartesian product across all dimensions. | 2 pipelines × 2 links × 6 rates each → 6⁴ = 1296 runs |

For **isolated sweeps** (one pipeline compressed, the other fixed at `none`), use `product` mode. With `none`, `effective_rates()` returns `[0.0]` (length 1), so the product collapses cleanly: 6 × 6 × 1 × 1 = 36 runs, sweeping both links of the compressed pipeline independently while the other pipeline is always uncompressed.

#### Designing contention sub-experiments

A recommended pattern for studying contention effects:

| Sub-experiment | Purpose |
|----------------|---------|
| `baseline` | No compression anywhere — establishes the pure compute-contention baseline |
| `topk_both_paired` | Symmetric: same rate on all (pipeline, link) dims simultaneously — measures joint degradation |
| `topk_resnet_only` | Asymmetric: sweep resnet compression with llama uncompressed — isolates resnet's degradation curve |
| `topk_llama_only` | Asymmetric: sweep llama compression with resnet uncompressed — isolates llama's degradation curve |

Comparing `topk_both_paired` against the two asymmetric sweeps at the same rate reveals whether simultaneous compression causes additional degradation beyond compressing each model in isolation.

---

## Profiles

Multi-model profiles are structurally identical to single-model profiles. A dedicated `linear-3-multi/` directory is used to give the containers distinct hostnames (`multi-a`, `multi-b`, `multi-c`) so they can coexist with single-model containers on the same Compose network.

```yaml
# profiles/linear-3-multi/100mbps.yaml
nodes:
  - name: A
    host: multi-a
    port: 8000
    resources:
      gpu: 1
  - name: B
    host: multi-b
    port: 8000
    resources:
      gpu: 1
  - name: C
    host: multi-c
    port: 8000
    resources:
      gpu: 1
links:
  - from: A
    to: B
    bandwidth_mbps: 100
  - from: B
    to: C
    bandwidth_mbps: 100
```

**Bidirectional traffic**: TC rules are egress-only. When two pipelines flow in opposite directions on the same physical links (e.g., resnet A→B→C and a hypothetical model C→B→A), you must declare both directions in the profile:

```yaml
links:
  - from: A
    to: B
    bandwidth_mbps: 100
  - from: B
    to: A       # reverse direction for the opposite-flow pipeline
    bandwidth_mbps: 100
  - from: B
    to: C
    bandwidth_mbps: 100
  - from: C
    to: B
    bandwidth_mbps: 100
```

For the `resnet56_llama_mmlu` experiment both pipelines flow A→B→C, so unidirectional `linear-3-multi` profiles are sufficient.

---

## Generating Experiments

```bash
# Single spec + profile
python tools/generate.py --multi \
    --spec multispecs/resnet56_llama_mmlu \
    --profile profiles/linear-3-multi/100mbps.yaml

# All compatible spec/profile combinations
python tools/generate.py --multi --all

# Specific sub-experiments only
python tools/generate.py --multi \
    --spec multispecs/resnet56_llama_mmlu \
    --profile profiles/linear-3-multi/100mbps.yaml \
    --sub-experiments baseline topk_both_paired

# Validate and show resolved runs
python tools/generate.py --multi --all --validate --show-runs
```

Generated experiment names follow `{spec_name}_{topology}_{profile_name}`, for example:
```
experiments/multi/resnet56_llama_mmlu_linear-3-multi_100mbps/
```

The tool writes:
- `experiment.yaml` — fully resolved `MultiExperimentConfig` (nodes with host/port, resolved sub-experiments)
- `infra.yaml` — verbatim profile copy

---

## Multi-Model Node Server

The multi-model node server (`framework/nodes/compute/multi/server.py`) replaces the single-model server for multi-pipeline deployments.

### Startup

Configured entirely via environment variables:

| Variable | Description |
|----------|-------------|
| `NODE_NAME` | Name of this node (e.g. `A`) |
| `EXPERIMENT_CONFIG_PATH` | Path to the generated `experiment.yaml` |
| `PARTITIONS_BASE_DIR` | Base directory for partition files |
| `DEVICE` | PyTorch device (default: `cuda` if available, else `cpu`) |
| `PROBE_INTERVAL_S` | Link probe interval in seconds (default: `30.0`) |

At startup the server:
1. Loads `MultiExperimentConfig` from `EXPERIMENT_CONFIG_PATH`.
2. For each pipeline that has partitions on this node, builds a `PipelineNodeState` containing the loaded partition modules, the pipeline's flow position (first/last node), the next node URL, and initial compression config (none).
3. Starts the FIFO worker loop and per-link probe loops.

Partitions are loaded by model type:
- `resnet56` — `torch.jit.load`
- `llama-*` — `torch.load` with the `models/llama` helper module imported first

### Queue and Worker

All incoming tasks (from any pipeline) enter a single `asyncio.Queue`. A single async worker dequeues tasks one at a time and processes them sequentially:

```
POST /infer (resnet-a task)  ──┐
POST /infer (llama task)     ──┤── Queue ──► Worker ──► forward pass ──► forward/callback
POST /infer (resnet-a task)  ──┘
```

A task's `pipeline_id` field routes it to the correct `PipelineNodeState` for decompression, forward pass, and re-compression.

### Per-Task Processing

For each dequeued task the worker:

1. **Decompresses** incoming activations using the pipeline's current incoming compression config (skipped on the first node).
2. **Runs forward pass** through the pipeline's partition(s) on this node.
3. **Compresses** the output using the pipeline's current outgoing compression config (skipped on the last node).
4. **Forwards** the result to the next node (`POST /infer` with `pipeline_id`) or POSTs to the callback URL (last node).
5. **Emits events**: `QueueSnapshotEvent` (dequeue), `ForwardPassEvent`, `DecompressEvent`, `CompressEvent`, `SendEvent`, `TaskNodeTimingEvent`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/infer` | Enqueue a task. Body: `MultiInferRequest` (includes `pipeline_id`). |
| `GET` | `/health` | Liveness check. Returns `{"status": "ok"}`. |
| `GET` | `/status` | Queue depth and processing state. Returns `{"queue_length": N, "processing": bool, "queue_by_pipeline": {...}}`. |
| `POST` | `/config` | Update compression config for one `(pipeline_id, direction)` pair. Waits for the queue to drain before applying. Body: `MultiConfigUpdate`. |
| `GET` | `/probe` | Trigger a link probe manually. |

### Config Updates

`POST /config` accepts a `MultiConfigUpdate`:

```json
{
  "pipeline_id": "resnet-a",
  "direction": "outgoing",
  "method": "topk",
  "rate": 0.25,
  "drain_timeout_s": 90.0
}
```

The node waits until its queue is empty and no task is being processed before applying the new config. This ensures in-flight tasks complete under the previous config.

---

## Multi-Model Orchestrator

### Controller (`multi_controller.py`)

Mirrors `controller.py` for the single-model case.

**`wait_for_multi_nodes_ready(exp, timeout_s, node_host)`**
Polls `GET /health` on all nodes. Raises `RuntimeError` if any node is unreachable after `timeout_s`.

**`wait_for_multi_nodes_idle(exp, timeout_s, node_host)`**
Polls `GET /status` until `queue_length == 0` and `processing == False` on every node.

**`push_multi_run_config(exp, run, drain_timeout_s, node_host)`**
For each `ResolvedPipelineLinkConfig` in the run, sends configs concurrently:
- Sending node: `POST /config` with `direction=outgoing`
- Receiving node: `POST /config` with `direction=incoming`

Both requests include the `pipeline_id` field so the node updates only that pipeline's config.

### Runner (`multi_runner.py`)

```bash
python -m framework.nodes.orchestrator.multi_runner \
    experiments/multi/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --callback-host orchestrator \
    --callback-port 8080
```

| Flag | Default | Description |
|------|---------|-------------|
| `--callback-host` | `localhost` / `CALLBACK_HOST` | Hostname nodes use to reach the result callback |
| `--callback-port` | `8080` / `CALLBACK_PORT` | Callback server port |
| `--result-timeout` | `300.0` | Per-task result wait timeout (seconds) |
| `--dry-run` | — | Log sweep plan without executing |
| `--node-host` | `NODE_HOST` env | Override hostname for all nodes |
| `--metrics-host` | `METRICS_HOST` env | Override metrics server hostname |

The runner:
1. Loads `MultiExperimentConfig` from `experiment.yaml`.
2. Starts a `MetricsEmitter` connected to the metrics server.
3. Executes sub-experiments sequentially.
4. For each resolved run: pushes configs via `push_multi_run_config`, then drives the workload.

### Workload Patterns

The workload pattern controls how tasks are submitted across pipelines. It is set once per experiment in `experiment.yaml` and applies to every run.

#### `fill`

Maintains a sliding window of `window_per_pipeline` in-flight tasks per pipeline. Each time a task completes, the next task from that pipeline's dataset is submitted immediately. `window_per_pipeline: 1` is equivalent to closed-loop (one in-flight task per pipeline at all times).

```yaml
workload:
  pattern: fill
  window_per_pipeline: 4
  mix:
    resnet-a: 0.5
    llama-mmlu-a: 0.5
```

Use this to keep all pipelines continuously loaded. All pipeline datasets are iterated concurrently; the experiment ends when every pipeline's dataset is exhausted.

#### `fixed_rate`

Submits tasks at a constant inter-arrival interval of `1 / arrival_rate` seconds. At each step the pipeline is chosen by sampling from `mix` proportions (renormalized over non-exhausted pipelines).

```yaml
workload:
  pattern: fixed_rate
  arrival_rate: 10.0   # 10 tasks/second total
  mix:
    resnet-a: 0.6
    llama-mmlu-a: 0.4
```

#### `poisson`

Same as `fixed_rate` but inter-arrival times are drawn from `Exp(arrival_rate)`, producing a Poisson arrival process.

```yaml
workload:
  pattern: poisson
  arrival_rate: 10.0
  mix:
    resnet-a: 0.5
    llama-mmlu-a: 0.5
```

#### `mix`

The `mix` dict assigns a fractional weight to each pipeline. Weights must sum to 1.0. Under `fill`, the mix ratio controls the relative number of in-flight slots but since each pipeline has its own window it is effectively ignored — all pipelines run at their own window simultaneously. Under rate-based patterns, the mix controls the fraction of total arrivals directed to each pipeline.

### Callback Server

The runner embeds the same FastAPI callback server as the single-model data client (`POST /result` on `callback_port`). The last node of any pipeline POSTs a `ResultPayload` (task_id + base64 data) to this endpoint. The runner resolves the pending future for the matching `task_id` and decodes the result using the pipeline's model-specific decoder:

| Model type | Output | Decoding |
|------------|--------|----------|
| `resnet*` | Logits `[B, C]` | `argmax(dim=1)` → predicted class indices |
| `llama-*` + wikitext2 | Logits `[B, L, V]` | Shift-label cross-entropy → (nll_sum, token_count) |
| `llama-*` + mmlu | Logits `[B, L, V]` | Last-position logits at A/B/C/D token IDs → argmax |

---

## Metrics

Multi-model experiments emit four additional event types beyond the standard set. All events are streamed to the central metrics server alongside the existing node-side events.

### `task_node_timing`

Emitted once per task per node. Records timestamps at each processing stage so derived timings (queue wait, compute time, compression time) can be computed in analysis.

| Field | Description |
|-------|-------------|
| `pipeline_id` | Pipeline this task belongs to |
| `task_id` | Unique task identifier |
| `node_id` | Node name |
| `enqueue_time` | Wall time when task entered the queue |
| `queue_length_at_enqueue` | Queue depth at enqueue (including this task) |
| `compute_start` | Wall time when worker began the forward pass |
| `compute_end` | Wall time when forward pass completed |
| `compress_start` | Wall time when compression began |
| `compress_end` | Wall time when compression completed |
| `sent_time` | Wall time when the forwarded/callback request was sent |

Derived metrics:
- **queue_wait** = `compute_start - enqueue_time`
- **compute_time** = `compute_end - compute_start`
- **compression_time** = `compress_end - compress_start`
- **total_node_time** = `sent_time - enqueue_time`

### `queue_snapshot`

Emitted at each enqueue and dequeue on a multi-model node. Captures the instantaneous queue state including a per-pipeline breakdown.

| Field | Description |
|-------|-------------|
| `node_id` | Node name |
| `queue_length` | Total queue depth at the moment of the event |
| `queue_by_pipeline` | Dict of `{pipeline_id: count}` for all queued tasks |
| `trigger` | `"enqueue"` or `"dequeue"` |
| `task_id` | Task that triggered this event |
| `pipeline_id` | Pipeline of the triggering task |

### `task_e2e`

Emitted by the orchestrator when each task completes. Measures total wall-clock latency from submission to result receipt.

| Field | Description |
|-------|-------------|
| `pipeline_id` | Pipeline this task belongs to |
| `task_id` | Unique task identifier |
| `submit_time` | Wall time when the task was POSTed to the first node |
| `receive_time` | Wall time when the result arrived at the callback server |
| `latency_ms` | `(receive_time - submit_time) × 1000` |

### `run_throughput`

Emitted once per completed sweep run. Captures aggregate and per-pipeline throughput and latency percentiles.

| Field | Description |
|-------|-------------|
| `wall_time_s` | Total elapsed time for the run (seconds) |
| `total_tasks` | Total tasks completed across all pipelines |
| `tasks_per_second` | `total_tasks / wall_time_s` |
| `per_pipeline` | Dict of `{pipeline_id: PipelineThroughputStats}` |

`PipelineThroughputStats` fields: `tasks`, `tasks_per_second`, `p50_ms`, `p90_ms`, `p99_ms`.

---

## Run IDs

Multi-model run IDs encode all `(pipeline_id, from_node, to_node)` triples. Descriptors are sorted by `(pipeline_id, from_node, to_node)` and joined with `--`:

```
# baseline: no compression on any pipeline or link
none-llama-mmlu-a-A-B--none-llama-mmlu-a-B-C--none-resnet-a-A-B--none-resnet-a-B-C

# topk_both_paired at rate 0.25
topk-llama-mmlu-a-A-B_0.25--topk-llama-mmlu-a-B-C_0.25--topk-resnet-a-A-B_0.25--topk-resnet-a-B-C_0.25

# topk_resnet_only: resnet topk A→B 0.25, B→C 0.50; llama uncompressed
none-llama-mmlu-a-A-B--none-llama-mmlu-a-B-C--topk-resnet-a-A-B_0.25--topk-resnet-a-B-C_0.50
```

Run IDs grow longer than single-model run IDs but remain deterministic and self-describing. The `--show-runs` flag during validation truncates the ID display at 60 characters to keep output readable.

---

## Docker Image

Multi-model nodes use `docker/Dockerfile.compute-multi`, which extends the base PyTorch runtime with both ResNet and Llama dependencies:

```
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

pip install torchvision transformers accelerate ...

COPY models/resnet/   # TorchScript partition helpers
COPY models/llama/    # Llama partition helpers
CMD python -m framework.nodes.compute.multi.server
```

Build:
```bash
docker build -f docker/Dockerfile.compute-multi -t dnn-compute-multi:latest .
```

The `HF_HOME=/hf_cache` environment variable points to the HuggingFace weight cache. Mount a volume at `/hf_cache` to avoid re-downloading model weights on each container start.

---

## Validation

```bash
# Validate a generated multi-model experiment
python -m framework.validate \
    experiments/multi/resnet56_llama_mmlu_linear-3-multi_100mbps

# Show all resolved sweep runs
python -m framework.validate \
    experiments/multi/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --show-runs

# Via generate tool
python tools/generate.py --multi --all --validate --show-runs
```

The validator detects multi-model experiments by the presence of the `pipelines` key in `experiment.yaml` and uses `MultiExperimentConfig` for parsing.

---

## Python API

```python
from pathlib import Path
from framework.utils.loader import is_multi_experiment, load_multi_experiment_config

path = Path("experiments/multi/resnet56_llama_mmlu_linear-3-multi_100mbps/experiment.yaml")
if is_multi_experiment(path):
    exp = load_multi_experiment_config(path)
    for pipeline in exp.pipelines:
        print(pipeline.name, pipeline.model, pipeline.flow)
    for sub_exp in exp.sub_experiments:
        runs = exp.resolve_sweep(sub_exp)
        print(sub_exp.name, len(runs), "runs")
```

Schemas are defined as Pydantic v2 models in `framework/datamodels/multi_experiment.py`:

| Class | Description |
|-------|-------------|
| `MultiExperimentConfig` | Top-level generated config (nodes, pipelines, datasets, workload, sub_experiments) |
| `PipelineConfig` | One pipeline instance (name, model, partitions, flow) |
| `MultiNodeConfig` | Physical node with host/port |
| `WorkloadConfig` | Submission pattern and mix |
| `MultiResolvedSubExperiment` | One resolved sub-experiment entry |
| `MultiResolvedRun` | One fully expanded sweep run |
| `ResolvedPipelineLinkConfig` | Concrete compression config for one (pipeline, link) pair |
