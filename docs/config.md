# Configuration

Experiments are defined using a two-axis structure that separates *what* runs from *where* it runs:

- **`specs/`** — logical experiment definitions: model, partitioning scheme, node topology, and compression sub-experiments
- **`profiles/`** — infrastructure definitions: node resources, network conditions, and topology address/port mapping

These two axes are composed by `tools/generate.py` to produce a fully resolved experiment directory under `experiments/` (which is gitignored). The orchestrator runs directly against the generated output.

```
specs/                      ← what (model, partitions, compression)
profiles/                   ← where (resources, network, addresses)
          ↘          ↙
       tools/generate.py
              ↓
        experiments/        ← generated, gitignored
```

---

## Specs

### Directory structure

Specs are organized as a two-level hierarchy under `specs/`:

```
specs/
  {model}/
    experiment.yaml          ← model-level defaults (model name, dataset, metrics_server)
    sub_experiments.yaml     ← compression sub-experiments shared across all partition splits
    {partition-split}/
      experiment.yaml        ← partition-split overrides (nodes with partition assignments)
      sub_experiments.yaml   ← optional: override or extend sub-experiments for this split
```

Each level's YAML is deep-merged with parent levels, with child values taking precedence. Sub-experiments are merged by name: a child entry with the same name replaces the parent entry; new child entries are added without affecting parent entries.

### Model-level spec (`specs/{model}/experiment.yaml`)

Defines defaults shared by all partition schemes for this model:

```yaml
model: resnet56
dataset:
  name: cifar10
  path: .datasets/cifar10
  batch_size: 100
  max_in_flight: 10
metrics_server:
  host: metrics
  port: 9100
```

### Partition-split spec (`specs/{model}/{split}/experiment.yaml`)

Defines the node topology and partition assignments for a specific split:

```yaml
nodes:
  - name: A
    host: node-a
    port: 8000
    partitions: [p1]
  - name: B
    host: node-b
    port: 8001
    partitions: [p2, p3]
  - name: C
    host: node-c
    port: 8002
    partitions: [p4, p5]
```

Partitions assigned to a node must be a contiguous subsequence of the model's full partition list. Node names must match the names used in the corresponding profile.

### Sub-experiments (`specs/{model}/sub_experiments.yaml`)

Defines the named compression variants to run. Sub-experiments are shared across all partition splits unless overridden:

```yaml
sub_experiments:

  baseline:
    links:
      - from: A
        to: B
        compression: none
      - from: B
        to: C
        compression: none

  topk_paired:
    baselines:
      - sub_experiment: baseline
      - spec: resnet56/single-node
        sub_experiment: baseline
        profile: single-node/default
    sweep_mode: paired
    sweep:
      - links:
          - from: A
            to: B
            compression: topk
            rates: [0.1, 0.3, 0.5]
          - from: B
            to: C
            compression: topk
            rates: [0.1, 0.3, 0.5]
```

Each sub-experiment entry can contain:

| Field | Description |
|-------|-------------|
| `links` | Fixed link compression config (no sweep) |
| `sweep_mode` | `paired` (default) or `product` — see [Sweep config](#sweep-config) |
| `sweep` | List of sweep entries that expand into multiple runs |
| `baselines` | List of baseline references for analysis comparison |
| `dataset` | Dataset override for this sub-experiment (optional; used by Llama experiments with per-sub-experiment datasets) |

A sub-experiment uses either `links` (for a fixed single run) or `sweep` (for multiple runs), not both. The `baseline` sub-experiment above uses `links: [none, none]` which produces a single run with `run_id: none-A-B--none-B-C`.

---

## Profiles

### Directory structure

Profiles are organized by topology under `profiles/`:

```
profiles/
  {topology}/
    {name}.yaml
```

For example:

```
profiles/
  single-node/
    default.yaml
  linear-3/
    default.yaml     ← 1 Gbps LAN
    100mbps.yaml     ← 100 Mbps WAN
    wan.yaml         ← WAN with delay/jitter/loss + resource constraints
```

A profile bundles the physical/network properties of a deployment topology: node addresses and ports, CPU/GPU/memory resource limits, and link traffic shaping (bandwidth, delay, jitter, loss). Profiles are model-agnostic — any model whose spec uses the same node names (`A`, `B`, `C`, …) can use the same profile.

### Profile file format

```yaml
nodes:
  - name: A
    host: node-a
    port: 8000
    resources:
      cpu: 2
      memory: 4Gi
      gpu: 0
  - name: B
    host: node-b
    port: 8001
links:
  - from: A
    to: B
    bandwidth_mbps: 100
    delay_ms: 20
    jitter_ms: 5
    loss_pct: 0.5
```

All resource and link fields are optional. See [Infra Config](#infra-config) for full field documentation.

---

## Generating experiments

Use `tools/generate.py` to compose a spec and profile into a runnable experiment directory:

```bash
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml

# Include only specific sub-experiments
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml \
    --sub-experiments baseline topk_paired

# Custom output directory
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml \
    --output-dir /tmp/experiments
```

The tool prints the generated experiment directory path on success (e.g. `experiments/resnet56_equal-split_linear-3_100mbps`).

### Experiment naming

Generated experiment names follow the pattern:

```
{model}_{partition-split}_{topology}_{profile-name}
```

Examples:
- `resnet56_equal-split_linear-3_100mbps`
- `resnet56_equal-split_linear-3_default`
- `llama_equal-split_linear-3_default`

### Generated output

The tool writes two files into `experiments/{name}/`:

- **`experiment.yaml`** — merged spec with resolved sub-experiment entries and baseline strings
- **`infra.yaml`** — the profile verbatim (InfraConfig format)

The generated `experiment.yaml` uses the *generated format* (`sub_experiments` list at top level), which the orchestrator detects automatically. Legacy hand-written experiments in the flat `experiments/{name}/` format continue to work without changes.

### Baseline materialisation

When a sub-experiment references a baseline, `tools/generate.py` automatically materialises the referenced experiment if it does not already exist. This is recursive — baselines can themselves reference further baselines. The tool detects and prevents cycles.

---

## Sub-experiments and baselines

### What sub-experiments are

Each sub-experiment is a named group of sweep runs within a single generated experiment. When the orchestrator runs a generated experiment, it executes all sub-experiments sequentially. All sub-experiment runs write to the same metrics directory (`metrics_data/{experiment_name}/`), differentiated by run ID.

Sub-experiments are designed to group related compression variants that should be analyzed together — for example, all paired topk runs at different rates in one sub-experiment, and all product topk runs in another.

### Baseline references

Sub-experiments can declare baseline references that analysis tools use for accuracy comparison. Two reference forms are supported:

**Same-spec baseline** — references another sub-experiment in the same spec at the same profile:

```yaml
baselines:
  - sub_experiment: baseline
```

The generate tool resolves this to `{experiment_name}/baseline` using the current spec and profile. No extra arguments needed.

**Cross-spec baseline** — references a sub-experiment in a different spec, typically a different topology:

```yaml
baselines:
  - spec: resnet56/single-node
    sub_experiment: baseline
    profile: resnet56/single-node/default
```

This is used to compare a distributed (partitioned) experiment against a single-node baseline. The spec, sub-experiment name, and profile are all specified explicitly. The generate tool materialises the referenced experiment (e.g. `resnet56_single-node_single-node_default`) if it does not already exist.

Both forms are fully self-contained — all information needed to materialise the referenced experiment is encoded directly in the reference.

### Distributed vs single-node baselines

A common pattern is for a distributed sweep sub-experiment to declare two baselines:

1. **Distributed no-compression baseline** — same spec (same partition split), `none` compression on all links. This controls for the overhead of partitioned inference itself.

2. **Single-node baseline** — cross-spec reference to a single-node spec. This represents the theoretical maximum accuracy with no network communication at all.

```yaml
topk_paired:
  baselines:
    - sub_experiment: baseline              # distributed, no compression
    - spec: resnet56/single-node
      sub_experiment: baseline
      profile: resnet56/single-node/default  # single-node, no network
  sweep_mode: paired
  sweep:
    ...
```

When the analysis tool sees these references, it loads results from both materialised experiments and uses them as reference columns in the accuracy comparison table.

### Per-sub-experiment datasets

For models with multiple evaluation datasets (e.g. Llama with MMLU and WikiText-2), sub-experiments can override the model-level dataset:

```yaml
sub_experiments:

  baseline_mmlu:
    dataset:
      name: mmlu
      path: .datasets/mmlu
    links:
      - from: early
        to: middle
        compression: none
      - from: middle
        to: late
        compression: none

  baseline_wikitext:
    dataset:
      name: wikitext2
      path: .datasets/wikitext2
    links:
      - from: early
        to: middle
        compression: none
      - from: middle
        to: late
        compression: none
```

When a sub-experiment specifies `dataset`, it overrides the model-level dataset for that sub-experiment only.

---

## Experiment config (generated format)

The generated `experiment.yaml` contains:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Derived experiment name |
| `model` | string | Model identifier |
| `nodes` | list | Pipeline nodes with host/port/partitions |
| `dataset` | object | Default dataset (from model-level spec) |
| `metrics_server` | object | Metrics server address |
| `sub_experiments` | list | Resolved sub-experiment entries |

Each resolved sub-experiment entry contains:

| Field | Description |
|-------|-------------|
| `name` | Sub-experiment identifier |
| `links` | Fixed link config (for non-sweep sub-experiments) |
| `sweep_mode` | `paired` or `product` |
| `sweep` | Sweep entries |
| `baselines` | List of `experiment_name/sub_experiment_name` strings |
| `dataset` | Dataset override (when present) |

Baseline strings in the generated output are fully resolved: `{experiment_name}/{sub_experiment_name}`.

---

## Experiment config (legacy format)

Legacy hand-written experiments remain supported. The orchestrator detects the format automatically: if `experiment.yaml` contains a `sub_experiments` key, it uses the generated format path; otherwise it uses the legacy single-sweep path.

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique experiment name |
| `model` | string | yes | Model identifier (e.g. `resnet56`) |
| `nodes` | list | yes | Pipeline nodes |
| `links` | list | no | Default inter-node compression config |
| `sweep` | list | no | Sweep entries; if empty, runs once with default links |
| `sweep_mode` | string | no | `paired` (default) or `product` |
| `dataset` | object | yes | Dataset config for the data client |
| `metrics_server` | object | yes | Metrics server address |
| `baselines` | list | no | Names of baseline experiments to compare against |

---

## Node config

```yaml
nodes:
  - name: A
    host: node-a
    port: 8000
    partitions: [p1, p2]
```

Partitions assigned to a node must be a contiguous subsequence of the model's full partition list.

## Link config

```yaml
links:
  - from: A
    to: B
    compression: topk
    rate: 0.3
```

`compression` must be one of:

| Method | `rate` | Extra fields | Semantics |
|--------|--------|--------------|-----------|
| `none` | not used | — | No compression |
| `topk` | fraction of values kept | — | Keep top-k by magnitude |
| `randomk` | fraction of values kept | — | Keep random-k |
| `quantization` | bit-width fraction | — | Uniform quantization |
| `llmint8` | not used | `outlier_precision`, `regular_precision` | LLM.int8 mixed-precision |

`rate` is required for `topk`, `randomk`, and `quantization`. For `llmint8`, use `outlier_precision` and `regular_precision` instead:

```yaml
links:
  - from: A
    to: B
    compression: llmint8
    outlier_precision: fp16   # fp16 or int8 (default: fp16)
    regular_precision: int8   # fp16, int8, int4, or int2 (default: int8)
```

## Sweep config

The sweep defines the compression configurations to try across runs. Each sweep entry covers all links and expands into one or more runs depending on `sweep_mode`.

```yaml
sweep_mode: paired   # or product
sweep:
  - links:
      - from: A
        to: B
        compression: none
  - links:
      - from: A
        to: B
        compression: topk
        rates: [0.1, 0.3, 0.5]
      - from: B
        to: C
        compression: topk
        rates: [0.1, 0.3, 0.5]
```

**paired mode**: rates at the same index are applied together across all links. All links in a sweep entry must have the same number of rates. A sweep entry with `rates: [0.1, 0.3, 0.5]` on two links generates 3 runs.

**product mode**: cartesian product of rates across links. Two links each with `rates: [0.1, 0.3, 0.5]` generate 9 runs.

A link can specify a single `rate` (scalar) or multiple `rates` (list), not both.

## Run IDs

Each resolved sweep run is assigned a `run_id` that encodes the full compression configuration across all links. Run IDs are used as keys throughout the metrics store (NDJSON files) and analysis output — they uniquely identify a run within an experiment.

The format is one descriptor per link, joined by `--`, with links sorted alphabetically by `(from, to)`:

```
{method}-{from}-{to}_{param}--{method}-{from}-{to}_{param}--...
```

Where `param` is:
- `{rate:.2f}` for `topk`, `randomk`, `quantization`
- `{outlier_precision}-{regular_precision}` for `llmint8`
- omitted for `none`

Examples:

```
# paired topk sweep, both links at rate 0.10
topk-A-B_0.10--topk-B-C_0.10

# mixed methods
topk-A-B_0.10--quantization-B-C_0.25

# llmint8 on first link, none on second
llmint8-A-B_fp16-int8--none-B-C

# distributed baseline, no compression
none-A-B--none-B-C

# single-node baseline, no links
single-node
```

Run IDs are deterministic — the same compression configuration always produces the same run ID regardless of sweep mode or link definition order. Sub-experiments within the same experiment should use compression configurations that produce non-overlapping run IDs to avoid blending records in the metrics store.

## Dataset config

```yaml
dataset:
  name: cifar10
  path: /data/cifar10
  batch_size: 100
  max_in_flight: 10
```

`path` is the path inside the orchestrator container. The deploy tool mounts the host dataset directory (passed via `--dataset-dir`) at this path.

---

## Infra Config

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `nodes` | list | yes | Node resource specifications |
| `links` | list | no | Link traffic shaping specifications |
| `inherits` | string | no | Path to a base infra config to inherit from |

### Node resources

```yaml
nodes:
  - name: A
    resources:
      cpu: 2        # optional — omit for no CPU limit
      memory: 4Gi   # optional — omit for no memory limit
      gpu: 0        # optional — omit or set 0 for no GPU reservation
```

All three fields are optional and default to no limit / no reservation. Omitting the `resources` block entirely is equivalent to leaving all fields unset. Only fields that are present are written into the generated Docker Compose or Kubernetes manifests — absent fields produce no constraint in the deployment.

### Link traffic shaping

```yaml
links:
  - from: A
    to: B
    bandwidth_mbps: 100   # egress rate cap (Mbit/s)
    delay_ms: 20          # added latency (ms)
    jitter_ms: 5          # latency jitter (ms) — requires delay_ms
    loss_pct: 0.1         # random packet loss (%)
```

All four fields are optional. A link with none set produces no tc rules. Any combination is valid — for example, delay without a rate cap, or loss without delay.

When any link parameter is set on a node's outgoing link, the deploy tool automatically:
- Injects `TC_LINK_<N>_*` environment variables into the node container
- Adds `cap_add: [NET_ADMIN]` (Docker) or `securityContext.capabilities.add: [NET_ADMIN]` (k8s)

At container startup, `entrypoint.sh` resolves the downstream node's hostname to an IP and installs per-link HTB and netem qdiscs. Each outgoing link gets an independent tc class so links to different downstream nodes are shaped separately.

### Inheritance

A child infra config can inherit from a base config and override specific fields:

```yaml
inherits: ../base/resnet56_infra.yaml

links:
  - from: A
    to: B
    bandwidth_mbps: 10
    delay_ms: 20
```

Inheritance is resolved at load time. Nodes and links are merged by name and from/to pair respectively — child entries replace base entries with the same key. Inheritance is supported at any depth.

### Fairness check

When running a sweep experiment, the loader compares the sweep infra config against each referenced baseline's infra config and warns if node resources or link bandwidth differ on shared nodes/links. This check is skipped for the single-node baseline since it necessarily uses a different node topology.

---

## Validation CLI

```bash
python -m framework.validate experiments/resnet56_equal-split_linear-3_100mbps
python -m framework.validate experiments/resnet56_equal-split_linear-3_100mbps --show-runs

# via Makefile
make validate EXPERIMENT=experiments/resnet56_equal-split_linear-3_100mbps
make validate EXPERIMENT=experiments/resnet56_equal-split_linear-3_100mbps SHOW_RUNS=1
```

## Python API

```python
from pathlib import Path
from framework.utils.loader import (
    load_experiment_dir,
    load_generated_experiment_config,
    is_generated_experiment,
    check_infra_fairness,
)

# Generated format
path = Path("experiments/resnet56_equal-split_linear-3_100mbps/experiment.yaml")
if is_generated_experiment(path):
    generated = load_generated_experiment_config(path)
    for sub_exp in generated.sub_experiments:
        print(sub_exp.name, sub_exp.baselines)

# Legacy format
exp, infra = load_experiment_dir(Path("experiments/resnet56_topk_sweep"))
runs = exp.resolve_sweep()
order = exp.node_order()
```

Schemas are defined as Pydantic v2 models in:
- `framework/datamodels/experiment.py` — `ExperimentConfig`, `NodeConfig`, `ResolvedRun`, …
- `framework/datamodels/infra.py` — `InfraConfig`, `InfraNodeConfig`, `InfraLinkConfig`
- `framework/datamodels/spec.py` — `GeneratedExperimentConfig`, `ResolvedSubExperiment`, `BaselineRef`
- `framework/utils/loader.py` — `load_experiment_dir`, `load_generated_experiment_config`, `is_generated_experiment`, `check_infra_fairness`
