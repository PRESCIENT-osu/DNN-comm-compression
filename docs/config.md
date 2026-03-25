# Configuration

Experiments are defined by two YAML files in an experiment directory under `experiments/<name>/`:

- `experiment.yaml` — logical experiment definition (model, topology, compression, sweep, dataset)
- `infra.yaml` — infrastructure definition (node resources, link traffic shaping)

These are kept separate so the same logical experiment can be run under different infrastructure conditions without modifying the experiment definition.

## Experiment Config

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
| `baseline` | string | no | Tags this experiment as a baseline (`single_node` or `distributed_no_compression`) |

### Node config

```yaml
nodes:
  - name: A
    host: container_a
    port: 8000
    partitions: [p1, p2]
```

Partitions assigned to a node must be a contiguous subsequence of the model's full partition list.

### Link config

```yaml
links:
  - from: A
    to: B
    compression: topk
    rate: 0.3
```

`compression` must be one of: `none`, `topk`, `packbits`, `randomk`. `rate` is required for all methods except `none`.

### Sweep config

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

### Dataset config

```yaml
dataset:
  name: cifar10
  path: /data/cifar10
  batch_size: 100
  max_in_flight: 10
```

`path` is the path inside the orchestrator container. The deploy tool mounts the host dataset directory (passed via `--dataset-dir`) at this path. Configure it to a fixed well-known path (e.g. `/data/cifar10`) and pass the actual host path at deploy time.

### Baselines

```yaml
baselines:
  - resnet56_single_node_baseline
  - resnet56_distributed_baseline
```

Analysis scripts use baseline results from the metrics server as reference points for accuracy comparisons. If a baseline has not been run, its accuracy column shows `N/A`.

To tag an experiment as a baseline itself:

```yaml
baseline: single_node   # or distributed_no_compression
```

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

To constrain only one dimension (e.g. reserve a GPU but leave CPU and memory unconstrained):

```yaml
nodes:
  - name: A
    resources:
      gpu: 1
```

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

## Validation CLI

```bash
python -m framework.validate experiments/resnet56_topk_sweep
python -m framework.validate experiments/resnet56_topk_sweep --show-runs

# via Makefile
make validate EXPERIMENT=experiments/resnet56_topk_sweep
make validate EXPERIMENT=experiments/resnet56_topk_sweep SHOW_RUNS=1
```

## Python API

```python
from pathlib import Path
from framework.utils.loader import load_experiment_dir, check_infra_fairness

exp, infra = load_experiment_dir(Path("experiments/resnet56_topk_sweep"))

# Expand sweep into concrete runs
runs = exp.resolve_sweep()

# Get pipeline node order
order = exp.node_order()
```

Schemas are defined as Pydantic v2 models in:
- `framework/datamodels/experiment.py` — `ExperimentConfig`, `NodeConfig`, `ResolvedRun`, …
- `framework/datamodels/infra.py` — `InfraConfig`, `InfraNodeConfig`, `InfraLinkConfig`
- `framework/utils/loader.py` — `load_experiment_dir`, `load_infra_config`, `check_infra_fairness`
