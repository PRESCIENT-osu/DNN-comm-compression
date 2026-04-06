# Optimization Experiments

Optimization experiments are a distinct mode of operation that sits on top of the multi-model infrastructure. Instead of sweeping a fixed grid of compression rates, the optimizer selects a compression ratio η per pipeline per link at each *slot* — a discrete time step covering a fixed number of inference batches — using one of several online resource allocation algorithms. The goal is to learn a compression policy that meets per-pipeline throughput targets while maximising inference accuracy.

---

## Concepts

### Slots

Time is divided into fixed-length slots. During each slot, the orchestrator:

1. Probes link capacities (throughput measurements).
2. Calls the optimizer to select η per pipeline per link.
3. Pushes the resulting compression config to all nodes.
4. Submits `batches_per_slot` inference batches and collects results.
5. Updates optimizer state (dual variables, channel estimators).

The number of slots and batches per slot are set in the `optimization_loop` section of the optspec.

### Tasks and Links (InferenceTask model)

The optimizer's internal model represents each pipeline as an `InferenceTask` with:

- **τ_i** — per-node compute latency (seconds), measured during the profiling slot.
- **a_i** — mean activation payload bytes transmitted on each inter-node link, measured during the profiling slot.
- **η_i ∈ [η_min, η_max]** — compression ratio on link i (1 = no compression, lower = more compression).
- **R_k** — throughput target (tasks/second), set in the optspec.
- **w_k** — WFQ compute weight, set in the optspec.
- **A_k(η)** — task accuracy as a function of compression ratios, provided as a callable.

### Accuracy Oracle

For the `no_csi` and `csi_aware` optimizers (and no-CSI multi-task baselines) the optimizer calls **A_k(η)** and **∇A_k(η)** when solving the per-slot subproblem. These callables are provided by the Stein gradient oracle wrapping a full-model simulation pipeline:

- **ResNet56** — `SimulatedResNetPipeline` loads the full `ResNet56` nn.Module from a checkpoint and installs `top-k` sparsification hooks at partition boundary layers. Accuracy is evaluated on a random CIFAR-10 subset (`n_fast_samples`).
- **Llama** — `SimulatedLlamaPipeline` loads the full HuggingFace model and uses `LLMActivationCompressor` hooks at decoder layer boundaries. Accuracy is evaluated with an MMLU or WikiText-2 evaluator.

The **Stein gradient oracle** estimates ∇A_k(η) using antithetic perturbations:

```
∇A_k(η) ≈ (1 / 2Nσ) Σ_{j=1}^{N} z_j · [A_k(η + σz_j) - A_k(η − σz_j)]
```

where z_j ~ N(0, I). This requires 2N model evaluations per gradient call. The `SteinOracleConfig` controls σ (`sigma`), N, and `n_fast_samples`.

For baseline sub-experiments whose optimizers do not call A_k (CSI-aware baselines, max/no-compression), trivial dummy callables are installed — the overhead is zero.

### Channel Estimation

No-CSI optimizers and estimated-CSI baselines maintain a per-link channel capacity estimator that is updated with measured throughput after each slot probe. Available estimators:

| Type | Description |
|------|-------------|
| `last_observation` | Most recent probe value |
| `mean` | Running mean over all observations |
| `running_min` | Running minimum over all observations |
| `moving_average` | Mean over a sliding window |
| `lcb` | Mean − z·σ lower confidence bound |

---

## Directory Structure

```
optspecs/                           ← optimization experiment definitions (source)
  {name}/
    experiment.yaml                 ← topology, pipelines, tasks, links, loop config
    sub_experiments.yaml            ← ordered list of sub-experiment configs

experiments/opt/                    ← generated output (gitignored)
  {name}/
    experiment.yaml                 ← fully resolved, consumed by opt_runner

framework/optimizer/
  inference_optimizer_adapter.py    ← adapter: builds InferenceTask, wraps external optimizers
  channel_estimators.py             ← per-link capacity estimators (framework side)
  compression_mapper.py             ← maps continuous η → discrete method + params
  accuracy_model.py                 ← surrogate accuracy model (sklearn backend)
  evaluators.py                     ← Llama MMLU + WikiText evaluator wrappers

framework/nodes/orchestrator/
  opt_runner.py                     ← optimization slot loop, artifact store, CLI entry point
  artifacts.py                      ← artifact read/write with config hash validation
  link_prober.py                    ← orchestrator-side link probing

models/resnet/simulation/
  pipeline.py                       ← SimulatedResNetPipeline (full-model, hook-based)

models/llama/simulation/
  pipeline.py                       ← SimulatedLlamaPipeline (full HF model, hook-based)

external/Inference_Optimizer/src/   ← external optimizer library (git submodule)
  core/task.py                      ← InferenceTask dataclass
  core/toy_A.py                     ← Stein gradient oracle (grad_oracle)
  optimizers/no_csi.py              ← NoCSISingleTaskOptimizer, NoCSIMultiTaskOptimizer
  optimizers/csi_aware.py           ← CSIAwareSingleTaskOptimizer, CSIAwareMultiTaskOptimizer
  optimizers/baseline.py            ← all baseline optimizer classes
  optimizers/estimators.py          ← vector channel estimators
```

---

## Optspec Format

Optimization specs live under `optspecs/` and are paired with a profile by `tools/generate.py --opt`.

### `experiment.yaml`

```yaml
nodes:
  - A
  - B
  - C

pipelines:
  - name: resnet-a
    model: resnet
    partitions:
      A: [p1]
      B: [p2, p3]
      C: [p4, p5]
    flow: [A, B, C]
    simulation_path: models/resnet/resnet56.th   # required for accuracy_model and Stein oracle

  - name: llama-mmlu-a
    model: llama
    partitions:
      A: [p1]
      B: [p2]
      C: [p3]
    flow: [A, B, C]
    simulation_path: models/llama/llama-3.1-8b   # required for accuracy_model and Stein oracle

datasets:                  # used by actual inference tasks (seed: 42)
  resnet:
    name: cifar10
    path: .datasets/cifar10
    batch_size: 100
    max_in_flight: 10
    max_samples: 1000
    seed: 42
  llama:
    name: mmlu
    path: .datasets/mmlu
    tokenizer_path: models/llama/.partitions/tokenizer
    batch_size: 4
    max_in_flight: 4
    subjects: [college_computer_science, high_school_mathematics]
    samples_per_subject: 20
    seed: 42

# Required when any sub-experiment uses stein_config.
# Different seeds prevent data overlap with `datasets` and accuracy model datasets.
stein_datasets:
  resnet:
    name: cifar10
    path: .datasets/cifar10
    batch_size: 100
    max_samples: 512
    seed: 200
  llama:
    name: mmlu
    path: .datasets/mmlu
    tokenizer_path: models/llama/.partitions/tokenizer
    batch_size: 4
    subjects: [college_computer_science, high_school_mathematics]
    samples_per_subject: 10
    seed: 201

workload:
  pattern: fill
  window_per_pipeline: 4
  mix:
    resnet-a: 0.5
    llama-mmlu-a: 0.5

# Per-pipeline throughput targets and WFQ weights.
tasks:
  resnet-a:
    throughput_target: 5.0    # R_k (tasks/second)
    task_weight: 0.4          # w_k (proportional WFQ compute share)
  llama-mmlu-a:
    throughput_target: 0.2
    task_weight: 0.6

# Per-link optimizer search space.
links:
  - from: A
    to: B
    eta_min: 0.05
    eta_max: 1.0
    allowed_methods: [topk, llmint8]
    llmint8_mapping:
      resnet-a:
        entries:
          - feature_k_values: [0.85]
            outlier_values: [0.01]
      llama-mmlu-a:
        entries:
          - feature_k_values: [0.90]
            outlier_values: [0.005]
  - from: B
    to: C
    eta_min: 0.05
    eta_max: 1.0
    allowed_methods: [topk]

optimization_loop:
  n_slots: 100
  batches_per_slot: 10
  profiling_batches: 20
  link_probe_interval_slots: 1
  dual_step_size: 0.01

metrics_server:
  host: metrics
  port: 9100
```

**`tasks`** — one entry per pipeline name. `throughput_target` (R_k) is the minimum tasks/second the optimizer tries to sustain. `task_weight` (w_k) is used as a WFQ scheduling weight and as the per-task objective weight in multi-task optimizers.

**`links`** — each entry sets the η search space for one inter-node link:
- `eta_min` — most compressed (floor on accuracy loss).
- `eta_max` — least compressed; `1.0` means no compression.
- `allowed_methods` — compression methods the optimizer may use (`topk`, `llmint8`).
- `llmint8_mapping` — per-pipeline nearest-neighbour codec lookup table. Required when `llmint8` is in `allowed_methods`.

**`optimization_loop`**:

| Field | Description |
|-------|-------------|
| `n_slots` | Total optimizer slots to run |
| `batches_per_slot` | Inference batches submitted per slot |
| `profiling_batches` | Batches during the profiling slot (η=1.0) |
| `link_probe_interval_slots` | Probe links every N slots |
| `dual_step_size` | Lagrangian dual step size ε (no-CSI algorithms) |

### `sub_experiments.yaml`

Sub-experiments are executed **in order**. Later sub-experiments may depend on artifacts from earlier ones. The file must always begin with a `profiling` sub-experiment.

Surrogate variant (accuracy model trained via simulation sweep, then used by optimizer):

```yaml
sub_experiments:
  - name: profiling
    type: profiling

  - name: accuracy_model_resnet
    type: accuracy_model
    pipeline_id: resnet-a
    model_type: gbm
    sweep_design: random
    n_sweep_samples: 60
    dataset:
      name: cifar10
      path: .datasets/cifar10
      batch_size: 100
      max_samples: 100
      seed: 100

  - name: no_csi_mu_sweep
    type: no_csi
    mu_sweep: [0.5, 1.0, 3.0, 10.0]
    accuracy_model_refs: [accuracy_model_resnet]
    channel_estimator:
      type: moving_average
      window_size: 10
      warmup_value_bps: 1.0e8

  - name: max_compression_single
    type: max_compression_single

  - name: historical_average_ce
    type: historical_average_ce
    channel_estimator:
      type: mean
      warmup_value_bps: 1.0e8
```

Stein oracle variant (A_k(η) evaluated on-the-fly; `stein_datasets` must be set in `experiment.yaml`):

```yaml
sub_experiments:
  - name: profiling
    type: profiling

  - name: no_csi_sweep
    type: no_csi
    mu_sweep: [0.5, 1.0, 3.0, 10.0]
    stein_config:
      sigma: 0.05
      N: 50
      n_fast_samples: 512
    channel_estimator:
      type: moving_average
      window_size: 10
      warmup_value_bps: 1.0e8

  - name: csi_aware
    type: csi_aware
    stein_config:
      sigma: 0.05
      N: 50
      n_fast_samples: 512
```

---

## Sub-Experiment Types

### Profiling (`profiling`)

**Must appear first.** Runs `profiling_batches` inference rounds at η=η_max (no compression) on all links, then probes all links. Stores:
- Per-pipeline per-node mean compute latency τ_i (seconds).
- Per-link mean activation payload size a_i (bytes).
- Nominal link throughput at η=1.0.

Artifacts are written to `{artifacts_dir}/profiling/{name}.json` and reused across subsequent sub-experiments. If the artifact already exists and its config hash matches, the profiling slot is skipped.

### Accuracy Model (`accuracy_model`)

Trains a surrogate A_k(η) for one pipeline via a **simulation-based sweep**. The pipeline's `simulation_path` must be set in `experiment.yaml`. The sweep evaluates `n_sweep_samples` distinct per-link η vectors through the simulation pipeline and fits a sklearn model to the resulting (η_vector, accuracy) pairs.

```yaml
- name: accuracy_model_resnet
  type: accuracy_model
  pipeline_id: resnet-a
  model_type: gbm          # sklearn backend: linear_monotonic, poly2, poly3, gbm, rf, mlp, mlp_small
  sweep_design: random     # random | diagonal
  n_sweep_samples: 60      # number of (η_vector, accuracy) pairs to collect
  dataset:                 # separate dataset slice — different seed from main tasks
    name: cifar10
    path: .datasets/cifar10
    batch_size: 100
    max_samples: 100
    seed: 100
```

The `dataset` block uses a different `seed` than the main `datasets` entry and the `stein_datasets` entry so that accuracy model training data does not overlap with actual inference task data or Stein oracle evaluation data.

The fitted model is pickled to `{artifacts_dir}/accuracy_models/{pipeline_id}_{name}.pkl` with a companion `{name}.json` storing the config hash for cache invalidation. If both files exist and the hash matches, the sweep is skipped and the cached model is loaded.

For sub-experiments using a Stein oracle (`stein_config` set), A_k(η) is evaluated on-the-fly — no `accuracy_model` phase is needed.

### No-CSI Optimizer (`no_csi`)

Primal-dual online optimizer. Does not require knowledge of the instantaneous channel state; instead uses a channel estimator and a Lagrangian dual variable λ_k per pipeline that penalises throughput violations.

`mu_sweep` is a list of penalty weights μ. Each μ value produces **one independent run** with its own dual variable state. All runs share the same profiling artifacts.

With surrogate accuracy models (one per pipeline):

```yaml
- name: no_csi_mu_sweep
  type: no_csi
  mu_sweep: [0.5, 1.0, 3.0, 10.0]
  accuracy_model_refs:            # names of accuracy_model sub-experiments to load
    - accuracy_model_resnet
    - accuracy_model_llama
  bcd_iterations: 10
  channel_estimator:
    type: moving_average
    window_size: 10
    warmup_value_bps: 1.0e8
```

With Stein oracle (no pre-training needed):

```yaml
- name: no_csi_sweep
  type: no_csi
  mu_sweep: [0.5, 1.0, 3.0, 10.0]
  stein_config:
    sigma: 0.05
    N: 50
    n_fast_samples: 512
  channel_estimator:
    type: moving_average
    window_size: 10
    warmup_value_bps: 1.0e8
```

`accuracy_model_refs` is a list of `accuracy_model` sub-experiment names. Each referenced sub-experiment covers one pipeline (determined by its `pipeline_id`); the runner merges them into a single `{pipeline_id → AccuracyModel}` map before passing to the optimizer. `stein_config` takes priority over `accuracy_model_refs` when both are set.

For single-pipeline experiments this uses `NoCSISingleTaskOptimizer`; for multi-pipeline experiments it uses `NoCSIMultiTaskOptimizer` (block-coordinate descent).

### CSI-Aware Optimizer (`csi_aware`)

Closed-form optimal η given instantaneous link capacity. For single-pipeline: η_i* = clip(c_i / (R·a_i), η_min, η_max). For multi-pipeline: solves a convex program over s_comm shares.

```yaml
- name: csi_aware
  type: csi_aware
  stein_config:
    sigma: 0.05
    N: 50
    n_fast_samples: 512
  channel_estimator:
    type: lcb
    window_size: 20
    z: 1.5
    warmup_value_bps: 1.0e8
```

The `stein_config` field is optional for baselines that do not evaluate A_k but is required for `no_csi` and `csi_aware` when simulation pipelines are active.

### Baseline Sub-Experiments

All baseline types use the same multi-model node infrastructure and the same slot loop. They differ only in how η is selected each slot.

#### Single-task baselines

| Type | Description |
|------|-------------|
| `max_compression_single` | η = η_min on every link every slot (maximum compression) |
| `no_compression_single` | η = 1.0 on every link every slot (no compression) |
| `uniform_compression_single` | Single uniform η derived from instantaneous c_t: `min(1, min_i c_i / (R·a_i))` |
| `estimated_csi_single` | Per-link η derived from c_hat: `min(1, c_hat_i / (R·a_i))`. Estimator type selects the variant (myopic = `last_observation`, conservative = `running_min`, etc.) |

#### Multi-task baselines

| Type | Description |
|------|-------------|
| `max_compression_multi` | η = η_min with equal static resource shares across pipelines |
| `no_compression_multi` | η = 1.0 with equal static resource shares |
| `static_equal_share` | Equal s_comp / s_comm splits; η derived from c_t and equal share |
| `proportional_resource` | Resource shares proportional to τ_k / a_k; η derived from c_t |
| `strict_priority_greedy` | Greedy allocation by w_k priority; η derived from available bandwidth |
| `decoupled_descent` | No-CSI: equal-split resources, per-task SLSQP subproblem, dual queues λ_k |
| `queue_proportional` | No-CSI: queue-proportional resource allocation, dual queues λ_k |
| `historical_average_ce` | No-CSI: feeds historical mean c_hat into the CSI-aware multi-task solver |

Baselines that maintain dual queues (`decoupled_descent`, `queue_proportional`) have:
- `mu` — Lagrangian penalty weight
- `epsilon` — initial and minimum dual variable value

All baselines that use a channel estimator accept a `channel_estimator` block identical to `no_csi`.

---

## Generating Optimizer Experiments

```bash
# Single optspec + profile
python tools/generate.py --opt \
    --spec optspecs/resnet56_llama_mmlu \
    --profile profiles/linear-3-multi/100mbps.yaml

# All compatible optspec/profile combinations
python tools/generate.py --opt --all
```

Generated experiments are placed under `experiments/opt/` with the name pattern `{spec_name}_{topology}_{profile_name}`:

```
experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps/
  experiment.yaml     ← fully resolved GeneratedOptExperimentConfig
  infra.yaml          ← verbatim profile copy
```

The generator:
- Injects node host/port from the profile.
- Sets `artifacts_dir` to `artifacts/{exp_name}` (relative to the working directory where `opt_runner` is invoked).
- Injects `channel_estimator.experiment_name_contains` from the profile filename stem so probe history is scoped to the correct hardware profile.

---

## Deploying Optimizer Experiments

Optimizer experiments share the same multi-model node infrastructure (`dnn-compute-multi`) and additionally require the `dnn-orchestrator` image. Build all images first:

```bash
make build-multi
make build-metrics
make build-orchestrator
```

### Docker

Generate a `docker-compose.yml` and optionally apply it immediately:

```bash
python -m framework.deploy \
    --experiment experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --opt \
    --target docker \
    --partitions-dir /absolute/path/to/.partitions \
    --dataset-dir /absolute/path/to/.datasets \
    --artifacts-dir /absolute/path/to/artifacts/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --apply
```

| Flag | Default | Description |
|------|---------|-------------|
| `--partitions-dir` | *(required)* | Host path to the partitions base directory (contains `resnet/`, `llama/` subdirs) |
| `--dataset-dir` | *(required)* | Host path to the base `.datasets/` directory |
| `--artifacts-dir` | `artifacts/` | Host path that maps to the container's `artifacts/{exp_name}/` directory |
| `--metrics-dir` | `metrics_data` | Host path for metrics NDJSON storage |
| `--apply` | — | Run `docker compose up -d` immediately after generating the manifest |

The manifest is written to `experiments/opt/{exp_name}/deploy/docker-compose.yml`.

The orchestrator container mounts `--artifacts-dir` at `/app/artifacts/{exp_name}/` (read-write). All profiling results, accuracy model pickles, and estimator state files are written there and persist on the host across container restarts. `--artifacts-dir` defaults to `./artifacts/` but should point to the experiment-specific subdirectory so the ArtifactStore's layout lands cleanly:

```
/absolute/path/to/artifacts/resnet56_llama_mmlu_linear-3-multi_100mbps/
  profiling/
  accuracy_models/
    resnet-a_accuracy_model_resnet.pkl
    resnet-a_accuracy_model_resnet.json
    llama-mmlu-a_accuracy_model_llama.pkl
    llama-mmlu-a_accuracy_model_llama.json
  estimators/
  slots/
```

### Kubernetes

Generate a `manifests.yaml` and optionally apply it:

```bash
python -m framework.deploy \
    --experiment experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --opt \
    --target k8s \
    --partitions-dir /absolute/path/to/.partitions \
    --dataset-dir /absolute/path/to/.datasets \
    --artifacts-dir /absolute/path/to/artifacts/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --namespace my-namespace \
    --apply
```

The manifest is written to `experiments/opt/{exp_name}/deploy/manifests.yaml`. It includes:
- One `Pod` per compute node (`dnn-compute-multi` image)
- A `Service` per compute node
- A metrics `Pod` + `Service` (`dnn-metrics` image)
- One orchestrator `Job` (`dnn-orchestrator` image)

**Artifact storage on Kubernetes** — the orchestrator Job uses a `hostPath` volume mounted at `/app/artifacts/{exp_name}/` inside the pod. Artifacts are written to `--artifacts-dir` on the **filesystem of whichever cluster node the orchestrator pod is scheduled on**. This has two implications:

1. The directory must exist on that node before the Job starts. Create it manually or via an `initContainer` if needed.
2. `hostPath` volumes are node-local and not shared. If you need artifacts to survive pod rescheduling or be accessible from multiple nodes, use a `PersistentVolumeClaim` with `ReadWriteMany` access (e.g., NFS or a cloud-provider managed disk) and patch the generated manifest accordingly.

`--artifacts-dir` should always be specified explicitly for Kubernetes — the default (`./artifacts/`) is a relative path and will resolve to an unpredictable location on the node's filesystem.

---

## Running Optimizer Experiments

```bash
python -m framework.nodes.orchestrator.opt_runner \
    experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps \
    --callback-host orchestrator \
    --callback-port 8080
```

| Flag | Default | Description |
|------|---------|-------------|
| `--callback-host` | `localhost` / `CALLBACK_HOST` | Hostname nodes use to reach the callback server |
| `--callback-port` | `8080` / `CALLBACK_PORT` | Callback server port |
| `--result-timeout` | `300.0` | Per-task result wait timeout (seconds) |
| `--dry-run` | — | Log the plan without executing |
| `--node-host` | `NODE_HOST` env | Override hostname for all nodes |
| `--metrics-host` | `METRICS_HOST` env | Override metrics server hostname |

The runner requires the same multi-model node infrastructure as `multi_runner.py`. Nodes must be running `framework.nodes.compute.multi.server`.

### Sub-experiment execution order

Sub-experiments run **sequentially** in definition order. Within each sub-experiment, the optimizer's slot loop runs fully before the next sub-experiment begins. For `no_csi`, each μ in `mu_sweep` produces an independent sequential run.

### Profiling artifact caching

If a profiling artifact already exists at `{artifacts_dir}/profiling/{name}.json` with a matching config hash, the profiling slot is skipped and τ/a values are loaded from the cached file. This allows interrupted runs to resume without re-profiling.

---

## Metrics Emitted

The opt_runner emits standard multi-model events plus three optimization-specific event types:

### `opt_slot`

Emitted once per optimization slot.

| Field | Description |
|-------|-------------|
| `slot_id` | Slot index |
| `eta_per_pipeline_per_link` | η selected by the optimizer for each pipeline on each link |
| `d_excess_per_task` | Per-pipeline throughput shortfall (max(0, R_k − achieved_rps)) |
| `c_hat_per_link` | Channel capacity estimate used at decision time |
| `optimizer_type` | Sub-experiment name (e.g. `no_csi_mu_sweep_mu1.0`) |
| `solve_time_ms` | Time taken to call the optimizer |
| `infeasible` | True when the optimizer declared infeasibility and η_max fallback was used |
| `sub_experiment_name` | Sub-experiment name for grouping |

### `throughput_constraint`

Emitted once per pipeline per slot.

| Field | Description |
|-------|-------------|
| `pipeline_id` | Pipeline name |
| `target_rps` | Throughput target R_k |
| `achieved_rps` | Achieved tasks/second this slot |
| `satisfied` | True when achieved_rps ≥ target_rps |
| `violation_magnitude` | max(0, target_rps − achieved_rps) |
| `cumulative_violations` | Total slots violated so far for this pipeline |

### `task_accuracy`

Emitted once per pipeline per slot during optimization, and once per simulation sample during `accuracy_model` sweeps.

| Field | Description |
|-------|-------------|
| `pipeline_id` | Pipeline name |
| `compression_rate` | Mean η across all links (scalar summary) |
| `accuracy` | Top-1 accuracy (ResNet) or MMLU accuracy (Llama) |
| `eta_per_link` | `{link_id: η}` per-link vector; set during `accuracy_model` sweeps, `null` during slot loop |
| `slot_id` | Slot index; `null` during accuracy model sweeps |
| `sub_experiment_name` | Sub-experiment that emitted this record |

---

## Simulation Pipelines

Simulation pipelines are used by the Stein oracle to evaluate A_k(η) without running real distributed inference. They load the full model locally on the orchestrator, install compression hooks at partition boundary layers, and run forward passes.

### `SimulatedResNetPipeline`

```python
from models.resnet.simulation.pipeline import SimulatedResNetPipeline
from torch.utils.data import DataLoader

pipeline = SimulatedResNetPipeline(
    checkpoint_path=Path("models/resnet/.partitions/resnet56.th"),
    partitions={"A": ["p1"], "B": ["p2", "p3"], "C": ["p4", "p5"]},
    flow=["A", "B", "C"],
    test_loader=test_loader,  # CIFAR-10 DataLoader
)

# Evaluate at η = [0.5, 0.3]  (2 inter-node links: A→B, B→C)
eta = torch.tensor([0.5, 0.3])
acc = pipeline.accuracy(eta, n_samples=512)   # fast: random 512-sample subset
acc = pipeline.accuracy(eta, n_samples=None)  # full: entire test loader
```

Compression is top-k magnitude sparsification, matching the deployed `TopK` compressor. Hooks are installed on `model.layer1/2/3` as pre- or post-forward hooks depending on which partition sends on each link.

### `SimulatedLlamaPipeline`

```python
from models.llama.simulation.pipeline import SimulatedLlamaPipeline
from framework.optimizer.evaluators import MMLUEvaluatorWrapper

fast_eval = MMLUEvaluatorWrapper(tokenizer, subjects=[...], samples_per_subject=10)
full_eval = MMLUEvaluatorWrapper(tokenizer, subjects=[...], samples_per_subject=50)

pipeline = SimulatedLlamaPipeline(
    model_name="meta-llama/Llama-3.1-8B",
    partitions={"A": ["p1"], "B": ["p2"], "C": ["p3"]},
    flow=["A", "B", "C"],
    fast_evaluator=fast_eval,
    full_evaluator=full_eval,
    activation_strategy="topk_per_token",
)

eta = torch.tensor([0.5])   # 1 inter-node link: A→B only (B→C is p3's output)
acc = pipeline.accuracy(eta)          # fast evaluator
acc = pipeline.accuracy(eta, full=True)  # full evaluator
```

Hook positions follow `partition_llama.py` boundary indices: cut at decoder layer `n//3` (p1 boundary) and `2*n//3` (p2 boundary).

---

## Docker Image

The orchestrator image requires a GPU for Stein oracle evaluations:

```
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# torchvision (ResNet simulation), transformers + accelerate (Llama), scipy (SLSQP)
pip install torchvision transformers datasets accelerate scipy scikit-learn

COPY framework/
COPY models/resnet/    # SimulatedResNetPipeline
COPY models/llama/     # SimulatedLlamaPipeline
COPY external/Inference_Optimizer/src/   # optimizer + oracle library
```

Build:
```bash
docker build -f docker/Dockerfile.orchestrator -t dnn-orchestrator:latest .
```

Mount a volume at `/hf_cache` to cache HuggingFace model weights across container restarts.

---

## Python API

```python
from pathlib import Path
from framework.utils.loader import load_opt_experiment_config
from framework.optimizer.inference_optimizer_adapter import (
    build_global_order,
    build_inference_tasks,
    build_adapter,
    probe_dict_to_c_t_vector,
)

exp = load_opt_experiment_config(Path("experiments/opt/.../experiment.yaml"))

global_order = build_global_order(exp)       # ["A", "B", "C"]

tasks, tid_to_pid, pid_to_tid = build_inference_tasks(
    exp=exp,
    tau_per_node={"resnet-a": {"A": 0.02, "B": 0.04, "C": 0.03}},
    a_per_link_bytes={"A-B": 614400.0, "B-C": 204800.0},
    global_order=global_order,
)

# Build adapter for a no_csi sub-experiment with mu=1.0
sub_exp = exp.sub_experiments[2]   # NoCsiSubExperiment
adapter = build_adapter(sub_exp, tasks, tid_to_pid, pid_to_tid, global_order, exp, mu=1.0)

# Single slot
c_t = probe_dict_to_c_t_vector({"A-B": 1e8, "B-C": 1e8}, global_order)
eta_per_pipeline_per_link, infeasible = adapter.step(t=0, c_t=c_t)
adapter.observe_capacity(c_t)
adapter.update_dual(t=0, actual_delays={0: 0.22, 1: 5.2})
```

Relevant Pydantic schemas (from `framework.datamodels.opt_experiment`):

| Class | Description |
|-------|-------------|
| `GeneratedOptExperimentConfig` | Top-level generated config |
| `OptTaskConfig` | Throughput target + WFQ weight per pipeline |
| `OptLinkConfig` | η bounds, allowed methods, LLMInt8 mapping |
| `OptimizationLoopConfig` | Slot loop hyperparameters |
| `SteinOracleConfig` | Stein oracle σ, N, n_fast_samples |
| `ChannelEstimatorConfig` | Estimator type, window size, warmup |
| `ProfilingSubExperiment` | Profiling phase config |
| `NoCsiSubExperiment` | No-CSI optimizer config |
| `CsiAwareSubExperiment` | CSI-aware optimizer config |
| `*SingleSubExperiment` | Single-task baseline configs |
| `*MultiSubExperiment` / `*SubExperiment` | Multi-task baseline configs |
