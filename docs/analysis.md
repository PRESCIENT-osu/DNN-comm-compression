# Analysis

Post-processing tools for aggregating metrics, generating plots, and cleaning up old results. Implemented in `framework/analysis/`.

## Prerequisites

Results must exist in the metrics server storage before running analysis:

- `metrics_data/<experiment_id>/result.ndjson` — written by the metrics server as `ResultEvent`s during each run

The orchestrator streams results to the metrics server during the sweep. Analysis reads directly from that file — no separate results directory is needed.

Baselines referenced in the experiment config should also have their results present. If they are missing, analysis prints `N/A` for baseline accuracy columns.

## Analyze

Prints three ASCII tables to stdout, all sorted by compression rate:

- **Accuracy** — per-run top-1 accuracy with one reference column per baseline
- **Latency Breakdown** — mean forward-pass, compress, decompress, send, and end-to-end durations (ms/batch)
- **Activation Sizes** — mean input/output bytes per compression event and compression ratio

```bash
python -m framework.analysis.analyze \
  --experiment resnet56_topk_sweep \
  [--experiments-dir experiments] \
  [--metrics-dir metrics_data]
```

Each table shows `compression` and `rate` as the first two columns. For multi-link experiments the rate column joins both link rates with `+` (e.g. `0.10+0.30`). Baseline columns show per-run accuracy when the baseline is itself a sweep experiment, or the overall accuracy when it is a single-run baseline.

## Visualize

Generates static PNG plots saved to `experiments/<name>/plots/` by default.

### 2-D plots (all sweep types)

| File | Contents |
|------|----------|
| `accuracy_vs_rate.png` | Top-1 accuracy vs compression rate, one line per method, dashed baseline reference lines |
| `latency_breakdown.png` | Stacked bar chart of forward/compress/decompress/send per run; end-to-end shown as a separate marker |
| `activation_sizes.png` | Grouped bar chart of input and output activation KB per run |
| `accuracy_latency_pareto.png` | Scatter of accuracy vs mean end-to-end latency with Pareto frontier and baseline stars |
| `per_class_accuracy.png` | Per-class accuracy heatmap across all runs (ResNet56/CIFAR-10 only) |
| `latency_cdf.png` | Empirical CDF of per-batch end-to-end latency for each run |
| `compression_overhead.png` | Stacked bar showing compress+decompress as a fraction of total end-to-end latency |

### 3-D plots (product sweep with exactly 2 links)

| File | Contents |
|------|----------|
| `surface_accuracy.png` | Accuracy surface over the 2-link rate grid |
| `surface_accuracy_loss.png` | Accuracy loss relative to the distributed baseline surface |
| `scatter3d_bandwidth_accuracy.png` | 3-D scatter of link-0 rate × link-1 rate × accuracy |
| `scatter3d_pareto.png` | 3-D Pareto scatter minimising latency and bandwidth, maximising accuracy |
| `surface_latency.png` | Mean end-to-end latency surface over the 2-link rate grid |

3-D plots are silently skipped for paired sweeps or any sweep with fewer than 2 links.

All plots are silently skipped when required data is not available.

```bash
python -m framework.analysis.visualize \
  --experiment resnet56_topk_sweep \
  [--experiments-dir experiments] \
  [--metrics-dir metrics_data] \
  [--output-dir experiments/resnet56_topk_sweep/plots]
```

## Topology

Renders an infra config as a directed graph and saves it as a PNG (or PDF/SVG). Nodes are placed using a layered BFS layout — source nodes on the left, sink nodes on the right, with fan-out and fan-in nodes in intermediate layers. Edge labels show bandwidth, delay, and loss where configured. Resource annotations (CPU/memory/GPU) appear below each node when set.

```bash
python -m framework.analysis.topology \
  --infra experiments/resnet56_topk_sweep/infra.yaml \
  --output topology.png \
  [--title "My Topology"]
```

The `--output` flag accepts any format supported by matplotlib (`.png`, `.pdf`, `.svg`). If omitted the plot is displayed interactively.

## Cleanup

Removes stale metrics and result files for an experiment:

```bash
python -m framework.analysis.cleanup --experiment resnet56_topk_sweep
```

> **Note:** Re-running the same experiment without cleaning up first will append new records to existing NDJSON files, blending results across runs and producing incorrect metrics. Always clean up before re-running an experiment.

## Typical Workflow

```bash
# 1. Generate the experiment (baselines are materialised automatically)
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml

# 2. Deploy and run
python -m framework.deploy \
  --experiment experiments/resnet56_equal-split_linear-3_100mbps \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply

# Wait for orchestrator to finish, then tear down
docker compose -f experiments/resnet56_equal-split_linear-3_100mbps/deploy/docker-compose.yml down

# 3. Analyse results
python -m framework.analysis.analyze --experiment resnet56_equal-split_linear-3_100mbps
python -m framework.analysis.visualize --experiment resnet56_equal-split_linear-3_100mbps

# 4. (Optional) Visualise the infrastructure topology
python -m framework.analysis.topology \
  --infra experiments/resnet56_equal-split_linear-3_100mbps/infra.yaml \
  --output experiments/resnet56_equal-split_linear-3_100mbps/topology.png
```
