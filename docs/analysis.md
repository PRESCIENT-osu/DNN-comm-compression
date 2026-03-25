# Analysis

Post-processing tools for aggregating metrics, generating plots, and cleaning up old results. Implemented in `framework/analysis/`.

## Prerequisites

Results must exist in the metrics server storage before running analysis:

- `metrics_data/<experiment_id>/result.ndjson` — written by the metrics server as `ResultEvent`s during each run

The orchestrator streams results to the metrics server during the sweep. Analysis reads directly from that file — no separate results directory is needed.

Baselines referenced in the experiment config should also have their results present. If they are missing, analysis prints `N/A` for baseline accuracy columns.

## Analyze

Prints three ASCII tables to stdout:

- **Accuracy** — per-run top-1 accuracy with one reference column per baseline
- **Latency Breakdown** — mean forward-pass, compress, decompress, send, and end-to-end durations (ms/batch)
- **Activation Sizes** — mean input/output bytes per compression event and compression ratio

```bash
python -m framework.analysis.analyze \
  --experiment resnet56_topk_sweep \
  [--experiments-dir experiments] \
  [--metrics-dir metrics_data]
```

## Visualize

Generates static PNG plots saved to `experiments/<name>/plots/` by default:

| File | Contents |
|------|----------|
| `accuracy_vs_rate.png` | Top-1 accuracy vs compression rate, one line per method, dashed baseline reference lines |
| `latency_breakdown.png` | Stacked bar chart of forward/compress/decompress/send per run; end-to-end shown as a separate marker |
| `activation_sizes.png` | Grouped bar chart of input and output activation KB per run |

Plots are skipped silently when no data is available.

```bash
python -m framework.analysis.visualize \
  --experiment resnet56_topk_sweep \
  [--experiments-dir experiments] \
  [--metrics-dir metrics_data] \
  [--output-dir experiments/resnet56_topk_sweep/plots]
```

## Typical Workflow

```bash
# 1. Deploy and run baselines
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_single_node_baseline \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply

# Wait for orchestrator to finish, then tear down
docker compose -f experiments/resnet56_single_node_baseline/deploy/docker-compose.yml down

# Repeat for the distributed baseline
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_distributed_baseline \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply

docker compose -f experiments/resnet56_distributed_baseline/deploy/docker-compose.yml down

# 2. Run the sweep experiment
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply

docker compose -f experiments/resnet56_topk_sweep/deploy/docker-compose.yml down

# 3. Analyse results
python -m framework.analysis.analyze --experiment resnet56_topk_sweep
python -m framework.analysis.visualize --experiment resnet56_topk_sweep
```
