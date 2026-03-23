# Analysis

Post-processing tools for aggregating metrics, generating plots, and cleaning up old results. Implemented in `framework/analysis/`.

## Prerequisites

Results must exist before running analysis:
- `experiments/<name>/results/<run_id>/records.jsonl` — written by the orchestrator after each run
- `metrics_data/<experiment_id>/<event_type>.ndjson` — written by the metrics server during runs

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

## Cleanup

Removes per-run result directories older than a threshold. Age is determined by the `records.jsonl` modification time.

```bash
# Dry run (shows what would be deleted)
python -m framework.analysis.cleanup \
  --experiment resnet56_topk_sweep \
  --older-than 7 \
  --dry-run

# Delete for real
python -m framework.analysis.cleanup \
  --experiment resnet56_topk_sweep \
  --older-than 7
```

`metrics_data/` files are not modified by cleanup — they are append-only NDJSON logs managed by the metrics server.

## Typical Workflow

```bash
# 1. Run baselines first
python -m framework.orchestrator.runner experiments/resnet56_single_node_baseline
python -m framework.orchestrator.runner experiments/resnet56_distributed_baseline

# 2. Run the sweep experiment
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep

# 3. Analyse results
python -m framework.analysis.analyze --experiment resnet56_topk_sweep
python -m framework.analysis.visualize --experiment resnet56_topk_sweep

# 4. Clean up after 7 days
python -m framework.analysis.cleanup --experiment resnet56_topk_sweep --older-than 7
```
