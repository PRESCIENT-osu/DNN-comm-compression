from __future__ import annotations

import argparse
import logging
from pathlib import Path

from framework.analysis._common import (
    aggregate_metrics,
    compute_accuracy,
    load_records,
)
from framework.config.loader import load_experiment_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def visualize(
    experiment_name: str,
    experiments_dir: Path,
    metrics_dir: Path,
    output_dir: Path,
) -> None:
    """Generate and save static matplotlib plots for an experiment.

    Produces up to three PNG files in ``output_dir``:

    - ``accuracy_vs_rate.png``: top-1 accuracy per compression rate, one line
      per compression method, dashed reference lines for each baseline.
    - ``latency_breakdown.png``: stacked bar chart of mean forward-pass,
      compress, decompress, and send durations per run.
    - ``activation_sizes.png``: grouped bar chart of mean input and output
      activation sizes (KB) per run.

    Plots are silently skipped when the required data is not present.

    Args:
        experiment_name: Name matching the experiment directory.
        experiments_dir: Root directory containing all experiment directories.
        metrics_dir: Root directory for metrics NDJSON files.
        output_dir: Directory to write PNG files into (created if absent).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from exc

    experiment_dir = experiments_dir / experiment_name
    exp = load_experiment_config(experiment_dir / "experiment.yaml")
    runs = exp.resolve_sweep()
    run_map = {r.run_id: r for r in runs}

    run_records = load_records(metrics_dir, experiment_name)
    metrics = aggregate_metrics(metrics_dir, experiment_name)

    baseline_accuracies: dict[str, float | None] = {}
    for baseline_name in exp.baselines:
        bl_records = load_records(metrics_dir, baseline_name)
        if bl_records:
            all_records = [r for recs in bl_records.values() for r in recs]
            baseline_accuracies[baseline_name] = compute_accuracy(all_records)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Plot 1: Accuracy vs compression rate
    # ------------------------------------------------------------------ #
    method_pts: dict[str, list[tuple[float, float]]] = {}
    for run_id, records in run_records.items():
        acc = compute_accuracy(records)
        if acc is None:
            continue
        resolved = run_map.get(run_id)
        if resolved is None or not resolved.links:
            method, rate = "none", 0.0
        else:
            method = resolved.links[0].compression.value
            rate = resolved.links[0].rate
        method_pts.setdefault(method, []).append((rate, acc * 100))

    if method_pts:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, pts in sorted(method_pts.items()):
            pts.sort()
            xs, ys = zip(*pts, strict=False)
            ax.plot(xs, ys, marker="o", label=method)
        for bl_name, bl_acc in baseline_accuracies.items():
            if bl_acc is not None:
                ax.axhline(bl_acc * 100, linestyle="--", label=bl_name)
        ax.set_xlabel("Compression rate")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Accuracy vs Compression Rate — {experiment_name}")
        ax.legend()
        fig.tight_layout()
        out = output_dir / "accuracy_vs_rate.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 2: Latency breakdown (stacked bar)
    # ------------------------------------------------------------------ #
    if metrics:
        run_ids = sorted(metrics.keys())
        forward = [metrics[r]["forward_ms"] or 0.0 for r in run_ids]
        compress = [metrics[r]["compress_ms"] or 0.0 for r in run_ids]
        decompress = [metrics[r]["decompress_ms"] or 0.0 for r in run_ids]
        send = [metrics[r]["send_ms"] or 0.0 for r in run_ids]

        x = list(range(len(run_ids)))
        fig, ax = plt.subplots(figsize=(max(6, len(run_ids) * 0.9), 5))
        ax.bar(x, forward, label="forward")
        bottom = forward[:]
        ax.bar(x, compress, bottom=bottom, label="compress")
        bottom = [a + b for a, b in zip(bottom, compress, strict=False)]
        ax.bar(x, decompress, bottom=bottom, label="decompress")
        bottom = [a + b for a, b in zip(bottom, decompress, strict=False)]
        ax.bar(x, send, bottom=bottom, label="send")
        ax.set_xticks(x)
        ax.set_xticklabels(run_ids, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Duration (ms/batch, mean)")
        ax.set_title(f"Latency Breakdown — {experiment_name}")
        ax.legend()
        fig.tight_layout()
        out = output_dir / "latency_breakdown.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 3: Activation sizes
    # ------------------------------------------------------------------ #
    size_entries = [
        (run_id, m)
        for run_id, m in sorted(metrics.items())
        if m["input_bytes"] is not None
    ]
    if size_entries:
        run_ids_sz = [r for r, _ in size_entries]
        input_kb = [m["input_bytes"] / 1024 for _, m in size_entries]
        output_kb = [
            (m["output_bytes"] / 1024 if m["output_bytes"] is not None else 0.0)
            for _, m in size_entries
        ]
        x = list(range(len(run_ids_sz)))
        fig, ax = plt.subplots(figsize=(max(6, len(run_ids_sz) * 0.9), 5))
        ax.bar([xi - 0.2 for xi in x], input_kb, width=0.4, label="input (KB)")
        ax.bar([xi + 0.2 for xi in x], output_kb, width=0.4, label="output (KB)")
        ax.set_xticks(x)
        ax.set_xticklabels(run_ids_sz, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Bytes (KB, mean)")
        ax.set_title(f"Activation Sizes — {experiment_name}")
        ax.legend()
        fig.tight_layout()
        out = output_dir / "activation_sizes.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)


def main() -> None:
    """CLI entry point for the visualization tool."""
    parser = argparse.ArgumentParser(
        description="Generate plots for a DNN compression experiment."
    )
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root experiments directory (default: experiments/)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("metrics_data"),
        help="Root metrics directory (default: metrics_data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: experiments/<name>/plots/)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (args.experiments_dir / args.experiment / "plots")
    visualize(args.experiment, args.experiments_dir, args.metrics_dir, output_dir)


if __name__ == "__main__":
    main()
