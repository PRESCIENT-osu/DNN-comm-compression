from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from framework.analysis._common import (
    aggregate_metrics,
    compute_accuracy,
    compute_per_class_accuracy,
    load_raw_end_to_end,
    load_records,
)
from framework.datamodels.experiment import SweepMode
from framework.utils.loader import load_experiment_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CIFAR10_CLASSES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return Pareto-optimal (x, y) pairs that minimise x and maximise y."""
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier: list[tuple[float, float]] = []
    best_y = float("-inf")
    for x, y in sorted_pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def _pareto3d(points: list[tuple[float, float, float]]) -> list[bool]:
    """Return a boolean mask: True if the point is NOT dominated.

    Dominance is defined as: minimise x, maximise y, minimise z.
    Point P is dominated if there exists Q with Q.x ≤ P.x, Q.y ≥ P.y,
    Q.z ≤ P.z and at least one strict inequality.

    Args:
        points: List of (x, y, z) tuples.

    Returns:
        List of booleans, True where the corresponding point is Pareto-optimal.
    """
    dominated = [False] * len(points)
    for i, (xi, yi, zi) in enumerate(points):
        if dominated[i]:
            continue
        for j, (xj, yj, zj) in enumerate(points):
            if i == j:
                continue
            if xj <= xi and yj >= yi and zj <= zi and (xj < xi or yj > yi or zj < zi):
                dominated[i] = True
                break
    return [not d for d in dominated]


def _build_surface_grid(
    run_map: dict[str, Any],
    metric_by_run_id: dict[str, float | None],
) -> tuple[Any, Any, Any] | None:
    """Build numpy meshgrid arrays for a 2-link product sweep surface plot.

    Extracts per-link rates from run_map and arranges the corresponding metric
    values onto a regular grid.  Cells with no matching run_id are filled with
    NaN so matplotlib silently skips them.

    Args:
        run_map: Mapping of run_id to ResolvedRun (must have exactly 2 links).
        metric_by_run_id: Mapping of run_id to scalar metric value.

    Returns:
        Tuple (X, Y, Z) of 2-D numpy arrays, or None if fewer than 4 data
        points are available.
    """
    import numpy as np

    triples: list[tuple[float, float, float]] = []
    for run_id, value in metric_by_run_id.items():
        run = run_map.get(run_id)
        if run is None or len(run.links) < 2 or value is None:
            continue
        triples.append((run.links[0].rate, run.links[1].rate, value))

    if len(triples) < 4:
        return None

    rates0 = sorted({t[0] for t in triples})
    rates1 = sorted({t[1] for t in triples})
    lookup = {(t[0], t[1]): t[2] for t in triples}

    X, Y = np.meshgrid(rates0, rates1, indexing="ij")
    Z = np.array(
        [[lookup.get((r0, r1), np.nan) for r1 in rates1] for r0 in rates0],
        dtype=float,
    )
    return X, Y, Z


def visualize(
    experiment_name: str,
    experiments_dir: Path,
    metrics_dir: Path,
    output_dir: Path,
) -> None:
    """Generate and save static matplotlib plots for an experiment.

    Produces up to twelve PNG files in ``output_dir``:

    2-D plots (all sweep types):
      - ``accuracy_vs_rate.png``
      - ``latency_breakdown.png``
      - ``activation_sizes.png``
      - ``accuracy_latency_pareto.png``
      - ``per_class_accuracy.png`` (ResNet/CIFAR-10 only)
      - ``latency_cdf.png``
      - ``compression_overhead.png``

    3-D plots (product sweep with 2 links only):
      - ``surface_accuracy.png``
      - ``surface_accuracy_loss.png``
      - ``scatter3d_bandwidth_accuracy.png``
      - ``scatter3d_pareto.png``
      - ``surface_latency.png``

    Plots are silently skipped when required data is not present.

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
    raw_end_to_end = load_raw_end_to_end(metrics_dir, experiment_name)

    baseline_accuracies: dict[str, float | None] = {}
    baseline_latencies: dict[str, float | None] = {}
    for baseline_name in exp.baselines:
        bl_records = load_records(metrics_dir, baseline_name)
        if bl_records:
            all_bl = [r for recs in bl_records.values() for r in recs]
            baseline_accuracies[baseline_name] = compute_accuracy(all_bl)
        else:
            baseline_accuracies[baseline_name] = None
        bl_metrics = aggregate_metrics(metrics_dir, baseline_name)
        if bl_metrics:
            lats = [
                m["end_to_end_ms"]
                for m in bl_metrics.values()
                if m["end_to_end_ms"] is not None
            ]
            baseline_latencies[baseline_name] = sum(lats) / len(lats) if lats else None
        else:
            baseline_latencies[baseline_name] = None

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

    # ------------------------------------------------------------------ #
    # Plot 4: Accuracy–latency Pareto scatter
    # ------------------------------------------------------------------ #
    pareto_pts: list[tuple[float, float]] = []
    pareto_labels: list[str] = []
    for run_id, records in run_records.items():
        acc = compute_accuracy(records)
        lat = (metrics.get(run_id) or {}).get("end_to_end_ms")
        if acc is None or lat is None:
            continue
        pareto_pts.append((lat, acc * 100))
        pareto_labels.append(run_id)

    if pareto_pts:
        fig, ax = plt.subplots(figsize=(9, 6))
        xs, ys = zip(*pareto_pts, strict=False)
        ax.scatter(xs, ys, zorder=3, label="sweep runs")
        for (x_pt, y_pt), lbl in zip(pareto_pts, pareto_labels, strict=False):
            ax.annotate(
                lbl,
                (x_pt, y_pt),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
            )
        frontier = _pareto_frontier(pareto_pts)
        if len(frontier) > 1:
            fx, fy = zip(*frontier, strict=False)
            ax.plot(fx, fy, "--", color="gray", linewidth=1, label="Pareto frontier")
        for bl_name, bl_acc in baseline_accuracies.items():
            bl_lat = baseline_latencies.get(bl_name)
            if bl_acc is not None and bl_lat is not None:
                ax.scatter(
                    [bl_lat],
                    [bl_acc * 100],
                    marker="*",
                    s=200,
                    zorder=4,
                    label=bl_name,
                )
        ax.set_xlabel("Mean end-to-end latency (ms/batch)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Accuracy–Latency Tradeoff — {experiment_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = output_dir / "accuracy_latency_pareto.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 5: Per-class accuracy heatmap (CIFAR-10 / ResNet only)
    # ------------------------------------------------------------------ #
    if exp.model.lower() == "resnet56" and run_records:
        sorted_run_ids = sorted(run_records.keys())
        heatmap_data: list[list[float]] = []
        for run_id in sorted_run_ids:
            per_class = compute_per_class_accuracy(run_records[run_id], n_classes=10)
            heatmap_data.append([per_class.get(c) or 0.0 for c in range(10)])

        if heatmap_data:
            fig, ax = plt.subplots(figsize=(12, max(3, len(sorted_run_ids) * 0.5 + 1)))
            im = ax.imshow(
                heatmap_data, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn"
            )
            ax.set_xticks(range(10))
            ax.set_xticklabels(_CIFAR10_CLASSES, rotation=45, ha="right")
            ax.set_yticks(range(len(sorted_run_ids)))
            ax.set_yticklabels(sorted_run_ids, fontsize=8)
            fig.colorbar(im, ax=ax, label="Per-class accuracy")
            ax.set_title(f"Per-Class Accuracy — {experiment_name}")
            fig.tight_layout()
            out = output_dir / "per_class_accuracy.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 6: End-to-end latency CDF
    # ------------------------------------------------------------------ #
    if raw_end_to_end:
        fig, ax = plt.subplots(figsize=(9, 5))
        for run_id, latencies in sorted(raw_end_to_end.items()):
            if not latencies:
                continue
            sorted_lats = sorted(latencies)
            n = len(sorted_lats)
            cum_probs = [(i + 1) / n for i in range(n)]
            ax.plot(sorted_lats, cum_probs, label=run_id)
        ax.axhline(0.50, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8)
        ax.text(ax.get_xlim()[0], 0.51, "P50", fontsize=7, color="gray", va="bottom")
        ax.text(ax.get_xlim()[0], 0.96, "P95", fontsize=7, color="gray", va="bottom")
        ax.set_xlabel("End-to-end latency (ms/batch)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(f"Latency CDF — {experiment_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = output_dir / "latency_cdf.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 7: Compression overhead fraction
    # ------------------------------------------------------------------ #
    overhead_entries: list[tuple[str, Any]] = [
        (run_id, m)
        for run_id, m in sorted(metrics.items())
        if m["compress_ms"] is not None
        and m["end_to_end_ms"] is not None
        and m["end_to_end_ms"] > 0
    ]
    if overhead_entries:
        run_ids_oh = [r for r, _ in overhead_entries]
        overhead_pct = [
            (m["compress_ms"] + (m["decompress_ms"] or 0.0)) / m["end_to_end_ms"] * 100
            for _, m in overhead_entries
        ]
        x = list(range(len(run_ids_oh)))
        fig, ax = plt.subplots(figsize=(max(6, len(run_ids_oh) * 0.9), 5))
        ax.bar(x, overhead_pct)
        ax.set_xticks(x)
        ax.set_xticklabels(run_ids_oh, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Compression overhead (% of end-to-end)")
        ax.set_title(f"Compression Overhead Fraction — {experiment_name}")
        fig.tight_layout()
        out = output_dir / "compression_overhead.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ================================================================== #
    # 3-D plots — product sweep with exactly 2 links only
    # ================================================================== #
    is_product_2link = exp.sweep_mode == SweepMode.PRODUCT and any(
        len(r.links) == 2 for r in run_map.values()
    )
    if not is_product_2link:
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

    sample_run = next(r for r in run_map.values() if len(r.links) == 2)
    link0_lbl = f"rate {sample_run.links[0].from_node}→{sample_run.links[0].to_node}"
    link1_lbl = f"rate {sample_run.links[1].from_node}→{sample_run.links[1].to_node}"

    # Pre-compute per-run scalar metrics used by multiple 3-D plots
    acc_by_run: dict[str, float | None] = {
        run_id: compute_accuracy(records) for run_id, records in run_records.items()
    }
    lat_by_run: dict[str, float | None] = {
        run_id: (metrics.get(run_id) or {}).get("end_to_end_ms") for run_id in run_map
    }
    # output_bytes from compress events = mean compressed activation size;
    # used as a bandwidth proxy — lower means more compression applied
    bw_by_run: dict[str, float | None] = {
        run_id: (metrics.get(run_id) or {}).get("output_bytes") for run_id in run_map
    }

    # Global baseline accuracy for the loss surface
    bl_acc_global: float | None = None
    for bl_acc_val in baseline_accuracies.values():
        if bl_acc_val is not None:
            bl_acc_global = bl_acc_val
            break

    # ------------------------------------------------------------------ #
    # Plot 8: Accuracy surface — z = accuracy(rate_AB, rate_BC)
    # ------------------------------------------------------------------ #
    grid = _build_surface_grid(run_map, acc_by_run)
    if grid is not None:
        X, Y, Z = grid
        fig = plt.figure(figsize=(10, 7))
        ax3 = fig.add_subplot(111, projection="3d")
        surf = ax3.plot_surface(X, Y, Z * 100, cmap="RdYlGn", alpha=0.9)
        fig.colorbar(surf, ax=ax3, shrink=0.5, label="Accuracy (%)")
        ax3.set_xlabel(link0_lbl)
        ax3.set_ylabel(link1_lbl)
        ax3.set_zlabel("Accuracy (%)")
        ax3.set_title(f"Accuracy Surface — {experiment_name}")
        ax3.view_init(elev=30, azim=225)
        fig.tight_layout()
        out = output_dir / "surface_accuracy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 9: Accuracy loss surface — z = baseline_accuracy - accuracy
    # ------------------------------------------------------------------ #
    if bl_acc_global is not None:
        loss_by_run: dict[str, float | None] = {
            run_id: (bl_acc_global - acc) if acc is not None else None
            for run_id, acc in acc_by_run.items()
        }
        grid = _build_surface_grid(run_map, loss_by_run)
        if grid is not None:
            X, Y, Z = grid
            fig = plt.figure(figsize=(10, 7))
            ax3 = fig.add_subplot(111, projection="3d")
            surf = ax3.plot_surface(X, Y, Z * 100, cmap="Reds", alpha=0.9)
            fig.colorbar(surf, ax=ax3, shrink=0.5, label="Accuracy loss (pp)")
            ax3.set_xlabel(link0_lbl)
            ax3.set_ylabel(link1_lbl)
            ax3.set_zlabel("Accuracy loss (pp)")
            ax3.set_title(f"Accuracy Loss vs Baseline — {experiment_name}")
            ax3.view_init(elev=30, azim=225)
            fig.tight_layout()
            out = output_dir / "surface_accuracy_loss.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 10: 3-D scatter — (rate_AB, rate_BC, accuracy), colour = bandwidth
    # ------------------------------------------------------------------ #
    scatter_pts: list[tuple[float, float, float, float]] = []
    for run_id, run in run_map.items():
        if len(run.links) < 2:
            continue
        acc = acc_by_run.get(run_id)
        bw = bw_by_run.get(run_id)
        if acc is None or bw is None:
            continue
        scatter_pts.append((run.links[0].rate, run.links[1].rate, acc * 100, bw))

    if scatter_pts:
        r0s, r1s, accs, bws = zip(*scatter_pts, strict=False)
        fig = plt.figure(figsize=(10, 7))
        ax3 = fig.add_subplot(111, projection="3d")
        sc = ax3.scatter(r0s, r1s, accs, c=bws, cmap="plasma_r", s=60, depthshade=True)
        fig.colorbar(sc, ax=ax3, shrink=0.5, label="Compressed activation size (bytes)")
        ax3.set_xlabel(link0_lbl)
        ax3.set_ylabel(link1_lbl)
        ax3.set_zlabel("Accuracy (%)")
        ax3.set_title(f"Accuracy vs Rate (colour = bandwidth) — {experiment_name}")
        ax3.view_init(elev=25, azim=210)
        fig.tight_layout()
        out = output_dir / "scatter3d_bandwidth_accuracy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 11: 3-D Pareto scatter — (latency, accuracy, bandwidth)
    # Pareto-optimal points are highlighted: minimise latency and bandwidth,
    # maximise accuracy.
    # ------------------------------------------------------------------ #
    pareto3_pts: list[tuple[float, float, float]] = []
    pareto3_ids: list[str] = []
    for run_id in run_map:
        lat = lat_by_run.get(run_id)
        acc = acc_by_run.get(run_id)
        bw = bw_by_run.get(run_id)
        if lat is None or acc is None or bw is None:
            continue
        pareto3_pts.append((lat, acc * 100, bw))
        pareto3_ids.append(run_id)

    if pareto3_pts:
        on_frontier = _pareto3d(pareto3_pts)
        lats3, accs3, bws3 = zip(*pareto3_pts, strict=False)
        colors = ["gold" if f else "steelblue" for f in on_frontier]
        sizes = [80 if f else 30 for f in on_frontier]
        fig = plt.figure(figsize=(10, 7))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.scatter(lats3, accs3, bws3, c=colors, s=sizes, depthshade=True)
        # Proxy handles for the legend
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gold",
                markersize=9,
                label="Pareto-optimal",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="steelblue",
                markersize=6,
                label="Dominated",
            ),
        ]
        ax3.legend(handles=legend_handles, fontsize=8)
        ax3.set_xlabel("Latency (ms/batch)")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_zlabel("Compressed size (bytes)")
        ax3.set_title(f"3-D Pareto Frontier — {experiment_name}")
        ax3.view_init(elev=20, azim=200)
        fig.tight_layout()
        out = output_dir / "scatter3d_pareto.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    # ------------------------------------------------------------------ #
    # Plot 12: Latency surface — z = end_to_end_ms(rate_AB, rate_BC)
    # ------------------------------------------------------------------ #
    grid = _build_surface_grid(run_map, lat_by_run)
    if grid is not None:
        X, Y, Z = grid
        fig = plt.figure(figsize=(10, 7))
        ax3 = fig.add_subplot(111, projection="3d")
        surf = ax3.plot_surface(X, Y, Z, cmap="viridis_r", alpha=0.9)
        fig.colorbar(surf, ax=ax3, shrink=0.5, label="Latency (ms/batch)")
        ax3.set_xlabel(link0_lbl)
        ax3.set_ylabel(link1_lbl)
        ax3.set_zlabel("Latency (ms/batch)")
        ax3.set_title(f"Latency Surface — {experiment_name}")
        ax3.view_init(elev=30, azim=225)
        fig.tight_layout()
        out = output_dir / "surface_latency.png"
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
