from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from framework.analysis._common import (
    aggregate_metrics,
    compute_accuracy,
    load_records,
)
from framework.utils.loader import load_experiment_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def analyze(
    experiment_name: str,
    experiments_dir: Path,
    metrics_dir: Path,
) -> None:
    """Aggregate results and metrics for an experiment and print comparison tables.

    Loads inference records from ``metrics_dir/<name>/result.ndjson`` and timing
    metrics from ``metrics_dir/<name>/``, then prints three tables:

    - **Accuracy**: per-run accuracy with baseline reference columns.
    - **Latency breakdown**: mean forward/compress/decompress/send times per run.
    - **Activation sizes**: mean input/output bytes and compression ratio per run.

    Args:
        experiment_name: Name matching the experiment directory.
        experiments_dir: Root directory containing all experiment directories.
        metrics_dir: Root directory for metrics NDJSON files.
    """
    experiment_dir = experiments_dir / experiment_name
    if not experiment_dir.exists():
        logger.error("Experiment directory not found: %s", experiment_dir)
        sys.exit(1)

    exp = load_experiment_config(experiment_dir / "experiment.yaml")
    run_records = load_records(metrics_dir, experiment_name)
    metrics = aggregate_metrics(metrics_dir, experiment_name)

    if not run_records and not metrics:
        logger.warning(
            "No results or metrics found for experiment '%s'. "
            "Run the experiment first.",
            experiment_name,
        )

    baseline_accuracies: dict[str, float | None] = {}
    baseline_per_run: dict[str, dict[str, float | None]] = {}
    for baseline_name in dict.fromkeys(exp.baselines):  # deduplicate, preserve order
        bl_records = load_records(metrics_dir, baseline_name)
        if bl_records:
            all_records = [r for recs in bl_records.values() for r in recs]
            baseline_accuracies[baseline_name] = compute_accuracy(all_records)
            baseline_per_run[baseline_name] = {
                rid: compute_accuracy(recs) for rid, recs in bl_records.items()
            }
        else:
            baseline_accuracies[baseline_name] = None
            baseline_per_run[baseline_name] = {}

    runs = exp.resolve_sweep()
    run_map = {r.run_id: r for r in runs}

    def _sort_key(run_id: str) -> tuple[float, ...]:
        run = run_map.get(run_id)
        if run and run.links:
            return tuple(lk.rate for lk in run.links)
        return (0.0,)

    def _compression_rate(run_id: str) -> tuple[str, str]:
        run = run_map.get(run_id)
        if run and run.links:
            methods = "+".join(lk.compression.value for lk in run.links)
            rates = "+".join(f"{lk.rate:.2f}" for lk in run.links)
        else:
            methods = "none"
            rates = "-"
        return methods, rates

    # ------------------------------------------------------------------ #
    # Accuracy table
    # ------------------------------------------------------------------ #
    acc_headers = ["compression", "rate", "accuracy"]
    acc_headers += list(baseline_accuracies.keys())
    acc_rows = []
    for run_id, records in sorted(run_records.items(), key=lambda kv: _sort_key(kv[0])):
        accuracy = compute_accuracy(records)
        methods, rates = _compression_rate(run_id)
        row = [
            methods,
            rates,
            f"{accuracy * 100:.2f}%" if accuracy is not None else "-",
        ]
        for bl_name, bl_acc in baseline_accuracies.items():
            per_run = baseline_per_run.get(bl_name, {})
            cell = per_run.get(run_id, bl_acc)
            row.append(f"{cell * 100:.2f}%" if cell is not None else "N/A")
        acc_rows.append(row)

    if acc_rows:
        _print_table(acc_headers, acc_rows, f"Accuracy — {experiment_name}")

    # ------------------------------------------------------------------ #
    # Latency breakdown table
    # ------------------------------------------------------------------ #
    lat_headers = [
        "compression",
        "rate",
        "forward_ms",
        "compress_ms",
        "decompress_ms",
        "send_ms",
        "end_to_end_ms",
    ]
    lat_rows = [
        [
            *_compression_rate(run_id),
            _fmt(m["forward_ms"]),
            _fmt(m["compress_ms"]),
            _fmt(m["decompress_ms"]),
            _fmt(m["send_ms"]),
            _fmt(m["end_to_end_ms"]),
        ]
        for run_id, m in sorted(metrics.items(), key=lambda kv: _sort_key(kv[0]))
    ]
    if lat_rows:
        _print_table(
            lat_headers,
            lat_rows,
            f"Latency Breakdown (mean ms/batch) — {experiment_name}",
        )

    # ------------------------------------------------------------------ #
    # Activation size table
    # ------------------------------------------------------------------ #
    size_headers = [
        "compression",
        "rate",
        "input_bytes",
        "output_bytes",
        "compression_ratio",
    ]
    size_rows = []
    for run_id, m in sorted(metrics.items(), key=lambda kv: _sort_key(kv[0])):
        if m["input_bytes"] is None:
            continue
        in_b = m["input_bytes"]
        out_b = m["output_bytes"]
        ratio = f"{in_b / out_b:.2f}x" if out_b is not None and out_b > 0 else "-"
        size_rows.append(
            [
                *_compression_rate(run_id),
                f"{in_b:.0f}",
                f"{out_b:.0f}" if out_b is not None else "-",
                ratio,
            ]
        )
    if size_rows:
        _print_table(size_headers, size_rows, f"Activation Sizes — {experiment_name}")


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _fmt(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "-"


def _print_table(headers: list[str], rows: list[list[str]], title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    col_widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    """CLI entry point for the analysis tool."""
    parser = argparse.ArgumentParser(
        description="Aggregate metrics and print comparison tables for an experiment."
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
    args = parser.parse_args()
    analyze(args.experiment, args.experiments_dir, args.metrics_dir)


if __name__ == "__main__":
    main()
