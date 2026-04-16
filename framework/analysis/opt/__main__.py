"""CLI entry point for optimizer experiment analysis.

Two modes (can be combined in one invocation):

1. Sub-experiment performance table (requires --metrics-dir and --exp-dir):

   python -m framework.analysis.opt \\
       --metrics-dir metrics_data/opt_single_resnet_topk_am_linear-3-opt_100mbps \\
       --exp-dir experiments/opt/opt_single_resnet_topk_linear-3-opt_100mbps \\
       --output results/opt_single_resnet_topk_100mbps

   Outputs:
       {output}/tables/sub_exp_table.csv

2. Accuracy model plots (requires --model-dir):

   python -m framework.analysis.opt \\
       --model-dir artifacts/shared/accuracy_models \\
       --output results/opt_single_resnet_topk_100mbps

   Outputs:
       {output}/plots/am_sweep_scatter.png
       {output}/plots/am_sweep_contour_{surrogate}.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run optimizer experiment analysis."""
    parser = argparse.ArgumentParser(
        description="Analyse optimizer experiment metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    table_group = parser.add_argument_group("sub-experiment table")
    table_group.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Metrics directory for one experiment run (contains opt_slot.ndjson etc.).",
    )
    table_group.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help="Generated experiment directory (contains experiment.yaml). "
        "Required with --metrics-dir to supply task weights.",
    )

    am_group = parser.add_argument_group("accuracy model plots")
    am_group.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Shared artifacts directory containing saved accuracy model .pkl files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Root output directory (default: results/).",
    )

    args = parser.parse_args()

    ran_something = False

    # ------------------------------------------------------------------
    # Sub-experiment performance table
    # ------------------------------------------------------------------
    if args.metrics_dir is not None:
        if args.exp_dir is None:
            logger.error("--exp-dir is required when --metrics-dir is provided")
            sys.exit(1)
        if not args.metrics_dir.exists():
            logger.error("metrics-dir not found: %s", args.metrics_dir)
            sys.exit(1)
        if not args.exp_dir.exists():
            logger.error("exp-dir not found: %s", args.exp_dir)
            sys.exit(1)

        from framework.analysis.opt.tables import build_sub_exp_table
        from framework.utils.loader import load_opt_experiment_config

        exp_config = load_opt_experiment_config(args.exp_dir / "experiment.yaml")
        logger.info("Loaded experiment config: %s", exp_config.name)

        sub_exp_df = build_sub_exp_table(args.metrics_dir, exp_config)
        logger.info("  %d sub-experiments", len(sub_exp_df))

        if not sub_exp_df.empty:
            tables_dir = args.output / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            out_path = tables_dir / "sub_exp_table.csv"
            sub_exp_df.to_csv(out_path, index=False)
            logger.info("  Wrote %s", out_path)
            _print_sub_exp_table(sub_exp_df)

        ran_something = True

    # ------------------------------------------------------------------
    # Accuracy model plots
    # ------------------------------------------------------------------
    if args.model_dir is not None:
        if not args.model_dir.exists():
            logger.error("model-dir not found: %s", args.model_dir)
            sys.exit(1)

        from framework.analysis.opt.plots.accuracy_model import (
            plot_accuracy_sweep_surface,
        )
        from framework.analysis.opt.tables import build_accuracy_sweep_table

        sentinel = Path("/nonexistent")
        sweep_df = build_accuracy_sweep_table(sentinel, sentinel, args.model_dir)
        logger.info("Loaded %d sweep samples from %s", len(sweep_df), args.model_dir)

        plots_dir = args.output / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_accuracy_sweep_surface(sweep_df, plots_dir, args.model_dir)
        logger.info("Accuracy model plots written to %s", plots_dir)

        ran_something = True

    if not ran_something:
        parser.print_help()
        sys.exit(1)

    logger.info("Done. Output: %s", args.output)


def _print_sub_exp_table(df) -> None:
    """Print the sub-experiment table to stdout."""
    import pandas as pd  # noqa: PLC0415

    cols = [
        "sub_experiment_name",
        "sub_exp_type",
        "compression_scheme",
        "backend",
        "surrogate_type",
        "estimator_type",
        "mu",
        "baseline_variant",
        "n_slots",
        "avg_utility",
        "avg_achieved_rps",
        "min_achieved_rps",
        "max_achieved_rps",
        "avg_delay_ms",
        "excess_delay_ms",
        "delay_ratio",
    ]
    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()
    for col in (
        "avg_utility",
        "avg_achieved_rps",
        "min_achieved_rps",
        "max_achieved_rps",
        "avg_delay_ms",
        "excess_delay_ms",
        "delay_ratio",
    ):
        if col in display.columns:
            display[col] = display[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"  # noqa: B023
            )

    print("\n" + "=" * 100)
    print("SUB-EXPERIMENT PERFORMANCE TABLE")
    print("=" * 100)
    print(display.to_string(index=False))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
