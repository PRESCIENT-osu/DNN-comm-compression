from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from framework.config.loader import (
    check_infra_fairness,
    load_experiment_dir,
    load_infra_config,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate(experiment_dir: Path, show_runs: bool) -> bool:
    """Validate an experiment directory and return True if valid.

    Args:
        experiment_dir: Path to the experiment directory.
        show_runs: If True, print the resolved sweep runs.

    Returns:
        True if all validation passed, False otherwise.
    """
    ok = True

    print(f"Validating: {experiment_dir}")

    try:
        exp, infra = load_experiment_dir(experiment_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return False
    except ValidationError as e:
        print(f"  ERROR in config:\n{e}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    print(f"  experiment : {exp.name}")
    print(f"  model      : {exp.model}")
    print(f"  baseline   : {exp.baseline.value if exp.baseline else 'no'}")

    try:
        order = exp.node_order()
        print(f"  node order : {' -> '.join(order)}")
    except ValueError as e:
        print(f"  ERROR in node topology: {e}")
        ok = False

    try:
        runs = exp.resolve_sweep()
        print(f"  sweep runs : {len(runs)}")
        if show_runs:
            for run in runs:
                link_summary = ", ".join(
                    f"{lk.from_node}->{lk.to_node} "
                    f"{lk.compression.value}"
                    + (f"@{lk.rate:.2f}" if lk.compression.value != "none" else "")
                    for lk in run.links
                )
                print(
                    f"    [{run.run_id}] {link_summary if link_summary else 'no links'}"
                )
    except ValueError as e:
        print(f"  ERROR in sweep expansion: {e}")
        ok = False

    if exp.baselines:
        print(f"  baselines  : {', '.join(exp.baselines)}")
        experiments_root = experiment_dir.parent
        for baseline_name in exp.baselines:
            baseline_dir = experiments_root / baseline_name
            baseline_infra_path = baseline_dir / "infra.yaml"
            if not baseline_dir.exists():
                print(f"  WARN: baseline '{baseline_name}' directory not found")
                continue
            if not baseline_infra_path.exists():
                print(f"  WARN: baseline '{baseline_name}' has no infra.yaml")
                continue
            if exp.baseline and exp.baseline.value == "single_node":
                continue
            try:
                baseline_infra = load_infra_config(baseline_infra_path)
                warnings = check_infra_fairness(infra, baseline_infra, baseline_name)
                for w in warnings:
                    print(f"  WARN (fairness): {w}")
            except Exception as e:
                print(f"  WARN: could not load baseline '{baseline_name}' infra: {e}")

    status = "OK" if ok else "FAILED"
    print(f"  result     : {status}")
    return ok


def main() -> None:
    """Entry point for config validation CLI."""
    parser = argparse.ArgumentParser(
        description="Validate an experiment config directory."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to the experiment directory containing experiment.yaml and infra.yaml",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Print each resolved sweep run",
    )
    args = parser.parse_args()

    ok = validate(args.experiment_dir, show_runs=args.show_runs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
