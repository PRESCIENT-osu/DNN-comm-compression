from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from framework.datamodels.experiment import ExperimentConfig
from framework.nodes.orchestrator.runner import _sub_experiment_to_exp_config
from framework.utils.loader import (
    check_infra_fairness,
    is_generated_experiment,
    is_multi_experiment,
    load_experiment_dir,
    load_generated_experiment_config,
    load_infra_config,
    load_multi_experiment_config,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _show_sweep(exp: ExperimentConfig, show_runs: bool) -> bool:
    """Resolve and print sweep runs for one ExperimentConfig. Returns False on error."""
    try:
        runs = exp.resolve_sweep()
        print(f"    sweep runs : {len(runs)}")
        if show_runs:
            for run in runs:
                link_summary = ", ".join(
                    f"{lk.from_node}->{lk.to_node} "
                    f"{lk.compression.value}"
                    + (f"@{lk.rate:.2f}" if lk.compression.value != "none" else "")
                    for lk in run.links
                )
                print(
                    f"      [{run.run_id}] {link_summary if link_summary else 'no links'}"
                )
        return True
    except ValueError as e:
        print(f"    ERROR in sweep expansion: {e}")
        return False


def _check_baselines(
    baselines: list[str],
    infra_path: Path,
    experiments_root: Path,
) -> None:
    """Warn on missing baseline dirs and infra fairness differences."""
    for ref in baselines:
        exp_name = ref.split("/")[0]
        baseline_dir = experiments_root / exp_name
        baseline_infra_path = baseline_dir / "infra.yaml"
        if not baseline_dir.exists():
            print(f"    WARN: baseline '{ref}' directory not found")
            continue
        if not baseline_infra_path.exists():
            print(f"    WARN: baseline '{ref}' has no infra.yaml")
            continue
        try:
            infra = load_infra_config(infra_path)
            baseline_infra = load_infra_config(baseline_infra_path)
            warnings = check_infra_fairness(infra, baseline_infra, ref)
            for w in warnings:
                print(f"    WARN (fairness): {w}")
        except Exception as e:
            print(f"    WARN: could not load baseline '{ref}' infra: {e}")


def validate(experiment_dir: Path, show_runs: bool) -> bool:
    """Validate an experiment directory and return True if valid."""
    ok = True
    exp_yaml = experiment_dir / "experiment.yaml"
    infra_yaml = experiment_dir / "infra.yaml"

    print(f"Validating: {experiment_dir}")

    if is_multi_experiment(exp_yaml):
        try:
            multi = load_multi_experiment_config(exp_yaml)
        except (FileNotFoundError, ValidationError, Exception) as e:
            print(f"  ERROR: {e}")
            return False

        print(f"  experiment : {multi.name}")
        print(f"  pipelines  : {', '.join(p.name for p in multi.pipelines)}")
        print(f"  nodes      : {' -> '.join(n.name for n in multi.nodes)}")

        for sub_exp in multi.sub_experiments:
            print(f"  [{sub_exp.name}]")
            try:
                runs = multi.resolve_sweep(sub_exp)
                print(f"    sweep runs : {len(runs)}")
                if show_runs:
                    for run in runs:
                        link_summary = ", ".join(
                            f"{lk.pipeline_id} {lk.from_node}->{lk.to_node} "
                            f"{lk.compression.value}"
                            + (
                                f"@{lk.rate:.2f}"
                                if lk.compression.value != "none"
                                else ""
                            )
                            for lk in run.links
                        )
                        print(
                            f"      [{run.run_id[:60]}...] {link_summary if link_summary else 'no links'}"
                        )
            except ValueError as e:
                print(f"    ERROR in sweep expansion: {e}")
                ok = False

    elif is_generated_experiment(exp_yaml):
        try:
            generated = load_generated_experiment_config(exp_yaml)
        except (FileNotFoundError, ValidationError, Exception) as e:
            print(f"  ERROR: {e}")
            return False

        print(f"  experiment : {generated.name}")
        print(f"  model      : {generated.model}")

        try:
            # node_order is the same for all sub-experiments — derive from first
            first_exp = _sub_experiment_to_exp_config(
                generated, generated.sub_experiments[0]
            )
            order = first_exp.node_order()
            print(f"  node order : {' -> '.join(order)}")
        except (ValueError, IndexError) as e:
            print(f"  ERROR in node topology: {e}")
            ok = False

        for sub_exp in generated.sub_experiments:
            print(f"  [{sub_exp.name}]")
            exp = _sub_experiment_to_exp_config(generated, sub_exp)
            if not _show_sweep(exp, show_runs):
                ok = False
            if sub_exp.baselines:
                print(f"    baselines  : {', '.join(sub_exp.baselines)}")
                _check_baselines(sub_exp.baselines, infra_yaml, experiment_dir.parent)

    else:
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

        if not _show_sweep(exp, show_runs):
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
                    warnings = check_infra_fairness(
                        infra, baseline_infra, baseline_name
                    )
                    for w in warnings:
                        print(f"  WARN (fairness): {w}")
                except Exception as e:
                    print(
                        f"  WARN: could not load baseline '{baseline_name}' infra: {e}"
                    )

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
