from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cleanup(
    experiment_name: str,
    experiments_dir: Path,
    older_than_days: int,
    dry_run: bool,
) -> None:
    """Delete result run directories older than a threshold.

    Scans ``experiments/<name>/results/`` for per-run subdirectories and
    removes those whose modification time is older than ``older_than_days``.
    The age check uses the ``records.jsonl`` mtime when present, otherwise
    the directory mtime.

    Args:
        experiment_name: Name matching the experiment directory.
        experiments_dir: Root directory containing all experiment directories.
        older_than_days: Remove run directories older than this many days.
        dry_run: If True, log what would be deleted without deleting anything.
    """
    cutoff = time.time() - older_than_days * 86400
    results_dir = experiments_dir / experiment_name / "results"

    if not results_dir.exists():
        logger.info("No results directory found for '%s'", experiment_name)
        return

    deleted = 0
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ref = run_dir / "records.jsonl"
        mtime = ref.stat().st_mtime if ref.exists() else run_dir.stat().st_mtime
        if mtime >= cutoff:
            continue
        age_days = (time.time() - mtime) / 86400
        if dry_run:
            logger.info("[dry-run] Would delete %s (age %.1f days)", run_dir, age_days)
        else:
            shutil.rmtree(run_dir)
            logger.info("Deleted %s (age %.1f days)", run_dir, age_days)
        deleted += 1

    if deleted == 0:
        logger.info(
            "No result directories older than %d day(s) found for '%s'",
            older_than_days,
            experiment_name,
        )
    elif dry_run:
        logger.info("Dry run: %d director(y/ies) would be deleted", deleted)
    else:
        logger.info("Deleted %d result director(y/ies)", deleted)


def main() -> None:
    """CLI entry point for the cleanup tool."""
    parser = argparse.ArgumentParser(
        description="Remove old result records from an experiment."
    )
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root experiments directory (default: experiments/)",
    )
    parser.add_argument(
        "--older-than",
        type=int,
        default=7,
        dest="older_than_days",
        help="Delete run directories older than this many days (default: 7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without deleting",
    )
    args = parser.parse_args()
    cleanup(args.experiment, args.experiments_dir, args.older_than_days, args.dry_run)


if __name__ == "__main__":
    main()
