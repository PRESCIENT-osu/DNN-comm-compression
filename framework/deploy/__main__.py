from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from framework.deploy import docker_backend, k8s_backend
from framework.utils.loader import (
    load_experiment_dir,
    load_infra_config,
    load_multi_experiment_config,
    load_opt_experiment_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for the deployment tool.

    Generates Docker Compose or Kubernetes manifests from an experiment
    directory.  Writes manifests to ``experiments/<name>/deploy/`` for
    inspection, and optionally applies them immediately with ``--apply``.
    """
    parser = argparse.ArgumentParser(
        description="Generate deployment manifests for a DNN compression experiment."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to the experiment directory (containing experiment.yaml and infra.yaml)",
    )
    parser.add_argument(
        "--target",
        choices=["docker", "k8s"],
        required=True,
        help="Deployment target",
    )
    parser.add_argument(
        "--image",
        default="dnn-compression:latest",
        help="Container image name / tag (default: dnn-compression:latest)",
    )
    parser.add_argument(
        "--partitions-dir",
        type=Path,
        required=True,
        help=(
            "Single-model: host path to the model .partitions directory. "
            "Multi-model (--multi): host path to the partitions base directory "
            "containing {model_name}/ subdirectories."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("metrics_data"),
        help="Host path for metrics storage (default: metrics_data)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Single-model: host path to the dataset directory. "
            "Multi-model (--multi): host path to the base .datasets directory "
            "containing all dataset subdirectories."
        ),
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace (k8s target only, default: default)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--multi",
        action="store_true",
        help="Generate manifests for a multi-model experiment (experiments/multi/)",
    )
    mode_group.add_argument(
        "--opt",
        action="store_true",
        help="Generate manifests for an optimization experiment (experiments/opt/)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help=(
            "Host path for optimizer artifact storage — profiling results, "
            "accuracy models, estimator state (--opt only, default: artifacts/)"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply manifests immediately after generating them",
    )
    args = parser.parse_args()

    if not args.experiment.exists():
        logger.error("Experiment directory not found: %s", args.experiment)
        sys.exit(1)

    experiment_config_path = args.experiment / "experiment.yaml"
    deploy_dir = args.experiment / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    infra = load_infra_config(args.experiment / "infra.yaml")

    if args.opt:
        exp_opt = load_opt_experiment_config(experiment_config_path)

        if args.dataset_dir is None:
            logger.error("--dataset-dir is required for optimization experiments")
            sys.exit(1)

        if args.target == "docker":
            content = docker_backend.generate_opt(
                exp=exp_opt,
                infra=infra,
                image=args.image,
                partitions_base_dir=args.partitions_dir,
                metrics_data_dir=args.metrics_dir,
                experiment_config_path=experiment_config_path,
                dataset_base_dir=args.dataset_dir,
                artifacts_dir=args.artifacts_dir,
            )
            out_path = deploy_dir / "docker-compose.yml"
            out_path.write_text(content)
            logger.info("Generated: %s", out_path)

            if args.apply:
                logger.info("Running: docker compose up -d")
                subprocess.run(
                    ["docker", "compose", "up", "-d"],
                    cwd=deploy_dir,
                    check=True,
                )

        elif args.target == "k8s":
            content = k8s_backend.generate_opt(
                exp=exp_opt,
                infra=infra,
                image=args.image,
                partitions_base_dir=args.partitions_dir,
                metrics_data_dir=args.metrics_dir,
                experiment_config_path=experiment_config_path,
                namespace=args.namespace,
                dataset_base_dir=args.dataset_dir,
                artifacts_dir=args.artifacts_dir,
            )
            out_path = deploy_dir / "manifests.yaml"
            out_path.write_text(content)
            logger.info("Generated: %s", out_path)

            if args.apply:
                logger.info(
                    "Running: kubectl apply -f manifests.yaml -n %s", args.namespace
                )
                subprocess.run(
                    ["kubectl", "apply", "-f", "manifests.yaml", "-n", args.namespace],
                    cwd=deploy_dir,
                    check=True,
                )

    elif args.multi:
        exp = load_multi_experiment_config(experiment_config_path)

        if args.dataset_dir is None:
            logger.error("--dataset-dir is required for multi-model experiments")
            sys.exit(1)

        if args.target == "docker":
            content = docker_backend.generate_multi(
                exp=exp,
                infra=infra,
                image=args.image,
                partitions_base_dir=args.partitions_dir,
                metrics_data_dir=args.metrics_dir,
                experiment_config_path=experiment_config_path,
                dataset_base_dir=args.dataset_dir,
            )
            out_path = deploy_dir / "docker-compose.yml"
            out_path.write_text(content)
            logger.info("Generated: %s", out_path)

            if args.apply:
                logger.info("Running: docker compose up -d")
                subprocess.run(
                    ["docker", "compose", "up", "-d"],
                    cwd=deploy_dir,
                    check=True,
                )

        elif args.target == "k8s":
            content = k8s_backend.generate_multi(
                exp=exp,
                infra=infra,
                image=args.image,
                partitions_base_dir=args.partitions_dir,
                metrics_data_dir=args.metrics_dir,
                experiment_config_path=experiment_config_path,
                namespace=args.namespace,
                dataset_base_dir=args.dataset_dir,
            )
            out_path = deploy_dir / "manifests.yaml"
            out_path.write_text(content)
            logger.info("Generated: %s", out_path)

            if args.apply:
                logger.info(
                    "Running: kubectl apply -f manifests.yaml -n %s", args.namespace
                )
                subprocess.run(
                    ["kubectl", "apply", "-f", "manifests.yaml", "-n", args.namespace],
                    cwd=deploy_dir,
                    check=True,
                )

    else:
        exp_single, _infra = load_experiment_dir(args.experiment)

        if args.target == "docker":
            content = docker_backend.generate(
                exp=exp_single,
                infra=infra,
                image=args.image,
                partitions_dir=args.partitions_dir,
                metrics_data_dir=args.metrics_dir,
                experiment_config_path=experiment_config_path,
                dataset_dir=args.dataset_dir,
            )
            out_path = deploy_dir / "docker-compose.yml"
            out_path.write_text(content)
            logger.info("Generated: %s", out_path)

            if args.apply:
                logger.info("Running: docker compose up -d")
                subprocess.run(
                    ["docker", "compose", "up", "-d"],
                    cwd=deploy_dir,
                    check=True,
                )

        elif args.target == "k8s":
            content = k8s_backend.generate(
                exp=exp_single,
                infra=infra,
                image=args.image,
                partitions_dir=args.partitions_dir,
                metrics_data_dir=args.metrics_dir,
                experiment_config_path=experiment_config_path,
                namespace=args.namespace,
                dataset_dir=args.dataset_dir,
            )
            out_path = deploy_dir / "manifests.yaml"
            out_path.write_text(content)
            logger.info("Generated: %s", out_path)

            if args.apply:
                logger.info(
                    "Running: kubectl apply -f manifests.yaml -n %s", args.namespace
                )
                subprocess.run(
                    ["kubectl", "apply", "-f", "manifests.yaml", "-n", args.namespace],
                    cwd=deploy_dir,
                    check=True,
                )


if __name__ == "__main__":
    main()
