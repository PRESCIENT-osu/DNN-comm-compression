from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from framework.datamodels.experiment import ExperimentConfig
from framework.datamodels.infra import InfraConfig, InfraLinkConfig


def _compute_node_module(model: str) -> str:
    """Return the server module path for the given model name."""
    if model.lower().startswith("llama"):
        return "framework.nodes.compute.llama.server"
    return "framework.nodes.compute.resnet.server"


def generate(
    exp: ExperimentConfig,
    infra: InfraConfig,
    image: str,
    partitions_dir: Path,
    metrics_data_dir: Path,
    experiment_config_path: Path,
    dataset_dir: Path,
) -> str:
    """Generate a docker-compose.yml for the experiment.

    One service is created per pipeline node plus one for the metrics server.
    Per-link tc rules are applied via HTB qdiscs using TC_LINK_<N>_* environment
    variables read by entrypoint.sh.  Nodes with outgoing links that have any
    traffic-shaping parameters receive ``cap_add: [NET_ADMIN]`` to allow tc to run.

    Args:
        exp: Experiment configuration.
        infra: Infrastructure configuration.
        image: Docker image name to use for all services.
        partitions_dir: Host path to the ``.partitions`` directory.
        metrics_data_dir: Host path for metrics storage.
        experiment_config_path: Host path to the experiment YAML file.
        dataset_dir: Host path to the dataset directory.

    Returns:
        YAML string for docker-compose.yml.
    """
    outgoing_links: dict[str, list[InfraLinkConfig]] = defaultdict(list)
    for link in infra.links:
        if any([link.bandwidth_mbps, link.delay_ms, link.loss_pct]):
            outgoing_links[link.from_node].append(link)

    node_host_map = {n.name: n.host for n in exp.nodes}
    services: dict[str, Any] = {}

    # --- Metrics server ---
    services["metrics"] = {
        "image": image,
        "entrypoint": ["/app/entrypoint.sh"],
        "command": [
            "python",
            "-m",
            "framework.nodes.metrics.server",
            "--host",
            "0.0.0.0",
            "--port",
            str(exp.metrics_server.port),
            "--storage-dir",
            "/app/metrics_data",
        ],
        "ports": [f"{exp.metrics_server.port}:{exp.metrics_server.port}"],
        "volumes": [
            f"{metrics_data_dir.resolve()}:/app/metrics_data",
        ],
        "networks": ["pipeline"],
        "restart": "unless-stopped",
    }

    node_module = _compute_node_module(exp.model)

    # --- Node services ---
    for node in exp.nodes:
        infra_node = next((n for n in infra.nodes if n.name == node.name), None)
        env: dict[str, Any] = {
            "NODE_NAME": node.name,
            "EXPERIMENT_CONFIG_PATH": "/app/experiment.yaml",
            "PARTITIONS_DIR": "/app/.partitions",
            "METRICS_SERVER_URL": (
                f"http://{exp.metrics_server.host}:{exp.metrics_server.port}"
            ),
            "PORT": str(node.port),
        }

        for i, link in enumerate(outgoing_links.get(node.name, [])):
            prefix = f"TC_LINK_{i}"
            env[f"{prefix}_HOST"] = node_host_map[link.to_node]
            if link.bandwidth_mbps is not None:
                env[f"{prefix}_MBPS"] = str(int(link.bandwidth_mbps))
            if link.delay_ms is not None:
                env[f"{prefix}_DELAY_MS"] = str(link.delay_ms)
            if link.jitter_ms is not None:
                env[f"{prefix}_JITTER_MS"] = str(link.jitter_ms)
            if link.loss_pct is not None:
                env[f"{prefix}_LOSS_PCT"] = str(link.loss_pct)

        svc: dict[str, Any] = {
            "image": image,
            "hostname": node.host,
            "environment": env,
            "ports": [f"{node.port}:{node.port}"],
            "volumes": [
                f"{partitions_dir.resolve()}:/app/.partitions:ro",
                f"{experiment_config_path.resolve()}:/app/experiment.yaml:ro",
            ],
            "depends_on": ["metrics"],
            "networks": ["pipeline"],
            "restart": "unless-stopped",
        }

        if outgoing_links.get(node.name):
            svc["cap_add"] = ["NET_ADMIN"]

        if infra_node:
            if infra_node.resources.memory is not None:
                svc["mem_limit"] = infra_node.resources.memory
            if infra_node.resources.cpu is not None:
                svc["cpus"] = float(infra_node.resources.cpu)
            if infra_node.resources.gpu > 0:
                svc["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": infra_node.resources.gpu,
                                    "capabilities": ["gpu"],
                                }
                            ]
                        }
                    }
                }

        svc["command"] = ["python", "-m", node_module]
        services[node.host] = svc

    node_service_names = [node.host for node in exp.nodes]
    services["orchestrator"] = {
        "image": image,
        "hostname": "orchestrator",
        "command": [
            "python",
            "-m",
            "framework.nodes.orchestrator.runner",
            "/app/experiment_dir",
            "--callback-host",
            "orchestrator",
            "--callback-port",
            "8080",
        ],
        "environment": {"PYTHONUNBUFFERED": "1"},
        "volumes": [
            f"{experiment_config_path.parent.resolve()}:/app/experiment_dir:ro",
            f"{dataset_dir.resolve()}:{exp.dataset.path}:ro",
        ],
        "depends_on": ["metrics"] + node_service_names,
        "networks": ["pipeline"],
    }

    compose: dict[str, Any] = {
        "version": "3.8",
        "networks": {"pipeline": {"driver": "bridge"}},
        "services": services,
    }

    return yaml.dump(compose, default_flow_style=False, sort_keys=False)
