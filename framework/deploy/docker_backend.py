from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from framework.config.experiment_schema import ExperimentConfig
from framework.config.infra_schema import InfraConfig


def generate(
    exp: ExperimentConfig,
    infra: InfraConfig,
    image: str,
    partitions_dir: Path,
    metrics_data_dir: Path,
    experiment_config_path: Path,
) -> str:
    """Generate a docker-compose.yml for the experiment.

    One service is created per pipeline node plus one for the metrics server.
    Link bandwidth limits from the infra config are applied via ``tc netem``
    using the ``TC_BANDWIDTH_MBPS`` environment variable read by the container
    entrypoint script.  Nodes with outgoing links that have bandwidth limits
    receive ``cap_add: [NET_ADMIN]`` to allow tc to run.

    Args:
        exp: Experiment configuration.
        infra: Infrastructure configuration.
        image: Docker image name to use for all services.
        partitions_dir: Host path to the ``.partitions`` directory.
        metrics_data_dir: Host path for metrics storage.
        experiment_config_path: Host path to the experiment YAML file.

    Returns:
        YAML string for docker-compose.yml.
    """
    outgoing_bw: dict[str, float] = {}
    for link in infra.links:
        if link.bandwidth_mbps is not None:
            outgoing_bw[link.from_node] = link.bandwidth_mbps

    # node_map = {n.name: n for n in exp.nodes}
    services: dict[str, Any] = {}

    # --- Metrics server ---
    services["metrics"] = {
        "image": image,
        "entrypoint": ["/app/entrypoint.sh"],
        "command": [
            "python",
            "-m",
            "framework.metrics_server.server",
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

        bw = outgoing_bw.get(node.name)
        if bw is not None:
            env["TC_BANDWIDTH_MBPS"] = str(int(bw))

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

        if bw is not None:
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

        services[node.host] = svc

    compose: dict[str, Any] = {
        "version": "3.8",
        "networks": {"pipeline": {"driver": "bridge"}},
        "services": services,
    }

    return yaml.dump(compose, default_flow_style=False, sort_keys=False)
