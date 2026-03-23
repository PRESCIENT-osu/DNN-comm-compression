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
    dataset_dir: Path,
    metrics_data_dir: Path,
    experiment_config_path: Path,
    namespace: str = "default",
) -> str:
    """Generate Kubernetes manifests for the experiment.

    Produces a single YAML document stream containing:
      - A ConfigMap with the experiment config
      - One Pod + Service per pipeline node
      - A metrics server Pod + Service

    Bandwidth limits from the infra config are applied via Cilium
    ``kubernetes.io/egress-bandwidth`` pod annotations.  Partitions and the
    dataset are mounted from the host via hostPath volumes (suitable for
    KinD local development).

    Args:
        exp: Experiment configuration.
        infra: Infrastructure configuration.
        image: Docker image name to use for all pods.
        partitions_dir: Host path to the ``.partitions`` directory.
        dataset_dir: Host path to the dataset directory.
        metrics_data_dir: Host path for metrics storage.
        experiment_config_path: Host path to the experiment YAML file.
        namespace: Kubernetes namespace to deploy into.

    Returns:
        Multi-document YAML string suitable for ``kubectl apply -f``.
    """
    docs: list[dict[str, Any]] = []

    docs.append(_configmap(exp, experiment_config_path, namespace))
    docs.append(_metrics_pod(exp, infra, image, metrics_data_dir, namespace))
    docs.append(_metrics_service(exp, namespace))

    outgoing_bw: dict[str, float] = {}
    for link in infra.links:
        if link.bandwidth_mbps is not None:
            outgoing_bw[link.from_node] = link.bandwidth_mbps

    for node in exp.nodes:
        infra_node = next((n for n in infra.nodes if n.name == node.name), None)
        bw = outgoing_bw.get(node.name)
        docs.append(
            _node_pod(
                node_cfg=node,
                infra_node=infra_node,
                exp=exp,
                image=image,
                partitions_dir=partitions_dir,
                dataset_dir=dataset_dir,
                namespace=namespace,
                egress_bandwidth_mbps=bw,
            )
        )
        docs.append(_node_service(node, namespace))

    return "---\n".join(
        yaml.dump(doc, default_flow_style=False, sort_keys=False) for doc in docs
    )


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------


def _configmap(
    exp: ExperimentConfig,
    experiment_config_path: Path,
    namespace: str,
) -> dict[str, Any]:
    with open(experiment_config_path) as f:
        config_content = f.read()
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": f"{exp.name}-config", "namespace": namespace},
        "data": {"experiment.yaml": config_content},
    }


def _metrics_pod(
    exp: ExperimentConfig,
    infra: InfraConfig,
    image: str,
    metrics_data_dir: Path,
    namespace: str,
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "metrics-server",
            "namespace": namespace,
            "labels": {"app": "metrics-server"},
        },
        "spec": {
            "containers": [
                {
                    "name": "metrics-server",
                    "image": image,
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
                    "ports": [{"containerPort": exp.metrics_server.port}],
                    "env": [{"name": "PYTHONUNBUFFERED", "value": "1"}],
                    "volumeMounts": [
                        {
                            "name": "metrics-data",
                            "mountPath": "/app/metrics_data",
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "metrics-data",
                    "hostPath": {"path": str(metrics_data_dir.resolve())},
                }
            ],
        },
    }


def _metrics_service(exp: ExperimentConfig, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": exp.metrics_server.host, "namespace": namespace},
        "spec": {
            "selector": {"app": "metrics-server"},
            "ports": [
                {
                    "port": exp.metrics_server.port,
                    "targetPort": exp.metrics_server.port,
                }
            ],
        },
    }


def _node_pod(
    node_cfg: Any,
    infra_node: Any,
    exp: ExperimentConfig,
    image: str,
    partitions_dir: Path,
    dataset_dir: Path,
    namespace: str,
    egress_bandwidth_mbps: float | None,
) -> dict[str, Any]:
    pod_name = f"node-{node_cfg.name.lower()}"
    annotations: dict[str, str] = {}
    if egress_bandwidth_mbps is not None:
        annotations["kubernetes.io/egress-bandwidth"] = f"{int(egress_bandwidth_mbps)}M"

    env = [
        {"name": "NODE_NAME", "value": node_cfg.name},
        {"name": "EXPERIMENT_CONFIG_PATH", "value": "/app/config/experiment.yaml"},
        {"name": "PARTITIONS_DIR", "value": "/app/.partitions"},
        {
            "name": "METRICS_SERVER_URL",
            "value": f"http://{exp.metrics_server.host}:{exp.metrics_server.port}",
        },
        {"name": "PORT", "value": str(node_cfg.port)},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]

    resources: dict[str, Any] = {}
    if infra_node:
        r = infra_node.resources
        req: dict[str, str] = {}
        lim: dict[str, str] = {}
        if r.cpu is not None:
            req["cpu"] = str(r.cpu)
            lim["cpu"] = str(r.cpu)
        if r.memory is not None:
            req["memory"] = r.memory
            lim["memory"] = r.memory
        if r.gpu > 0:
            lim["nvidia.com/gpu"] = str(r.gpu)
            req["nvidia.com/gpu"] = str(r.gpu)
        if req or lim:
            resources = {"requests": req, "limits": lim}

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": {"app": pod_name, "experiment": exp.name},
            "annotations": annotations,
        },
        "spec": {
            "hostname": node_cfg.host,
            "containers": [
                {
                    "name": pod_name,
                    "image": image,
                    "command": ["python", "-m", "framework.node.server"],
                    "ports": [{"containerPort": node_cfg.port}],
                    "env": env,
                    "resources": resources,
                    "volumeMounts": [
                        {
                            "name": "partitions",
                            "mountPath": "/app/.partitions",
                            "readOnly": True,
                        },
                        {
                            "name": "dataset",
                            "mountPath": "/app/data",
                            "readOnly": True,
                        },
                        {
                            "name": "experiment-config",
                            "mountPath": "/app/config",
                            "readOnly": True,
                        },
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "partitions",
                    "hostPath": {"path": str(partitions_dir.resolve())},
                },
                {
                    "name": "dataset",
                    "hostPath": {"path": str(dataset_dir.resolve())},
                },
                {
                    "name": "experiment-config",
                    "configMap": {"name": f"{exp.name}-config"},
                },
            ],
        },
    }


def _node_service(node_cfg: Any, namespace: str) -> dict[str, Any]:
    pod_name = f"node-{node_cfg.name.lower()}"
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": node_cfg.host, "namespace": namespace},
        "spec": {
            "selector": {"app": pod_name},
            "ports": [{"port": node_cfg.port, "targetPort": node_cfg.port}],
        },
    }
