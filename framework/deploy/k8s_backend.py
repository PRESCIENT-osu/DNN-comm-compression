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


def _derive_image(compute_image: str, role: str) -> str:
    """Derive a role-specific image name from the compute image tag."""
    tag = compute_image.split(":")[-1] if ":" in compute_image else "latest"
    return f"dnn-{role}:{tag}"


def _abs_container_path(path: str) -> str:
    """Return an absolute container path, prepending /app for relative paths."""
    return path if Path(path).is_absolute() else f"/app/{path}"


def generate(
    exp: ExperimentConfig,
    infra: InfraConfig,
    image: str,
    partitions_dir: Path,
    metrics_data_dir: Path,
    experiment_config_path: Path,
    namespace: str = "default",
    dataset_dir: Path | None = None,
) -> str:
    """Generate Kubernetes manifests for the experiment.

    Produces a single YAML document stream containing:
      - A ConfigMap with the experiment config
      - One Pod + Service per pipeline node
      - A metrics server Pod + Service

    Per-link tc rules are applied via HTB qdiscs using TC_LINK_<N>_* environment
    variables read by entrypoint.sh.  Node Services are headless so DNS resolves
    directly to pod IPs, which is required for tc u32 filters to match correctly.
    Partitions are mounted from the host via hostPath volumes (suitable for kind
    local development).

    Args:
        exp: Experiment configuration.
        infra: Infrastructure configuration.
        image: Docker image name to use for all pods.
        partitions_dir: Host path to the ``.partitions`` directory.
        metrics_data_dir: Host path for metrics storage.
        experiment_config_path: Host path to the experiment YAML file.
        namespace: Kubernetes namespace to deploy into.
        dataset_dir: Host path to the dataset directory.

    Returns:
        Multi-document YAML string suitable for ``kubectl apply -f``.
    """
    docs: list[dict[str, Any]] = []
    metrics_image = _derive_image(image, "metrics")
    orchestrator_image = _derive_image(image, "orchestrator")
    dataset_container_path = _abs_container_path(exp.dataset.path)
    experiments_dir = experiment_config_path.parent.parent

    docs.append(_metrics_pod(exp, infra, metrics_image, metrics_data_dir, namespace))
    docs.append(_metrics_service(exp, infra, namespace))

    outgoing_links: dict[str, list[InfraLinkConfig]] = defaultdict(list)
    for link in infra.links:
        if any([link.bandwidth_mbps, link.delay_ms, link.loss_pct]):
            outgoing_links[link.from_node].append(link)

    node_host_map = {n.name: n.host for n in exp.nodes}
    node_module = _compute_node_module(exp.model)

    for node in exp.nodes:
        infra_node = next((n for n in infra.nodes if n.name == node.name), None)
        docs.append(
            _node_pod(
                node_cfg=node,
                infra_node=infra_node,
                exp=exp,
                image=image,
                partitions_dir=partitions_dir,
                experiments_dir=experiments_dir,
                namespace=namespace,
                outgoing_links=outgoing_links.get(node.name, []),
                node_host_map=node_host_map,
                node_module=node_module,
            )
        )
        docs.append(_node_service(node, infra_node, namespace))

    if dataset_dir is not None:
        docs.append(
            _orchestrator_job(
                exp=exp,
                image=orchestrator_image,
                experiment_config_path=experiment_config_path,
                dataset_dir=dataset_dir,
                dataset_container_path=dataset_container_path,
                namespace=namespace,
            )
        )

    return "---\n".join(
        yaml.dump(doc, default_flow_style=False, sort_keys=False) for doc in docs
    )


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------


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
                        "framework.nodes.metrics.server",
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


def _metrics_service(
    exp: ExperimentConfig, infra: InfraConfig, namespace: str
) -> dict[str, Any]:
    port_spec: dict[str, Any] = {
        "port": exp.metrics_server.port,
        "targetPort": exp.metrics_server.port,
    }
    spec: dict[str, Any] = {
        "selector": {"app": "metrics-server"},
        "ports": [port_spec],
    }
    if infra.metrics_node_port is not None:
        spec["type"] = "NodePort"
        port_spec["nodePort"] = infra.metrics_node_port
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": exp.metrics_server.host, "namespace": namespace},
        "spec": spec,
    }


def _node_pod(
    node_cfg: Any,
    infra_node: Any,
    exp: ExperimentConfig,
    image: str,
    partitions_dir: Path,
    experiments_dir: Path,
    namespace: str,
    outgoing_links: list[InfraLinkConfig],
    node_host_map: dict[str, str],
    node_module: str,
) -> dict[str, Any]:
    pod_name = f"node-{node_cfg.name.lower()}"

    env = [
        {"name": "NODE_NAME", "value": node_cfg.name},
        {
            "name": "EXPERIMENT_CONFIG_PATH",
            "value": f"/app/experiments/{exp.name}/experiment.yaml",
        },
        {"name": "PARTITIONS_DIR", "value": "/app/.partitions"},
        {
            "name": "METRICS_SERVER_URL",
            "value": f"http://{exp.metrics_server.host}:{exp.metrics_server.port}",
        },
        {"name": "PORT", "value": str(node_cfg.port)},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]

    for i, link in enumerate(outgoing_links):
        prefix = f"TC_LINK_{i}"
        env.append({"name": f"{prefix}_HOST", "value": node_host_map[link.to_node]})
        if link.bandwidth_mbps is not None:
            env.append(
                {"name": f"{prefix}_MBPS", "value": str(int(link.bandwidth_mbps))}
            )
        if link.delay_ms is not None:
            env.append({"name": f"{prefix}_DELAY_MS", "value": str(link.delay_ms)})
        if link.jitter_ms is not None:
            env.append({"name": f"{prefix}_JITTER_MS", "value": str(link.jitter_ms)})
        if link.loss_pct is not None:
            env.append({"name": f"{prefix}_LOSS_PCT", "value": str(link.loss_pct)})

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

    container: dict[str, Any] = {
        "name": pod_name,
        "image": image,
        "args": ["python", "-m", node_module],
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
                "name": "experiments",
                "mountPath": "/app/experiments",
                "readOnly": True,
            },
        ],
    }

    if outgoing_links:
        container["securityContext"] = {"capabilities": {"add": ["NET_ADMIN"]}}

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": {"app": pod_name, "experiment": exp.name},
        },
        "spec": {
            "hostname": node_cfg.host,
            "containers": [container],
            "volumes": [
                {
                    "name": "partitions",
                    "hostPath": {"path": str(partitions_dir.resolve())},
                },
                {
                    "name": "experiments",
                    "hostPath": {"path": str(experiments_dir.resolve())},
                },
            ],
        },
    }


def _node_service(node_cfg: Any, infra_node: Any, namespace: str) -> dict[str, Any]:
    pod_name = f"node-{node_cfg.name.lower()}"
    port_spec: dict[str, Any] = {
        "port": node_cfg.port,
        "targetPort": node_cfg.port,
    }
    spec: dict[str, Any] = {
        "clusterIP": "None",
        "selector": {"app": pod_name},
        "ports": [port_spec],
    }
    if infra_node is not None and infra_node.node_port is not None:
        spec["type"] = "NodePort"
        port_spec["nodePort"] = infra_node.node_port
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": node_cfg.host, "namespace": namespace},
        "spec": spec,
    }


def _orchestrator_job(
    exp: ExperimentConfig,
    image: str,
    experiment_config_path: Path,
    dataset_dir: Path,
    dataset_container_path: str,
    namespace: str,
) -> dict[str, Any]:
    """Generate a Kubernetes Job manifest for the experiment orchestrator.

    The Job runs the sweep, collects results via its callback server, and
    exits when complete.  CALLBACK_HOST is injected from the pod's own IP
    via the Downward API so node pods can POST results back without a Service.

    Args:
        exp: Experiment configuration.
        image: Docker image name.
        experiment_config_path: Host path to experiment.yaml (grandparent experiments/ dir is mounted).
        dataset_dir: Host path to the dataset directory.
        dataset_container_path: Absolute container path where dataset_dir is mounted.
        namespace: Kubernetes namespace.

    Returns:
        Job manifest dict.
    """
    job_name = f"{exp.name.lower().replace('_', '-')}-orchestrator"
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "orchestrator",
                            "image": image,
                            "args": [
                                "python",
                                "-m",
                                "framework.nodes.orchestrator.runner",
                                f"/app/experiments/{exp.name}",
                            ],
                            "env": [
                                {
                                    "name": "CALLBACK_HOST",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "status.podIP"}
                                    },
                                },
                                {"name": "CALLBACK_PORT", "value": "8080"},
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                            ],
                            "volumeMounts": [
                                {
                                    "name": "experiment-dir",
                                    "mountPath": "/app/experiments",
                                    "readOnly": True,
                                },
                                {
                                    "name": "dataset",
                                    "mountPath": dataset_container_path,
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "experiment-dir",
                            "hostPath": {
                                "path": str(
                                    experiment_config_path.parent.parent.resolve()
                                )
                            },
                        },
                        {
                            "name": "dataset",
                            "hostPath": {"path": str(dataset_dir.resolve())},
                        },
                    ],
                }
            },
        },
    }
