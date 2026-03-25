from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from framework.datamodels.experiment import ExperimentConfig
from framework.datamodels.infra import InfraConfig

logger = logging.getLogger(__name__)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to the experiment YAML file.

    Returns:
        Validated ExperimentConfig instance.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    raw = data.get("experiment", data)
    return ExperimentConfig.model_validate(raw)


def load_infra_config(path: Path) -> InfraConfig:
    """Load an infra config from a YAML file, resolving inheritance.

    If the config contains an ``inherits`` key, the referenced base config
    is loaded recursively and the child config is merged on top.  The
    ``inherits`` path is resolved relative to the directory containing the
    child config file.

    Args:
        path: Path to the infra YAML file.

    Returns:
        Fully resolved InfraConfig instance with no inherits key.
    """
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    inherits_path = data.pop("inherits", None)
    if inherits_path:
        base_path = Path(inherits_path)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = load_infra_config(base_path)
        data = _merge_infra(base, data)

    return InfraConfig.model_validate(data)


def load_experiment_dir(experiment_dir: Path) -> tuple[ExperimentConfig, InfraConfig]:
    """Load both configs from an experiment directory.

    Args:
        experiment_dir: Directory containing experiment.yaml and infra.yaml.

    Returns:
        Tuple of (ExperimentConfig, InfraConfig).

    Raises:
        FileNotFoundError: If either config file is missing.
    """
    exp_path = experiment_dir / "experiment.yaml"
    infra_path = experiment_dir / "infra.yaml"
    if not exp_path.exists():
        raise FileNotFoundError(f"experiment.yaml not found in {experiment_dir}")
    if not infra_path.exists():
        raise FileNotFoundError(f"infra.yaml not found in {experiment_dir}")
    return load_experiment_config(exp_path), load_infra_config(infra_path)


def check_infra_fairness(
    sweep_infra: InfraConfig,
    baseline_infra: InfraConfig,
    baseline_name: str,
) -> list[str]:
    """Compare sweep and baseline infra configs for fairness.

    Returns a list of warning messages for any properties that differ
    between the two configs on shared nodes and links.  An empty list
    means the configs are equivalent on all comparable properties.

    The single-node baseline is expected to differ in node count and is
    not subject to fairness checks by the caller.

    Args:
        sweep_infra: Infra config for the sweep experiment.
        baseline_infra: Infra config for the baseline experiment.
        baseline_name: Human-readable baseline name for warning messages.

    Returns:
        List of warning strings describing differences.
    """
    warnings: list[str] = []

    sweep_nodes = {n.name: n for n in sweep_infra.nodes}
    baseline_nodes = {n.name: n for n in baseline_infra.nodes}

    for name, node in sweep_nodes.items():
        if name not in baseline_nodes:
            continue
        base_node = baseline_nodes[name]
        if node.resources != base_node.resources:
            warnings.append(
                f"Node '{name}' resources differ between sweep and baseline "
                f"'{baseline_name}': sweep={node.resources.model_dump()}, "
                f"baseline={base_node.resources.model_dump()}"
            )

    sweep_links = {(lk.from_node, lk.to_node): lk for lk in sweep_infra.links}
    baseline_links = {(lk.from_node, lk.to_node): lk for lk in baseline_infra.links}

    for pair, link in sweep_links.items():
        if pair not in baseline_links:
            continue
        base_link = baseline_links[pair]
        if link.bandwidth_mbps != base_link.bandwidth_mbps:
            warnings.append(
                f"Link {pair[0]}→{pair[1]} bandwidth differs between sweep and "
                f"baseline '{baseline_name}': "
                f"sweep={link.bandwidth_mbps} Mbps, "
                f"baseline={base_link.bandwidth_mbps} Mbps"
            )

    return warnings


def _merge_infra(base: InfraConfig, override: dict[str, Any]) -> dict[str, Any]:
    """Merge a child infra dict on top of a resolved base InfraConfig.

    Nodes and links are merged by name and from/to pair respectively.
    Child entries take precedence over base entries.

    Args:
        base: The resolved base InfraConfig.
        override: Raw dict from the child infra YAML (without inherits key).

    Returns:
        Merged dict suitable for InfraConfig.model_validate.
    """
    merged_nodes: dict[str, Any] = {n.name: n.model_dump() for n in base.nodes}
    for node in override.get("nodes", []):
        merged_nodes[node["name"]] = node

    merged_links: dict[tuple[str, str], Any] = {
        (lk.from_node, lk.to_node): lk.model_dump(by_alias=True) for lk in base.links
    }
    for link in override.get("links", []):
        from_key = link.get("from") or link.get("from_node")
        to_key = link.get("to") or link.get("to_node")
        merged_links[(from_key, to_key)] = link

    return {
        "nodes": list(merged_nodes.values()),
        "links": list(merged_links.values()),
    }
