"""Experiment generation tool.

Composes a spec + profile into a fully resolved experiment directory ready
for deployment.  All sub-experiments defined in the spec are included.
Referenced baseline experiments are materialised recursively.

Usage::

    # Single experiment
    python tools/generate.py \\
        --spec specs/resnet56/equal-split \\
        --profile profiles/linear-3/100mbps.yaml

    # Single experiment, specific sub-experiments only
    python tools/generate.py \\
        --spec specs/resnet56/equal-split \\
        --profile profiles/linear-3/100mbps.yaml \\
        --sub-experiments baseline topk_paired

    # All compatible spec/profile combinations
    python tools/generate.py --all

    # All combinations with validation and run listing
    python tools/generate.py --all --validate --show-runs

    # Single multi-model experiment
    python tools/generate.py --multi \\
        --spec multispecs/resnet56_llama \\
        --profile profiles/linear-3-bidir/100mbps.yaml

    # All compatible multi-model spec/profile combinations
    python tools/generate.py --multi --all

Output is written to experiments/<name>/ (single-model) or
experiments/multi/<name>/ (multi-model).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPECS_DIR = Path("specs")
MULTISPECS_DIR = Path("multispecs")
OPTSPECS_DIR = Path("optspecs")
PROFILES_DIR = Path("profiles")
EXPERIMENTS_DIR = Path("experiments")
MULTI_EXPERIMENTS_DIR = Path("experiments") / "multi"
OPT_EXPERIMENTS_DIR = Path("experiments") / "opt"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base; override takes precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def load_spec(spec_dir: Path) -> dict[str, Any]:
    """Walk the spec hierarchy root→leaf and deep-merge experiment.yaml layers.

    Args:
        spec_dir: Path to the spec directory (e.g. specs/resnet56/equal-split).

    Returns:
        Merged experiment config dict.
    """
    parts = spec_dir.relative_to(SPECS_DIR).parts
    merged: dict[str, Any] = {}
    current = SPECS_DIR
    for part in parts:
        current = current / part
        yaml_path = current / "experiment.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Spec file not found: {yaml_path}")
        merged = _deep_merge(merged, _load_yaml(yaml_path))
    return merged


def load_sub_experiments(spec_dir: Path) -> dict[str, Any]:
    """Walk the spec hierarchy and resolve sub_experiments.yaml layers.

    If a level has its own sub_experiments.yaml it completely replaces
    everything inherited from parent levels.  Levels without a
    sub_experiments.yaml pass through the parent's entries unchanged.

    Args:
        spec_dir: Path to the spec directory (e.g. specs/resnet56/equal-split).

    Returns:
        Resolved dict of sub-experiment name → sub-experiment config dict.
    """
    parts = spec_dir.relative_to(SPECS_DIR).parts
    merged: dict[str, Any] = {}
    current = SPECS_DIR
    for part in parts:
        current = current / part
        yaml_path = current / "sub_experiments.yaml"
        if yaml_path.exists():
            data = _load_yaml(yaml_path)
            merged = data.get("sub_experiments", {})
    return merged


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


def load_profile(profile_file: Path) -> dict[str, Any]:
    """Load a profile YAML file.

    Args:
        profile_file: Direct path to the profile YAML file
            (e.g. profiles/linear-3/100mbps.yaml).

    Returns:
        Profile dict (InfraConfig-compatible).
    """
    if not profile_file.exists():
        raise FileNotFoundError(f"Profile not found: {profile_file}")
    return _load_yaml(profile_file)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_compatibility(spec: dict[str, Any], profile: dict[str, Any]) -> None:
    """Ensure node names in spec and profile match exactly.

    Args:
        spec: Merged spec dict (must contain a nodes list).
        profile: Profile dict.

    Raises:
        ValueError: If node name sets differ.
    """
    spec_nodes = {n["name"] for n in spec.get("nodes", [])}
    profile_nodes = {n["name"] for n in profile.get("nodes", [])}
    if not profile_nodes:
        return  # profile specifies no nodes — no constraint to check
    if spec_nodes != profile_nodes:
        raise ValueError(
            f"Node name mismatch between spec and profile.\n"
            f"  Spec nodes:    {sorted(spec_nodes)}\n"
            f"  Profile nodes: {sorted(profile_nodes)}"
        )


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def derive_name(spec_dir: Path, profile_file: Path) -> str:
    """Derive the canonical experiment name from spec and profile paths.

    Format: {model}_{partition-split}_{topology}_{profile-name}

    Args:
        spec_dir: Path to the spec directory (e.g. specs/resnet56/equal-split).
        profile_file: Path to the profile YAML file (e.g. profiles/linear-3/100mbps.yaml).

    Returns:
        Experiment name string.
    """
    spec_parts = spec_dir.relative_to(SPECS_DIR).parts
    model = spec_parts[0]
    split = spec_parts[1] if len(spec_parts) > 1 else "default"

    profile_parts = profile_file.relative_to(PROFILES_DIR).parts
    topology = profile_parts[0]
    profile_name = profile_file.stem

    return f"{model}_{split}_{topology}_{profile_name}"


# ---------------------------------------------------------------------------
# Baseline resolution
# ---------------------------------------------------------------------------


def resolve_baseline(
    ref: dict[str, Any],
    current_spec: Path,
    current_profile: Path,
    output_dir: Path,
    _seen: set[str],
) -> str:
    """Resolve a baseline reference to an 'experiment_name/sub_experiment' string.

    Materialises the referenced experiment if it does not already exist.
    Baseline refs in sub_experiments.yaml use short relative paths
    (e.g. spec: resnet56/single-node, profile: single-node/default) which are
    resolved against SPECS_DIR and PROFILES_DIR respectively.

    Args:
        ref: Baseline reference dict with sub_experiment and optional spec/profile.
        current_spec: Spec directory of the experiment being generated.
        current_profile: Profile file of the experiment being generated.
        output_dir: Root experiments directory.
        _seen: Set of already-materialising experiment names (cycle guard).

    Returns:
        String of the form 'experiment_name/sub_experiment_name'.
    """
    sub_exp_name = ref["sub_experiment"]
    ref_spec = SPECS_DIR / ref["spec"] if "spec" in ref else current_spec
    ref_profile = (
        PROFILES_DIR / f"{ref['profile']}.yaml" if "profile" in ref else current_profile
    )

    baseline_exp_name = derive_name(ref_spec, ref_profile)
    baseline_dir = output_dir / baseline_exp_name

    if baseline_exp_name not in _seen and not baseline_dir.exists():
        logger.info("Materialising baseline: %s", baseline_exp_name)
        _materialise(ref_spec, ref_profile, output_dir, _seen=_seen)

    return f"{baseline_exp_name}/{sub_exp_name}"


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


def _materialise(
    spec_dir: Path,
    profile_file: Path,
    output_dir: Path,
    sub_experiment_filter: list[str] | None = None,
    _seen: set[str] | None = None,
) -> Path:
    """Core generation logic — resolves and writes one experiment directory.

    Args:
        spec_dir: Path to the spec directory (e.g. specs/resnet56/equal-split).
        profile_file: Path to the profile YAML file (e.g. profiles/linear-3/100mbps.yaml).
        output_dir: Root directory for generated experiments.
        sub_experiment_filter: If set, only include these sub-experiment names.
        _seen: Cycle guard — set of experiment names currently being materialised.

    Returns:
        Path to the generated experiment directory.
    """
    if _seen is None:
        _seen = set()

    spec = load_spec(spec_dir)
    all_sub_experiments = load_sub_experiments(spec_dir)
    profile = load_profile(profile_file)

    if not all_sub_experiments:
        raise ValueError(f"No sub_experiments.yaml found under spec '{spec_dir}'")

    validate_compatibility(spec, profile)

    exp_name = derive_name(spec_dir, profile_file)
    _seen.add(exp_name)

    if sub_experiment_filter:
        missing = set(sub_experiment_filter) - set(all_sub_experiments)
        if missing:
            raise ValueError(
                f"Sub-experiments not found in spec '{spec_dir}': {sorted(missing)}\n"
                f"Available: {sorted(all_sub_experiments)}"
            )
        selected = {
            k: v for k, v in all_sub_experiments.items() if k in sub_experiment_filter
        }
    else:
        selected = all_sub_experiments

    # Resolve each sub-experiment
    resolved_sub_experiments: list[dict[str, Any]] = []
    for name, sub_exp in selected.items():
        resolved_baselines: list[str] = []
        for ref in sub_exp.get("baselines", []):
            bl = resolve_baseline(ref, spec_dir, profile_file, output_dir, _seen)
            resolved_baselines.append(bl)

        entry: dict[str, Any] = {"name": name}
        if sub_exp.get("links"):
            entry["links"] = sub_exp["links"]
        if sub_exp.get("sweep"):
            entry["sweep_mode"] = sub_exp.get("sweep_mode", "paired")
            entry["sweep"] = sub_exp["sweep"]
        if resolved_baselines:
            entry["baselines"] = resolved_baselines
        if sub_exp.get("dataset"):
            entry["dataset"] = sub_exp["dataset"]

        resolved_sub_experiments.append(entry)

    # Build generated experiment.yaml
    generated: dict[str, Any] = {
        "name": exp_name,
        "model": spec["model"],
        "nodes": spec["nodes"],
    }
    if "dataset" in spec:
        generated["dataset"] = spec["dataset"]
    generated["metrics_server"] = spec["metrics_server"]
    generated["sub_experiments"] = resolved_sub_experiments

    # Write output
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    _dump_yaml(generated, exp_dir / "experiment.yaml")
    _dump_yaml(profile, exp_dir / "infra.yaml")

    logger.info("Generated experiment: %s", exp_dir)
    return exp_dir


# ---------------------------------------------------------------------------
# Generate-all helpers
# ---------------------------------------------------------------------------


def _leaf_specs() -> list[Path]:
    """Return all spec directories whose merged config defines nodes.

    A spec directory is a runnable target when the fully merged experiment
    config (walking root→leaf) contains a non-empty ``nodes`` list.  This
    filters out model-level directories that only define shared defaults.
    """
    candidates = sorted(p.parent for p in SPECS_DIR.rglob("experiment.yaml"))
    return [p for p in candidates if load_spec(p).get("nodes")]


def _generate_all(
    output_dir: Path,
    sub_experiment_filter: list[str] | None = None,
) -> list[Path]:
    """Generate experiments for all compatible spec/profile pairs.

    Skips incompatible combinations (node name mismatches) silently.

    Args:
        output_dir: Root directory for generated experiments.
        sub_experiment_filter: If set, include only these sub-experiment names.

    Returns:
        List of generated experiment directory paths.
    """
    specs = _leaf_specs()
    profiles = sorted(PROFILES_DIR.rglob("*.yaml"))

    logger.info(
        "Found %d runnable spec(s) and %d profile(s) — trying %d combination(s)",
        len(specs),
        len(profiles),
        len(specs) * len(profiles),
    )

    generated: list[Path] = []
    skipped = 0

    for spec in specs:
        for profile in profiles:
            try:
                exp_dir = _materialise(spec, profile, output_dir, sub_experiment_filter)
                generated.append(exp_dir)
            except ValueError as e:
                if "Node name mismatch" in str(e) or "No sub_experiments" in str(e):
                    skipped += 1
                else:
                    logger.error("Failed %s + %s: %s", spec, profile, e)
            except FileNotFoundError as e:
                logger.error("Failed %s + %s: %s", spec, profile, e)

    logger.info(
        "Generated %d experiment(s), skipped %d incompatible combination(s)",
        len(generated),
        skipped,
    )
    return generated


# ---------------------------------------------------------------------------
# Multi-model generation
# ---------------------------------------------------------------------------


def load_multi_spec(spec_dir: Path) -> dict[str, Any]:
    """Load a multi-model spec from multispecs/<name>/experiment.yaml.

    Unlike single-model specs, multi-model specs are flat — no hierarchical
    merge is performed.

    Args:
        spec_dir: Path to the multispec directory (e.g. multispecs/resnet56_llama).

    Returns:
        Spec config dict.

    Raises:
        FileNotFoundError: If experiment.yaml is not found in spec_dir.
    """
    yaml_path = spec_dir / "experiment.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Multi-model spec not found: {yaml_path}")
    return _load_yaml(yaml_path)


def load_multi_sub_experiments(spec_dir: Path) -> dict[str, Any]:
    """Load sub-experiments from multispecs/<name>/sub_experiments.yaml.

    Args:
        spec_dir: Path to the multispec directory.

    Returns:
        Dict of sub-experiment name → sub-experiment config dict.

    Raises:
        FileNotFoundError: If sub_experiments.yaml is not found.
    """
    yaml_path = spec_dir / "sub_experiments.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Multi-model sub_experiments not found: {yaml_path}")
    data = _load_yaml(yaml_path)
    return data.get("sub_experiments", {})


def validate_multi_compatibility(spec: dict[str, Any], profile: dict[str, Any]) -> None:
    """Ensure node names in multi-model spec and profile match exactly.

    In multi-model specs, nodes is a list of name strings rather than full
    node config dicts.

    Args:
        spec: Multi-model spec dict (nodes is a list of name strings).
        profile: Profile dict.

    Raises:
        ValueError: If node name sets differ.
    """
    spec_nodes = set(spec.get("nodes", []))
    profile_nodes = {n["name"] for n in profile.get("nodes", [])}
    if not profile_nodes:
        return
    if spec_nodes != profile_nodes:
        raise ValueError(
            f"Node name mismatch between multi-model spec and profile.\n"
            f"  Spec nodes:    {sorted(spec_nodes)}\n"
            f"  Profile nodes: {sorted(profile_nodes)}"
        )


def derive_multi_name(spec_dir: Path, profile_file: Path) -> str:
    """Derive the canonical experiment name for a multi-model experiment.

    Format: {spec_name}_{topology}_{profile_name}

    Args:
        spec_dir: Path to the multispec directory (e.g. multispecs/resnet56_llama).
        profile_file: Path to the profile YAML file (e.g. profiles/linear-3-bidir/100mbps.yaml).

    Returns:
        Experiment name string.
    """
    spec_name = spec_dir.relative_to(MULTISPECS_DIR).parts[0]
    profile_parts = profile_file.relative_to(PROFILES_DIR).parts
    topology = profile_parts[0]
    profile_name = profile_file.stem
    return f"{spec_name}_{topology}_{profile_name}"


def _materialise_multi(
    spec_dir: Path,
    profile_file: Path,
    output_dir: Path,
    sub_experiment_filter: list[str] | None = None,
) -> Path:
    """Generate one multi-model experiment directory.

    Merges the multi-model spec with a profile to produce a fully resolved
    experiment.yaml and infra.yaml in experiments/multi/<name>/.

    Node host/port values are taken from the profile; the spec provides only
    node names.  Sub-experiment baselines are stored as-is (already resolved
    strings of the form 'experiment_name/sub_experiment_name').

    Args:
        spec_dir: Path to the multispec directory (e.g. multispecs/resnet56_llama).
        profile_file: Path to the profile YAML file.
        output_dir: Root directory for generated multi-model experiments.
        sub_experiment_filter: If set, only include these sub-experiment names.

    Returns:
        Path to the generated experiment directory.
    """
    spec = load_multi_spec(spec_dir)
    all_sub_experiments = load_multi_sub_experiments(spec_dir)
    profile = load_profile(profile_file)

    validate_multi_compatibility(spec, profile)

    exp_name = derive_multi_name(spec_dir, profile_file)

    if sub_experiment_filter:
        missing = set(sub_experiment_filter) - set(all_sub_experiments)
        if missing:
            raise ValueError(
                f"Sub-experiments not found in spec '{spec_dir}': {sorted(missing)}\n"
                f"Available: {sorted(all_sub_experiments)}"
            )
        selected = {
            k: v for k, v in all_sub_experiments.items() if k in sub_experiment_filter
        }
    else:
        selected = all_sub_experiments

    # Build node configs by merging spec node names with profile host/port
    profile_node_map = {n["name"]: n for n in profile.get("nodes", [])}
    nodes: list[dict[str, Any]] = []
    for node_name in spec.get("nodes", []):
        profile_node = profile_node_map.get(node_name, {})
        nodes.append(
            {
                "name": node_name,
                "host": profile_node.get("host", node_name),
                "port": profile_node.get("port", 8000),
            }
        )

    # Resolve sub-experiments (baselines are already resolved strings)
    resolved_sub_experiments: list[dict[str, Any]] = []
    for name, sub_exp in selected.items():
        entry: dict[str, Any] = {"name": name}
        if sub_exp.get("links"):
            entry["links"] = sub_exp["links"]
        if sub_exp.get("sweep"):
            entry["sweep_mode"] = sub_exp.get("sweep_mode", "paired")
            entry["sweep"] = sub_exp["sweep"]
        if sub_exp.get("baselines"):
            entry["baselines"] = sub_exp["baselines"]
        resolved_sub_experiments.append(entry)

    # Build generated experiment.yaml
    generated: dict[str, Any] = {
        "name": exp_name,
        "nodes": nodes,
        "pipelines": spec["pipelines"],
        "datasets": spec["datasets"],
        "workload": spec["workload"],
        "metrics_server": spec["metrics_server"],
        "sub_experiments": resolved_sub_experiments,
    }

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    _dump_yaml(generated, exp_dir / "experiment.yaml")
    _dump_yaml(profile, exp_dir / "infra.yaml")

    logger.info("Generated multi-model experiment: %s", exp_dir)
    return exp_dir


def _leaf_multi_specs() -> list[Path]:
    """Return all multispec directories that have both experiment.yaml and sub_experiments.yaml.

    Returns:
        Sorted list of valid multispec directory paths.
    """
    if not MULTISPECS_DIR.exists():
        return []
    candidates = sorted(
        p.parent
        for p in MULTISPECS_DIR.rglob("experiment.yaml")
        if (p.parent / "sub_experiments.yaml").exists()
    )
    return candidates


def _generate_all_multi(
    output_dir: Path,
    sub_experiment_filter: list[str] | None = None,
) -> list[Path]:
    """Generate experiments for all compatible multispec/profile pairs.

    Skips incompatible combinations (node name mismatches) silently.

    Args:
        output_dir: Root directory for generated multi-model experiments.
        sub_experiment_filter: If set, include only these sub-experiment names.

    Returns:
        List of generated experiment directory paths.
    """
    specs = _leaf_multi_specs()
    profiles = sorted(PROFILES_DIR.rglob("*.yaml"))

    logger.info(
        "Found %d multi-model spec(s) and %d profile(s) — trying %d combination(s)",
        len(specs),
        len(profiles),
        len(specs) * len(profiles),
    )

    generated: list[Path] = []
    skipped = 0

    for spec in specs:
        for profile in profiles:
            try:
                exp_dir = _materialise_multi(
                    spec, profile, output_dir, sub_experiment_filter
                )
                generated.append(exp_dir)
            except ValueError as e:
                if "Node name mismatch" in str(e) or "Sub-experiments not found" in str(
                    e
                ):
                    skipped += 1
                else:
                    logger.error("Failed %s + %s: %s", spec, profile, e)
            except (FileNotFoundError, KeyError) as e:
                logger.error("Failed %s + %s: %s", spec, profile, e)

    logger.info(
        "Generated %d multi-model experiment(s), skipped %d incompatible combination(s)",
        len(generated),
        skipped,
    )
    return generated


# ---------------------------------------------------------------------------
# Optimization experiment generation
# ---------------------------------------------------------------------------


def load_opt_spec(spec_dir: Path) -> dict[str, Any]:
    """Load an optimization spec from optspecs/<name>/experiment.yaml.

    Opt specs are flat (no hierarchical merge).

    Args:
        spec_dir: Path to the optspec directory (e.g. optspecs/resnet56_llama_mmlu).

    Returns:
        Spec config dict.

    Raises:
        FileNotFoundError: If experiment.yaml is not found.
    """
    yaml_path = spec_dir / "experiment.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Opt spec not found: {yaml_path}")
    return _load_yaml(yaml_path)


def load_opt_sub_experiments(spec_dir: Path) -> list[dict[str, Any]]:
    """Load sub-experiments for an optspec.

    Unlike multispecs (which use a named dict), opt sub-experiments are an
    ordered list because execution order is semantically meaningful.

    Two layouts are supported:
    - Separate file: ``sub_experiments.yaml`` alongside ``experiment.yaml``.
    - Combined file: ``sub_experiments`` key inside ``experiment.yaml``.

    The separate file takes precedence if both are present.

    Args:
        spec_dir: Path to the optspec directory.

    Returns:
        Ordered list of sub-experiment config dicts.

    Raises:
        FileNotFoundError: If neither source provides sub-experiments.
    """
    sub_yaml = spec_dir / "sub_experiments.yaml"
    if sub_yaml.exists():
        data = _load_yaml(sub_yaml)
        return data.get("sub_experiments", [])

    exp_yaml = spec_dir / "experiment.yaml"
    if exp_yaml.exists():
        data = _load_yaml(exp_yaml)
        if "sub_experiments" in data:
            return data["sub_experiments"]

    raise FileNotFoundError(
        f"Opt sub_experiments not found for spec '{spec_dir}': "
        f"provide sub_experiments.yaml or a sub_experiments key in experiment.yaml"
    )


def validate_opt_compatibility(spec: dict[str, Any], profile: dict[str, Any]) -> None:
    """Ensure node names in the opt spec and profile match exactly.

    In opt specs, nodes is a list of name strings (same as multispecs).

    Args:
        spec: Opt spec dict (nodes is a list of name strings).
        profile: Profile dict.

    Raises:
        ValueError: If node name sets differ.
    """
    spec_nodes = set(spec.get("nodes", []))
    profile_nodes = {n["name"] for n in profile.get("nodes", [])}
    if not profile_nodes:
        return
    if spec_nodes != profile_nodes:
        raise ValueError(
            f"Node name mismatch between opt spec and profile.\n"
            f"  Spec nodes:    {sorted(spec_nodes)}\n"
            f"  Profile nodes: {sorted(profile_nodes)}"
        )


def derive_opt_name(spec_dir: Path, profile_file: Path) -> str:
    """Derive the canonical experiment name for an optimization experiment.

    Format: {spec_name}_{topology}_{profile_name}

    This matches the multi-model naming convention so profile suffixes
    (e.g. ``_100mbps``) can be used as ``experiment_name_contains`` filters
    when querying the metrics server for link probe history.

    Args:
        spec_dir: Path to the optspec directory (e.g. optspecs/resnet56_llama_mmlu).
        profile_file: Path to the profile YAML file.

    Returns:
        Experiment name string.
    """
    spec_name = spec_dir.relative_to(OPTSPECS_DIR).parts[0]
    profile_parts = profile_file.relative_to(PROFILES_DIR).parts
    topology = profile_parts[0]
    profile_name = profile_file.stem
    return f"{spec_name}_{topology}_{profile_name}"


def _inject_channel_estimator_context(
    sub_experiments: list[dict[str, Any]],
    profile_name: str,
) -> list[dict[str, Any]]:
    """Inject ``experiment_name_contains`` into channel_estimator configs.

    For any sub-experiment that has a ``channel_estimator`` with
    ``history_source: metrics_server`` and no ``experiment_name_contains``
    already set, sets ``experiment_name_contains`` to the profile filename
    stem (e.g. ``"100mbps"``).  This scopes probe history queries to the
    correct hardware profile without modifying InfraConfig or existing events.

    Args:
        sub_experiments: List of raw sub-experiment config dicts.
        profile_name: Profile filename stem (e.g. ``"100mbps"``).

    Returns:
        Deep copy of the list with context injected.
    """
    import copy

    result = copy.deepcopy(sub_experiments)
    for sub_exp in result:
        ce = sub_exp.get("channel_estimator")
        if isinstance(ce, dict) and ce.get("history_source") == "metrics_server":
            if "experiment_name_contains" not in ce:
                ce["experiment_name_contains"] = profile_name
    return result


def _materialise_opt(
    spec_dir: Path,
    profile_file: Path,
    output_dir: Path,
    sub_experiment_filter: list[str] | None = None,
) -> Path:
    """Generate one optimization experiment directory.

    Merges the opt spec with a profile to produce a fully resolved
    experiment.yaml and infra.yaml in experiments/opt/<name>/.

    Node host/port values are taken from the profile; the spec provides only
    node names.  Channel estimator configs with ``history_source: metrics_server``
    have ``experiment_name_contains`` injected from the profile filename stem.

    Args:
        spec_dir: Path to the optspec directory.
        profile_file: Path to the profile YAML file.
        output_dir: Root directory for generated opt experiments.
        sub_experiment_filter: If set, only include sub-experiments whose
            ``name`` field is in this list.

    Returns:
        Path to the generated experiment directory.
    """
    spec = load_opt_spec(spec_dir)
    all_sub_experiments = load_opt_sub_experiments(spec_dir)
    profile = load_profile(profile_file)

    validate_opt_compatibility(spec, profile)

    exp_name = derive_opt_name(spec_dir, profile_file)
    profile_name = profile_file.stem

    if sub_experiment_filter:
        names_in_spec = [s.get("name") for s in all_sub_experiments]
        missing = set(sub_experiment_filter) - set(names_in_spec)
        if missing:
            raise ValueError(
                f"Sub-experiments not found in spec '{spec_dir}': {sorted(missing)}\n"
                f"Available: {sorted(n for n in names_in_spec if n)}"
            )
        selected = [
            s for s in all_sub_experiments if s.get("name") in sub_experiment_filter
        ]
    else:
        selected = all_sub_experiments

    # Inject profile context into channel estimator configs.
    selected = _inject_channel_estimator_context(selected, profile_name)

    # Build node configs by merging spec node names with profile host/port.
    profile_node_map = {n["name"]: n for n in profile.get("nodes", [])}
    nodes: list[dict[str, Any]] = []
    for node_name in spec.get("nodes", []):
        profile_node = profile_node_map.get(node_name, {})
        nodes.append(
            {
                "name": node_name,
                "host": profile_node.get("host", node_name),
                "port": profile_node.get("port", 8000),
            }
        )

    # Build generated experiment.yaml.
    generated: dict[str, Any] = {
        "name": exp_name,
        "nodes": nodes,
        "pipelines": spec["pipelines"],
        "datasets": spec["datasets"],
        "workload": spec["workload"],
        "tasks": spec["tasks"],
        "links": spec["links"],
        "optimization_loop": spec["optimization_loop"],
        "metrics_server": spec["metrics_server"],
        "artifacts_dir": f"artifacts/{exp_name}",
        "shared_artifacts_dir": "artifacts/shared",
        "sub_experiments": selected,
    }

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    _dump_yaml(generated, exp_dir / "experiment.yaml")
    _dump_yaml(profile, exp_dir / "infra.yaml")

    logger.info("Generated opt experiment: %s", exp_dir)
    return exp_dir


def _leaf_opt_specs() -> list[Path]:
    """Return all optspec directories that have a resolvable sub-experiments source.

    Accepts either a separate ``sub_experiments.yaml`` or a ``sub_experiments``
    key inside ``experiment.yaml``.

    Returns:
        Sorted list of valid optspec directory paths.
    """
    if not OPTSPECS_DIR.exists():
        return []

    def _has_sub_experiments(spec_dir: Path) -> bool:
        if (spec_dir / "sub_experiments.yaml").exists():
            return True
        exp_yaml = spec_dir / "experiment.yaml"
        if exp_yaml.exists():
            data = _load_yaml(exp_yaml)
            return "sub_experiments" in data
        return False

    candidates = sorted(
        p.parent
        for p in OPTSPECS_DIR.rglob("experiment.yaml")
        if _has_sub_experiments(p.parent)
    )
    return candidates


def _generate_all_opt(
    output_dir: Path,
    sub_experiment_filter: list[str] | None = None,
) -> list[Path]:
    """Generate experiments for all compatible optspec/profile pairs.

    Skips incompatible combinations (node name mismatches) silently.

    Args:
        output_dir: Root directory for generated opt experiments.
        sub_experiment_filter: If set, include only sub-experiments with
            these names.

    Returns:
        List of generated experiment directory paths.
    """
    specs = _leaf_opt_specs()
    profiles = sorted(PROFILES_DIR.rglob("*.yaml"))

    logger.info(
        "Found %d opt spec(s) and %d profile(s) — trying %d combination(s)",
        len(specs),
        len(profiles),
        len(specs) * len(profiles),
    )

    generated: list[Path] = []
    skipped = 0

    for spec in specs:
        for profile in profiles:
            try:
                exp_dir = _materialise_opt(
                    spec, profile, output_dir, sub_experiment_filter
                )
                generated.append(exp_dir)
            except ValueError as e:
                if "Node name mismatch" in str(e) or "Sub-experiments not found" in str(
                    e
                ):
                    skipped += 1
                else:
                    logger.error("Failed %s + %s: %s", spec, profile, e)
            except (FileNotFoundError, KeyError) as e:
                logger.error("Failed %s + %s: %s", spec, profile, e)

    logger.info(
        "Generated %d opt experiment(s), skipped %d incompatible combination(s)",
        len(generated),
        skipped,
    )
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fully resolved experiment directories from specs and profiles. "
            "Use --spec/--profile for a single experiment or --all for every "
            "compatible combination.  Add --multi for multispecs/ or --opt for "
            "optspecs/."
        )
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help=(
            "Generate multi-model experiments from multispecs/ "
            "(output goes to experiments/multi/ by default)"
        ),
    )
    parser.add_argument(
        "--opt",
        action="store_true",
        help=(
            "Generate optimization experiments from optspecs/ "
            "(output goes to experiments/opt/ by default)"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all compatible spec/profile combinations",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help=(
            "Path to the spec directory "
            "(e.g. specs/resnet56/equal-split or multispecs/resnet56_llama)"
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Path to the profile YAML file (e.g. profiles/linear-3/100mbps.yaml)",
    )
    parser.add_argument(
        "--sub-experiments",
        nargs="+",
        metavar="NAME",
        help="Only include these sub-experiments (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Root output directory "
            f"(default: {EXPERIMENTS_DIR} for single-model, "
            f"{MULTI_EXPERIMENTS_DIR} for --multi)"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each generated experiment after generation",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Print resolved sweep runs during validation (implies --validate)",
    )
    args = parser.parse_args()

    if args.show_runs:
        args.validate = True

    if args.multi and args.opt:
        print("ERROR: --multi and --opt are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (
        OPT_EXPERIMENTS_DIR
        if args.opt
        else (MULTI_EXPERIMENTS_DIR if args.multi else EXPERIMENTS_DIR)
    )

    if args.all and (args.spec or args.profile):
        print("ERROR: --all cannot be used with --spec or --profile", file=sys.stderr)
        sys.exit(1)

    if not args.all and not (args.spec and args.profile):
        print("ERROR: provide --spec and --profile, or use --all", file=sys.stderr)
        sys.exit(1)

    if args.opt:
        if args.all:
            generated = _generate_all_opt(output_dir, args.sub_experiments)
            if not generated:
                print("ERROR: no opt experiments were generated", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                exp_dir = _materialise_opt(
                    spec_dir=args.spec,
                    profile_file=args.profile,
                    output_dir=output_dir,
                    sub_experiment_filter=args.sub_experiments,
                )
                print(exp_dir)
                generated = [exp_dir]
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
    elif args.multi:
        if args.all:
            generated = _generate_all_multi(output_dir, args.sub_experiments)
            if not generated:
                print(
                    "ERROR: no multi-model experiments were generated", file=sys.stderr
                )
                sys.exit(1)
        else:
            try:
                exp_dir = _materialise_multi(
                    spec_dir=args.spec,
                    profile_file=args.profile,
                    output_dir=output_dir,
                    sub_experiment_filter=args.sub_experiments,
                )
                print(exp_dir)
                generated = [exp_dir]
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
    else:
        if args.all:
            generated = _generate_all(output_dir, args.sub_experiments)
            if not generated:
                print("ERROR: no experiments were generated", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                exp_dir = _materialise(
                    spec_dir=args.spec,
                    profile_file=args.profile,
                    output_dir=output_dir,
                    sub_experiment_filter=args.sub_experiments,
                )
                print(exp_dir)
                generated = [exp_dir]
            except (FileNotFoundError, ValueError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)

    if args.validate:
        from framework.validate import validate

        all_ok = True
        for exp_dir in sorted(generated):
            print("=" * 60)
            if not validate(exp_dir, show_runs=args.show_runs):
                all_ok = False
            print()
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
