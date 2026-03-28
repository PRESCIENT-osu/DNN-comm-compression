"""Experiment generation tool.

Composes a spec + profile into a fully resolved experiment directory ready
for deployment.  All sub-experiments defined in the spec are included.
Referenced baseline experiments are materialised recursively.

Usage::

    python tools/generate.py \\
        --spec resnet56/equal-split \\
        --profile resnet56/linear-3/100mbps

    python tools/generate.py \\
        --spec resnet56/equal-split \\
        --profile resnet56/linear-3/100mbps \\
        --sub-experiments baseline topk_paired

Output is written to experiments/<name>/ which should be gitignored.
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
PROFILES_DIR = Path("profiles")
EXPERIMENTS_DIR = Path("experiments")


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


def load_spec(spec_path: str) -> dict[str, Any]:
    """Walk the spec hierarchy root→leaf and deep-merge experiment.yaml layers.

    Args:
        spec_path: Slash-separated path relative to specs/ (e.g. resnet56/equal-split).

    Returns:
        Merged experiment config dict.
    """
    parts = Path(spec_path).parts
    merged: dict[str, Any] = {}
    current = SPECS_DIR
    for part in parts:
        current = current / part
        yaml_path = current / "experiment.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Spec file not found: {yaml_path}")
        merged = _deep_merge(merged, _load_yaml(yaml_path))
    return merged


def load_sub_experiments(spec_path: str) -> dict[str, Any]:
    """Walk the spec hierarchy and merge sub_experiments.yaml layers.

    Child entries override parent entries with the same name; new child
    entries are addded without affecting parent entries.

    Args:
        spec_path: Slash-separated path relative to specs/.

    Returns:
        Merged dict of sub-experiment name → sub-experiment config dict.
    """
    parts = Path(spec_path).parts
    merged: dict[str, Any] = {}
    current = SPECS_DIR
    for part in parts:
        current = current / part
        yaml_path = current / "sub_experiments.yaml"
        if yaml_path.exists():
            data = _load_yaml(yaml_path)
            child_entries = data.get("sub_experiments", {})
            merged = {**merged, **child_entries}
    return merged


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


def load_profile(profile_path: str) -> dict[str, Any]:
    """Load a profile YAML file.

    Args:
        profile_path: Slash-separated path relative to profiles/ without
            extension (e.g. resnet56/linear-3/100mbps).

    Returns:
        Profile dict (InfraConfig-compatible).
    """
    full_path = PROFILES_DIR / f"{profile_path}.yaml"
    if not full_path.exists():
        raise FileNotFoundError(f"Profile not found: {full_path}")
    return _load_yaml(full_path)


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


def derive_name(spec_path: str, profile_path: str) -> str:
    """Derive the canonical experiment name from spec and profile paths.

    Format: {model}_{partition-split}_{profile-topology}_{profile-name}

    Args:
        spec_path: e.g. resnet56/equal-split
        profile_path: e.g. resnet56/linear-3/100mbps

    Returns:
        Experiment name string.
    """
    spec_parts = Path(spec_path).parts
    profile_parts = Path(profile_path).parts

    model = spec_parts[0]
    split = spec_parts[1] if len(spec_parts) > 1 else "default"
    # profile_parts: (model, topology, name) — drop model prefix
    topology = profile_parts[1] if len(profile_parts) > 1 else profile_parts[0]
    profile_name = profile_parts[2] if len(profile_parts) > 2 else "default"

    return f"{model}_{split}_{topology}_{profile_name}"


# ---------------------------------------------------------------------------
# Baseline resolution
# ---------------------------------------------------------------------------


def resolve_baseline(
    ref: dict[str, Any],
    current_spec: str,
    current_profile: str,
    output_dir: Path,
    _seen: set[str],
) -> str:
    """Resolve a baseline reference to an 'experiment_name/sub_experiment' string.

    Materialises the referenced experiment if it does not already exist.

    Args:
        ref: Baseline reference dict with sub_experiment and optional spec/profile.
        current_spec: Spec path of the experiment being generated.
        current_profile: Profile path of the experiment being generated.
        output_dir: Root experiments directory.
        _seen: Set of already-materialising experiment names (cycle guard).

    Returns:
        String of the form 'experiment_name/sub_experiment_name'.
    """
    sub_exp_name = ref["sub_experiment"]
    ref_spec = ref.get("spec", current_spec)
    ref_profile = ref.get("profile", current_profile)

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
    spec_path: str,
    profile_path: str,
    output_dir: Path,
    sub_experiment_filter: list[str] | None = None,
    _seen: set[str] | None = None,
) -> Path:
    """Core generation logic — resolves and writes one experiment directory.

    Args:
        spec_path: Spec path (e.g. resnet56/equal-split).
        profile_path: Profile path (e.g. resnet56/linear-3/100mbps).
        output_dir: Root directory for generated experiments.
        sub_experiment_filter: If set, only include these sub-experiment names.
        _seen: Cycle guard — set of experiment names currently being materialised.

    Returns:
        Path to the generated experiment directory.
    """
    if _seen is None:
        _seen = set()

    spec = load_spec(spec_path)
    all_sub_experiments = load_sub_experiments(spec_path)
    profile = load_profile(profile_path)

    if not all_sub_experiments:
        raise ValueError(f"No sub_experiments.yaml found under spec '{spec_path}'")

    validate_compatibility(spec, profile)

    exp_name = derive_name(spec_path, profile_path)
    _seen.add(exp_name)

    if sub_experiment_filter:
        missing = set(sub_experiment_filter) - set(all_sub_experiments)
        if missing:
            raise ValueError(
                f"Sub-experiments not found in spec '{spec_path}': {sorted(missing)}\n"
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
            bl = resolve_baseline(ref, spec_path, profile_path, output_dir, _seen)
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
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fully resolved experiment directory from a spec and profile. "
            "All sub-experiments in the spec are included by default."
        )
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Spec path relative to specs/ (e.g. resnet56/equal-split)",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Profile path relative to profiles/ without extension "
        "(e.g. resnet56/linear-3/100mbps)",
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
        default=EXPERIMENTS_DIR,
        help=f"Root output directory (default: {EXPERIMENTS_DIR})",
    )
    args = parser.parse_args()

    try:
        exp_dir = _materialise(
            spec_path=args.spec,
            profile_path=args.profile,
            output_dir=args.output_dir,
            sub_experiment_filter=args.sub_experiments,
        )
        print(exp_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
