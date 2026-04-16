"""Sub-experiment metadata parsing for optimizer experiment analysis.

Maps sub_experiment_name strings to structured metadata used for grouping
and labelling across all analysis and plotting functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

import pandas as pd


@dataclass
class SubExpMeta:
    """Structured metadata derived from a sub_experiment_name string."""

    sub_exp_type: (
        str  # profiling | accuracy_model | no_csi | csi_aware | baseline | unknown
    )
    backend: str | None  # am | stein
    surrogate_type: str | None  # gbm | poly2 | poly3 | rf | mlp  (AM no_csi only)
    estimator_type: str | None  # ma | lcb | mean  (Stein no_csi only)
    mu: float | None  # Lyapunov penalty weight (no_csi only)
    baseline_variant: str | None  # display name with _single suffix removed


# Maps the raw sub_experiment_name to a clean display label.
# The _single suffix is dropped per the analysis convention.
_BASELINE_MAP: dict[str, str] = {
    "max_compression_single": "max_compression",
    "no_compression_single": "no_compression",
    "uniform_compression_single": "uniform_compression",
    "myopic_single": "myopic",
    "conservative_single": "conservative",
    "movingavg_single": "movingavg",
    # Multi-task baselines (for future experiments)
    "max_compression_multi": "max_compression",
    "no_compression_multi": "no_compression",
    "static_equal_share": "static_equal_share",
    "proportional_resource": "proportional_resource",
    "strict_priority_greedy": "strict_priority_greedy",
    "decoupled_descent": "decoupled_descent",
    "queue_proportional": "queue_proportional",
    "historical_average_ce": "historical_average_ce",
}

_NO_CSI_AM_RE = re.compile(r"no_csi_mu_sweep_am_(\w+)_mu([\d.]+)$")
_NO_CSI_STEIN_RE = re.compile(r"no_csi_mu_sweep_ce_(\w+)_mu([\d.]+)$")
_ACCURACY_MODEL_RE = re.compile(r"accuracy_model_(?:\w+?)_(\w+)$")

_UNKNOWN = SubExpMeta("unknown", None, None, None, None, None)


def parse_sub_exp_meta(name: str) -> SubExpMeta:
    """Parse a sub_experiment_name string into structured metadata.

    Args:
        name: Raw sub_experiment_name field from a metric event.

    Returns:
        SubExpMeta with all available fields populated.
    """
    if name == "profiling":
        return SubExpMeta("profiling", None, None, None, None, None)

    if name == "csi_aware":
        return SubExpMeta("csi_aware", None, None, None, None, None)

    if name in _BASELINE_MAP:
        return SubExpMeta("baseline", None, None, None, None, _BASELINE_MAP[name])

    m = _NO_CSI_AM_RE.match(name)
    if m:
        return SubExpMeta("no_csi", "am", m.group(1), None, float(m.group(2)), None)

    m = _NO_CSI_STEIN_RE.match(name)
    if m:
        return SubExpMeta("no_csi", "stein", None, m.group(1), float(m.group(2)), None)

    m = _ACCURACY_MODEL_RE.match(name)
    if m:
        return SubExpMeta("accuracy_model", "am", m.group(1), None, None, None)

    return _UNKNOWN


_META_FIELDS = [f.name for f in fields(SubExpMeta)]


def inject_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Add structured metadata columns derived from sub_experiment_name.

    Adds one column per SubExpMeta field. Existing meta columns are
    overwritten.  Rows with null sub_experiment_name get all-None meta.

    Args:
        df: DataFrame with a ``sub_experiment_name`` column.

    Returns:
        Copy of df with meta columns appended.
    """
    if df.empty or "sub_experiment_name" not in df.columns:
        return df

    df = df.copy()
    metas = df["sub_experiment_name"].apply(
        lambda n: parse_sub_exp_meta(n) if pd.notna(n) else _UNKNOWN
    )
    for field_name in _META_FIELDS:
        df[field_name] = metas.apply(lambda m, f=field_name: getattr(m, f))

    return df
