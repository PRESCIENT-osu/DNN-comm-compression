"""Table builders for optimizer experiment analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from framework.analysis.opt._meta import inject_meta
from framework.analysis.opt.loader import (
    load_opt_slots,
    load_task_accuracy,
    load_throughput_constraints,
)
from framework.datamodels.opt_experiment import GeneratedOptExperimentConfig

logger = logging.getLogger(__name__)


def _exp_id(directory: Path) -> str:
    return directory.name


_BASELINE_DISPLAY: dict[str, str] = {
    "max_compression_single": "max_compression",
    "no_compression_single": "no_compression",
    "uniform_compression_single": "uniform_compression",
    "myopic_single": "myopic",
    "conservative_single": "conservative",
    "movingavg_single": "movingavg",
    "max_compression_multi": "max_compression",
    "no_compression_multi": "no_compression",
    "static_equal_share": "static_equal_share",
    "proportional_resource": "proportional_resource",
    "strict_priority_greedy": "strict_priority_greedy",
    "decoupled_descent": "decoupled_descent",
    "queue_proportional": "queue_proportional",
    "historical_average_ce": "historical_average_ce",
}


def _inject_meta_from_config(
    df: pd.DataFrame,
    exp_config: GeneratedOptExperimentConfig,
) -> pd.DataFrame:
    """Inject sub-experiment metadata columns derived from the experiment config.

    More robust than ``_meta.inject_meta``: event sub_experiment_names are
    matched to config entries by longest-prefix.  Optimization run names have
    the form ``{config.name}__{scheme}`` or ``{config.name}__{scheme}_mu{mu}``;
    baseline names are exactly ``{config.name}``.

    Args:
        df: DataFrame with a ``sub_experiment_name`` column.
        exp_config: Parsed experiment config.

    Returns:
        Copy of ``df`` with metadata columns appended.
    """
    if df.empty or "sub_experiment_name" not in df.columns:
        return df

    # Map accuracy_model sub-experiment name → surrogate_type
    surrogate_by_am_name: dict[str, str] = {
        s.name: s.model_type
        for s in exp_config.sub_experiments
        if s.type == "accuracy_model"
    }

    # Sort by name length descending for longest-prefix match
    config_entries = sorted(
        exp_config.sub_experiments, key=lambda s: len(s.name), reverse=True
    )

    _empty: dict = {
        "sub_exp_type": "unknown",
        "backend": None,
        "surrogate_type": None,
        "estimator_type": None,
        "mu": None,
        "baseline_variant": None,
        "compression_scheme": None,
    }

    def _meta(event_name: str) -> dict:
        if not isinstance(event_name, str):
            return _empty

        matched = next(
            (
                s
                for s in config_entries
                if event_name == s.name or event_name.startswith(s.name + "__")
            ),
            None,
        )
        if matched is None:
            return _empty

        cfg_type = matched.type
        suffix = event_name[
            len(matched.name) :
        ]  # "" for baselines, "__{scheme}[_mu{mu}]" for opt

        # Extract compression_scheme and μ from suffix.
        # Suffix format: "__{scheme}" or "__{scheme}_mu{mu}"
        compression_scheme = None
        mu = None
        if suffix.startswith("__"):
            inner = suffix[2:]  # strip leading __
            if "_mu" in inner:
                scheme_part, mu_str = inner.rsplit("_mu", 1)
                compression_scheme = scheme_part or None
                try:
                    mu = float(mu_str)
                except ValueError:
                    pass
            else:
                compression_scheme = inner or None

        if cfg_type in _BASELINE_DISPLAY:
            return {
                **_empty,
                "sub_exp_type": "baseline",
                "baseline_variant": _BASELINE_DISPLAY[cfg_type],
                "compression_scheme": compression_scheme,
            }

        if cfg_type == "csi_aware":
            return {
                **_empty,
                "sub_exp_type": "csi_aware",
                "compression_scheme": compression_scheme,
            }

        if cfg_type == "no_csi":
            backend = "am" if matched.accuracy_model_refs else "stein"
            surrogate_type = None
            if matched.accuracy_model_refs:
                surrogate_type = surrogate_by_am_name.get(
                    matched.accuracy_model_refs[0]
                )
            estimator_type = str(matched.channel_estimator.type)
            return {
                **_empty,
                "sub_exp_type": "no_csi",
                "backend": backend,
                "surrogate_type": surrogate_type,
                "estimator_type": estimator_type,
                "mu": mu,
                "compression_scheme": compression_scheme,
            }

        if cfg_type == "estimated_csi_single":
            estimator_type = str(matched.channel_estimator.type)
            _estimator_to_variant: dict[str, str] = {
                "last_observation": "myopic",
                "running_min": "conservative",
                "moving_average": "movingavg",
            }
            baseline_variant = _estimator_to_variant.get(estimator_type, estimator_type)
            return {
                **_empty,
                "sub_exp_type": "baseline",
                "baseline_variant": baseline_variant,
                "compression_scheme": compression_scheme,
            }

        return {
            **_empty,
            "sub_exp_type": cfg_type,
            "compression_scheme": compression_scheme,
        }

    metas = df["sub_experiment_name"].apply(_meta)
    df = df.copy()
    for col in (
        "sub_exp_type",
        "backend",
        "surrogate_type",
        "estimator_type",
        "mu",
        "baseline_variant",
        "compression_scheme",
    ):
        df[col] = metas.apply(lambda m, c=col: m.get(c))
    return df


def build_sub_exp_table(
    metrics_dir: Path,
    exp_config: GeneratedOptExperimentConfig,
) -> pd.DataFrame:
    """Build per-sub-experiment performance table for an optimizer experiment.

    Computes four metrics per optimization sub-experiment (sub_exp_type in
    no_csi, csi_aware, baseline):

    - ``avg_utility``: time-averaged weighted accuracy
      (1/T) * Σₜ Σₖ wₖ Aₖ(ηₖ(t)), where T is the slot count from opt_slot
      events and wₖ comes from ``exp_config.tasks``.
    - ``avg_achieved_rps``: mean achieved throughput (rps) per (slot, pipeline)
      pair — compare against ``tasks[pid].throughput_target``.
    - ``avg_delay_ms``: mean throughput-based actual delay (ms) per (slot,
      pipeline) pair — mean(1000 / achieved_rps).
    - ``excess_delay_ms``: time-averaged sum of per-pipeline delay excess —
      (1/T) * Σₜ Σₖ (1000/achieved_rps − 1000/target_rps) in ms.
      Negative = system met the target; positive = throughput shortfall.
    - ``delay_ratio``: mean normalised delay — mean(target_rps / achieved_rps).
      Values < 1 mean the target was exceeded; > 1 means shortfall.

    Args:
        metrics_dir: Directory containing NDJSON metric event files.
        exp_config: Parsed experiment config; provides per-pipeline
            ``task_weight`` (wₖ) values.

    Returns:
        One row per sub_experiment_name with the four metric columns, ``n_slots``,
        and _meta columns (sub_exp_type, backend, surrogate_type, etc.).
        Rows for profiling and accuracy_model sub-experiments are excluded.
    """
    exp_id = _exp_id(metrics_dir)

    # task weights: pipeline_id → wₖ
    task_weights: dict[str, float] = {
        pid: cfg.task_weight for pid, cfg in exp_config.tasks.items()
    }

    _GRP = ["sub_experiment_name", "run_index"]

    # ------------------------------------------------------------------
    # T per (sub-experiment, run): count distinct slot_ids from opt_slot events
    # ------------------------------------------------------------------
    slots = load_opt_slots(metrics_dir / "opt_slot.ndjson", exp_id)
    if slots.empty:
        return pd.DataFrame()

    slots_filtered = slots[slots["sub_experiment_name"].notna()]
    n_slots: pd.Series = (
        slots_filtered.groupby(_GRP)["slot_id"].nunique().rename("n_slots")
    )

    # ------------------------------------------------------------------
    # avg_utility = (1/T) * Σₜ Σₖ wₖ Aₖ(ηₖ(t))
    # ------------------------------------------------------------------
    _, acc_slot = load_task_accuracy(metrics_dir / "task_accuracy.ndjson", exp_id)

    avg_utility: pd.Series = pd.Series(dtype=float, name="avg_utility")
    if not acc_slot.empty and "sub_experiment_name" in acc_slot.columns:
        acc = acc_slot[acc_slot["sub_experiment_name"].notna()].copy()
        acc["w_k"] = acc["pipeline_id"].map(task_weights).fillna(1.0)
        weighted_sum = (
            (acc["w_k"] * acc["accuracy"])
            .groupby([acc["sub_experiment_name"], acc["run_index"]])
            .sum()
        )
        # Divide by T from opt_slot (authoritative slot count)
        avg_utility = (weighted_sum / n_slots).rename("avg_utility")

    # ------------------------------------------------------------------
    # Delay metrics from throughput_constraint events
    # D_act,k(t) = 1000 / achieved_rps  (ms per inference, throughput-based)
    # ------------------------------------------------------------------
    constraints = load_throughput_constraints(
        metrics_dir / "throughput_constraint.ndjson", exp_id
    )

    avg_delay: pd.Series = pd.Series(dtype=float, name="avg_delay_ms")
    avg_achieved_rps: pd.Series = pd.Series(dtype=float, name="avg_achieved_rps")
    excess_delay: pd.Series = pd.Series(dtype=float, name="excess_delay_ms")
    delay_ratio: pd.Series = pd.Series(dtype=float, name="delay_ratio")

    if not constraints.empty:
        c = constraints.copy()
        c["_actual_delay_ms"] = 1000.0 / c["achieved_rps"]
        c["_excess"] = c["_actual_delay_ms"] - 1000.0 / c["target_rps"]
        c["_ratio"] = c["_actual_delay_ms"] * c["target_rps"] / 1000.0

        # avg_delay_ms: mean actual delay (ms) over all (slot, pipeline) pairs
        avg_delay = c.groupby(_GRP)["_actual_delay_ms"].mean().rename("avg_delay_ms")

        # avg_achieved_rps: mean achieved throughput (rps) over all
        # (slot, pipeline) pairs.  Reported alongside target_rps in tasks config.
        avg_achieved_rps = (
            c.groupby(_GRP)["achieved_rps"].mean().rename("avg_achieved_rps")
        )

        # delay_ratio: mean(target_rps / achieved_rps); <1 means target exceeded
        delay_ratio = c.groupby(_GRP)["_ratio"].mean().rename("delay_ratio")

        # excess_delay_ms = (1/T) * Σₜ Σₖ (D_act,k(t) − 1/R_k(t))
        excess_delay = (
            c.groupby([*_GRP, "slot_id"])["_excess"].sum().groupby(_GRP).sum() / n_slots
        ).rename("excess_delay_ms")

    # ------------------------------------------------------------------
    # Combine on (sub_experiment_name, run_index) index
    # ------------------------------------------------------------------
    result: pd.DataFrame = n_slots.to_frame()
    for series in (avg_utility, avg_achieved_rps, avg_delay, excess_delay, delay_ratio):
        if not series.empty:
            result = result.join(series, how="left")

    result = result.reset_index()
    result = _inject_meta_from_config(result, exp_config)

    # Keep only optimization sub-experiments (drop profiling, accuracy_model)
    result = result[
        result["sub_exp_type"].isin(["no_csi", "csi_aware", "baseline"])
    ].copy()

    # Append _run{n} suffix to sub_experiment_name so each re-run is a
    # distinct labelled row.  Do this after metadata injection so the
    # config-based name matching is unaffected.
    result["sub_experiment_name"] = result.apply(
        lambda r: f"{r['sub_experiment_name']}_run{r['run_index']}", axis=1
    )
    result = result.drop(columns=["run_index"])

    return result


def build_accuracy_sweep_table(
    am_dir: Path,
    stein_dir: Path,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    """Load accuracy model sweep training data from events or cached artifacts.

    Sweep events (``TaskAccuracyEvent`` with ``slot_id=None``) are only emitted
    when the accuracy model is trained fresh.  If the model was loaded from a
    cached ``.pkl``, no events are emitted.  When ``model_dir`` is supplied and
    no event data is found, training data is recovered from the ``X_train`` /
    ``y_train`` arrays stored inside each ``.pkl`` artifact.

    Args:
        am_dir: AM experiment directory.
        stein_dir: Stein experiment directory.
        model_dir: Directory of saved ``.pkl`` accuracy model artifacts.
            Used as fallback when ndjson events are absent.

    Returns:
        DataFrame with columns ``eta_*``, ``accuracy``, ``surrogate_type``.
    """
    parts: list[pd.DataFrame] = []
    for directory in (am_dir, stein_dir):
        exp_id = _exp_id(directory)
        path = directory / "task_accuracy.ndjson"
        if not path.exists():
            continue
        sweep_df, _ = load_task_accuracy(path, exp_id)
        if not sweep_df.empty:
            parts.append(sweep_df)

    if parts:
        result = pd.concat(parts, ignore_index=True)
        result = inject_meta(result)
        if "sub_exp_type" in result.columns:
            result = result[result["sub_exp_type"] == "accuracy_model"].copy()
        return result

    # Fallback: recover training data from pkl artifacts
    if model_dir is None or not model_dir.exists():
        return pd.DataFrame()

    import pickle  # noqa: PLC0415

    known_surrogates = {"gbm", "poly2", "poly3", "rf", "mlp"}
    seen: set[str] = set()
    artifact_parts: list[pd.DataFrame] = []

    for pkl_path in sorted(model_dir.glob("*.pkl")):
        surrogate = next(
            (s for s in known_surrogates if f"__{s}__" in pkl_path.name),
            None,
        )
        if surrogate is None or surrogate in seen:
            continue
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)  # noqa: S301
        except Exception as exc:
            logger.warning("Could not load %s: %s", pkl_path.name, exc)
            continue
        X = d.get("X_train")
        y = d.get("y_train")
        if X is None or y is None:
            logger.warning("%s has no X_train/y_train — skipping", pkl_path.name)
            continue
        seen.add(surrogate)
        json_path = pkl_path.with_suffix(".json")
        link_names: list[str] = []
        if json_path.exists():
            import json as _json  # noqa: PLC0415

            try:
                meta = _json.loads(json_path.read_text())
                flow: list[str] = meta.get("flow", [])
                link_names = [
                    f"eta_{flow[i]}_{flow[i + 1]}" for i in range(len(flow) - 1)
                ]
            except Exception:
                pass
        if len(link_names) != X.shape[1]:
            link_names = [f"eta_link{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=link_names)
        df["accuracy"] = y
        df["surrogate_type"] = surrogate
        artifact_parts.append(df)

    if not artifact_parts:
        return pd.DataFrame()

    return pd.concat(artifact_parts, ignore_index=True)
