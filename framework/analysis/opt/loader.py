"""Raw NDJSON loaders for optimizer experiment metric files.

Each function loads one NDJSON file type and returns a flat DataFrame.
Sub-experiment metadata columns are NOT injected here; call
``tables.build_*`` functions which inject metadata after the final join.

All loaders are tolerant of missing files (return empty DataFrame).
When a metrics file contains data from multiple experiment runs, each loader
assigns a ``run_index`` column (1-based) per sub-experiment by detecting
slot counter resets: a new run is recorded whenever slot_id falls below the
maximum slot_id seen so far within that sub-experiment.  This is exact for
experiments with T > 1 slots; T = 1 is indistinguishable from a single run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from framework.analysis._common import load_ndjson


def _assign_run_index(
    df: pd.DataFrame, sub_exp_col: str, slot_col: str
) -> pd.DataFrame:
    """Add a 1-based ``run_index`` column detecting slot counter resets.

    For each sub-experiment group (rows in original file order), increments
    the run counter whenever ``slot_id`` falls strictly below the running
    maximum seen so far within that group, which signals a reset from a
    re-run.  Rows with null slot_id keep the current run counter value.

    Args:
        df: DataFrame with ``sub_exp_col`` and ``slot_col`` columns.
        sub_exp_col: Column name for the sub-experiment identifier.
        slot_col: Column name for the slot index.

    Returns:
        Copy of ``df`` with a ``run_index`` (int) column appended.
    """
    if df.empty:
        return df.assign(run_index=pd.Series(dtype=int))

    # Map original index label → run_index so we can handle arbitrary
    # row ordering from groupby without losing positional alignment.
    run_map: dict = {}
    for _, group in df.groupby(sub_exp_col, sort=False):
        run_idx = 1
        max_slot = -1
        for idx, slot in zip(group.index, group[slot_col], strict=False):
            if pd.notna(slot):
                s = int(slot)
                if s < max_slot:
                    run_idx += 1
                    max_slot = s
                else:
                    max_slot = max(max_slot, s)
            run_map[idx] = run_idx

    df = df.copy()
    df["run_index"] = df.index.map(run_map)
    return df


# Regex for parsing slot_id out of task_id strings.
# Task IDs from opt_runner have format "{run_id}_s{slot_id}_{pipeline_id}_{batch}_{uuid}",
# so the slot id is in the middle, not at the end.
_SLOT_ID_RE = re.compile(r"_s(\d+)_")


def _safe_load(path: Path) -> list[dict]:
    """Load NDJSON lines, returning an empty list if the file is missing."""
    if not path.exists():
        return []
    return list(load_ndjson(path))


def _link_col(link_id: str, prefix: str) -> str:
    """'A-B', 'eta' → 'eta_A_B'."""
    return f"{prefix}_{link_id.replace('-', '_')}"


def load_opt_slots(path: Path, experiment_id: str) -> pd.DataFrame:
    """Load opt_slot.ndjson → one row per (slot_id, pipeline_id).

    Explodes the nested per-pipeline and per-link dicts into flat columns.
    c_hat columns are converted from bps to Mbps for consistency with
    link_probe throughput values.

    Args:
        path: Path to opt_slot.ndjson.
        experiment_id: Experiment directory name used as the identifier.

    Returns:
        DataFrame with columns: experiment_id, sub_experiment_name, slot_id,
        pipeline_id, lambda, d_excess, throughput_shortfall, eta_{link},
        mean_eta, c_hat_{link} (Mbps), mean_c_hat (Mbps), infeasible,
        solve_time_ms, timestamp.
    """
    rows = []
    for event in _safe_load(path):
        sub_exp = event.get("sub_experiment_name")
        slot_id = event["slot_id"]
        infeasible = event.get("infeasible", False)
        solve_time_ms = event["solve_time_ms"]
        timestamp = event["timestamp"]
        c_hat_bps = event.get("c_hat_per_link", {})
        lambda_per_task = event.get("lambda_per_task", {})
        d_excess_per_task = event.get("d_excess_per_task", {})
        tp_shortfall = event.get("throughput_shortfall_per_pipeline", {})

        for pid, eta_dict in event.get("eta_per_pipeline_per_link", {}).items():
            row: dict = {
                "experiment_id": experiment_id,
                "sub_experiment_name": sub_exp,
                "slot_id": slot_id,
                "pipeline_id": pid,
                "infeasible": infeasible,
                "solve_time_ms": solve_time_ms,
                "timestamp": timestamp,
                "lambda": lambda_per_task.get(pid, float("nan")),
                "d_excess": d_excess_per_task.get(pid, float("nan")),
                "throughput_shortfall": tp_shortfall.get(pid, float("nan")),
            }
            for link_id, eta in eta_dict.items():
                row[_link_col(link_id, "eta")] = eta
            row["mean_eta"] = (
                sum(eta_dict.values()) / len(eta_dict) if eta_dict else float("nan")
            )
            # Convert bps → Mbps
            for link_id, bps in c_hat_bps.items():
                row[_link_col(link_id, "c_hat")] = bps / 1e6
            row["mean_c_hat"] = (
                sum(c_hat_bps.values()) / len(c_hat_bps) / 1e6
                if c_hat_bps
                else float("nan")
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    return _assign_run_index(df, "sub_experiment_name", "slot_id")


def load_throughput_constraints(path: Path, experiment_id: str) -> pd.DataFrame:
    """Load throughput_constraint.ndjson → one row per (slot_id, pipeline_id).

    Args:
        path: Path to throughput_constraint.ndjson.
        experiment_id: Experiment identifier.

    Returns:
        DataFrame with columns: experiment_id, sub_experiment_name, slot_id,
        pipeline_id, target_rps, achieved_rps, satisfied, violation_magnitude,
        cumulative_violations.
    """
    rows = []
    for event in _safe_load(path):
        rows.append(
            {
                "experiment_id": experiment_id,
                "sub_experiment_name": event.get("sub_experiment_name"),
                "slot_id": event["slot_id"],
                "pipeline_id": event["pipeline_id"],
                "target_rps": event["target_rps"],
                "achieved_rps": event["achieved_rps"],
                "satisfied": event["satisfied"],
                "violation_magnitude": event["violation_magnitude"],
                "cumulative_violations": event["cumulative_violations"],
            }
        )
    df = pd.DataFrame(rows)
    return _assign_run_index(df, "sub_experiment_name", "slot_id")


def load_task_accuracy(
    path: Path, experiment_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load task_accuracy.ndjson, split into sweep and slot DataFrames.

    Events with slot_id=None come from accuracy_model sweep sub-experiments.
    Events with slot_id set come from optimization loop slots.

    Args:
        path: Path to task_accuracy.ndjson.
        experiment_id: Experiment identifier.

    Returns:
        Tuple (sweep_df, slot_df).  sweep_df has no slot_id column.
        slot_df has a slot_id column.  Both include eta_{link} columns
        (flattened from eta_per_link), mean_eta, accuracy, n_samples.
    """
    sweep_rows: list[dict] = []
    slot_rows: list[dict] = []

    for event in _safe_load(path):
        eta_per_link = event.get("eta_per_link") or {}
        base: dict = {
            "experiment_id": experiment_id,
            "sub_experiment_name": event.get("sub_experiment_name"),
            "pipeline_id": event["pipeline_id"],
            "compression_method": event["compression_method"],
            "compression_rate": event["compression_rate"],
            "accuracy": event["accuracy"],
            "n_samples": event["n_samples"],
        }
        for link_id, eta in eta_per_link.items():
            base[_link_col(link_id, "eta")] = eta
        if eta_per_link:
            base["mean_eta"] = sum(eta_per_link.values()) / len(eta_per_link)
        else:
            base["mean_eta"] = event.get("compression_rate", float("nan"))

        if event.get("slot_id") is None:
            sweep_rows.append(base)
        else:
            slot_rows.append({**base, "slot_id": event["slot_id"]})

    slot_df = pd.DataFrame(slot_rows)
    slot_df = _assign_run_index(slot_df, "sub_experiment_name", "slot_id")
    return pd.DataFrame(sweep_rows), slot_df


def load_run_throughput(path: Path, experiment_id: str) -> pd.DataFrame:
    """Load run_throughput.ndjson → one row per (slot_id, pipeline_id).

    Args:
        path: Path to run_throughput.ndjson.
        experiment_id: Experiment identifier.

    Returns:
        DataFrame with columns: experiment_id, sub_experiment_name, slot_id,
        pipeline_id, tasks_per_second, p50_ms, p90_ms, p99_ms, total_tasks,
        wall_time_s.
    """
    rows = []
    for event in _safe_load(path):
        sub_exp = event.get("sub_experiment_name")
        slot_id = event.get("slot_id")
        wall_time_s = event.get("wall_time_s", float("nan"))
        for pid, stats in event.get("per_pipeline", {}).items():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "sub_experiment_name": sub_exp,
                    "slot_id": slot_id,
                    "pipeline_id": pid,
                    "tasks_per_second": stats["tasks_per_second"],
                    "p50_ms": stats["p50_ms"],
                    "p90_ms": stats["p90_ms"],
                    "p99_ms": stats["p99_ms"],
                    "total_tasks": stats["tasks"],
                    "wall_time_s": wall_time_s,
                }
            )
    df = pd.DataFrame(rows)
    return _assign_run_index(df, "sub_experiment_name", "slot_id")


def load_link_probes(path: Path, experiment_id: str) -> pd.DataFrame:
    """Load link_probe.ndjson → one row per (slot_id, link_id), slot probes only.

    Background probes (slot_id=None) are excluded.  throughput_mbps is
    already in Mbps, matching the c_hat columns in the slot table.

    Args:
        path: Path to link_probe.ndjson.
        experiment_id: Experiment identifier.

    Returns:
        DataFrame with columns: experiment_id, sub_experiment_name, slot_id,
        link_id, throughput_mbps, rtt_ms.
    """
    rows = []
    for event in _safe_load(path):
        if event.get("slot_id") is None:
            continue
        rows.append(
            {
                "experiment_id": experiment_id,
                "sub_experiment_name": event.get("sub_experiment_name"),
                "slot_id": event["slot_id"],
                "link_id": f"{event['from_node']}-{event['to_node']}",
                "throughput_mbps": event.get("throughput_mbps"),
                "rtt_ms": event["rtt_ms"],
            }
        )
    df = pd.DataFrame(rows)
    return _assign_run_index(df, "sub_experiment_name", "slot_id")


def load_sub_experiments(path: Path, experiment_id: str) -> pd.DataFrame:
    """Load sub_experiment.ndjson → one row per sub-experiment completion.

    Args:
        path: Path to sub_experiment.ndjson.
        experiment_id: Experiment identifier.

    Returns:
        DataFrame with columns: experiment_id, sub_experiment_name,
        duration_s, n_runs, timestamp.
    """
    rows = []
    for event in _safe_load(path):
        rows.append(
            {
                "experiment_id": experiment_id,
                "sub_experiment_name": event.get("sub_experiment_name"),
                "duration_s": event["duration_s"],
                "n_runs": event["n_runs"],
                "timestamp": event["timestamp"],
            }
        )
    return pd.DataFrame(rows)


def load_task_e2e(path: Path, experiment_id: str) -> pd.DataFrame:
    """Load task_e2e.ndjson → one row per completed task.

    Profiling and warmup events (sub_experiment_name=None) are excluded.
    slot_id is parsed from the ``_s{N}`` suffix in task_id where present.

    Args:
        path: Path to task_e2e.ndjson.
        experiment_id: Experiment identifier.

    Returns:
        DataFrame with columns: experiment_id, sub_experiment_name, slot_id
        (int or None), pipeline_id, task_id, latency_ms, timestamp.
    """
    rows = []
    for event in _safe_load(path):
        sub_exp = event.get("sub_experiment_name")
        if sub_exp is None:
            continue
        task_id = event.get("task_id", "")
        m = _SLOT_ID_RE.search(task_id)
        slot_id = int(m.group(1)) if m else None
        rows.append(
            {
                "experiment_id": experiment_id,
                "sub_experiment_name": sub_exp,
                "slot_id": slot_id,
                "pipeline_id": event["pipeline_id"],
                "task_id": task_id,
                "latency_ms": event["latency_ms"],
                "timestamp": event["timestamp"],
            }
        )
    return pd.DataFrame(rows)


def load_task_node_timing(path: Path, experiment_id: str) -> pd.DataFrame:
    """Load task_node_timing.ndjson → one row per (task_id, node_id).

    Derives per-stage durations from absolute timestamps.
    Note: TaskNodeTimingEvent does not carry sub_experiment_name; join on
    task_id with load_task_e2e() to add sub-experiment grouping.

    Args:
        path: Path to task_node_timing.ndjson.
        experiment_id: Experiment identifier.

    Returns:
        DataFrame with columns: experiment_id, pipeline_id, task_id, node_id,
        queue_length_at_enqueue, queue_wait_ms, compute_ms, compress_ms,
        send_ms, total_node_ms.
    """
    rows = []
    for event in _safe_load(path):
        t_enq = event["enqueue_time"]
        t_cs = event["compute_start"]
        t_ce = event["compute_end"]
        t_ps = event["compress_start"]
        t_pe = event["compress_end"]
        t_sent = event["sent_time"]
        rows.append(
            {
                "experiment_id": experiment_id,
                "pipeline_id": event["pipeline_id"],
                "task_id": event["task_id"],
                "node_id": event["node_id"],
                "queue_length_at_enqueue": event["queue_length_at_enqueue"],
                "queue_wait_ms": (t_cs - t_enq) * 1e3,
                "compute_ms": (t_ce - t_cs) * 1e3,
                "compress_ms": (t_pe - t_ps) * 1e3,
                "send_ms": (t_sent - t_pe) * 1e3,
                "total_node_ms": (t_sent - t_enq) * 1e3,
            }
        )
    return pd.DataFrame(rows)
