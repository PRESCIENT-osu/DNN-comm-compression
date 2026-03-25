from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    """Read a NDJSON file and return a list of parsed objects.

    Args:
        path: Path to the NDJSON file.

    Returns:
        List of parsed JSON objects; empty list if the file does not exist.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records(
    metrics_dir: Path, experiment_id: str
) -> dict[str, list[dict[str, Any]]]:
    """Load per-run inference records from result.ndjson in the metrics store.

    Args:
        metrics_dir: Root metrics directory (parent of experiment subdirectories).
        experiment_id: Experiment name used as subdirectory within metrics_dir.

    Returns:
        Mapping of run_id to list of record dicts with ``ground_truth`` and
        ``predicted`` fields suitable for :func:`compute_accuracy`.
    """
    path = metrics_dir / experiment_id / "result.ndjson"
    run_records: dict[str, list[dict[str, Any]]] = {}
    for event in load_ndjson(path):
        run_id = event.get("run_id")
        if run_id is None:
            continue
        # ResultEvent stores ground truth as 'actual'; normalise to 'ground_truth'
        if "actual" in event and "ground_truth" not in event:
            event = dict(event)
            event["ground_truth"] = event["actual"]
        run_records.setdefault(run_id, []).append(event)
    return run_records


def compute_accuracy(records: list[dict[str, Any]]) -> float | None:
    """Compute top-1 accuracy from a list of inference records.

    Args:
        records: List of record dicts with ``ground_truth`` and ``predicted`` lists.

    Returns:
        Accuracy in [0, 1], or None if records is empty.
    """
    correct = 0
    total = 0
    for r in records:
        for gt, pred in zip(r["ground_truth"], r["predicted"], strict=False):
            if gt == pred:
                correct += 1
            total += 1
    if total == 0:
        return None
    return correct / total


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate_metrics(
    metrics_dir: Path, experiment_id: str
) -> dict[str, dict[str, Any]]:
    """Aggregate per-run timing and size metrics from the metrics NDJSON files.

    Reads ``forward_pass.ndjson``, ``compress.ndjson``, ``decompress.ndjson``,
    ``send.ndjson``, and ``end_to_end.ndjson`` from ``metrics_dir/<experiment_id>/``
    and returns mean values grouped by run_id.

    Args:
        metrics_dir: Root metrics directory (parent of experiment subdirectories).
        experiment_id: Experiment name used as subdirectory within metrics_dir.

    Returns:
        Mapping of run_id to a dict with keys:
        ``forward_ms``, ``compress_ms``, ``decompress_ms``, ``send_ms``,
        ``end_to_end_ms``, ``input_bytes``, ``output_bytes`` — all floats or None.
    """
    exp_dir = metrics_dir / experiment_id
    forward_pass = load_ndjson(exp_dir / "forward_pass.ndjson")
    compress = load_ndjson(exp_dir / "compress.ndjson")
    decompress = load_ndjson(exp_dir / "decompress.ndjson")
    send = load_ndjson(exp_dir / "send.ndjson")
    end_to_end = load_ndjson(exp_dir / "end_to_end.ndjson")

    runs: dict[str, dict[str, list[float]]] = {}

    def _ensure(run_id: str) -> dict[str, list[float]]:
        if run_id not in runs:
            runs[run_id] = {
                "forward_ms": [],
                "compress_ms": [],
                "decompress_ms": [],
                "send_ms": [],
                "end_to_end_ms": [],
                "input_bytes": [],
                "output_bytes": [],
            }
        return runs[run_id]

    for ev in forward_pass:
        _ensure(ev["run_id"])["forward_ms"].append(ev["duration_ms"])

    for ev in compress:
        r = _ensure(ev["run_id"])
        r["compress_ms"].append(ev["duration_ms"])
        r["input_bytes"].append(ev["input_bytes"])
        r["output_bytes"].append(ev["output_bytes"])

    for ev in decompress:
        _ensure(ev["run_id"])["decompress_ms"].append(ev["duration_ms"])

    for ev in send:
        _ensure(ev["run_id"])["send_ms"].append(ev["duration_ms"])

    for ev in end_to_end:
        _ensure(ev["run_id"])["end_to_end_ms"].append(ev["duration_ms"])

    return {
        run_id: {
            "forward_ms": _mean(data["forward_ms"]),
            "compress_ms": _mean(data["compress_ms"]),
            "decompress_ms": _mean(data["decompress_ms"]),
            "send_ms": _mean(data["send_ms"]),
            "end_to_end_ms": _mean(data["end_to_end_ms"]),
            "input_bytes": _mean(data["input_bytes"]),
            "output_bytes": _mean(data["output_bytes"]),
        }
        for run_id, data in runs.items()
    }
