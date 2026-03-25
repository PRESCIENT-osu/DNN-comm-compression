from __future__ import annotations

from typing import Any


class RunRecord:
    """A single completed inference result for one batch."""

    def __init__(
        self,
        request_id: str,
        batch_idx: int,
        ground_truth: list[int],
        predicted: list[int],
        experiment_id: str,
        run_id: str,
        timestamp: float,
    ) -> None:
        self.request_id = request_id
        self.batch_idx = batch_idx
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.timestamp = timestamp

    def to_dict(self) -> dict[str, Any]:
        """Serialise record to a JSON-compatible dict."""
        return {
            "request_id": self.request_id,
            "batch_idx": self.batch_idx,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
        }


class LlamaRunRecord:
    """A single completed inference result for one Llama batch.

    For perplexity runs (WikiText-2): nll_sum and token_count are populated.
    For accuracy runs (MMLU): ground_truth and predicted are populated.
    """

    def __init__(
        self,
        request_id: str,
        batch_idx: int,
        metric_type: str,
        experiment_id: str,
        run_id: str,
        timestamp: float,
        ground_truth: list[int] | None = None,
        predicted: list[int] | None = None,
        nll_sum: float = 0.0,
        token_count: int = 0,
    ) -> None:
        self.request_id = request_id
        self.batch_idx = batch_idx
        self.metric_type = metric_type
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.timestamp = timestamp
        self.ground_truth = ground_truth or []
        self.predicted = predicted or []
        self.nll_sum = nll_sum
        self.token_count = token_count

    def to_dict(self) -> dict[str, Any]:
        """Serialise record to a JSON-compatible dict."""
        return {
            "request_id": self.request_id,
            "batch_idx": self.batch_idx,
            "metric_type": self.metric_type,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "nll_sum": self.nll_sum,
            "token_count": self.token_count,
        }
