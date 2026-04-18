from __future__ import annotations

from pydantic import BaseModel

from framework.datamodels.experiment import CompressionMethod


class InferRequest(BaseModel):
    """Payload for POST /infer on a compute node."""

    task_id: str
    callback_url: str
    experiment_id: str
    run_id: str
    data: bytes  # raw binary (compressed activation or pickled input)
    attention_mask: str | None = (
        None  # base64-encoded pickled bool tensor [B, L]; MMLU only
    )
    input_ids: str | None = (
        None  # base64-encoded pickled int tensor [B, L]; WikiText only
    )
    metric_type: str | None = None  # "perplexity" or "accuracy"; None for ResNet
    answer_token_ids: str | None = (
        None  # comma-separated token IDs for MMLU answer choices
    )


class ConfigUpdate(BaseModel):
    """Payload for POST /config on a compute node."""

    direction: str  # "incoming" or "outgoing"
    method: CompressionMethod
    rate: float = 0.0
    drain_timeout_s: float = 60.0
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"


class ResultPayload(BaseModel):
    """Payload POSTed by the last compute node to the orchestrator callback."""

    task_id: str
    data: bytes  # pickled output tensor (raw binary)


class MultiInferRequest(BaseModel):
    """Payload for POST /infer on a multi-model compute node."""

    task_id: str
    pipeline_id: str
    callback_url: str
    experiment_id: str
    run_id: str
    data: bytes  # raw binary (compressed activation or pickled input)
    attention_mask: str | None = (
        None  # base64-encoded pickled bool tensor [B, L]; MMLU only
    )
    input_ids: str | None = (
        None  # base64-encoded pickled int tensor [B, L]; WikiText only
    )
    metric_type: str | None = None  # "perplexity" or "accuracy"; None for ResNet
    answer_token_ids: str | None = (
        None  # comma-separated token IDs for MMLU answer choices
    )


class MultiConfigUpdate(BaseModel):
    """Payload for POST /config on a multi-model compute node."""

    pipeline_id: str
    direction: str  # "incoming" or "outgoing"
    method: CompressionMethod
    rate: float = 0.0
    drain_timeout_s: float = 60.0
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"
