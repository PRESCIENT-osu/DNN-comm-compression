from __future__ import annotations

from pydantic import BaseModel

from framework.datamodels.experiment import CompressionMethod


class InferRequest(BaseModel):
    """Payload for POST /infer on a compute node."""

    task_id: str
    callback_url: str
    experiment_id: str
    run_id: str
    data: str  # base64-encoded bytes (compressed activation or raw input)
    attention_mask: str | None = (
        None  # base64-encoded pickled bool tensor [B, L]; MMLU only
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
    data: str  # base64-encoded pickled output tensor


class MultiInferRequest(BaseModel):
    """Payload for POST /infer on a multi-model compute node."""

    task_id: str
    pipeline_id: str
    callback_url: str
    experiment_id: str
    run_id: str
    data: str  # base64-encoded bytes (compressed activation or raw input)


class MultiConfigUpdate(BaseModel):
    """Payload for POST /config on a multi-model compute node."""

    pipeline_id: str
    direction: str  # "incoming" or "outgoing"
    method: CompressionMethod
    rate: float = 0.0
    drain_timeout_s: float = 60.0
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"
