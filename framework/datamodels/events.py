from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Metric event types emitted during distributed inference."""

    FORWARD_PASS = "forward_pass"
    COMPRESS = "compress"
    DECOMPRESS = "decompress"
    SEND = "send"
    RESULT = "result"
    LINK_PROBE = "link_probe"
    END_TO_END = "end_to_end"


class BaseEvent(BaseModel):
    """Common fields shared by all metric events."""

    event_type: EventType
    experiment_id: str
    run_id: str
    timestamp: float = Field(default_factory=time.time)
    request_id: str | None = None


class ForwardPassEvent(BaseEvent):
    """Emitted after a node completes a forward pass through its partitions."""

    event_type: Literal[EventType.FORWARD_PASS] = EventType.FORWARD_PASS
    node: str
    duration_ms: float
    device: str


class CompressEvent(BaseEvent):
    """Emitted after a node compresses an outgoing activation tensor."""

    event_type: Literal[EventType.COMPRESS] = EventType.COMPRESS
    node: str
    method: str
    rate: float
    input_bytes: int
    output_bytes: int
    duration_ms: float


class DecompressEvent(BaseEvent):
    """Emitted after a node decompresses a received activation tensor."""

    event_type: Literal[EventType.DECOMPRESS] = EventType.DECOMPRESS
    node: str
    method: str
    duration_ms: float


class SendEvent(BaseEvent):
    """Emitted after a node finishes sending an activation payload."""

    event_type: Literal[EventType.SEND] = EventType.SEND
    from_node: str
    to_node: str
    payload_bytes: int
    duration_ms: float


class ResultEvent(BaseEvent):
    """Emitted by the orchestrator when a completed inference result is received."""

    event_type: Literal[EventType.RESULT] = EventType.RESULT
    predicted: Any
    actual: Any


class EndToEndLatencyEvent(BaseEvent):
    """Emitted by the orchestrator after a batch completes the full pipeline.

    Measures wall-clock time from when the batch was sent to the first node
    until the result was received back at the orchestrator callback server.
    """

    event_type: Literal[EventType.END_TO_END] = EventType.END_TO_END
    duration_ms: float
    batch_size: int


class LinkProbeEvent(BaseEvent):
    """Emitted by the background link prober between experiment runs."""

    event_type: Literal[EventType.LINK_PROBE] = EventType.LINK_PROBE
    from_node: str
    to_node: str
    rtt_ms: float
    throughput_mbps: float | None = None


MetricEvent = Annotated[
    ForwardPassEvent
    | CompressEvent
    | DecompressEvent
    | SendEvent
    | ResultEvent
    | LinkProbeEvent
    | EndToEndLatencyEvent,
    Field(discriminator="event_type"),
]
