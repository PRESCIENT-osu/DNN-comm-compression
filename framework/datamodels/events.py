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
    # Multi-model event types
    TASK_NODE_TIMING = "task_node_timing"
    QUEUE_SNAPSHOT = "queue_snapshot"
    TASK_E2E = "task_e2e"
    RUN_THROUGHPUT = "run_throughput"


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
    """Emitted by the orchestrator when a completed inference result is received.

    For classification (ResNet, MMLU): predicted and actual carry lists of
    class indices.  For perplexity (WikiText-2): nll_sum and token_count carry
    the per-batch NLL accumulator values; predicted and actual are omitted.
    """

    event_type: Literal[EventType.RESULT] = EventType.RESULT
    predicted: Any = None
    actual: Any = None
    nll_sum: float | None = None
    token_count: int | None = None


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


class PipelineThroughputStats(BaseModel):
    """Per-pipeline throughput and latency statistics for one sweep run."""

    tasks: int
    tasks_per_second: float
    p50_ms: float
    p90_ms: float
    p99_ms: float


class TaskNodeTimingEvent(BaseEvent):
    """Emitted once per task per node in the multi-model pipeline.

    Records timing at each stage of node processing so queue wait, compute
    time, and compression time can be derived per task.
    """

    event_type: Literal[EventType.TASK_NODE_TIMING] = EventType.TASK_NODE_TIMING
    pipeline_id: str
    task_id: str
    node_id: str
    enqueue_time: float
    queue_length_at_enqueue: int
    compute_start: float
    compute_end: float
    compress_start: float
    compress_end: float
    sent_time: float


class QueueSnapshotEvent(BaseEvent):
    """Emitted at each enqueue and dequeue on a multi-model node.

    Captures queue depth at the moment of the event, broken down by pipeline,
    enabling analysis of per-pipeline queue occupancy over time.
    """

    event_type: Literal[EventType.QUEUE_SNAPSHOT] = EventType.QUEUE_SNAPSHOT
    node_id: str
    queue_length: int
    queue_by_pipeline: dict[str, int]
    trigger: Literal["enqueue", "dequeue"]
    task_id: str
    pipeline_id: str


class TaskE2EEvent(BaseEvent):
    """Emitted by the multi-model orchestrator when a task completes end-to-end.

    Measures wall-clock time from when the task was submitted to the first node
    until the result was received at the orchestrator callback.
    """

    event_type: Literal[EventType.TASK_E2E] = EventType.TASK_E2E
    pipeline_id: str
    task_id: str
    submit_time: float
    receive_time: float
    latency_ms: float


class RunThroughputEvent(BaseEvent):
    """Emitted by the multi-model orchestrator at the end of each sweep run.

    Captures aggregate throughput and per-pipeline latency percentiles for the
    completed run.
    """

    event_type: Literal[EventType.RUN_THROUGHPUT] = EventType.RUN_THROUGHPUT
    wall_time_s: float
    total_tasks: int
    tasks_per_second: float
    per_pipeline: dict[str, PipelineThroughputStats]


MetricEvent = Annotated[
    ForwardPassEvent
    | CompressEvent
    | DecompressEvent
    | SendEvent
    | ResultEvent
    | LinkProbeEvent
    | EndToEndLatencyEvent
    | TaskNodeTimingEvent
    | QueueSnapshotEvent
    | TaskE2EEvent
    | RunThroughputEvent,
    Field(discriminator="event_type"),
]
