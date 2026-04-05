from __future__ import annotations

import time
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class EventType(StrEnum):
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
    SUB_EXPERIMENT = "sub_experiment"
    # Optimization loop event types
    TASK_ACCURACY = "task_accuracy"
    OPT_SLOT = "opt_slot"
    THROUGHPUT_CONSTRAINT = "throughput_constraint"
    CHANNEL_ESTIMATE_QUALITY = "channel_estimate_quality"


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


class SubExperimentEvent(BaseEvent):
    """Emitted by the orchestrator when a sub-experiment (sweep) completes.

    Captures the wall-clock duration of a full sub-experiment sweep from the
    first run dispatch to the last result received.
    """

    event_type: Literal[EventType.SUB_EXPERIMENT] = EventType.SUB_EXPERIMENT
    sub_experiment_name: str | None
    duration_s: float
    n_runs: int


class TaskAccuracyEvent(BaseEvent):
    """Emitted by the orchestrator at the end of each optimization slot or sweep run.

    Captures the (compression_method, compression_rate) → accuracy mapping for a
    single pipeline.  During optimization loops the slot_id field is set so that
    per-slot accuracy can be joined with OptSlotEvent for Pareto front analysis.
    During sweep sub-experiments slot_id is None.
    """

    event_type: Literal[EventType.TASK_ACCURACY] = EventType.TASK_ACCURACY
    pipeline_id: str
    task_id: str
    compression_method: str
    compression_rate: float  # η scalar summary; use OptSlotEvent for per-link vector
    accuracy: float  # top-1 for classification; negative perplexity for LM
    n_samples: int
    slot_id: int | None = None  # set during optimization loop; None for sweep runs


class OptSlotEvent(BaseEvent):
    """Emitted once per optimization slot by the opt_runner.

    Captures the optimizer's decisions and internal state for that slot,
    enabling downstream convergence analysis of η, λ, and constraint slack.
    Paired with ThroughputConstraintEvent and TaskAccuracyEvent (same slot_id)
    to reconstruct the full (accuracy, throughput) operating point per slot.
    """

    event_type: Literal[EventType.OPT_SLOT] = EventType.OPT_SLOT
    slot_id: int
    eta_per_pipeline_per_link: dict[
        str, dict[str, float]
    ]  # pipeline_id → link_id → η chosen this slot
    lambda_per_task: dict[str, float]  # task_id → dual variable λ_k
    d_excess_per_task: dict[str, float]  # task_id → D_actual - 1/R_k (seconds)
    c_hat_per_link: dict[
        str, float
    ]  # link_id → channel estimate at decision time (bps)
    optimizer_type: str
    solve_time_ms: float  # wall time for optimizer.step()
    infeasible: bool = (
        False  # True when optimizer returned None (infeasible); η fell back to eta_max
    )
    sub_experiment_name: str | None = None


class ThroughputConstraintEvent(BaseEvent):
    """Emitted once per slot per pipeline by the opt_runner.

    Records whether the throughput target R_k was satisfied this slot and by
    how much.  Cumulative violations count is maintained by the runner and
    carried here for convenience so offline analysis does not need to replay
    the event stream in order.
    """

    event_type: Literal[EventType.THROUGHPUT_CONSTRAINT] = (
        EventType.THROUGHPUT_CONSTRAINT
    )
    slot_id: int
    pipeline_id: str
    task_id: str
    target_rps: float  # R_k configured for this pipeline
    achieved_rps: float  # tasks completed this slot / slot wall time
    satisfied: bool
    violation_magnitude: float  # max(0, target - achieved) / target; 0 when satisfied
    cumulative_violations: int  # count of violated slots since experiment start
    sub_experiment_name: str | None = None


class ChannelEstimateQualityEvent(BaseEvent):
    """Emitted by the link prober after each probe interval.

    Compares the channel estimator's prediction at the time of the optimizer
    decision against the throughput actually measured by the probe.  Enables
    downstream analysis of estimator bias, variance, and staleness effects.
    """

    event_type: Literal[EventType.CHANNEL_ESTIMATE_QUALITY] = (
        EventType.CHANNEL_ESTIMATE_QUALITY
    )
    slot_id: (
        int | None
    )  # slot during which the probe was triggered; None if between slots
    from_node: str
    to_node: str
    c_hat_bps: float  # estimator prediction at decision time
    c_actual_bps: float  # throughput measured by this probe
    absolute_error_bps: float  # |c_hat - c_actual|
    relative_error: float  # |c_hat - c_actual| / c_actual; 0 when c_actual == 0
    estimator_type: str  # e.g. "moving_average", "lcb", "last_observation"
    n_observations: int  # number of samples the estimator has accumulated


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
    | RunThroughputEvent
    | SubExperimentEvent
    | TaskAccuracyEvent
    | OptSlotEvent
    | ThroughputConstraintEvent
    | ChannelEstimateQualityEvent,
    Field(discriminator="event_type"),
]
