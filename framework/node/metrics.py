from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class MetricsEmitter:
    """Async metrics emitter that streams events to the central metrics server.

    Events are placed on an in-memory asyncio queue by the critical path and
    drained by a background worker that batches and POSTs them to the metrics
    server.  The critical path never blocks on network I/O.

    If the queue is full, incoming events are dropped with a warning.  If the
    metrics server is unavailable, the worker retries with exponential backoff
    before dropping the batch.
    """

    def __init__(
        self,
        server_url: str,
        buffer_size: int = 1000,
        batch_size: int = 50,
        flush_interval_s: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        """Initialise the emitter.

        Args:
            server_url: Base URL of the metrics server (e.g. ``http://metrics:9100``).
            buffer_size: Maximum number of events to buffer before dropping.
            batch_size: Maximum number of events to send in a single request.
            flush_interval_s: How often to flush the queue even when not full.
            max_retries: Number of send attempts before dropping a batch.
        """
        self._server_url = server_url.rstrip("/")
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._max_retries = max_retries
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=buffer_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._stopped = False

    def emit(self, event: BaseEvent) -> None:
        """Enqueue a metric event for async transmission.

        Non-blocking.  If the buffer is full the event is dropped and a
        warning is logged.

        Args:
            event: The metric event to emit.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "Metrics buffer full (%d events), dropping %s event",
                self._buffer_size,
                event.event_type.value,
            )

    async def start(self) -> None:
        """Start the background worker task."""
        self._stopped = False
        self._worker_task = asyncio.create_task(self._worker(), name="metrics-worker")
        logger.info("Metrics emitter started (server=%s)", self._server_url)

    async def stop(self) -> None:
        """Stop the worker and flush remaining events."""
        self._stopped = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self._flush_remaining()
        logger.info("Metrics emitter stopped")

    async def _worker(self) -> None:
        while not self._stopped:
            batch = await self._collect_batch()
            if batch:
                await self._send_with_retry(batch)

    async def _collect_batch(self) -> list[BaseEvent]:
        """Collect up to batch_size events within flush_interval_s."""
        batch: list[BaseEvent] = []
        deadline = asyncio.get_event_loop().time() + self._flush_interval_s
        while len(batch) < self._batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(event)
            except TimeoutError:
                break
        return batch

    async def _flush_remaining(self) -> None:
        """Drain and send all events still in the queue."""
        batch: list[BaseEvent] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            if len(batch) >= self._batch_size:
                await self._send_with_retry(batch)
                batch = []
        if batch:
            await self._send_with_retry(batch)

    async def _send_with_retry(self, batch: list[BaseEvent]) -> None:
        """Send a batch of events to the metrics server with exponential backoff."""
        payload = [e.model_dump(mode="json") for e in batch]
        delay = 1.0
        # print(f'Metrics server url: {self._server_url}')
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        f"{self._server_url}/metrics",
                        json=payload,
                    )
                    resp.raise_for_status()
                return
            except Exception as exc:
                if attempt == self._max_retries:
                    logger.warning(
                        "Dropping batch of %d events after %d failed attempts: %s",
                        len(batch),
                        self._max_retries,
                        exc,
                    )
                else:
                    logger.debug(
                        "Metrics send attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt,
                        self._max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
