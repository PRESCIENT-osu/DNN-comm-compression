"""Multi-model compute node server.

Hosts partitions for multiple named pipelines on a single node.  Incoming
inference requests from any pipeline are queued into a shared FIFO queue and
processed serially by a single worker coroutine.  This exposes real compute
contention between pipelines, which is measured via per-task timing and
queue-snapshot metrics.

Partition layout per pipeline is declared in the experiment config.  Each
pipeline independently declares its execution flow and compression config.

Environment variables:
    NODE_NAME: Name of this node as declared in the experiment config.
    EXPERIMENT_CONFIG_PATH: Path to the generated multi-model experiment.yaml.
    PARTITIONS_BASE_DIR: Base directory; partitions for model M are loaded
        from ``{PARTITIONS_BASE_DIR}/{M}/{partition_id}.pt``.
    METRICS_SERVER_URL: URL of the metrics ingestion server (default:
        ``http://metrics:9100``).
    PROBE_INTERVAL_S: Seconds between link probe cycles (default: ``30``).
    METRICS_BUFFER_SIZE: Max buffered metric events before blocking (default:
        ``1000``).
    PORT: HTTP port to listen on (default: ``8000``).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import pickle
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from framework.datamodels.api import MultiConfigUpdate, MultiInferRequest
from framework.datamodels.events import (
    CompressEvent,
    DecompressEvent,
    ForwardPassEvent,
    LinkProbeEvent,
    QueueSnapshotEvent,
    SendEvent,
    TaskNodeTimingEvent,
)
from framework.datamodels.experiment import CompressionMethod
from framework.datamodels.multi_experiment import PipelineLinkCompression
from framework.nodes.compute.common.compressor import get_compressor
from framework.nodes.metrics.emitter import MetricsEmitter
from framework.utils.loader import load_multi_experiment_config

logger = logging.getLogger(__name__)

_PROBE_PAYLOAD_BYTES = 100 * 1024  # 100 KB throughput probe payload


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QueueItem:
    """A single inference task waiting in the node's FIFO queue.

    Args:
        request: The incoming inference request.
        enqueue_time: Wall-clock time when the item was enqueued (seconds).
        queue_length_at_enqueue: Number of items already in the queue at
            enqueue time (not including this item).
    """

    request: MultiInferRequest
    enqueue_time: float
    queue_length_at_enqueue: int


class PipelineNodeState:
    """Runtime state for one pipeline on this node.

    Args:
        pipeline_id: Pipeline identifier.
        model: Model type string (e.g. ``"resnet56"``).
        partitions: Loaded partition modules in execution order.
        is_first_node: True if this node is the entry point for this pipeline.
        is_last_node: True if this node is the exit point for this pipeline.
        next_node_url: URL of the next node in this pipeline's flow, or None
            if this is the last node.
        next_node_name: Name of the next node, for metric tagging.
        incoming: Incoming link compression config (decompression applied here).
        outgoing: Outgoing link compression config (compression applied here).
    """

    def __init__(
        self,
        pipeline_id: str,
        model: str,
        partitions: list[Any],
        is_first_node: bool,
        is_last_node: bool,
        next_node_url: str | None,
        next_node_name: str | None,
        incoming: PipelineLinkCompression,
        outgoing: PipelineLinkCompression,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.model = model
        self.partitions = partitions
        self.is_first_node = is_first_node
        self.is_last_node = is_last_node
        self.next_node_url = next_node_url
        self.next_node_name = next_node_name
        self.incoming = incoming
        self.outgoing = outgoing


class MultiNodeState:
    """Mutable runtime state for the multi-model node server.

    Args:
        node_name: Name of this node.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        pipelines: Per-pipeline runtime state keyed by pipeline ID.
        emitter: Async metrics emitter.
        probe_interval_s: Seconds between link probe cycles.
    """

    def __init__(
        self,
        node_name: str,
        device: str,
        pipelines: dict[str, PipelineNodeState],
        emitter: MetricsEmitter,
        probe_interval_s: float,
    ) -> None:
        self.node_name = node_name
        self.device = device
        self.pipelines = pipelines
        self.emitter = emitter
        self.probe_interval_s = probe_interval_s

        self.last_experiment_id: str = "unknown"
        self.last_run_id: str = "none"

        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        # Set when queue is empty and worker is not processing; cleared otherwise.
        self.idle_event: asyncio.Event = asyncio.Event()
        self.idle_event.set()
        self._processing: bool = False

    def queue_by_pipeline(self) -> dict[str, int]:
        """Return a count of queued items per pipeline (excludes item being processed).

        Returns:
            Dict mapping pipeline_id to number of items currently in the queue.
        """
        counts: dict[str, int] = {pid: 0 for pid in self.pipelines}
        for item in list(self.queue._queue):  # type: ignore[attr-defined]
            pid = item.request.pipeline_id
            counts[pid] = counts.get(pid, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app() -> FastAPI:
    """Create and return a configured FastAPI multi-model node server app.

    Returns:
        Configured FastAPI application.
    """
    _state: list[MultiNodeState | None] = [None]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        _state[0] = await _initialize()
        state = _state[0]
        await state.emitter.start()
        asyncio.create_task(_worker_loop(state), name="worker")
        # Start one probe loop per unique outgoing link URL.
        seen_urls: set[str] = set()
        for ps in state.pipelines.values():
            if ps.next_node_url and ps.next_node_url not in seen_urls:
                seen_urls.add(ps.next_node_url)
                asyncio.create_task(
                    _probe_loop(
                        state, ps.next_node_url, ps.next_node_name or "unknown"
                    ),
                    name=f"probe-{ps.next_node_name}",
                )
        logger.info(
            "Multi-model node '%s' ready on device '%s' — pipelines: %s",
            state.node_name,
            state.device,
            list(state.pipelines.keys()),
        )
        yield
        await state.emitter.stop()

    app = FastAPI(lifespan=lifespan)

    # -----------------------------------------------------------------------
    # Inference API
    # -----------------------------------------------------------------------

    @app.post("/infer", status_code=202)
    async def infer(request: MultiInferRequest) -> JSONResponse:
        """Accept an inference task and enqueue it for the worker."""
        state = _state[0]
        assert state is not None
        if request.pipeline_id not in state.pipelines:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown pipeline_id '{request.pipeline_id}'",
            )
        state.last_experiment_id = request.experiment_id
        state.last_run_id = request.run_id

        enqueue_time = time.time()
        queue_len_before = state.queue.qsize()
        item = QueueItem(
            request=request,
            enqueue_time=enqueue_time,
            queue_length_at_enqueue=queue_len_before,
        )
        state.idle_event.clear()
        await state.queue.put(item)

        state.emitter.emit(
            QueueSnapshotEvent(
                experiment_id=request.experiment_id,
                run_id=request.run_id,
                node_id=state.node_name,
                queue_length=queue_len_before + 1,
                queue_by_pipeline=state.queue_by_pipeline(),
                trigger="enqueue",
                task_id=request.task_id,
                pipeline_id=request.pipeline_id,
            )
        )
        return JSONResponse({"task_id": request.task_id, "status": "accepted"})

    # -----------------------------------------------------------------------
    # Management API
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> JSONResponse:
        state = _state[0]
        assert state is not None
        return JSONResponse({"status": "ok", "node": state.node_name})

    @app.get("/status")
    async def status() -> JSONResponse:
        state = _state[0]
        assert state is not None
        return JSONResponse(
            {
                "node": state.node_name,
                "queue_length": state.queue.qsize(),
                "processing": state._processing,
                "device": state.device,
                "pipelines": {
                    pid: {
                        "model": ps.model,
                        "incoming": ps.incoming.compression.value,
                        "outgoing": ps.outgoing.compression.value,
                    }
                    for pid, ps in state.pipelines.items()
                },
            }
        )

    @app.post("/config")
    async def update_config(update: MultiConfigUpdate) -> JSONResponse:
        """Update the compression config for one pipeline direction on this node.

        Waits for the queue to fully drain before applying the update.
        """
        state = _state[0]
        assert state is not None
        if update.pipeline_id not in state.pipelines:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown pipeline_id '{update.pipeline_id}'",
            )
        if update.direction not in ("incoming", "outgoing"):
            raise HTTPException(
                status_code=400,
                detail=f"direction must be 'incoming' or 'outgoing', got '{update.direction}'",
            )
        try:
            await asyncio.wait_for(
                state.idle_event.wait(), timeout=update.drain_timeout_s
            )
        except TimeoutError:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Timed out waiting for queue to drain after "
                    f"{update.drain_timeout_s}s"
                ),
            ) from None

        new_compression = PipelineLinkCompression(
            compression=update.method,
            rate=update.rate if update.method != CompressionMethod.NONE else None,
            outlier_precision=update.outlier_precision,
            regular_precision=update.regular_precision,
        )
        ps = state.pipelines[update.pipeline_id]
        if update.direction == "incoming":
            ps.incoming = new_compression
        else:
            ps.outgoing = new_compression

        logger.info(
            "Config updated: pipeline=%s %s → method=%s rate=%s",
            update.pipeline_id,
            update.direction,
            update.method.value,
            update.rate,
        )
        return JSONResponse(
            {
                "status": "ok",
                "pipeline_id": update.pipeline_id,
                "direction": update.direction,
                "method": update.method.value,
                "rate": update.rate,
            }
        )

    # -----------------------------------------------------------------------
    # Probe API
    # -----------------------------------------------------------------------

    @app.get("/probe")
    async def probe_rtt() -> JSONResponse:
        return JSONResponse({"status": "ok", "timestamp": time.time()})

    @app.post("/probe")
    async def probe_throughput(request: Request) -> JSONResponse:
        body = await request.body()
        return JSONResponse({"received_bytes": len(body), "timestamp": time.time()})

    return app


# ---------------------------------------------------------------------------
# Background: single FIFO worker
# ---------------------------------------------------------------------------


async def _worker_loop(state: MultiNodeState) -> None:
    """Single worker coroutine — dequeues and processes one task at a time.

    Emits a QueueSnapshotEvent at dequeue and a TaskNodeTimingEvent after
    each task completes.

    Args:
        state: Shared multi-model node state.
    """
    while True:
        item = await state.queue.get()
        state._processing = True
        try:
            await _process_task(state, item)
        except Exception:
            logger.exception(
                "Unhandled error processing task '%s'", item.request.task_id
            )
        finally:
            state._processing = False
            state.queue.task_done()
            if state.queue.empty():
                state.idle_event.set()


async def _process_task(state: MultiNodeState, item: QueueItem) -> None:
    """Execute one inference task from the queue.

    Records per-stage timing and emits TaskNodeTimingEvent on completion.

    Args:
        state: Shared multi-model node state.
        item: Dequeued task item.
    """
    request = item.request
    ps = state.pipelines[request.pipeline_id]

    compute_start = time.time()

    state.emitter.emit(
        QueueSnapshotEvent(
            experiment_id=request.experiment_id,
            run_id=request.run_id,
            node_id=state.node_name,
            queue_length=state.queue.qsize(),
            queue_by_pipeline=state.queue_by_pipeline(),
            trigger="dequeue",
            task_id=request.task_id,
            pipeline_id=request.pipeline_id,
        )
    )

    try:
        raw_bytes = base64.b64decode(request.data)

        # Decompress incoming activation (skip for first node in this pipeline).
        if ps.is_first_node:
            tensor: torch.Tensor = pickle.loads(raw_bytes)
        else:
            t0 = time.perf_counter()
            compressor = get_compressor(
                ps.incoming.compression,
                outlier_precision=ps.incoming.outlier_precision,
                regular_precision=ps.incoming.regular_precision,
            )
            tensor = compressor.decompress(raw_bytes, state.device)
            state.emitter.emit(
                DecompressEvent(
                    experiment_id=request.experiment_id,
                    run_id=request.run_id,
                    request_id=request.task_id,
                    node=state.node_name,
                    method=ps.incoming.compression.value,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            )

        # Forward pass through this pipeline's partitions on this node.
        t0 = time.perf_counter()
        with torch.no_grad():
            for partition in ps.partitions:
                tensor = partition(tensor.to(state.device))
        compute_end = time.time()
        state.emitter.emit(
            ForwardPassEvent(
                experiment_id=request.experiment_id,
                run_id=request.run_id,
                request_id=request.task_id,
                node=state.node_name,
                duration_ms=(time.perf_counter() - t0) * 1000,
                device=state.device,
            )
        )

        # Last node: pickle result and post to callback.
        if ps.is_last_node:
            compress_start = compress_end = sent_time = time.time()
            result_bytes = pickle.dumps(tensor.cpu())
            await _send_result(request.task_id, request.callback_url, result_bytes)
            sent_time = time.time()
            _emit_timing(
                state,
                request,
                item,
                compute_start,
                compute_end,
                compress_start,
                compress_end,
                sent_time,
            )
            return

        # Compress and forward to next node.
        compress_start = time.time()
        t0 = time.perf_counter()
        out_compressor = get_compressor(
            ps.outgoing.compression,
            outlier_precision=ps.outgoing.outlier_precision,
            regular_precision=ps.outgoing.regular_precision,
        )
        input_bytes_size = tensor.numel() * tensor.element_size()
        compressed = out_compressor.compress(tensor, ps.outgoing.rate or 1.0)
        compress_end = time.time()
        state.emitter.emit(
            CompressEvent(
                experiment_id=request.experiment_id,
                run_id=request.run_id,
                request_id=request.task_id,
                node=state.node_name,
                method=ps.outgoing.compression.value,
                rate=ps.outgoing.rate or 1.0,
                input_bytes=input_bytes_size,
                output_bytes=len(compressed),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        assert ps.next_node_url is not None
        assert ps.next_node_name is not None
        t0 = time.perf_counter()
        await _forward_to_next(ps.next_node_url, request, compressed)
        sent_time = time.time()
        state.emitter.emit(
            SendEvent(
                experiment_id=request.experiment_id,
                run_id=request.run_id,
                request_id=request.task_id,
                from_node=state.node_name,
                to_node=ps.next_node_name,
                payload_bytes=len(compressed),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        _emit_timing(
            state,
            request,
            item,
            compute_start,
            compute_end,
            compress_start,
            compress_end,
            sent_time,
        )

    except Exception:
        logger.exception(
            "Error processing task '%s' pipeline '%s'",
            request.task_id,
            request.pipeline_id,
        )


def _emit_timing(
    state: MultiNodeState,
    request: MultiInferRequest,
    item: QueueItem,
    compute_start: float,
    compute_end: float,
    compress_start: float,
    compress_end: float,
    sent_time: float,
) -> None:
    """Emit a TaskNodeTimingEvent for a completed task.

    Args:
        state: Shared node state.
        request: The inference request.
        item: The original queue item (carries enqueue timing).
        compute_start: Wall-clock time when compute began (after dequeue).
        compute_end: Wall-clock time when forward pass completed.
        compress_start: Wall-clock time when compression began.
        compress_end: Wall-clock time when compression completed.
        sent_time: Wall-clock time when the payload was handed off.
    """
    state.emitter.emit(
        TaskNodeTimingEvent(
            experiment_id=request.experiment_id,
            run_id=request.run_id,
            pipeline_id=request.pipeline_id,
            task_id=request.task_id,
            node_id=state.node_name,
            enqueue_time=item.enqueue_time,
            queue_length_at_enqueue=item.queue_length_at_enqueue,
            compute_start=compute_start,
            compute_end=compute_end,
            compress_start=compress_start,
            compress_end=compress_end,
            sent_time=sent_time,
        )
    )


async def _forward_to_next(
    next_url: str, request: MultiInferRequest, compressed: bytes
) -> None:
    """POST compressed activation to the next node in the pipeline.

    Args:
        next_url: Full URL of the next node's /infer endpoint.
        request: Original inference request (task_id, callback_url, etc.).
        compressed: Compressed activation bytes.
    """
    payload: dict[str, Any] = {
        "task_id": request.task_id,
        "pipeline_id": request.pipeline_id,
        "callback_url": request.callback_url,
        "experiment_id": request.experiment_id,
        "run_id": request.run_id,
        "data": base64.b64encode(compressed).decode(),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(next_url, json=payload)
        resp.raise_for_status()


async def _send_result(task_id: str, callback_url: str, result_bytes: bytes) -> None:
    """POST the final inference result to the orchestrator callback.

    Args:
        task_id: Task identifier.
        callback_url: Orchestrator callback URL.
        result_bytes: Pickled output tensor bytes.
    """
    payload: dict[str, Any] = {
        "task_id": task_id,
        "data": base64.b64encode(result_bytes).decode(),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(callback_url, json=payload)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Background: link probers
# ---------------------------------------------------------------------------


async def _probe_loop(
    state: MultiNodeState, next_node_url: str, next_node_name: str
) -> None:
    """Periodically probe one outgoing link for RTT and throughput.

    Probes are sent only when the node is idle (queue empty, not processing).

    Args:
        state: Shared node state.
        next_node_url: Full /infer URL of the downstream node.
        next_node_name: Downstream node name for metric tagging.
    """
    probe_base_url = next_node_url.rsplit("/infer", 1)[0]
    while True:
        await state.idle_event.wait()
        await asyncio.sleep(0.5)
        if not state.idle_event.is_set():
            continue
        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.get(f"{probe_base_url}/probe")
            rtt_ms = (time.perf_counter() - t0) * 1000

            payload = bytes(_PROBE_PAYLOAD_BYTES)
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(f"{probe_base_url}/probe", content=payload)
            elapsed_s = time.perf_counter() - t0
            throughput_mbps = (_PROBE_PAYLOAD_BYTES * 8) / (elapsed_s * 1e6)

            state.emitter.emit(
                LinkProbeEvent(
                    experiment_id=state.last_experiment_id,
                    run_id=state.last_run_id,
                    from_node=state.node_name,
                    to_node=next_node_name,
                    rtt_ms=rtt_ms,
                    throughput_mbps=throughput_mbps,
                )
            )
            logger.debug(
                "Link probe %s→%s: rtt=%.1fms throughput=%.1fMbps",
                state.node_name,
                next_node_name,
                rtt_ms,
                throughput_mbps,
            )
        except Exception as exc:
            logger.warning(
                "Link probe %s→%s failed: %s", state.node_name, next_node_name, exc
            )
        await asyncio.sleep(state.probe_interval_s)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


async def _initialize() -> MultiNodeState:
    """Load experiment config, partitions, and build per-pipeline node state.

    Returns:
        Fully initialised MultiNodeState.

    Raises:
        RuntimeError: If required environment variables are missing or the
            node is not found in the experiment config.
    """
    node_name = _require_env("NODE_NAME")
    config_path = Path(_require_env("EXPERIMENT_CONFIG_PATH"))
    partitions_base_dir = Path(_require_env("PARTITIONS_BASE_DIR"))
    metrics_url = os.getenv("METRICS_SERVER_URL", "http://metrics:9100")
    probe_interval_s = float(os.getenv("PROBE_INTERVAL_S", "30"))
    buffer_size = int(os.getenv("METRICS_BUFFER_SIZE", "1000"))

    exp = load_multi_experiment_config(config_path)

    # Verify this node exists in the config.
    try:
        exp.node_for(node_name)
    except KeyError:
        raise RuntimeError(
            f"NODE_NAME='{node_name}' not found in experiment config at {config_path}"
        ) from None

    device = _detect_device()
    pipelines: dict[str, PipelineNodeState] = {}

    for pipeline in exp.pipelines:
        if node_name not in pipeline.partitions:
            # This pipeline has no partitions on this node — skip it.
            continue

        flow = pipeline.flow
        is_first_node = flow[0] == node_name
        is_last_node = flow[-1] == node_name

        # Determine next node in this pipeline's flow.
        next_node_url: str | None = None
        next_node_name: str | None = None
        if not is_last_node:
            next_name = flow[flow.index(node_name) + 1]
            next_cfg = exp.node_for(next_name)
            next_node_url = f"http://{next_cfg.host}:{next_cfg.port}/infer"
            next_node_name = next_name

        # Default compression — none until the orchestrator pushes a run config.
        _none = PipelineLinkCompression(compression=CompressionMethod.NONE)

        # Load partitions for this pipeline on this node.
        partition_ids = pipeline.partitions[node_name]
        partition_dir = partitions_base_dir / pipeline.model
        partitions = _load_partitions(
            pipeline.model, partition_ids, partition_dir, device
        )

        pipelines[pipeline.name] = PipelineNodeState(
            pipeline_id=pipeline.name,
            model=pipeline.model,
            partitions=partitions,
            is_first_node=is_first_node,
            is_last_node=is_last_node,
            next_node_url=next_node_url,
            next_node_name=next_node_name,
            incoming=_none,
            outgoing=_none,
        )
        logger.info(
            "Pipeline '%s' (model=%s): partitions=%s first=%s last=%s next=%s",
            pipeline.name,
            pipeline.model,
            partition_ids,
            is_first_node,
            is_last_node,
            next_node_name,
        )

    if not pipelines:
        raise RuntimeError(
            f"Node '{node_name}' has no pipeline partitions assigned in "
            f"experiment config at {config_path}"
        )

    emitter = MetricsEmitter(server_url=metrics_url, buffer_size=buffer_size)
    return MultiNodeState(
        node_name=node_name,
        device=device,
        pipelines=pipelines,
        emitter=emitter,
        probe_interval_s=probe_interval_s,
    )


# ---------------------------------------------------------------------------
# Partition loading
# ---------------------------------------------------------------------------


def _load_partitions(
    model: str,
    partition_ids: list[str],
    partition_dir: Path,
    device: str,
) -> list[Any]:
    """Load partition modules for one pipeline on this node.

    Dispatches to the appropriate loader based on model type:
    - ResNet variants use ``torch.jit.load`` (TorchScript).
    - Llama variants use ``torch.load`` (pickle; requires the partition module
      to be importable so that class references resolve correctly).

    Args:
        model: Model type string (e.g. ``"resnet56"`` or ``"llama-3.1-8b"``).
        partition_ids: Ordered list of partition identifiers to load.
        partition_dir: Directory containing ``<id>.pt`` files for this model.
        device: Torch device string.

    Returns:
        List of loaded modules in partition order.

    Raises:
        FileNotFoundError: If a partition file does not exist.
        ValueError: If the model type is not recognised.
    """
    model_lower = model.lower()
    if "llama" in model_lower:
        import models.llama.partition_llama as _  # noqa: F401 — registers module path for pickle

        def _load(path: Path) -> Any:
            m = torch.load(str(path), map_location=device, weights_only=False)
            m.eval()
            return m

    elif "resnet" in model_lower:

        def _load(path: Path) -> Any:
            m = torch.jit.load(str(path), map_location=device)
            m.eval()
            return m

    else:
        raise ValueError(
            f"Unrecognised model type '{model}'. "
            "Add a loader branch in _load_partitions for this model."
        )

    partitions = []
    for pid in partition_ids:
        path = partition_dir / f"{pid}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Partition file not found: {path}")
        partitions.append(_load(path))
        logger.info("Loaded partition '%s' from %s", pid, path)
    return partitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA available, using GPU")
        return "cuda"
    logger.info("No CUDA device found, using CPU")
    return "cpu"


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable '{key}' is not set")
    return value


# ---------------------------------------------------------------------------
# App instance and entry point
# ---------------------------------------------------------------------------


app = build_app()


def main() -> None:
    """Entry point for running the multi-model node server."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Multi-model DNN inference node server"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
