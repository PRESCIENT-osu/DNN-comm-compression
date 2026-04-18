from __future__ import annotations

import asyncio
import base64
import logging
import os
import pickle
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from framework.datamodels.api import ConfigUpdate, InferRequest
from framework.datamodels.events import (
    CompressEvent,
    DecompressEvent,
    ForwardPassEvent,
    LinkProbeEvent,
    SendEvent,
)
from framework.datamodels.experiment import CompressionMethod
from framework.nodes.compute.common.compressor import get_compressor
from framework.nodes.metrics.emitter import MetricsEmitter
from framework.utils.loader import load_experiment_config

logger = logging.getLogger(__name__)

_PROBE_PAYLOAD_BYTES = 100 * 1024  # 100 KB throughput probe payload


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class NodeState:
    """Mutable runtime state for a node server instance."""

    def __init__(
        self,
        node_name: str,
        partitions: list[Any],
        device: str,
        next_node_url: str | None,
        next_node_name: str | None,
        is_first_node: bool,
        is_last_node: bool,
        incoming_method: CompressionMethod,
        incoming_rate: float,
        incoming_outlier_precision: str,
        incoming_regular_precision: str,
        outgoing_method: CompressionMethod,
        outgoing_rate: float,
        outgoing_outlier_precision: str,
        outgoing_regular_precision: str,
        emitter: MetricsEmitter,
        probe_interval_s: float,
    ) -> None:
        self.node_name = node_name
        self.partitions = partitions
        self.device = device
        self.next_node_url = next_node_url
        self.next_node_name = next_node_name
        self.is_first_node = is_first_node
        self.is_last_node = is_last_node
        self.incoming_method = incoming_method
        self.incoming_rate = incoming_rate
        self.incoming_outlier_precision = incoming_outlier_precision
        self.incoming_regular_precision = incoming_regular_precision
        self.outgoing_method = outgoing_method
        self.outgoing_rate = outgoing_rate
        self.outgoing_outlier_precision = outgoing_outlier_precision
        self.outgoing_regular_precision = outgoing_regular_precision
        self.emitter = emitter
        self.probe_interval_s = probe_interval_s

        self._in_flight: int = 0
        self.idle_event: asyncio.Event = asyncio.Event()
        self.idle_event.set()

        self.last_experiment_id: str = "unknown"
        self.last_run_id: str = "none"

    @property
    def model_dtype(self) -> torch.dtype | None:
        """dtype of the loaded partitions, or None if partitions have no parameters."""
        for partition in self.partitions:
            param = next(iter(partition.parameters()), None)
            if param is not None:
                return param.dtype
        return None

    @property
    def in_flight(self) -> int:
        """Current number of requests being processed."""
        return self._in_flight

    def increment_in_flight(self) -> None:
        """Increment the in-flight counter and clear the idle event."""
        self._in_flight += 1
        self.idle_event.clear()

    def decrement_in_flight(self) -> None:
        """Decrement the in-flight counter and signal idle if it reaches zero."""
        self._in_flight = max(0, self._in_flight - 1)
        if self._in_flight == 0:
            self.idle_event.set()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(
    load_partitions_fn: Callable[[list[str], Path, str], list[Any]],
) -> FastAPI:
    """Create and return a configured FastAPI node server app.

    Args:
        load_partitions_fn: Callable that loads model partitions from disk.
            Signature: (partition_names, partitions_dir, device) -> list of modules.

    Returns:
        Configured FastAPI application.
    """
    _state: list[NodeState | None] = [None]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        _state[0] = await _initialize(load_partitions_fn)
        await _state[0].emitter.start()  # type: ignore[union-attr]
        if not _state[0].is_last_node:  # type: ignore[union-attr]
            asyncio.create_task(_probe_loop(_state[0]), name="link-prober")  # type: ignore[arg-type]
        logger.info(
            "Node '%s' ready on device '%s' (first=%s, last=%s)",
            _state[0].node_name,  # type: ignore[union-attr]
            _state[0].device,  # type: ignore[union-attr]
            _state[0].is_first_node,  # type: ignore[union-attr]
            _state[0].is_last_node,  # type: ignore[union-attr]
        )
        yield
        await _state[0].emitter.stop()  # type: ignore[union-attr]

    app = FastAPI(lifespan=lifespan)

    # -----------------------------------------------------------------------
    # Inference API
    # -----------------------------------------------------------------------

    @app.post("/infer", status_code=202)
    async def infer(http_request: Request) -> JSONResponse:
        state = _state[0]
        assert state is not None
        request = await _parse_infer_request(http_request)
        state.increment_in_flight()
        asyncio.create_task(
            _process_inference(state, request), name=f"infer-{request.task_id}"
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
                "in_flight": state.in_flight,
                "device": state.device,
                "incoming": {
                    "method": state.incoming_method.value,
                    "rate": state.incoming_rate,
                },
                "outgoing": {
                    "method": state.outgoing_method.value,
                    "rate": state.outgoing_rate,
                },
            }
        )

    @app.post("/config")
    async def update_config(update: ConfigUpdate) -> JSONResponse:
        state = _state[0]
        assert state is not None
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
                detail=f"Timed out waiting for in-flight requests to drain after {update.drain_timeout_s}s",
            ) from None
        if update.direction == "incoming":
            state.incoming_method = update.method
            state.incoming_rate = update.rate
            state.incoming_outlier_precision = update.outlier_precision
            state.incoming_regular_precision = update.regular_precision
        else:
            state.outgoing_method = update.method
            state.outgoing_rate = update.rate
            state.outgoing_outlier_precision = update.outlier_precision
            state.outgoing_regular_precision = update.regular_precision
        logger.info(
            "Config updated: %s → method=%s rate=%s",
            update.direction,
            update.method.value,
            update.rate,
        )
        return JSONResponse(
            {
                "status": "ok",
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
# Request parsing
# ---------------------------------------------------------------------------


async def _parse_infer_request(http_request: Request) -> InferRequest:
    """Parse an infer request from either JSON (legacy) or binary format."""
    content_type = http_request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await http_request.json()
        return InferRequest(
            task_id=body["task_id"],
            callback_url=body["callback_url"],
            experiment_id=body["experiment_id"],
            run_id=body["run_id"],
            data=base64.b64decode(body["data"]),
            attention_mask=body.get("attention_mask"),
            input_ids=body.get("input_ids"),
            metric_type=body.get("metric_type"),
            answer_token_ids=body.get("answer_token_ids"),
        )
    data = await http_request.body()
    headers = http_request.headers
    return InferRequest(
        task_id=headers["x-task-id"],
        callback_url=headers["x-callback-url"],
        experiment_id=headers["x-experiment-id"],
        run_id=headers["x-run-id"],
        data=data,
        attention_mask=headers.get("x-attention-mask"),
        input_ids=headers.get("x-input-ids"),
        metric_type=headers.get("x-metric-type"),
        answer_token_ids=headers.get("x-answer-token-ids"),
    )


# ---------------------------------------------------------------------------
# Background: inference processing
# ---------------------------------------------------------------------------


async def _process_inference(state: NodeState, request: InferRequest) -> None:
    state.last_experiment_id = request.experiment_id
    state.last_run_id = request.run_id
    try:
        raw_bytes = request.data

        attention_mask: torch.Tensor | None = None
        if request.attention_mask is not None:
            attention_mask = pickle.loads(base64.b64decode(request.attention_mask))

        if state.is_first_node:
            tensor: torch.Tensor = pickle.loads(raw_bytes)
        else:
            t0 = time.perf_counter()
            compressor = get_compressor(
                state.incoming_method,
                outlier_precision=state.incoming_outlier_precision,
                regular_precision=state.incoming_regular_precision,
            )
            tensor = compressor.decompress(raw_bytes, state.device)
            if state.model_dtype is not None and tensor.dtype != state.model_dtype:
                tensor = tensor.to(state.model_dtype)
            state.emitter.emit(
                DecompressEvent(
                    experiment_id=request.experiment_id,
                    run_id=request.run_id,
                    request_id=request.task_id,
                    node=state.node_name,
                    method=state.incoming_method.value,
                    input_bytes=len(raw_bytes),
                    output_bytes=tensor.numel() * tensor.element_size(),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            for partition in state.partitions:
                if attention_mask is not None:
                    tensor = partition(
                        tensor.to(state.device), attention_mask=attention_mask
                    )
                else:
                    tensor = partition(tensor.to(state.device))
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

        if state.is_last_node:
            result_bytes = _compute_result(tensor, request)
            await _send_result(request.task_id, request.callback_url, result_bytes)
            return

        t0 = time.perf_counter()
        out_compressor = get_compressor(
            state.outgoing_method,
            outlier_precision=state.outgoing_outlier_precision,
            regular_precision=state.outgoing_regular_precision,
        )
        input_bytes_size = tensor.numel() * tensor.element_size()
        compressed = out_compressor.compress(tensor, state.outgoing_rate)
        state.emitter.emit(
            CompressEvent(
                experiment_id=request.experiment_id,
                run_id=request.run_id,
                request_id=request.task_id,
                node=state.node_name,
                method=state.outgoing_method.value,
                rate=state.outgoing_rate,
                input_bytes=input_bytes_size,
                output_bytes=len(compressed),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        assert state.next_node_url is not None
        assert state.next_node_name is not None
        t0 = time.perf_counter()
        await _forward_to_next(state.next_node_url, request, compressed)
        state.emitter.emit(
            SendEvent(
                experiment_id=request.experiment_id,
                run_id=request.run_id,
                request_id=request.task_id,
                from_node=state.node_name,
                to_node=state.next_node_name,
                payload_bytes=len(compressed),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

    except Exception:
        logger.exception("Error processing inference request '%s'", request.task_id)
    finally:
        state.decrement_in_flight()


def _compute_result(tensor: torch.Tensor, request: InferRequest) -> bytes:
    """Compute the final result at the last node.

    For Llama WikiText/MMLU, computes the metric on-node to avoid
    transferring the full logit tensor.
    """
    if request.metric_type == "perplexity" and request.input_ids is not None:
        input_ids = pickle.loads(base64.b64decode(request.input_ids))
        shift_logits = tensor[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous().long().to(tensor.device)
        B, L_minus_1, V = shift_logits.shape
        nll = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="sum",
        ).item()
        return pickle.dumps((nll, B * L_minus_1))
    if request.metric_type == "accuracy" and request.answer_token_ids is not None:
        ids = [int(x) for x in request.answer_token_ids.split(",")]
        last_logits = tensor[:, -1, :]
        cand_logits = last_logits[:, ids]
        return pickle.dumps(cand_logits.argmax(dim=-1).tolist())
    return pickle.dumps(tensor.cpu())


async def _forward_to_next(
    next_url: str, request: InferRequest, compressed: bytes
) -> None:
    headers: dict[str, str] = {
        "x-task-id": request.task_id,
        "x-callback-url": request.callback_url,
        "x-experiment-id": request.experiment_id,
        "x-run-id": request.run_id,
        "content-type": "application/octet-stream",
    }
    if request.attention_mask is not None:
        headers["x-attention-mask"] = request.attention_mask
    if request.input_ids is not None:
        headers["x-input-ids"] = request.input_ids
    if request.metric_type is not None:
        headers["x-metric-type"] = request.metric_type
    if request.answer_token_ids is not None:
        headers["x-answer-token-ids"] = request.answer_token_ids
    timeout = httpx.Timeout(connect=10.0, write=120.0, read=120.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(next_url, content=compressed, headers=headers)
        resp.raise_for_status()


async def _send_result(task_id: str, callback_url: str, result_bytes: bytes) -> None:
    headers: dict[str, str] = {
        "x-task-id": task_id,
        "content-type": "application/octet-stream",
    }
    timeout = httpx.Timeout(connect=10.0, write=120.0, read=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(callback_url, content=result_bytes, headers=headers)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Background: link prober
# ---------------------------------------------------------------------------


async def _probe_loop(state: NodeState) -> None:
    while True:
        await state.idle_event.wait()
        await asyncio.sleep(0.5)
        if state.in_flight > 0 or state.next_node_url is None:
            continue
        probe_base_url = state.next_node_url.rsplit("/infer", 1)[0]
        rtt_ms: float | None = None
        throughput_mbps: float | None = None
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
                    to_node=state.next_node_name or "unknown",
                    rtt_ms=rtt_ms,
                    throughput_mbps=throughput_mbps,
                )
            )
            logger.debug(
                "Link probe %s→%s: rtt=%.1fms throughput=%.1fMbps",
                state.node_name,
                state.next_node_name,
                rtt_ms,
                throughput_mbps,
            )
        except Exception as exc:
            logger.warning(
                "Link probe %s→%s failed: %s",
                state.node_name,
                state.next_node_name,
                exc,
            )
        await asyncio.sleep(state.probe_interval_s)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


async def _initialize(
    load_partitions_fn: Callable[[list[str], Path, str], list[Any]],
) -> NodeState:
    node_name = _require_env("NODE_NAME")
    config_path = Path(_require_env("EXPERIMENT_CONFIG_PATH"))
    partitions_dir = Path(_require_env("PARTITIONS_DIR"))
    metrics_url = os.getenv("METRICS_SERVER_URL", "http://metrics:9100")
    probe_interval_s = float(os.getenv("PROBE_INTERVAL_S", "30"))
    buffer_size = int(os.getenv("METRICS_BUFFER_SIZE", "1000"))

    exp = load_experiment_config(config_path)
    node_cfg = next((n for n in exp.nodes if n.name == node_name), None)
    if node_cfg is None:
        raise RuntimeError(
            f"NODE_NAME='{node_name}' not found in experiment config at {config_path}"
        )

    order = exp.node_order()
    is_first_node = order[0] == node_name
    is_last_node = order[-1] == node_name

    incoming_link = next((lk for lk in exp.links if lk.to_node == node_name), None)
    incoming_method = (
        incoming_link.compression if incoming_link else CompressionMethod.NONE
    )
    incoming_rate = incoming_link.rate or 0.0 if incoming_link else 0.0
    incoming_outlier_precision = (
        incoming_link.outlier_precision if incoming_link else "fp16"
    )
    incoming_regular_precision = (
        incoming_link.regular_precision if incoming_link else "int8"
    )

    outgoing_link = next((lk for lk in exp.links if lk.from_node == node_name), None)
    outgoing_method = (
        outgoing_link.compression if outgoing_link else CompressionMethod.NONE
    )
    outgoing_rate = outgoing_link.rate or 0.0 if outgoing_link else 0.0
    outgoing_outlier_precision = (
        outgoing_link.outlier_precision if outgoing_link else "fp16"
    )
    outgoing_regular_precision = (
        outgoing_link.regular_precision if outgoing_link else "int8"
    )

    next_node_url: str | None = None
    next_node_name: str | None = None
    if not is_last_node:
        if outgoing_link:
            next_name = outgoing_link.to_node
        else:
            # Generated experiment format has no top-level links; fall back to
            # definition order.  NOTE: this assumes a strictly linear pipeline
            # (nodes listed in execution order).  Revisit if we ever support
            # topologies where partitions execute out of order (e.g. diamond).
            next_name = order[order.index(node_name) + 1]
        next_node_cfg = next((n for n in exp.nodes if n.name == next_name), None)
        if next_node_cfg:
            next_node_url = f"http://{next_node_cfg.host}:{next_node_cfg.port}/infer"
            next_node_name = next_node_cfg.name

    device = _detect_device()
    partitions = load_partitions_fn(node_cfg.partitions, partitions_dir, device)

    emitter = MetricsEmitter(
        server_url=metrics_url,
        buffer_size=buffer_size,
    )

    return NodeState(
        node_name=node_name,
        partitions=partitions,
        device=device,
        next_node_url=next_node_url,
        next_node_name=next_node_name,
        is_first_node=is_first_node,
        is_last_node=is_last_node,
        incoming_method=incoming_method,
        incoming_rate=incoming_rate,
        incoming_outlier_precision=incoming_outlier_precision,
        incoming_regular_precision=incoming_regular_precision,
        outgoing_method=outgoing_method,
        outgoing_rate=outgoing_rate,
        outgoing_outlier_precision=outgoing_outlier_precision,
        outgoing_regular_precision=outgoing_regular_precision,
        emitter=emitter,
        probe_interval_s=probe_interval_s,
    )


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
