from __future__ import annotations

import asyncio
import base64
import logging
import os
import pickle
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from framework.config.experiment_schema import CompressionMethod
from framework.config.loader import load_experiment_config
from framework.node.compressor import get_compressor
from framework.node.metrics import (
    CompressEvent,
    DecompressEvent,
    ForwardPassEvent,
    LinkProbeEvent,
    MetricsEmitter,
    SendEvent,
)

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
        partitions: list[torch.jit.ScriptModule],
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

        # Updated with each inference request for probe event context
        self.last_experiment_id: str = "unknown"
        self.last_run_id: str = "none"

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
# Request / response models
# ---------------------------------------------------------------------------


class InferRequest(BaseModel):
    """Payload for POST /infer."""

    task_id: str
    callback_url: str
    experiment_id: str
    run_id: str
    data: str  # base64-encoded bytes (compressed activation or raw input)


class ConfigUpdate(BaseModel):
    """Payload for POST /config."""

    direction: str  # "incoming" or "outgoing"
    method: CompressionMethod
    rate: float = 0.0
    drain_timeout_s: float = 60.0
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_state: NodeState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _state
    _state = await _initialize()
    await _state.emitter.start()
    if not _state.is_last_node:
        asyncio.create_task(_probe_loop(_state), name="link-prober")
    logger.info(
        "Node '%s' ready on device '%s' (first=%s, last=%s)",
        _state.node_name,
        _state.device,
        _state.is_first_node,
        _state.is_last_node,
    )
    yield
    await _state.emitter.stop()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------


@app.post("/infer", status_code=202)
async def infer(request: InferRequest) -> JSONResponse:
    """Accept an inference request and process it asynchronously.

    Returns immediately with 202 Accepted.  Processing (decompress,
    forward pass, compress, forward) happens in a background task.

    Args:
        request: Inference request payload.

    Returns:
        JSON with task_id and accepted status.
    """
    assert _state is not None
    _state.increment_in_flight()
    asyncio.create_task(
        _process_inference(_state, request), name=f"infer-{request.task_id}"
    )
    return JSONResponse({"task_id": request.task_id, "status": "accepted"})


# ---------------------------------------------------------------------------
# Management API
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    """Return node health status.

    Returns:
        JSON with status and node name.
    """
    assert _state is not None
    return JSONResponse({"status": "ok", "node": _state.node_name})


@app.get("/status")
async def status() -> JSONResponse:
    """Return current node status including in-flight count and compression config.

    Returns:
        JSON with node name, in-flight count, device, and compression config.
    """
    assert _state is not None
    return JSONResponse(
        {
            "node": _state.node_name,
            "in_flight": _state.in_flight,
            "device": _state.device,
            "incoming": {
                "method": _state.incoming_method.value,
                "rate": _state.incoming_rate,
            },
            "outgoing": {
                "method": _state.outgoing_method.value,
                "rate": _state.outgoing_rate,
            },
        }
    )


@app.post("/config")
async def update_config(update: ConfigUpdate) -> JSONResponse:
    """Update the compression config for the incoming or outgoing link.

    Waits for in-flight requests to drain before applying the new config.

    Args:
        update: Config update specifying direction, method, and rate.

    Returns:
        JSON with status and the applied config.

    Raises:
        HTTPException 400: If direction is not "incoming" or "outgoing".
        HTTPException 409: If drain timeout is exceeded.
    """
    assert _state is not None
    if update.direction not in ("incoming", "outgoing"):
        raise HTTPException(
            status_code=400,
            detail=f"direction must be 'incoming' or 'outgoing', got '{update.direction}'",
        )
    try:
        await asyncio.wait_for(_state.idle_event.wait(), timeout=update.drain_timeout_s)
    except TimeoutError:
        raise HTTPException(
            status_code=409,
            detail=f"Timed out waiting for in-flight requests to drain after {update.drain_timeout_s}s",
        ) from None
    if update.direction == "incoming":
        _state.incoming_method = update.method
        _state.incoming_rate = update.rate
        _state.incoming_outlier_precision = update.outlier_precision
        _state.incoming_regular_precision = update.regular_precision
    else:
        _state.outgoing_method = update.method
        _state.outgoing_rate = update.rate
        _state.outgoing_outlier_precision = update.outlier_precision
        _state.outgoing_regular_precision = update.regular_precision
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


# ---------------------------------------------------------------------------
# Probe API
# ---------------------------------------------------------------------------


@app.get("/probe")
async def probe_rtt() -> JSONResponse:
    """Respond to an RTT probe from the previous node's prober.

    Returns:
        JSON with status and server-side timestamp.
    """
    return JSONResponse({"status": "ok", "timestamp": time.time()})


@app.post("/probe")
async def probe_throughput(request: Request) -> JSONResponse:
    """Receive a throughput probe payload and return its size.

    Args:
        request: HTTP request carrying an arbitrary binary payload.

    Returns:
        JSON with received byte count and server-side timestamp.
    """
    body = await request.body()
    return JSONResponse({"received_bytes": len(body), "timestamp": time.time()})


# ---------------------------------------------------------------------------
# Background: inference processing
# ---------------------------------------------------------------------------


async def _process_inference(state: NodeState, request: InferRequest) -> None:
    """Execute the full inference pipeline for one request.

    Decompresses the incoming payload, runs the forward pass, compresses
    the output, and forwards it to the next node or the callback URL.
    Metrics are emitted at each step.  The in-flight counter is always
    decremented in the finally block.

    Args:
        state: Current node runtime state.
        request: The inference request to process.
    """
    state.last_experiment_id = request.experiment_id
    state.last_run_id = request.run_id
    try:
        raw_bytes = base64.b64decode(request.data)

        # --- Decompress (skip for first node) ---
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
            state.emitter.emit(
                DecompressEvent(
                    experiment_id=request.experiment_id,
                    run_id=request.run_id,
                    request_id=request.task_id,
                    node=state.node_name,
                    method=state.incoming_method.value,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            )

        # --- Forward pass ---
        t0 = time.perf_counter()
        with torch.no_grad():
            for partition in state.partitions:
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

        # --- Last node: send result to callback ---
        if state.is_last_node:
            result_bytes = pickle.dumps(tensor.cpu())
            await _send_result(request.task_id, request.callback_url, result_bytes)
            return

        # --- Compress ---
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

        # --- Forward to next node ---
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


async def _forward_to_next(
    next_url: str, request: InferRequest, compressed: bytes
) -> None:
    """Forward a compressed activation to the next node.

    Args:
        next_url: POST /infer URL of the next node.
        request: Original inference request (carries task_id, callback_url, etc.).
        compressed: Compressed activation bytes to forward.
    """
    payload: dict[str, Any] = {
        "task_id": request.task_id,
        "callback_url": request.callback_url,
        "experiment_id": request.experiment_id,
        "run_id": request.run_id,
        "data": base64.b64encode(compressed).decode(),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(next_url, json=payload)
        resp.raise_for_status()


async def _send_result(task_id: str, callback_url: str, result_bytes: bytes) -> None:
    """Send the final inference result back to the orchestrator callback.

    Args:
        task_id: Request identifier.
        callback_url: Orchestrator callback endpoint URL.
        result_bytes: Pickled result tensor.
    """
    payload: dict[str, Any] = {
        "task_id": task_id,
        "data": base64.b64encode(result_bytes).decode(),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(callback_url, json=payload)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Background: link prober
# ---------------------------------------------------------------------------


async def _probe_loop(state: NodeState) -> None:
    """Background task that probes the outgoing link when the node is idle.

    Waits for the idle event (in_flight == 0), performs RTT and throughput
    probes against the next node, emits a LinkProbeEvent, then sleeps for
    probe_interval_s before trying again.

    Args:
        state: Current node runtime state.
    """
    while True:
        await state.idle_event.wait()
        # Brief settle to avoid probing during the very start of idle
        await asyncio.sleep(0.5)
        if state.in_flight > 0 or state.next_node_url is None:
            continue
        probe_base_url = state.next_node_url.rsplit("/infer", 1)[0]
        rtt_ms: float | None = None
        throughput_mbps: float | None = None
        try:
            # RTT probe
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.get(f"{probe_base_url}/probe")
            rtt_ms = (time.perf_counter() - t0) * 1000

            # Throughput probe
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


async def _initialize() -> NodeState:
    """Build NodeState from environment variables and experiment config.

    Reads NODE_NAME, EXPERIMENT_CONFIG_PATH, PARTITIONS_DIR,
    METRICS_SERVER_URL, PROBE_INTERVAL_S, and METRICS_BUFFER_SIZE from
    the environment.  Loads the experiment config, resolves this node's
    partition assignments and link configs, loads partition models, and
    auto-detects the available device.

    Returns:
        Fully initialised NodeState ready for use.

    Raises:
        RuntimeError: If required environment variables are missing or the
            node name is not found in the experiment config.
    """
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

    # Resolve incoming link (from previous node → this node)
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

    # Resolve outgoing link (this node → next node)
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

    # Resolve next node URL
    next_node_url: str | None = None
    next_node_name: str | None = None
    if not is_last_node and outgoing_link:
        next_node_cfg = next(
            (n for n in exp.nodes if n.name == outgoing_link.to_node), None
        )
        if next_node_cfg:
            next_node_url = f"http://{next_node_cfg.host}:{next_node_cfg.port}/infer"
            next_node_name = next_node_cfg.name

    # Load partition models
    device = _detect_device()
    partitions = _load_partitions(node_cfg.partitions, partitions_dir, device)

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
    """Return the best available device string.

    Returns:
        ``"cuda"`` if a CUDA-capable GPU is available, otherwise ``"cpu"``.
    """
    if torch.cuda.is_available():
        logger.info("CUDA available, using GPU")
        return "cuda"
    logger.info("No CUDA device found, using CPU")
    return "cpu"


def _load_partitions(
    partition_names: list[str],
    partitions_dir: Path,
    device: str,
) -> list[torch.jit.ScriptModule]:
    """Load TorchScript partition models from disk.

    Args:
        partition_names: Ordered list of partition identifiers (e.g. ``["p2", "p3"]``).
        partitions_dir: Directory containing ``<name>.pt`` files.
        device: Device to map the loaded models to.

    Returns:
        List of loaded TorchScript modules in partition order.

    Raises:
        FileNotFoundError: If a partition file does not exist.
    """
    partitions = []
    for name in partition_names:
        path = partitions_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Partition file not found: {path}")
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        partitions.append(model)
        logger.info("Loaded partition '%s' from %s", name, path)
    return partitions


def _require_env(key: str) -> str:
    """Read a required environment variable.

    Args:
        key: Environment variable name.

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is not set.
    """
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable '{key}' is not set")
    return value


def main() -> None:
    """Entry point for running the node server from the command line."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="DNN inference node server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
