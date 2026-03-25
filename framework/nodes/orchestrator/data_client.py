from __future__ import annotations

import asyncio
import base64
import logging
import pickle
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from framework.datamodels.api import ResultPayload
from framework.datamodels.events import EndToEndLatencyEvent, ResultEvent
from framework.datamodels.experiment import ExperimentConfig
from framework.datamodels.results import RunRecord
from framework.nodes.metrics.emitter import MetricsEmitter
from framework.nodes.orchestrator.datasets import get_dataset

logger = logging.getLogger(__name__)


class DataClient:
    """Orchestrator-side data client.

    Manages a FastAPI callback server for the full experiment session and
    provides a run() method that sends dataset batches to the first pipeline
    node, collects results via the callback, and saves per-run records.

    Args:
        callback_host: Hostname or IP that pipeline nodes can reach to POST results.
        callback_port: Port for the callback server.
        result_timeout_s: Per-batch timeout waiting for a result.
    """

    def __init__(
        self,
        callback_host: str = "localhost",
        callback_port: int = 8080,
        result_timeout_s: float = 300.0,
    ) -> None:
        self._callback_host = callback_host
        self._callback_port = callback_port
        self._result_timeout_s = result_timeout_s
        self._pending: dict[str, asyncio.Future[ResultPayload]] = {}
        self._server_task: asyncio.Task[None] | None = None
        self._server: uvicorn.Server | None = None
        self._app = self._make_app()

    @property
    def callback_url(self) -> str:
        """Base callback URL reachable by pipeline nodes."""
        return f"http://{self._callback_host}:{self._callback_port}/result"

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None, None]:
        """Start the callback server for the duration of an experiment session.

        Usage::

            async with data_client.session():
                for run in runs:
                    records = await data_client.run(...)
        """
        await self._start_server()
        try:
            yield
        finally:
            await self._stop_server()

    async def run(
        self,
        exp: ExperimentConfig,
        run_id: str,
        first_node_url: str,
        emitter: MetricsEmitter,
    ) -> list[RunRecord]:
        """Send all dataset batches for one sweep run and collect results.

        Batches are sent concurrently, bounded by exp.dataset.max_in_flight.
        The semaphore is held until the result is received, ensuring at most
        max_in_flight batches are in the pipeline at any time.

        Args:
            exp: Experiment config (dataset, metrics server info).
            run_id: Current sweep run identifier for tagging.
            first_node_url: POST /infer URL of the first pipeline node.
            emitter: Metrics emitter for ResultEvent emission.

        Returns:
            List of RunRecord for all completed batches.
        """
        dataset = get_dataset(exp.dataset)
        semaphore = asyncio.Semaphore(exp.dataset.max_in_flight)
        records: list[RunRecord] = []
        lock = asyncio.Lock()

        async def _send_one(
            batch_idx: int, images: torch.Tensor, labels: list[int]
        ) -> None:
            task_id = f"{run_id}_{batch_idx}_{uuid.uuid4().hex[:6]}"
            async with semaphore:
                future: asyncio.Future[ResultPayload] = (
                    asyncio.get_event_loop().create_future()
                )
                self._pending[task_id] = future
                t0 = time.perf_counter()
                try:
                    await _send_batch_to_node(
                        task_id=task_id,
                        images=images,
                        first_node_url=first_node_url,
                        callback_url=self.callback_url,
                        experiment_id=exp.name,
                        run_id=run_id,
                    )
                    result_payload = await asyncio.wait_for(
                        future, timeout=self._result_timeout_s
                    )
                except TimeoutError:
                    logger.error(
                        "Timeout waiting for result of batch %d (task_id=%s)",
                        batch_idx,
                        task_id,
                    )
                    return
                except Exception:
                    logger.exception(
                        "Error sending batch %d (task_id=%s)", batch_idx, task_id
                    )
                    return
                finally:
                    self._pending.pop(task_id, None)

                emitter.emit(
                    EndToEndLatencyEvent(
                        experiment_id=exp.name,
                        run_id=run_id,
                        request_id=task_id,
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        batch_size=images.shape[0],
                    )
                )
                predicted = _decode_predictions(result_payload.data)
                record = RunRecord(
                    request_id=task_id,
                    batch_idx=batch_idx,
                    ground_truth=labels,
                    predicted=predicted,
                    experiment_id=exp.name,
                    run_id=run_id,
                    timestamp=time.time(),
                )
                emitter.emit(
                    ResultEvent(
                        experiment_id=exp.name,
                        run_id=run_id,
                        request_id=task_id,
                        predicted=predicted,
                        actual=labels,
                    )
                )
                async with lock:
                    records.append(record)

        tasks = [
            asyncio.create_task(_send_one(idx, images, labels))
            for idx, images, labels in dataset.batches()
        ]
        await asyncio.gather(*tasks)
        logger.info(
            "Run '%s' complete: %d/%d batches collected",
            run_id,
            len(records),
            dataset.num_batches(),
        )
        return records

    def _make_app(self) -> FastAPI:
        """Build the FastAPI callback app with a /result endpoint."""
        app = FastAPI()
        client_ref = self

        @app.post("/result")
        async def handle_result(payload: ResultPayload) -> JSONResponse:
            future = client_ref._pending.get(payload.task_id)
            if future and not future.done():
                future.set_result(payload)
            else:
                logger.warning(
                    "Received result for unknown or expired task_id '%s'",
                    payload.task_id,
                )
            return JSONResponse({"status": "ok"})

        @app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({"status": "ok"})

        return app

    async def _start_server(self) -> None:
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self._callback_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(
            self._server.serve(), name="callback-server"
        )
        # Wait for the server to become ready
        for _ in range(20):
            await asyncio.sleep(0.2)
            if self._server.started:
                break
        logger.info("Callback server ready on port %d", self._callback_port)

    async def _stop_server(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            await self._server_task
        logger.info("Callback server stopped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _send_batch_to_node(
    task_id: str,
    images: torch.Tensor,
    first_node_url: str,
    callback_url: str,
    experiment_id: str,
    run_id: str,
) -> None:
    """Pickle and POST a batch to the first pipeline node.

    Args:
        task_id: Unique request identifier.
        images: Batch image tensor [B, C, H, W].
        first_node_url: POST /infer URL of the first node.
        callback_url: URL the last node should POST results to.
        experiment_id: Experiment name for metrics tagging.
        run_id: Run identifier for metrics tagging.
    """
    raw = base64.b64encode(pickle.dumps(images)).decode()
    payload = {
        "task_id": task_id,
        "callback_url": callback_url,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "data": raw,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(first_node_url, json=payload)
        resp.raise_for_status()


def _decode_predictions(data: str) -> list[int]:
    """Decode base64-pickled logits tensor to predicted class indices.

    Args:
        data: Base64-encoded pickled tensor from the last node.

    Returns:
        List of predicted class indices (argmax of logits).
    """
    tensor: torch.Tensor = pickle.loads(base64.b64decode(data))
    return tensor.argmax(dim=1).tolist()
