"""Llama-specific data client supporting WikiText-2 perplexity and MMLU accuracy evaluation.

Mirrors DataClient from data_client.py but:
- Sends tokenized int64 input_ids to the first node (not image tensors)
- Receives float logit tensors [B, L, vocab_size] from the last node
- Computes perplexity (WikiText-2) or accuracy (MMLU) from returned logits
- LlamaRunRecord stores metric_type, nll_sum/token_count (perplexity) or
  ground_truth/predicted (accuracy) for downstream analysis
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import pickle
import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from framework.datamodels.api import ResultPayload
from framework.datamodels.events import EndToEndLatencyEvent, ResultEvent
from framework.datamodels.experiment import DatasetConfig, ExperimentConfig
from framework.datamodels.results import LlamaRunRecord
from framework.nodes.metrics.emitter import MetricsEmitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal dataset classes
# ---------------------------------------------------------------------------


class _WikiText2Batches:
    """Tokenize WikiText-2 test split and yield fixed-length input_id chunks.

    Args:
        config: Dataset config carrying max_seq_len and batch_size.
        tokenizer: Llama tokenizer.
    """

    def __init__(self, config: DatasetConfig, tokenizer: AutoTokenizer) -> None:
        if config.path:
            from datasets import load_from_disk  # type: ignore[import]

            raw = load_from_disk(str(config.path))["test"]
        else:
            from datasets import load_dataset  # type: ignore[import]

            raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(x for x in raw["text"] if len(x) > 20)
        tokens = tokenizer(text, return_tensors="pt").input_ids[0]  # [N]

        seq_len = config.max_seq_len
        # Truncate to multiple of seq_len
        n_complete = len(tokens) // seq_len
        chunks = tokens[: n_complete * seq_len].view(n_complete, seq_len)

        if config.seed is not None:
            generator = torch.Generator().manual_seed(config.seed)
            idx = torch.randperm(len(chunks), generator=generator)
        else:
            idx = torch.arange(len(chunks))
        if config.max_samples is not None:
            idx = idx[: config.max_samples]
        self._chunks = chunks[idx]
        self._batch_size = config.batch_size

    def batches(
        self,
    ) -> Iterator[tuple[int, torch.Tensor, list[int], torch.Tensor | None]]:
        """Yield (batch_idx, input_ids [B, L], [], None) — no labels or mask (fixed-length chunks)."""
        n = len(self._chunks)
        for i in range(0, n, self._batch_size):
            batch = self._chunks[i : i + self._batch_size]
            yield i // self._batch_size, batch, [], None

    def num_batches(self) -> int:
        """Return the total number of batches."""
        return math.ceil(len(self._chunks) / self._batch_size)


class _MMLUBatches:
    """Load MMLU questions, format and tokenize, yield batches.

    Args:
        config: Dataset config carrying subjects, samples_per_subject, batch_size.
        tokenizer: Llama tokenizer.
    """

    def __init__(self, config: DatasetConfig, tokenizer: AutoTokenizer) -> None:
        from datasets import load_dataset  # type: ignore[import]

        subjects = config.subjects or [
            "college_computer_science",
            "high_school_mathematics",
            "professional_law",
            "global_facts",
            "miscellaneous",
            "business_ethics",
        ]
        samples_per = config.samples_per_subject or 50

        rng = (
            torch.Generator().manual_seed(config.seed)
            if config.seed is not None
            else None
        )

        items: list[dict[str, Any]] = []
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test")
                n = min(len(ds), samples_per)
                if rng is not None:
                    indices = torch.randperm(len(ds), generator=rng).tolist()[:n]
                else:
                    indices = list(range(n))
                for i in indices:
                    item = ds[i]
                    items.append(
                        {
                            "question": item["question"],
                            "choices": item["choices"],
                            "answer": item["answer"],
                        }
                    )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Skipping MMLU subject %s: %s", subj, exc
                )

        if config.max_samples is not None:
            if rng is not None:
                order = torch.randperm(len(items), generator=rng).tolist()
                items = [items[i] for i in order[: config.max_samples]]
            else:
                items = items[: config.max_samples]

        self._items = items
        self._tokenizer = tokenizer
        self._tokenizer.padding_side = "left"
        self._batch_size = config.batch_size

    def _format(self, item: dict[str, Any]) -> str:
        choices = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(item["choices"])
        )
        return f"Question: {item['question']}\n{choices}\nAnswer:"

    def batches(
        self,
    ) -> Iterator[tuple[int, torch.Tensor, list[int], torch.Tensor | None]]:
        """Yield (batch_idx, input_ids [B, L], ground_truth_indices, attention_mask [B, L]).

        attention_mask is None when batch_size == 1 (no padding needed).
        """
        items = self._items
        for i in range(0, len(items), self._batch_size):
            batch_items = items[i : i + self._batch_size]
            prompts = [self._format(it) for it in batch_items]
            labels = [it["answer"] for it in batch_items]
            encoded = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # Only include mask when there is actual padding (batch_size > 1).
            mask = encoded.attention_mask if len(batch_items) > 1 else None
            yield i // self._batch_size, encoded.input_ids, labels, mask

    def num_batches(self) -> int:
        """Return the total number of batches."""
        return math.ceil(len(self._items) / self._batch_size)


# ---------------------------------------------------------------------------
# LlamaDataClient
# ---------------------------------------------------------------------------


class LlamaDataClient:
    """Orchestrator-side data client for Llama inference pipelines.

    Mirrors DataClient from data_client.py. Sends tokenized input_ids to the
    first pipeline node and decodes returned logits as either perplexity
    (WikiText-2) or accuracy (MMLU).

    Args:
        tokenizer_path: Path to saved tokenizer directory (produced by partition_llama.py).
        callback_host: Hostname pipeline nodes use to POST results back.
        callback_port: Port for the callback server.
        result_timeout_s: Per-batch timeout in seconds.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer_path: str,
        callback_host: str = "localhost",
        callback_port: int = 8080,
        result_timeout_s: float = 300.0,
    ) -> None:
        resolved_tokenizer_path = str(Path(tokenizer_path).resolve())
        self._tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Token IDs for A B C D (used by MMLU decoding)
        self._answer_token_ids: list[int] = [
            self._tokenizer.encode(" " + c, add_special_tokens=False)[-1]
            for c in ["A", "B", "C", "D"]
        ]
        # Build the dataset loader once — reused across all sweep runs.
        dataset_name = dataset_config.name.lower()
        if dataset_name == "wikitext2":
            self._loader: _WikiText2Batches | _MMLUBatches = _WikiText2Batches(
                dataset_config, self._tokenizer
            )
            self._metric_type = "perplexity"
        elif dataset_name == "mmlu":
            self._loader = _MMLUBatches(dataset_config, self._tokenizer)
            self._metric_type = "accuracy"
        else:
            raise ValueError(
                f"Unsupported Llama dataset '{dataset_config.name}'. Use 'wikitext2' or 'mmlu'."
            )
        self._callback_host = callback_host
        self._callback_port = callback_port
        self._result_timeout_s = result_timeout_s
        self._pending: dict[str, asyncio.Future[ResultPayload]] = {}
        self._pending_inputs: dict[str, torch.Tensor] = {}
        self._server_task: asyncio.Task[None] | None = None
        self._server: uvicorn.Server | None = None
        self._app = self._make_app()

    @property
    def callback_url(self) -> str:
        """Base callback URL reachable by pipeline nodes."""
        return f"http://{self._callback_host}:{self._callback_port}/result"

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None, None]:
        """Start the callback server for the duration of an experiment session."""
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
    ) -> list[LlamaRunRecord]:
        """Send all dataset batches for one sweep run and collect results.

        Args:
            exp: Experiment config (dataset config, metrics server info).
            run_id: Current sweep run identifier.
            first_node_url: POST /infer URL of the first pipeline node.
            emitter: Metrics emitter for ResultEvent emission.

        Returns:
            List of LlamaRunRecord for all completed batches.
        """
        loader = self._loader
        metric_type = self._metric_type
        semaphore = asyncio.Semaphore(exp.dataset.max_in_flight)
        records: list[LlamaRunRecord] = []
        lock = asyncio.Lock()

        async def _send_one(
            batch_idx: int,
            input_ids: torch.Tensor,
            labels: list[int],
            attention_mask: torch.Tensor | None,
        ) -> None:
            task_id = f"{run_id}_{batch_idx}_{uuid.uuid4().hex[:6]}"
            async with semaphore:
                future: asyncio.Future[ResultPayload] = (
                    asyncio.get_event_loop().create_future()
                )
                self._pending[task_id] = future
                self._pending_inputs[task_id] = input_ids
                t0 = time.perf_counter()
                try:
                    await _send_batch_to_node(
                        task_id=task_id,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        first_node_url=first_node_url,
                        callback_url=self.callback_url,
                        experiment_id=exp.name,
                        run_id=run_id,
                        metric_type=metric_type,
                        answer_token_ids=(
                            ",".join(str(t) for t in self._answer_token_ids)
                            if metric_type == "accuracy"
                            else None
                        ),
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
                    self._pending_inputs.pop(task_id, None)

                emitter.emit(
                    EndToEndLatencyEvent(
                        experiment_id=exp.name,
                        run_id=run_id,
                        request_id=task_id,
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        batch_size=input_ids.shape[0],
                    )
                )
                if metric_type == "perplexity":
                    nll_sum, token_count = _decode_wikitext_result(
                        result_payload.data, input_ids
                    )
                    record = LlamaRunRecord(
                        request_id=task_id,
                        batch_idx=batch_idx,
                        metric_type="perplexity",
                        experiment_id=exp.name,
                        run_id=run_id,
                        timestamp=time.time(),
                        nll_sum=nll_sum,
                        token_count=token_count,
                    )
                    emitter.emit(
                        ResultEvent(
                            experiment_id=exp.name,
                            run_id=run_id,
                            request_id=task_id,
                            nll_sum=nll_sum,
                            token_count=token_count,
                        )
                    )
                else:  # mmlu
                    predicted = _decode_mmlu_result(
                        result_payload.data, self._answer_token_ids
                    )
                    record = LlamaRunRecord(
                        request_id=task_id,
                        batch_idx=batch_idx,
                        metric_type="accuracy",
                        experiment_id=exp.name,
                        run_id=run_id,
                        timestamp=time.time(),
                        ground_truth=labels,
                        predicted=predicted,
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
            asyncio.create_task(_send_one(idx, input_ids, labels, mask))
            for idx, input_ids, labels, mask in loader.batches()
        ]
        await asyncio.gather(*tasks)
        logger.info(
            "Run '%s' complete: %d/%d batches collected",
            run_id,
            len(records),
            loader.num_batches(),
        )
        return records

    def _make_app(self) -> FastAPI:
        """Build the FastAPI callback app with a /result endpoint."""
        app = FastAPI()
        client_ref = self

        @app.post("/result")
        async def handle_result(request: Request) -> JSONResponse:
            task_id = request.headers.get("x-task-id", "")
            data = await request.body()
            payload = ResultPayload(task_id=task_id, data=data)
            future = client_ref._pending.get(payload.task_id)
            if future and not future.done():
                future.set_result(payload)
            else:
                logger.warning(
                    "Received result for unknown task_id '%s'", payload.task_id
                )
            return JSONResponse({"status": "ok"})

        @app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({"status": "ok"})

        return app

    async def _start_server(self) -> None:
        config = uvicorn.Config(
            self._app, host="0.0.0.0", port=self._callback_port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(
            self._server.serve(), name="callback-server"
        )
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
# Helper functions
# ---------------------------------------------------------------------------


async def _send_batch_to_node(
    task_id: str,
    input_ids: torch.Tensor,
    first_node_url: str,
    callback_url: str,
    experiment_id: str,
    run_id: str,
    pipeline_id: str = "",
    attention_mask: torch.Tensor | None = None,
    metric_type: str | None = None,
    answer_token_ids: str | None = None,
) -> None:
    """Pickle and POST a tokenized batch to the first pipeline node.

    Args:
        task_id: Unique request identifier.
        input_ids: Tokenized input tensor [B, L].
        first_node_url: POST /infer URL of the first node.
        callback_url: URL the last node should POST results to.
        experiment_id: Experiment name for metrics tagging.
        run_id: Run identifier for metrics tagging.
        pipeline_id: Pipeline identifier for multi-model nodes.
        attention_mask: Optional padding mask [B, L]; present only for MMLU
            batches with batch_size > 1.
        metric_type: ``"perplexity"`` or ``"accuracy"`` — tells the last node
            to compute the metric on-node instead of returning the full logit tensor.
        answer_token_ids: Comma-separated token IDs for MMLU answer choices.
    """
    data = pickle.dumps(input_ids)
    headers: dict[str, str] = {
        "x-task-id": task_id,
        "x-pipeline-id": pipeline_id,
        "x-callback-url": callback_url,
        "x-experiment-id": experiment_id,
        "x-run-id": run_id,
        "content-type": "application/octet-stream",
        "x-input-ids": base64.b64encode(pickle.dumps(input_ids)).decode(),
    }
    if attention_mask is not None:
        headers["x-attention-mask"] = base64.b64encode(
            pickle.dumps(attention_mask.bool())
        ).decode()
    if metric_type is not None:
        headers["x-metric-type"] = metric_type
    if answer_token_ids is not None:
        headers["x-answer-token-ids"] = answer_token_ids
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(first_node_url, content=data, headers=headers)
        resp.raise_for_status()


def _decode_wikitext_result(data: bytes, input_ids: torch.Tensor) -> tuple[float, int]:
    """Decode WikiText result — either pre-computed (nll, count) tuple or raw logits.

    Args:
        data: Pickled result from the last node. Either a ``(nll_sum, token_count)``
            tuple (on-node computation) or a logit tensor ``[B, L, V]`` (legacy).
        input_ids: Original input token IDs ``[B, L]``.

    Returns:
        Tuple of (sum of NLL over all tokens in this batch, token count).
    """
    result = pickle.loads(data)
    if isinstance(result, tuple):
        return result
    logits: torch.Tensor = result
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().long()
    B, L_minus_1, V = shift_logits.shape
    nll = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="sum",
    ).item()
    return nll, B * L_minus_1


def _decode_mmlu_result(data: bytes, answer_token_ids: list[int]) -> list[int]:
    """Decode MMLU result — either pre-computed answer indices or raw logits.

    Args:
        data: Pickled result from the last node. Either a ``list[int]`` of
            predicted answer indices (on-node computation) or a logit tensor
            ``[B, L, V]`` (legacy).
        answer_token_ids: Token IDs for ' A', ' B', ' C', ' D'.

    Returns:
        List of predicted answer indices (0=A, 1=B, 2=C, 3=D) per sample.
    """
    result = pickle.loads(data)
    if isinstance(result, list):
        return result
    logits: torch.Tensor = result
    last_logits = logits[:, -1, :]
    cand_logits = last_logits[:, answer_token_ids]
    return cand_logits.argmax(dim=-1).tolist()
