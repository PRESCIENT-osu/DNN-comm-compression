"""Multi-model experiment runner.

Drives task submission for multi-model pipeline experiments, supporting fill,
fixed_rate, and poisson workload patterns.  Collects per-task E2E timing and
emits RunThroughputEvent at the end of each sweep run.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import itertools
import logging
import math
import os
import pickle
import random
import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from framework.datamodels.api import ResultPayload
from framework.datamodels.events import (
    PipelineThroughputStats,
    RunThroughputEvent,
    TaskE2EEvent,
)
from framework.datamodels.experiment import DatasetConfig
from framework.datamodels.multi_experiment import (
    MultiExperimentConfig,
    MultiResolvedRun,
    MultiResolvedSubExperiment,
    PipelineConfig,
    WorkloadPattern,
)
from framework.nodes.metrics.emitter import MetricsEmitter
from framework.nodes.orchestrator.controller import run_already_completed
from framework.nodes.orchestrator.datasets import get_dataset
from framework.nodes.orchestrator.multi_controller import (
    push_multi_run_config,
    wait_for_multi_nodes_ready,
)
from framework.utils.loader import load_multi_experiment_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SlotResult — returned by MultiDataClient.run_slot()
# ---------------------------------------------------------------------------


@dataclass
class SlotResult:
    """Aggregated results for one optimization slot.

    Attributes:
        per_pipeline_latency_ms: Per-pipeline list of task E2E latencies (ms).
        per_pipeline_results: Per-pipeline list of (labels, decoded) pairs.
            For classification/MMLU pipelines, decoded is a list of predicted
            class indices.  For perplexity pipelines, decoded is a
            ``(nll_sum, token_count)`` tuple.
        wall_time_s: Wall-clock duration of the slot in seconds.
        perplexity_baselines: Baseline NLL per pipeline for WikiText normalization.
            Set by opt_runner during profiling; empty dict means no WikiText pipelines.
    """

    per_pipeline_latency_ms: dict[str, list[float]] = field(default_factory=dict)
    per_pipeline_results: dict[str, list[tuple[list[int], Any]]] = field(
        default_factory=dict
    )
    wall_time_s: float = 0.0
    perplexity_baselines: dict[str, float] = field(default_factory=dict)

    def achieved_rps(self, pipeline_id: str) -> float:
        """Return observed task throughput for a pipeline (tasks/second).

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Completed tasks divided by wall_time_s; 0.0 if wall_time_s is zero.
        """
        n = len(self.per_pipeline_latency_ms.get(pipeline_id, []))
        return n / self.wall_time_s if self.wall_time_s > 0 else 0.0

    def quality_metric(self, pipeline_id: str) -> float:
        """Return quality metric in [0, 1] for a pipeline (higher = better).

        For classification and MMLU pipelines, returns top-1 accuracy.
        For perplexity pipelines, decoded results are ``(nll_sum, token_count)``
        tuples; returns ``exp(-max(0, ratio - 1))`` where
        ``ratio = mean_nll / baseline_nll`` and baseline_nll is taken from
        ``self.perplexity_baselines``.  If no baseline is stored for the pipeline,
        falls back to ``exp(-mean_nll)`` clamped to [0, 1].

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Quality metric in [0, 1].
        """
        pairs = self.per_pipeline_results.get(pipeline_id, [])
        if not pairs:
            return 0.0
        _, first = pairs[0]
        # Perplexity path: decoded is a (nll_sum, token_count) tuple.
        if isinstance(first, tuple):
            total_nll = sum(float(nll) for _, (nll, _) in pairs)
            total_tokens = sum(int(tc) for _, (_, tc) in pairs)
            mean_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
            baseline = self.perplexity_baselines.get(pipeline_id)
            if baseline and baseline > 0:
                ratio = mean_nll / baseline
            else:
                ratio = mean_nll
            return math.exp(-max(0.0, ratio - 1.0))
        # Classification / MMLU path: decoded is a list of predicted indices.
        correct = sum(
            1
            for labels, predicted in pairs
            for gt, pred in zip(labels, predicted, strict=False)
            if gt == pred
        )
        total = sum(len(labels) for labels, _ in pairs)
        return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# _RunHandle — lightweight run identifier used by run_slot
# ---------------------------------------------------------------------------


class _RunHandle:
    """Minimal run descriptor accepted by _send_one in slot context.

    Args:
        run_id: Slot-scoped run identifier.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


async def run_multi_experiment(
    experiment_dir: Path,
    callback_host: str,
    callback_port: int,
    result_timeout_s: float,
    dry_run: bool,
    node_host: str | None = None,
    metrics_host: str | None = None,
    resume: bool = False,
) -> None:
    """Load and execute a full multi-model experiment.

    Sub-experiments are executed sequentially.  Within each sub-experiment the
    sweep is expanded and each run is executed using the configured workload
    pattern.

    Args:
        experiment_dir: Directory containing the generated experiment.yaml.
        callback_host: Hostname pipeline nodes use to reach the result callback.
        callback_port: Port for the orchestrator's callback server.
        result_timeout_s: Per-task result wait timeout in seconds.
        dry_run: If True, log the sweep plan without executing.
        node_host: Override hostname used to reach all nodes.
        metrics_host: Override hostname for the metrics server.
    """
    exp_yaml = experiment_dir / "experiment.yaml"
    exp = load_multi_experiment_config(exp_yaml)
    logger.info(
        "Multi-model experiment: %s  pipelines: %d  sub-experiments: %d",
        exp.name,
        len(exp.pipelines),
        len(exp.sub_experiments),
    )

    resolved_metrics_host = metrics_host or exp.metrics_server.host
    metrics_url = f"http://{resolved_metrics_host}:{exp.metrics_server.port}"
    emitter = MetricsEmitter(server_url=metrics_url)
    await emitter.start()

    for sub_exp in exp.sub_experiments:
        await _run_sub_experiment(
            exp=exp,
            sub_exp=sub_exp,
            callback_host=callback_host,
            callback_port=callback_port,
            result_timeout_s=result_timeout_s,
            dry_run=dry_run,
            node_host=node_host,
            emitter=emitter,
            metrics_url=metrics_url,
            resume=resume,
        )

    await emitter.stop()
    logger.info("Multi-model experiment '%s' complete", exp.name)


async def _run_sub_experiment(
    exp: MultiExperimentConfig,
    sub_exp: MultiResolvedSubExperiment,
    callback_host: str,
    callback_port: int,
    result_timeout_s: float,
    dry_run: bool,
    node_host: str | None,
    emitter: MetricsEmitter,
    metrics_url: str = "",
    resume: bool = False,
) -> None:
    """Execute one sub-experiment: resolve its sweep and run each resolved run.

    Args:
        exp: Top-level multi-model experiment config.
        sub_exp: Sub-experiment to execute.
        callback_host: Hostname pipeline nodes use to reach the result callback.
        callback_port: Port for the callback server.
        result_timeout_s: Per-task result wait timeout in seconds.
        dry_run: If True, log the sweep plan without executing.
        node_host: Optional node hostname override.
        emitter: Shared metrics emitter (already started).
    """
    runs = exp.resolve_sweep(sub_exp)
    logger.info("[%s] Sweep: %d run(s)", sub_exp.name, len(runs))
    for run in runs:
        link_summary = ", ".join(
            f"{lk.pipeline_id} {lk.from_node}→{lk.to_node} {lk.compression.value}"
            + (f"@{lk.rate:.2f}" if lk.compression.value != "none" else "")
            for lk in run.links
        )
        logger.info(
            "  [%s][%s] %s", sub_exp.name, run.run_id, link_summary or "no links"
        )

    if dry_run:
        logger.info("[%s] Dry run — skipping execution", sub_exp.name)
        return

    await wait_for_multi_nodes_ready(exp, node_host=node_host)

    client = MultiDataClient(
        exp=exp,
        callback_host=callback_host,
        callback_port=callback_port,
        result_timeout_s=result_timeout_s,
    )

    async with client.session():
        for run in runs:
            if resume and await run_already_completed(
                run.run_id, "run_throughput", metrics_url, exp.name
            ):
                logger.info("[%s] Skipping completed run: %s", sub_exp.name, run.run_id)
                continue

            logger.info("[%s] Starting run: %s", sub_exp.name, run.run_id)
            if run.links:
                await push_multi_run_config(exp, run, node_host=node_host)
            else:
                logger.info(
                    "[%s] No links to configure for run '%s'", sub_exp.name, run.run_id
                )
            await client.run(run=run, node_host=node_host, emitter=emitter)


# ---------------------------------------------------------------------------
# Data client
# ---------------------------------------------------------------------------


class MultiDataClient:
    """Orchestrator-side data client for multi-model pipeline experiments.

    Manages a shared FastAPI callback server and drives task submission
    according to the experiment's workload pattern (fill, fixed_rate, poisson).

    Args:
        exp: Multi-model experiment config.
        callback_host: Hostname pipeline nodes use to reach the result callback.
        callback_port: Port for the callback server.
        result_timeout_s: Per-task result wait timeout in seconds.
    """

    def __init__(
        self,
        exp: MultiExperimentConfig,
        callback_host: str = "localhost",
        callback_port: int = 8080,
        result_timeout_s: float = 300.0,
    ) -> None:
        self._exp = exp
        self._callback_host = callback_host
        self._callback_port = callback_port
        self._result_timeout_s = result_timeout_s
        self._pending: dict[str, asyncio.Future[ResultPayload]] = {}
        self._server_task: asyncio.Task[None] | None = None
        self._server: uvicorn.Server | None = None
        self._app = self._make_app()
        self._tokenizer_cache: dict[str, Any] = {}
        self._answer_token_ids_cache: dict[str, list[int]] = {}
        self._init_tokenizers()

    def _init_tokenizers(self) -> None:
        """Pre-load tokenizers for all Llama pipelines in the experiment."""
        for pipeline in self._exp.pipelines:
            model_key = pipeline.model
            if (
                model_key.lower().startswith("llama")
                and model_key not in self._tokenizer_cache
            ):
                from transformers import AutoTokenizer  # type: ignore[import]

                dataset_cfg = self._exp.datasets[model_key]
                tokenizer_path = dataset_cfg.tokenizer_path or dataset_cfg.path
                tok = AutoTokenizer.from_pretrained(tokenizer_path)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                self._tokenizer_cache[model_key] = tok
                self._answer_token_ids_cache[model_key] = [
                    tok.encode(" " + c, add_special_tokens=False)[-1]
                    for c in ["A", "B", "C", "D"]
                ]

    @property
    def callback_url(self) -> str:
        """Base callback URL reachable by pipeline nodes."""
        return f"http://{self._callback_host}:{self._callback_port}/result"

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None, None]:
        """Start the callback server for the duration of an experiment session.

        Usage::

            async with client.session():
                for run in runs:
                    await client.run(...)
        """
        await self._start_server()
        try:
            yield
        finally:
            await self._stop_server()

    async def run(
        self,
        run: MultiResolvedRun,
        node_host: str | None,
        emitter: MetricsEmitter,
    ) -> None:
        """Submit all tasks for one sweep run and collect results.

        Drives the workload pattern defined in ``exp.workload``.  Emits
        ``TaskE2EEvent`` per task and ``RunThroughputEvent`` at run completion.

        Args:
            run: Resolved sweep run with concrete (pipeline, link) compression configs.
            node_host: Optional node hostname override.
            emitter: Metrics emitter for event emission.
        """
        exp = self._exp
        loaders = self._build_loaders()
        latencies: dict[str, list[float]] = {p.name: [] for p in exp.pipelines}
        results: dict[str, list[tuple[list[int], Any]]] = {
            p.name: [] for p in exp.pipelines
        }
        lock = asyncio.Lock()
        wall_start = time.perf_counter()

        if exp.workload.pattern == WorkloadPattern.FILL:
            await self._submit_fill(
                run, loaders, node_host, latencies, results, lock, emitter
            )
        elif exp.workload.pattern == WorkloadPattern.FIXED_RATE:
            await self._submit_rate_based(
                run,
                loaders,
                node_host,
                latencies,
                results,
                lock,
                emitter,
                poisson=False,
            )
        else:  # POISSON
            await self._submit_rate_based(
                run, loaders, node_host, latencies, results, lock, emitter, poisson=True
            )

        wall_time_s = time.perf_counter() - wall_start
        total_tasks = sum(len(v) for v in latencies.values())
        tasks_per_second = total_tasks / wall_time_s if wall_time_s > 0 else 0.0

        per_pipeline: dict[str, PipelineThroughputStats] = {}
        for pipeline_id, lats in latencies.items():
            if lats:
                sorted_lats = sorted(lats)
                n = len(sorted_lats)
                per_pipeline[pipeline_id] = PipelineThroughputStats(
                    tasks=n,
                    tasks_per_second=n / wall_time_s if wall_time_s > 0 else 0.0,
                    p50_ms=sorted_lats[int(n * 0.50)],
                    p90_ms=sorted_lats[int(n * 0.90)],
                    p99_ms=sorted_lats[min(int(n * 0.99), n - 1)],
                )

        emitter.emit(
            RunThroughputEvent(
                experiment_id=exp.name,
                run_id=run.run_id,
                wall_time_s=wall_time_s,
                total_tasks=total_tasks,
                tasks_per_second=tasks_per_second,
                per_pipeline=per_pipeline,
            )
        )
        _log_run_summary(run.run_id, latencies, results)

    def _build_loaders(self) -> dict[str, _PipelineLoader]:
        """Build a dataset loader for each pipeline in the experiment.

        Returns:
            Mapping from pipeline name to its loader.
        """
        loaders: dict[str, _PipelineLoader] = {}
        for pipeline in self._exp.pipelines:
            dataset_cfg = self._exp.datasets[pipeline.model]
            loaders[pipeline.name] = _PipelineLoader(
                pipeline=pipeline,
                dataset_cfg=dataset_cfg,
                tokenizer=self._tokenizer_cache.get(pipeline.model),
                answer_token_ids=self._answer_token_ids_cache.get(pipeline.model),
            )
        return loaders

    async def _submit_fill(
        self,
        run: MultiResolvedRun,
        loaders: dict[str, _PipelineLoader],
        node_host: str | None,
        latencies: dict[str, list[float]],
        results: dict[str, list[tuple[list[int], Any]]],
        lock: asyncio.Lock,
        emitter: MetricsEmitter,
    ) -> None:
        """Submit tasks using the fill (sliding window) pattern.

        Maintains ``window_per_pipeline`` tasks in-flight per pipeline,
        replenishing immediately on each completion.  ``window_per_pipeline == 1``
        is equivalent to closed-loop (one in-flight per pipeline at all times).

        Args:
            run: Resolved sweep run.
            loaders: Per-pipeline dataset loaders.
            node_host: Optional node hostname override.
            latencies: Mutable per-pipeline latency lists (appended to).
            results: Mutable per-pipeline result lists (appended to).
            lock: Shared asyncio lock protecting latencies/results.
            emitter: Metrics emitter.
        """
        exp = self._exp
        window = exp.workload.window_per_pipeline or 1
        semaphores = {p.name: asyncio.Semaphore(window) for p in exp.pipelines}
        all_tasks: list[asyncio.Task[None]] = []

        for pipeline in exp.pipelines:
            node = exp.node_for(pipeline.flow[0])
            first_node_url = f"http://{node_host or node.host}:{node.port}/infer"
            loader = loaders[pipeline.name]

            for batch_idx, input_tensor, labels, attention_mask in loader.batches():
                task = asyncio.create_task(
                    self._send_one(
                        pipeline_id=pipeline.name,
                        batch_idx=batch_idx,
                        input_tensor=input_tensor,
                        labels=labels,
                        attention_mask=attention_mask,
                        loader=loader,
                        first_node_url=first_node_url,
                        run=run,
                        semaphore=semaphores[pipeline.name],
                        latencies=latencies,
                        results=results,
                        lock=lock,
                        emitter=emitter,
                    )
                )
                all_tasks.append(task)

        await asyncio.gather(*all_tasks)

    async def _submit_rate_based(
        self,
        run: MultiResolvedRun,
        loaders: dict[str, _PipelineLoader],
        node_host: str | None,
        latencies: dict[str, list[float]],
        results: dict[str, list[tuple[list[int], Any]]],
        lock: asyncio.Lock,
        emitter: MetricsEmitter,
        poisson: bool,
    ) -> None:
        """Submit tasks at a controlled inter-arrival rate.

        Pipeline selection at each step follows the workload mix ratios,
        renormalized over non-exhausted pipelines.  Tasks are fired without
        concurrency limits — the submission loop controls the arrival rate.

        Args:
            run: Resolved sweep run.
            loaders: Per-pipeline dataset loaders.
            node_host: Optional node hostname override.
            latencies: Mutable per-pipeline latency lists (appended to).
            results: Mutable per-pipeline result lists (appended to).
            lock: Shared asyncio lock protecting latencies/results.
            emitter: Metrics emitter.
            poisson: If True, draw inter-arrival times from Exp(arrival_rate).
                     If False, use constant 1/arrival_rate intervals.
        """
        exp = self._exp
        arrival_rate = exp.workload.arrival_rate or 1.0
        pipeline_names = [p.name for p in exp.pipelines]
        mix = exp.workload.mix

        node_urls: dict[str, str] = {}
        for pipeline in exp.pipelines:
            node = exp.node_for(pipeline.flow[0])
            node_urls[pipeline.name] = (
                f"http://{node_host or node.host}:{node.port}/infer"
            )

        batch_iters: dict[
            str, Iterator[tuple[int, torch.Tensor, list[int], torch.Tensor | None]]
        ] = {name: iter(loaders[name].batches()) for name in pipeline_names}
        exhausted: set[str] = set()
        active_tasks: list[asyncio.Task[None]] = []

        while True:
            available = [p for p in pipeline_names if p not in exhausted]
            if not available:
                break

            weights = [mix[p] for p in available]
            total_weight = sum(weights)
            norm_weights = [w / total_weight for w in weights]
            pipeline_id = random.choices(available, weights=norm_weights, k=1)[0]

            try:
                batch_idx, input_tensor, labels, attention_mask = next(
                    batch_iters[pipeline_id]
                )
            except StopIteration:
                exhausted.add(pipeline_id)
                continue

            task = asyncio.create_task(
                self._send_one(
                    pipeline_id=pipeline_id,
                    batch_idx=batch_idx,
                    input_tensor=input_tensor,
                    labels=labels,
                    attention_mask=attention_mask,
                    loader=loaders[pipeline_id],
                    first_node_url=node_urls[pipeline_id],
                    run=run,
                    semaphore=None,
                    latencies=latencies,
                    results=results,
                    lock=lock,
                    emitter=emitter,
                )
            )
            active_tasks.append(task)

            if poisson:
                interval = random.expovariate(arrival_rate)
            else:
                interval = 1.0 / arrival_rate
            await asyncio.sleep(interval)

        await asyncio.gather(*active_tasks)

    async def _send_one(
        self,
        pipeline_id: str,
        batch_idx: int,
        input_tensor: torch.Tensor,
        labels: list[int],
        loader: _PipelineLoader,
        first_node_url: str,
        run: MultiResolvedRun | _RunHandle,
        semaphore: asyncio.Semaphore | None,
        latencies: dict[str, list[float]],
        results: dict[str, list[tuple[list[int], Any]]],
        lock: asyncio.Lock,
        emitter: MetricsEmitter,
        sub_experiment_name: str | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Submit one task to its pipeline's first node and collect the result.

        Optionally gated by a semaphore (fill pattern).  Emits ``TaskE2EEvent``
        and appends latency and decoded result to the run accumulators.

        Args:
            pipeline_id: Pipeline this task belongs to.
            batch_idx: Batch index within the pipeline's dataset.
            input_tensor: Input tensor (image batch or input_ids).
            labels: Ground-truth labels for this batch (empty for perplexity).
            loader: Dataset loader for decoding the result.
            first_node_url: POST /infer URL of the pipeline's first node.
            run: Resolved sweep run (for run_id tagging).
            semaphore: Optional concurrency limit (fill pattern only).
            latencies: Mutable per-pipeline latency accumulator.
            results: Mutable per-pipeline result accumulator.
            lock: Shared asyncio lock protecting latencies/results.
            emitter: Metrics emitter.
            sub_experiment_name: Active sub-experiment name; None for standalone sweeps.
        """
        task_id = f"{run.run_id}_{pipeline_id}_{batch_idx}_{uuid.uuid4().hex[:6]}"

        async def _execute() -> None:
            future: asyncio.Future[ResultPayload] = (
                asyncio.get_event_loop().create_future()
            )
            self._pending[task_id] = future
            submit_time = time.time()
            try:
                await _post_infer(
                    task_id=task_id,
                    pipeline_id=pipeline_id,
                    input_tensor=input_tensor,
                    first_node_url=first_node_url,
                    callback_url=self.callback_url,
                    experiment_id=self._exp.name,
                    run_id=run.run_id,
                    attention_mask=attention_mask,
                    metric_type=loader.metric_type,
                    answer_token_ids=loader.answer_token_ids_csv,
                    input_ids_header=(
                        base64.b64encode(pickle.dumps(input_tensor)).decode()
                        if loader.metric_type == "perplexity"
                        else None
                    ),
                )
                result_payload = await asyncio.wait_for(
                    future, timeout=self._result_timeout_s
                )
            except TimeoutError:
                logger.error(
                    "Timeout waiting for result of task %s (pipeline=%s, batch=%d)",
                    task_id,
                    pipeline_id,
                    batch_idx,
                )
                return
            except Exception:
                logger.exception(
                    "Error sending task %s (pipeline=%s, batch=%d)",
                    task_id,
                    pipeline_id,
                    batch_idx,
                )
                return
            finally:
                self._pending.pop(task_id, None)

            receive_time = time.time()
            latency_ms = (receive_time - submit_time) * 1000.0

            emitter.emit(
                TaskE2EEvent(
                    experiment_id=self._exp.name,
                    run_id=run.run_id,
                    pipeline_id=pipeline_id,
                    task_id=task_id,
                    submit_time=submit_time,
                    receive_time=receive_time,
                    latency_ms=latency_ms,
                    sub_experiment_name=sub_experiment_name,
                )
            )

            decoded = loader.decode(result_payload.data, input_tensor)
            async with lock:
                latencies[pipeline_id].append(latency_ms)
                results[pipeline_id].append((labels, decoded))

        if semaphore is not None:
            async with semaphore:
                await _execute()
        else:
            await _execute()

    async def run_slot(
        self,
        n_batches: int,
        run_id: str,
        node_host: str | None,
        emitter: MetricsEmitter,
        slot_id: int | None = None,
        sub_experiment_name: str | None = None,
        perplexity_baselines: dict[str, float] | None = None,
    ) -> SlotResult:
        """Submit n_batches tasks for one optimization slot and collect results.

        Distributes batches across pipelines according to ``exp.workload.mix``.
        Uses the fill (sliding-window) concurrency pattern with
        ``window_per_pipeline`` in-flight tasks per pipeline.

        Args:
            n_batches: Total number of batches to submit across all pipelines.
            run_id: Slot-scoped run identifier (used for event tagging).
            node_host: Optional node hostname override.
            emitter: Metrics emitter for TaskE2EEvent emission.
            slot_id: Optimizer slot index; None for standalone multi-model sweeps.
            sub_experiment_name: Active sub-experiment name; None for standalone sweeps.
            perplexity_baselines: Baseline NLL per pipeline for WikiText normalization.
                Passed through to ``SlotResult`` for use in ``quality_metric()``.

        Returns:
            SlotResult with per-pipeline latencies and decoded results.
        """
        exp = self._exp
        loaders = self._build_loaders()
        latencies: dict[str, list[float]] = {p.name: [] for p in exp.pipelines}
        results: dict[str, list[tuple[list[int], Any]]] = {
            p.name: [] for p in exp.pipelines
        }
        lock = asyncio.Lock()
        wall_start = time.perf_counter()

        # Distribute n_batches across pipelines by mix weight.
        mix = exp.workload.mix or {}
        total_mix = sum(mix.get(p.name, 1.0) for p in exp.pipelines)
        counts: dict[str, int] = {}
        allocated = 0
        pipelines = list(exp.pipelines)
        for i, pipeline in enumerate(pipelines):
            w = mix.get(pipeline.name, 1.0)
            if i == len(pipelines) - 1:
                counts[pipeline.name] = max(0, n_batches - allocated)
            else:
                c = round(n_batches * w / total_mix)
                counts[pipeline.name] = c
                allocated += c

        window = exp.workload.window_per_pipeline or 1
        semaphores = {p.name: asyncio.Semaphore(window) for p in exp.pipelines}
        run_handle = _RunHandle(run_id)
        all_tasks: list[asyncio.Task[None]] = []

        for pipeline in exp.pipelines:
            node = exp.node_for(pipeline.flow[0])
            first_node_url = f"http://{node_host or node.host}:{node.port}/infer"
            loader = loaders[pipeline.name]
            count = counts[pipeline.name]

            for batch_idx, input_tensor, labels, attention_mask in itertools.islice(
                loader.batches(), count
            ):
                task = asyncio.create_task(
                    self._send_one(
                        pipeline_id=pipeline.name,
                        batch_idx=batch_idx,
                        input_tensor=input_tensor,
                        labels=labels,
                        attention_mask=attention_mask,
                        loader=loader,
                        first_node_url=first_node_url,
                        run=run_handle,
                        semaphore=semaphores[pipeline.name],
                        latencies=latencies,
                        results=results,
                        lock=lock,
                        emitter=emitter,
                        sub_experiment_name=sub_experiment_name,
                    )
                )
                all_tasks.append(task)

        await asyncio.gather(*all_tasks)
        wall_time_s = time.perf_counter() - wall_start

        per_pipeline: dict[str, PipelineThroughputStats] = {}
        total_tasks = 0
        for pid, lats in latencies.items():
            if lats:
                sorted_lats = sorted(lats)
                n = len(sorted_lats)
                total_tasks += n
                per_pipeline[pid] = PipelineThroughputStats(
                    tasks=n,
                    tasks_per_second=n / wall_time_s if wall_time_s > 0 else 0.0,
                    p50_ms=sorted_lats[int(n * 0.50)],
                    p90_ms=sorted_lats[int(n * 0.90)],
                    p99_ms=sorted_lats[min(int(n * 0.99), n - 1)],
                )
        emitter.emit(
            RunThroughputEvent(
                experiment_id=exp.name,
                run_id=run_id,
                wall_time_s=wall_time_s,
                total_tasks=total_tasks,
                tasks_per_second=total_tasks / wall_time_s if wall_time_s > 0 else 0.0,
                per_pipeline=per_pipeline,
                slot_id=slot_id,
                sub_experiment_name=sub_experiment_name,
            )
        )

        return SlotResult(
            per_pipeline_latency_ms=latencies,
            per_pipeline_results=results,
            wall_time_s=wall_time_s,
            perplexity_baselines=perplexity_baselines or {},
        )

    def _make_app(self) -> FastAPI:
        """Build the FastAPI callback app with /result and /health endpoints."""
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
# Pipeline dataset loader
# ---------------------------------------------------------------------------


class _PipelineLoader:
    """Dataset loader and result decoder for one pipeline.

    Wraps ``get_dataset`` (resnet-like models), ``_WikiText2Batches``, or
    ``_MMLUBatches`` (Llama models) behind a unified interface.

    Args:
        pipeline: Pipeline config (used for model type detection).
        dataset_cfg: Dataset config for the pipeline's model type.
        tokenizer: Pre-loaded tokenizer (required for Llama pipelines).
        answer_token_ids: Token IDs for A/B/C/D (required for MMLU).
    """

    def __init__(
        self,
        pipeline: PipelineConfig,
        dataset_cfg: DatasetConfig,
        tokenizer: Any | None,
        answer_token_ids: list[int] | None,
    ) -> None:
        self._pipeline_name = pipeline.name
        self._answer_token_ids = answer_token_ids
        model = pipeline.model.lower()

        if model.startswith("llama"):
            dataset_name = dataset_cfg.name.lower()
            if dataset_name == "wikitext2":
                from framework.nodes.orchestrator.llama_data_client import (
                    _WikiText2Batches,
                )

                self._loader = _WikiText2Batches(dataset_cfg, tokenizer)
                self._metric_type = "perplexity"
            elif dataset_name == "mmlu":
                from framework.nodes.orchestrator.llama_data_client import _MMLUBatches

                self._loader = _MMLUBatches(dataset_cfg, tokenizer)
                self._metric_type = "accuracy_llama"
            else:
                raise ValueError(
                    f"Unsupported Llama dataset '{dataset_cfg.name}'. "
                    "Use 'wikitext2' or 'mmlu'."
                )
        else:
            self._loader = get_dataset(dataset_cfg)
            self._metric_type = "accuracy"

    def batches(
        self,
    ) -> Iterator[tuple[int, torch.Tensor, list[int], torch.Tensor | None]]:
        """Yield (batch_idx, input_tensor, labels, attention_mask) tuples.

        attention_mask is None for ResNet and WikiText-2; present for MMLU
        batches with batch_size > 1.
        """
        if self._metric_type == "accuracy":
            # ResNet: underlying loader yields 3-tuples; synthesize None mask.
            for batch_idx, input_tensor, labels in self._loader.batches():
                yield batch_idx, input_tensor, labels, None
        else:
            # Llama loaders already yield 4-tuples.
            yield from self._loader.batches()

    @property
    def metric_type(self) -> str | None:
        """Return the metric type header value for on-node computation, or None for ResNet."""
        if self._metric_type == "perplexity":
            return "perplexity"
        if self._metric_type == "accuracy_llama":
            return "accuracy"
        return None

    @property
    def answer_token_ids_csv(self) -> str | None:
        """Return comma-separated answer token IDs for MMLU, or None."""
        if self._answer_token_ids is not None:
            return ",".join(str(t) for t in self._answer_token_ids)
        return None

    def num_batches(self) -> int:
        """Return the total number of batches in this pipeline's dataset."""
        return self._loader.num_batches()

    def decode(
        self, data: bytes, input_tensor: torch.Tensor | None
    ) -> tuple[float, int] | list[int]:
        """Decode a raw result payload into a metric-specific output.

        Args:
            data: Pickled result tensor from the last node (raw bytes).
            input_tensor: Original input sent to the first node.  Required for
                WikiText-2 perplexity (used to compute shift-labels).

        Returns:
            ``(nll_sum, token_count)`` for perplexity pipelines, or
            ``list[int]`` of predicted class/answer indices for accuracy pipelines.
        """
        if self._metric_type == "perplexity":
            from framework.nodes.orchestrator.llama_data_client import (
                _decode_wikitext_result,
            )

            return _decode_wikitext_result(data, input_tensor)
        elif self._metric_type == "accuracy_llama":
            from framework.nodes.orchestrator.llama_data_client import (
                _decode_mmlu_result,
            )

            return _decode_mmlu_result(data, self._answer_token_ids)  # type: ignore[arg-type]
        else:
            tensor: torch.Tensor = pickle.loads(data)
            return tensor.argmax(dim=1).tolist()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _post_infer(
    task_id: str,
    pipeline_id: str,
    input_tensor: torch.Tensor,
    first_node_url: str,
    callback_url: str,
    experiment_id: str,
    run_id: str,
    attention_mask: torch.Tensor | None = None,
    metric_type: str | None = None,
    answer_token_ids: str | None = None,
    input_ids_header: str | None = None,
) -> None:
    """Pickle and POST a task to a multi-model pipeline's first node.

    Args:
        task_id: Unique task identifier.
        pipeline_id: Pipeline this task belongs to.
        input_tensor: Input tensor (image batch or input_ids).
        first_node_url: POST /infer URL of the pipeline's first node.
        callback_url: URL the last node should POST results to.
        experiment_id: Experiment name for metrics tagging.
        run_id: Run identifier for metrics tagging.
        attention_mask: Optional padding mask [B, L]; present for MMLU batches
            with batch_size > 1.
        metric_type: ``"perplexity"`` or ``"accuracy"`` for on-node computation.
        answer_token_ids: Comma-separated token IDs for MMLU answer choices.
        input_ids_header: Base64-encoded pickled input_ids for WikiText perplexity.
    """
    data = pickle.dumps(input_tensor)
    headers: dict[str, str] = {
        "x-task-id": task_id,
        "x-pipeline-id": pipeline_id,
        "x-callback-url": callback_url,
        "x-experiment-id": experiment_id,
        "x-run-id": run_id,
        "content-type": "application/octet-stream",
    }
    if attention_mask is not None:
        headers["x-attention-mask"] = base64.b64encode(
            pickle.dumps(attention_mask.bool())
        ).decode()
    if metric_type is not None:
        headers["x-metric-type"] = metric_type
    if answer_token_ids is not None:
        headers["x-answer-token-ids"] = answer_token_ids
    if input_ids_header is not None:
        headers["x-input-ids"] = input_ids_header
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(first_node_url, content=data, headers=headers)
        resp.raise_for_status()


def _log_run_summary(
    run_id: str,
    latencies: dict[str, list[float]],
    results: dict[str, list[tuple[list[int], Any]]],
) -> None:
    """Log per-pipeline metric summaries for one completed sweep run.

    Args:
        run_id: Run identifier for log prefixes.
        latencies: Per-pipeline list of task E2E latencies in milliseconds.
        results: Per-pipeline list of (labels, decoded_result) pairs.
    """
    for pipeline_id, lats in latencies.items():
        if not lats:
            logger.warning("Run '%s' pipeline '%s': no results", run_id, pipeline_id)
            continue

        pipeline_results = results[pipeline_id]

        if pipeline_results:
            _, first_result = pipeline_results[0]
            if isinstance(first_result, tuple):
                # Perplexity: (nll_sum, token_count)
                total_nll = sum(r[0] for _, r in pipeline_results)
                total_tok = sum(r[1] for _, r in pipeline_results)
                ppl = math.exp(total_nll / total_tok) if total_tok > 0 else float("inf")
                logger.info(
                    "Run '%s' pipeline '%s' perplexity: %.2f",
                    run_id,
                    pipeline_id,
                    ppl,
                )
            elif isinstance(first_result, list):
                # Accuracy: list of predicted class indices
                correct = sum(
                    1
                    for labels, predicted in pipeline_results
                    for gt, pred in zip(labels, predicted, strict=False)
                    if gt == pred
                )
                total = sum(len(labels) for labels, _ in pipeline_results)
                accuracy = correct / total if total > 0 else 0.0
                logger.info(
                    "Run '%s' pipeline '%s' accuracy: %.2f%% (%d/%d)",
                    run_id,
                    pipeline_id,
                    accuracy * 100,
                    correct,
                    total,
                )

        avg_lat = sum(lats) / len(lats)
        logger.info(
            "Run '%s' pipeline '%s' latency: avg=%.1fms p50=%.1fms tasks=%d",
            run_id,
            pipeline_id,
            avg_lat,
            sorted(lats)[len(lats) // 2],
            len(lats),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the multi-model experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run a multi-model DNN compression experiment sweep."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to the experiment directory (containing experiment.yaml)",
    )
    parser.add_argument(
        "--callback-host",
        default=os.getenv("CALLBACK_HOST", "localhost"),
        help="Hostname pipeline nodes use to reach the result callback "
        "(default: CALLBACK_HOST env var or 'localhost')",
    )
    parser.add_argument(
        "--callback-port",
        type=int,
        default=int(os.getenv("CALLBACK_PORT", "8080")),
        help="Port for the orchestrator's callback server (default: 8080)",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=300.0,
        help="Per-task result wait timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the sweep plan without executing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have results in the metrics server",
    )
    parser.add_argument(
        "--node-host",
        default=os.getenv("NODE_HOST"),
        help="Override hostname used to reach all nodes (e.g. 'localhost' when "
        "running against Docker with published ports). Can also be set via "
        "NODE_HOST env var.",
    )
    parser.add_argument(
        "--metrics-host",
        default=os.getenv("METRICS_HOST"),
        help="Override hostname used to reach the metrics server. Defaults to "
        "metrics_server.host in the experiment config. Can also be set via "
        "METRICS_HOST env var.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_multi_experiment(
            experiment_dir=args.experiment_dir,
            callback_host=args.callback_host,
            callback_port=args.callback_port,
            result_timeout_s=args.result_timeout,
            dry_run=args.dry_run,
            node_host=args.node_host,
            metrics_host=args.metrics_host,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
