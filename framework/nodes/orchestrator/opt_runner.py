"""Optimization experiment runner.

Executes the ordered list of sub-experiments defined in a generated
``experiments/opt/<name>/experiment.yaml``.  Sub-experiments run sequentially;
each type shares infrastructure (data client, link prober, artifact store,
metrics emitter) and builds on artifacts produced by earlier phases.

Phase dispatch
--------------
- ``profiling``      — runs at η=1.0, measures nominal throughput and accuracy.
- ``accuracy_model`` — fits or reloads a surrogate A_k(η) per pipeline.
- ``no_csi``         — primal-dual slot loop, one run per μ in mu_sweep.
- ``csi_aware``      — closed-form η* slot loop using channel capacity estimates.
- ``random``         — uniform random η baseline slot loop.
- ``fifo``           — η=1.0 baseline slot loop.

CLI usage::

    python -m framework.nodes.orchestrator.opt_runner \\
        experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps \\
        --callback-host orchestrator \\
        --callback-port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch

from framework.datamodels.api import MultiConfigUpdate
from framework.datamodels.events import (
    OptSlotEvent,
    SubExperimentEvent,
    TaskAccuracyEvent,
    ThroughputConstraintEvent,
)
from framework.datamodels.experiment import CompressionMethod
from framework.datamodels.multi_experiment import WorkloadPattern
from framework.datamodels.opt_experiment import (
    AccuracyModelSubExperiment,
    CsiAwareSubExperiment,  # noqa: F401
    DecoupledDescentSubExperiment,  # noqa: F401
    EstimatedCsiSingleSubExperiment,  # noqa: F401
    GeneratedOptExperimentConfig,
    HistoricalAverageCESubExperiment,  # noqa: F401
    MaxCompressionMultiSubExperiment,  # noqa: F401
    MaxCompressionSingleSubExperiment,  # noqa: F401
    NoCompressionMultiSubExperiment,  # noqa: F401
    NoCompressionSingleSubExperiment,  # noqa: F401
    NoCsiSubExperiment,
    OptSubExperiment,
    ProfilingSubExperiment,
    ProportionalResourceSubExperiment,  # noqa: F401
    QueueProportionalSubExperiment,  # noqa: F401
    StaticEqualShareSubExperiment,  # noqa: F401
    StrictPriorityGreedySubExperiment,  # noqa: F401
    UniformCompressionSingleSubExperiment,  # noqa: F401
)
from framework.nodes.metrics.emitter import MetricsEmitter
from framework.nodes.orchestrator.artifacts import ArtifactStore
from framework.nodes.orchestrator.link_prober import LinkProber
from framework.nodes.orchestrator.multi_controller import (
    wait_for_multi_nodes_ready,
)
from framework.nodes.orchestrator.multi_runner import MultiDataClient
from framework.optimizer.accuracy_model import AccuracyModel, build_accuracy_model
from framework.optimizer.compression_mapper import CompressionMapper
from framework.optimizer.inference_optimizer_adapter import (
    BaseOptimizerAdapter,
    build_adapter,
    build_global_order,
    build_inference_tasks,
    probe_dict_to_c_t_vector,
)
from framework.utils.loader import load_opt_experiment_config

logger = logging.getLogger(__name__)

_CONFIG_DRAIN_TIMEOUT_S = 30.0
_CONFIG_HTTP_TIMEOUT_S = _CONFIG_DRAIN_TIMEOUT_S + 5.0


# ---------------------------------------------------------------------------
# Config push
# ---------------------------------------------------------------------------


async def push_opt_slot_config(
    eta_per_pipeline_per_link: dict[str, dict[str, float]],
    exp: GeneratedOptExperimentConfig,
    mapper: CompressionMapper,
    node_host: str | None = None,
    drain_timeout_s: float = _CONFIG_DRAIN_TIMEOUT_S,
) -> None:
    """Push per-pipeline compression config for a slot's η decisions to all nodes.

    For each link, resolves which pipelines traverse it, maps η to a concrete
    compression method via ``mapper``, and concurrently pushes outgoing config
    to the sending node and incoming config to the receiving node.

    Args:
        eta_per_pipeline_per_link: Optimizer-selected η per pipeline per link
            (keyed by pipeline_id → link_id → η).
        exp: Generated experiment config (provides node host/port and pipeline
            flow information).
        mapper: Compression mapper for translating η to method + params.
        node_host: Override hostname for all nodes (useful in Docker/k8s).
        drain_timeout_s: How long each node waits for its queue to drain
            before applying the new config.
    """
    node_map = {n.name: n for n in exp.nodes}

    # Build link_id → [pipeline_ids that traverse this link].
    link_pipelines: dict[str, list[str]] = {}
    for link in exp.links:
        traversing: list[str] = []
        for pipeline in exp.pipelines:
            flow = pipeline.flow
            for i in range(len(flow) - 1):
                if flow[i] == link.from_node and flow[i + 1] == link.to_node:
                    traversing.append(pipeline.name)
                    break
        link_pipelines[link.link_id] = traversing

    push_tasks: list[Any] = []
    async with httpx.AsyncClient(timeout=_CONFIG_HTTP_TIMEOUT_S) as client:
        for link in exp.links:
            link_id = link.link_id
            from_node = node_map[link.from_node]
            to_node = node_map[link.to_node]

            for pipeline_id in link_pipelines[link_id]:
                eta = eta_per_pipeline_per_link.get(pipeline_id, {}).get(
                    link_id, link.eta_max
                )
                decision = mapper.map(link_id, pipeline_id, eta)
                try:
                    method_enum = CompressionMethod(decision.method)
                except ValueError:
                    logger.warning(
                        "Unknown compression method %r for link %s pipeline %s; "
                        "falling back to topk",
                        decision.method,
                        link_id,
                        pipeline_id,
                    )
                    method_enum = CompressionMethod.TOPK

                outlier_prec = decision.params.get("outlier_precision", "fp16")
                regular_prec = decision.params.get("regular_precision", "int8")

                for node, direction in [
                    (from_node, "outgoing"),
                    (to_node, "incoming"),
                ]:
                    url = f"http://{node_host or node.host}:{node.port}/config"
                    payload = MultiConfigUpdate(
                        pipeline_id=pipeline_id,
                        direction=direction,
                        method=method_enum,
                        rate=decision.effective_eta,
                        drain_timeout_s=drain_timeout_s,
                        outlier_precision=outlier_prec,
                        regular_precision=regular_prec,
                    )
                    push_tasks.append(
                        _push_one_config(client, url, payload, node.name, pipeline_id)
                    )

        await asyncio.gather(*push_tasks)


async def _push_one_config(
    client: httpx.AsyncClient,
    url: str,
    payload: MultiConfigUpdate,
    node_name: str,
    pipeline_id: str,
) -> None:
    """POST one MultiConfigUpdate to a node, raising on failure.

    Args:
        client: Shared httpx client.
        url: Full /config URL.
        payload: Config update payload.
        node_name: Node name for log messages.
        pipeline_id: Pipeline name for log messages.
    """
    try:
        resp = await client.post(url, json=payload.model_dump())
        resp.raise_for_status()
        logger.debug(
            "Config pushed: node=%s pipeline=%s %s method=%s rate=%.3f",
            node_name,
            pipeline_id,
            payload.direction,
            payload.method.value,
            payload.rate,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to push {payload.direction} config for pipeline "
            f"'{pipeline_id}' to node '{node_name}' at {url}: {exc}"
        ) from exc


async def push_wfq_weights(
    s_comp_per_node: dict[str, dict[str, float]],
    exp: GeneratedOptExperimentConfig,
    node_host: str | None = None,
) -> None:
    """Push per-pipeline WFQ scheduling weights to each node.

    Called once per slot alongside ``push_opt_slot_config`` to apply the
    compute-share allocation (s_comp) produced by the optimizer.  Each node
    receives a ``POST /config/weights`` with the pipeline→weight mapping for
    the pipelines that traverse it.

    If ``s_comp_per_node`` is empty (e.g. infeasible slot), no request is sent
    and nodes retain their current weights.

    Note: s_comm (per-pipeline link bandwidth shares) is not actuated here —
    the HTTP transport does not support per-pipeline rate limiting at the link
    layer.  Nodes use the physical link bandwidth as-is; only compute shares
    (s_comp) are controlled via WFQ.  Implementing s_comm would require
    per-pipeline egress shaping at the application layer (TODO).

    Args:
        s_comp_per_node: ``{node_name: {pipeline_id: weight}}`` from
            ``extract_s_comp_per_pipeline_per_node``.
        exp: Generated experiment config (for node host/port lookup).
        node_host: Override hostname for all nodes.
    """
    if not s_comp_per_node:
        return

    node_map = {n.name: n for n in exp.nodes}
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for node_name, weights in s_comp_per_node.items():
            node = node_map.get(node_name)
            if node is None or not weights:
                continue
            url = f"http://{node_host or node.host}:{node.port}/config/weights"
            tasks.append(_push_node_weights(client, url, weights, node_name))
        if tasks:
            await asyncio.gather(*tasks)


async def _push_node_weights(
    client: httpx.AsyncClient,
    url: str,
    weights: dict[str, float],
    node_name: str,
) -> None:
    """POST WFQ weights to one node, logging on failure without raising.

    Args:
        client: Shared httpx client.
        url: Full ``/config/weights`` URL for the node.
        weights: ``{pipeline_id: weight}`` mapping to apply.
        node_name: Node name for log messages.
    """
    try:
        resp = await client.post(url, json={"weights": weights})
        resp.raise_for_status()
        logger.debug("WFQ weights pushed: node=%s weights=%s", node_name, weights)
    except Exception as exc:
        logger.warning(
            "Failed to push WFQ weights to node '%s': %s — keeping current weights",
            node_name,
            exc,
        )


# ---------------------------------------------------------------------------
# OptRunner
# ---------------------------------------------------------------------------


class OptRunner:
    """Executes an optimization experiment's ordered sub-experiments.

    Owns references to the data client, link prober, artifact store, and
    metrics emitter.  Sub-experiments run sequentially and share these
    resources.

    Args:
        exp: Generated optimization experiment config.
        data_client: Multi-model data client (must be inside a session context).
        link_prober: Orchestrator-side link prober.
        artifact_store: Artifact storage for profiling and model artifacts.
        emitter: Metrics emitter (already started).
        metrics_url: Base URL of the metrics server.
        node_host: Optional node hostname override.
    """

    def __init__(
        self,
        exp: GeneratedOptExperimentConfig,
        data_client: MultiDataClient,
        link_prober: LinkProber,
        artifact_store: ArtifactStore,
        emitter: MetricsEmitter,
        metrics_url: str,
        node_host: str | None = None,
    ) -> None:
        self._exp = exp
        self._data_client = data_client
        self._link_prober = link_prober
        self._artifacts = artifact_store
        self._emitter = emitter
        self._metrics_url = metrics_url
        self._node_host = node_host
        self._mapper = CompressionMapper(exp.links)
        self._pipeline_ids = [p.name for p in exp.pipelines]

        # In-memory cache of loaded accuracy models (keyed by (sub_exp.name, scheme)).
        self._accuracy_models: dict[tuple[str, str], dict[str, AccuracyModel]] = {}

        # Profiling artifact cache: nominal bps per link at η=1.0.
        self._nominal_bps_per_link: dict[str, float] = {}
        self._tau_per_node: dict[str, dict[str, float]] = {}
        self._a_per_link_bytes: dict[str, float] = {}

        # Baseline NLL per pipeline for WikiText perplexity normalization.
        # Populated during profiling (η=1.0) and passed to run_slot() calls.
        self._perplexity_baselines: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Execute all sub-experiments for each compression scheme in definition order.

        The scheme list is taken from ``exp.links[0].allowed_methods`` (or
        ``["topk"]`` when unset).  For each scheme the mapper is locked to that
        scheme via ``set_active_scheme`` and all sub-experiments are run in
        order.  ``ProfilingSubExperiment`` is skipped on subsequent scheme
        iterations because profiling is scheme-independent.

        Sub-experiments are run sequentially within each scheme.  An exception
        in any phase propagates immediately and halts the run.
        """
        schemes = (
            self._exp.links[0].allowed_methods
            if self._exp.links and self._exp.links[0].allowed_methods
            else ["topk"]
        )

        for scheme_idx, scheme in enumerate(schemes):
            self._mapper.set_active_scheme(scheme)
            logger.info(
                "=== Scheme: %s (%d/%d) ===", scheme, scheme_idx + 1, len(schemes)
            )

            for sub_exp in self._exp.sub_experiments:
                if scheme_idx > 0 and isinstance(sub_exp, ProfilingSubExperiment):
                    logger.info(
                        "[%s] Skipping profiling for scheme '%s' (already done)",
                        sub_exp.name,
                        scheme,
                    )
                    continue

                scoped_name = f"{sub_exp.name}__{scheme}"
                logger.info(
                    "=== Sub-experiment: %s (type=%s) scheme=%s ===",
                    scoped_name,
                    sub_exp.type,
                    scheme,
                )
                t_sub_start = time.perf_counter()
                await self._run_sub_experiment(sub_exp, scheme=scheme)
                duration_s = time.perf_counter() - t_sub_start
                logger.info(
                    "[%s] Sub-experiment complete in %.1fs", scoped_name, duration_s
                )
                if isinstance(sub_exp, ProfilingSubExperiment):
                    n_runs = 1
                elif isinstance(sub_exp, AccuracyModelSubExperiment):
                    n_runs = sub_exp.n_sweep_samples
                elif isinstance(sub_exp, NoCsiSubExperiment):
                    n_runs = self._exp.optimization_loop.n_slots * len(sub_exp.mu_sweep)
                else:
                    n_runs = self._exp.optimization_loop.n_slots
                self._emitter.emit(
                    SubExperimentEvent(
                        experiment_id=self._exp.name,
                        run_id=scoped_name,
                        sub_experiment_name=scoped_name,
                        duration_s=duration_s,
                        n_runs=n_runs,
                        compression_scheme=scheme,
                    )
                )
        logger.info("Optimization experiment '%s' complete", self._exp.name)

    async def _run_sub_experiment(
        self, sub_exp: OptSubExperiment, scheme: str = "topk"
    ) -> None:
        if isinstance(sub_exp, ProfilingSubExperiment):
            await self._run_profiling(sub_exp)
        elif isinstance(sub_exp, AccuracyModelSubExperiment):
            await self._run_accuracy_model(sub_exp, scheme=scheme)
        elif isinstance(sub_exp, NoCsiSubExperiment):
            # Build simulations once outside the mu loop to avoid reloading
            # the Llama model for every mu value.
            stein_cfg = getattr(sub_exp, "stein_config", None)
            pre_sims: dict[str, Any] | None = None
            if stein_cfg is not None:
                from framework.optimizer.simulation_factory import (  # noqa: PLC0415
                    build_simulations,
                )

                pre_sims = build_simulations(
                    self._exp,
                    stein_cfg,
                    self._mapper,
                    dataset_override=self._exp.stein_datasets,
                )
            try:
                for mu in sub_exp.mu_sweep:
                    await self._run_opt_with_adapter(
                        sub_exp, mu=mu, pre_built_simulations=pre_sims, scheme=scheme
                    )
            finally:
                if pre_sims:
                    for sim in pre_sims.values():
                        try:
                            sim.remove_hooks()
                        except Exception:
                            pass
        else:
            # All other optimization sub-experiments (baselines + CSI-aware).
            # CSI-aware never uses the Stein oracle (closed-form η*), so skip
            # simulation building even if stein_config is set.
            await self._run_opt_with_adapter(
                sub_exp,
                skip_stein=isinstance(sub_exp, CsiAwareSubExperiment),
                scheme=scheme,
            )

    # ------------------------------------------------------------------
    # Phase: profiling
    # ------------------------------------------------------------------

    async def _run_profiling(self, sub_exp: ProfilingSubExperiment) -> None:
        """Run profiling at η=1.0 and store nominal-throughput artifacts.

        Pushes no-compression config (η=eta_max) to all links, runs
        ``profiling_batches`` inference rounds, probes all links, and
        persists per-pipeline accuracy/latency + per-link nominal bps.

        Args:
            sub_exp: Profiling sub-experiment config.
        """
        artifact_path = self._artifacts.root / "profiling" / f"{sub_exp.name}.json"
        config_hash = self._artifacts.config_hash(
            {
                "profiling_batches": self._exp.optimization_loop.profiling_batches,
                "pipelines": [p.name for p in self._exp.pipelines],
            }
        )

        if self._artifacts.is_valid(artifact_path, config_hash):
            logger.info("[%s] Reusing cached profiling artifact", sub_exp.name)
            data = self._artifacts.read_json(artifact_path)
            self._nominal_bps_per_link = data.get("nominal_bps_per_link", {})
            self._tau_per_node = data.get("tau_per_node_s", {})
            self._a_per_link_bytes = data.get("a_per_link_bytes", {})
            self._perplexity_baselines = data.get("perplexity_baselines", {})
            return

        # Push η=eta_max (no compression) to all links.
        pipeline_ids = [p.name for p in self._exp.pipelines]
        eta_full = {
            pid: {lk.link_id: lk.eta_max for lk in self._exp.links}
            for pid in pipeline_ids
        }
        await push_opt_slot_config(
            eta_full, self._exp, self._mapper, self._node_host, _CONFIG_DRAIN_TIMEOUT_S
        )

        run_id = f"{self._exp.name}_{sub_exp.name}"

        # Warm-up: run several untimed batches so CUDA JIT compiles on every node
        # before timed profiling begins.  A single batch is insufficient — the
        # first profiling batch still hits kernel-compilation latency on intermediate
        # nodes (e.g. node B), inflating τ by 100–1000×.  Use a distinct run_id so
        # these events are excluded from _query_tau_and_a.
        await self._data_client.run_slot(
            n_batches=5,
            run_id=f"{run_id}_warmup",
            node_host=self._node_host,
            emitter=self._emitter,
        )

        slot_result = await self._data_client.run_slot(
            n_batches=self._exp.optimization_loop.profiling_batches,
            run_id=run_id,
            node_host=self._node_host,
            emitter=self._emitter,
        )

        # Compute WikiText perplexity baselines from the η=1.0 profiling slot.
        # For perplexity pipelines, decoded results are (nll_sum, token_count)
        # tuples; mean NLL at uncompressed η is the normalization denominator.
        perplexity_baselines: dict[str, float] = {}
        for pid, pairs in slot_result.per_pipeline_results.items():
            if not pairs:
                continue
            _, first_decoded = pairs[0]
            if isinstance(first_decoded, tuple):
                total_nll = sum(float(nll) for _, (nll, _) in pairs)
                total_tokens = sum(int(tc) for _, (_, tc) in pairs)
                if total_tokens > 0:
                    perplexity_baselines[pid] = total_nll / total_tokens
        self._perplexity_baselines = perplexity_baselines

        # Probe links at η=1.0 to measure nominal channel capacity.
        probe_results = await self._link_prober.probe_all(
            slot_id=None,
            experiment_id=self._exp.name,
            run_id=run_id,
            sub_experiment_name=sub_exp.name,
        )
        self._nominal_bps_per_link = probe_results

        # Query per-node compute times and per-link activation sizes from
        # the metrics server.  These are needed by InferenceTask construction
        # in the external optimizer adapter.  Brief sleep first to allow nodes
        # to flush their async metric events before we query.
        await asyncio.sleep(2.0)
        tau_per_node, a_per_link = await self._query_tau_and_a(
            run_id=run_id,
            experiment_id=self._exp.name,
        )

        per_pipeline: dict[str, Any] = {}
        for pipeline in self._exp.pipelines:
            pid = pipeline.name
            lats = slot_result.per_pipeline_latency_ms.get(pid, [])
            per_pipeline[pid] = {
                "accuracy": slot_result.quality_metric(pid),
                "avg_latency_ms": sum(lats) / len(lats) if lats else 0.0,
                "n_samples": len(lats),
            }

        artifact_data: dict[str, Any] = {
            "per_pipeline": per_pipeline,
            "nominal_bps_per_link": probe_results,
            "tau_per_node_s": tau_per_node,
            "a_per_link_bytes": a_per_link,
            "perplexity_baselines": perplexity_baselines,
        }
        self._artifacts.write_json(artifact_path, artifact_data, config_hash)
        self._tau_per_node: dict[str, dict[str, float]] = tau_per_node
        self._a_per_link_bytes: dict[str, float] = a_per_link
        logger.info(
            "[%s] Profiling complete: pipelines=%s links=%s tau_nodes=%s a_links=%s",
            sub_exp.name,
            list(per_pipeline.keys()),
            {k: f"{v / 1e6:.1f}Mbps" for k, v in probe_results.items()},
            list(tau_per_node.keys()),
            {k: f"{v / 1024:.1f}KB" for k, v in a_per_link.items()},
        )

    async def _query_tau_and_a(
        self,
        run_id: str,
        experiment_id: str,
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """Query the metrics server for profiling timing and activation size data.

        Fetches ``task_node_timing`` events (for per-node compute latency τ_i)
        and ``send`` events (for per-link activation size a_i) emitted during
        the profiling slot, then aggregates them.

        Args:
            run_id: Run identifier used to filter events to this slot.
            experiment_id: Experiment identifier (used as ``experiment_name_contains``
                filter so only this experiment's events are scanned).

        Returns:
            Tuple of:
              - ``tau_per_node_s``: ``{pipeline_id: {node_id: mean_compute_s}}``
              - ``a_per_link_bytes``: ``{link_id: mean_payload_bytes}``
        """
        tau_per_node: dict[str, dict[str, float]] = {}
        a_per_link: dict[str, float] = {}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # --- tau: per-node compute latency ---
                resp = await client.get(
                    f"{self._metrics_url}/metrics/query",
                    params={
                        "event_type": "task_node_timing",
                        "run_id": run_id,
                        "experiment_name_contains": experiment_id,
                        "limit": "10000",
                    },
                )
                resp.raise_for_status()
                timing_events: list[dict[str, Any]] = resp.json().get("events", [])

                # Accumulate compute durations per (pipeline_id, node_id).
                from collections import defaultdict  # noqa: PLC0415

                sums: dict[tuple[str, str], float] = defaultdict(float)
                counts: dict[tuple[str, str], int] = defaultdict(int)
                for ev in timing_events:
                    pid = ev.get("pipeline_id")
                    nid = ev.get("node_id")
                    cs = ev.get("compute_start")
                    ce = ev.get("compute_end")
                    if pid and nid and cs is not None and ce is not None:
                        sums[(pid, nid)] += float(ce) - float(cs)
                        counts[(pid, nid)] += 1

                for (pid, nid), total in sums.items():
                    tau_per_node.setdefault(pid, {})[nid] = total / counts[(pid, nid)]

                # --- a_i: per-link mean payload bytes ---
                resp = await client.get(
                    f"{self._metrics_url}/metrics/query",
                    params={
                        "event_type": "send",
                        "run_id": run_id,
                        "experiment_name_contains": experiment_id,
                        "limit": "10000",
                    },
                )
                resp.raise_for_status()
                send_events: list[dict[str, Any]] = resp.json().get("events", [])

                byte_sums: dict[str, float] = defaultdict(float)
                byte_counts: dict[str, int] = defaultdict(int)
                for ev in send_events:
                    fn = ev.get("from_node")
                    tn = ev.get("to_node")
                    pb = ev.get("payload_bytes")
                    if fn and tn and pb is not None:
                        link_id = f"{fn}-{tn}"
                        byte_sums[link_id] += float(pb)
                        byte_counts[link_id] += 1

                for link_id, total in byte_sums.items():
                    a_per_link[link_id] = total / byte_counts[link_id]

        except Exception as exc:
            logger.warning(
                "Could not query tau/a_i from metrics server (run_id=%s): %s",
                run_id,
                exc,
            )

        return tau_per_node, a_per_link

    # ------------------------------------------------------------------
    # Phase: accuracy model
    # ------------------------------------------------------------------

    async def _run_accuracy_model(
        self, sub_exp: AccuracyModelSubExperiment, scheme: str = "topk"
    ) -> None:
        """Train a surrogate accuracy model A_k(η) for one pipeline via simulation sweep.

        1. Build a content-addressed artifact path that encodes model, partition
           layout, surrogate type, compression scheme, and a hash of the full
           uniqueness set (including dataset seed).  Artifacts are stored under
           ``shared_root/accuracy_models/`` so they are reusable across experiments
           that use the same pipeline config and dataset.
        2. Return early if the ``.pkl`` artifact already exists (content-addressed
           path serves as the validity check).
        3. Otherwise run ``_run_accuracy_sweep_simulated`` to generate
           (η_vector, accuracy) pairs via the pipeline's simulation.
        4. Fit a ``SurrogateAccuracyModel`` and persist it alongside a ``.json``
           sidecar with human-readable metadata.

        Args:
            sub_exp: Accuracy model sub-experiment config.
            scheme: Active compression scheme for this sweep pass.
        """
        pipeline = self._exp.pipeline_for(sub_exp.pipeline_id)
        pipeline_links = self._pipeline_links(pipeline.flow)

        compression_method_per_link = {
            lk.link_id: self._mapper.map(
                lk.link_id, sub_exp.pipeline_id, lk.eta_min
            ).method
            for lk in pipeline_links
        }

        dataset_cfg = sub_exp.dataset
        base_path = self._artifacts.accuracy_model_path(
            model=pipeline.model,
            partitions=pipeline.partitions,
            flow=pipeline.flow,
            simulation_path=pipeline.simulation_path,
            surrogate_type=sub_exp.model_type,
            compression_method_per_link=compression_method_per_link,
            sweep_design=sub_exp.sweep_design,
            n_sweep_samples=sub_exp.n_sweep_samples,
            dataset_seed=dataset_cfg.seed,
            dataset_max_samples=dataset_cfg.max_samples,
        )
        pkl_path = base_path.parent / (base_path.name + ".pkl")
        meta_path = base_path.parent / (base_path.name + ".json")

        if not sub_exp.force_retrain and pkl_path.exists():
            logger.info(
                "[%s] Reusing shared accuracy model artifact: %s",
                sub_exp.name,
                pkl_path.name,
            )
            self._load_accuracy_model_from_artifact(sub_exp, pkl_path, scheme=scheme)
            return

        X, y = await self._run_accuracy_sweep_simulated(
            sub_exp, pipeline_links, scheme=scheme
        )

        if not X:
            logger.warning(
                "[%s] No accuracy samples collected; accuracy model unavailable",
                sub_exp.name,
            )
            return

        model = build_accuracy_model(sub_exp.model_type)
        model.fit(np.array(X), np.array(y))

        model.save(pkl_path)
        self._artifacts.write_json(
            meta_path,
            {
                "pipeline_id": sub_exp.pipeline_id,
                "model": pipeline.model,
                "partitions": pipeline.partitions,
                "flow": pipeline.flow,
                "surrogate_type": sub_exp.model_type,
                "compression_method_per_link": compression_method_per_link,
                "sweep_design": sub_exp.sweep_design,
                "n_sweep_samples": sub_exp.n_sweep_samples,
                "dataset_seed": dataset_cfg.seed,
                "dataset_max_samples": dataset_cfg.max_samples,
            },
        )
        logger.info(
            "[%s] Accuracy model fitted: pipeline=%s samples=%d artifact=%s",
            sub_exp.name,
            sub_exp.pipeline_id,
            len(y),
            pkl_path.name,
        )
        self._accuracy_models.setdefault((sub_exp.name, scheme), {})[
            sub_exp.pipeline_id
        ] = model

    async def _run_accuracy_sweep_simulated(
        self,
        sub_exp: AccuracyModelSubExperiment,
        pipeline_links: list[Any],
        scheme: str = "topk",
    ) -> tuple[list[list[float]], list[float]]:
        """Generate (η_vector, accuracy) training pairs via simulation.

        Samples ``n_sweep_samples`` η vectors according to ``sweep_design``,
        evaluates each via the pipeline's simulation object (loaded from
        ``simulation_path``), and emits a ``TaskAccuracyEvent`` per sample.

        Args:
            sub_exp: Accuracy model sub-experiment config.
            pipeline_links: List of ``OptLinkConfig`` for this pipeline's links,
                in flow order.
            scheme: Active compression scheme for this sweep pass.

        Returns:
            Tuple of (X, y) where X is a list of η-vectors and y is a list of
            accuracy scalars.
        """
        from framework.optimizer.simulation_factory import (  # noqa: PLC0415
            build_simulations,
        )

        model_key = self._exp.pipeline_for(sub_exp.pipeline_id).model.lower()
        dataset_override = {model_key: sub_exp.dataset}
        simulations = build_simulations(
            self._exp,
            stein_cfg=None,
            mapper=self._mapper,
            dataset_override=dataset_override,
        )
        sim = simulations.get(sub_exp.pipeline_id)
        if sim is None:
            logger.warning(
                "[%s] No simulation available for pipeline '%s'; "
                "cannot train accuracy model — set simulation_path in the pipeline config",
                sub_exp.name,
                sub_exp.pipeline_id,
            )
            return [], []

        n_links = len(pipeline_links)
        eta_min_vec = np.array([lk.eta_min for lk in pipeline_links])
        eta_max_vec = np.array([lk.eta_max for lk in pipeline_links])
        link_ids = [lk.link_id for lk in pipeline_links]

        rng = np.random.default_rng(42)
        if sub_exp.sweep_design == "diagonal":
            scalars = np.linspace(0.0, 1.0, sub_exp.n_sweep_samples)
            eta_samples = eta_min_vec + scalars[:, None] * (eta_max_vec - eta_min_vec)
        else:  # random
            u = rng.random((sub_exp.n_sweep_samples, n_links))
            eta_samples = eta_min_vec + u * (eta_max_vec - eta_min_vec)

        X: list[list[float]] = []
        y: list[float] = []

        for i, eta_vec in enumerate(eta_samples):
            try:
                acc = float(sim.accuracy(torch.tensor(eta_vec, dtype=torch.float32)))
            except Exception as exc:
                logger.warning(
                    "[%s] Simulation failed for sample %d: %s", sub_exp.name, i, exc
                )
                continue

            X.append(eta_vec.tolist())
            y.append(acc)

            eta_per_link = dict(zip(link_ids, eta_vec.tolist(), strict=False))
            first_link = pipeline_links[0] if pipeline_links else None
            comp_method = (
                self._mapper.map(
                    first_link.link_id, sub_exp.pipeline_id, float(eta_vec[0])
                ).method
                if first_link
                else "topk"
            )
            # For WikiText simulations, n_samples is the number of sliding-window
            # sequences evaluated; for classification/MMLU it is the batch size.
            fast_eval = getattr(sim, "fast_evaluator", None)
            n_samples = (
                fast_eval.n_sequences
                if fast_eval is not None and hasattr(fast_eval, "n_sequences")
                else sub_exp.dataset.batch_size
            )
            scoped_name = f"{sub_exp.name}__{scheme}"
            run_id = f"{self._exp.name}_{scoped_name}_sweep_{i}"
            self._emitter.emit(
                TaskAccuracyEvent(
                    experiment_id=self._exp.name,
                    run_id=run_id,
                    pipeline_id=sub_exp.pipeline_id,
                    task_id=f"{run_id}_agg",
                    compression_method=comp_method,
                    compression_rate=float(np.mean(eta_vec)),
                    accuracy=acc,
                    n_samples=n_samples,
                    eta_per_link=eta_per_link,
                    sub_experiment_name=scoped_name,
                    compression_scheme=scheme,
                )
            )
            await asyncio.sleep(0)
            logger.debug(
                "[%s] Sample %d/%d: η=%s → accuracy=%.4f",
                sub_exp.name,
                i + 1,
                sub_exp.n_sweep_samples,
                [f"{v:.3f}" for v in eta_vec],
                acc,
            )

        try:
            sim.remove_hooks()
        except Exception:
            pass

        return X, y

    def _pipeline_links(self, flow: list[str]) -> list[Any]:
        """Return the OptLinkConfig objects for consecutive node pairs in ``flow``.

        Args:
            flow: Ordered list of node names for a pipeline.

        Returns:
            List of matching OptLinkConfig objects in flow order.
        """
        link_map = {(lk.from_node, lk.to_node): lk for lk in self._exp.links}
        result = []
        for i in range(len(flow) - 1):
            lk = link_map.get((flow[i], flow[i + 1]))
            if lk is not None:
                result.append(lk)
        return result

    def _load_accuracy_model_from_artifact(
        self, sub_exp: AccuracyModelSubExperiment, pkl_path: Path, scheme: str = "topk"
    ) -> None:
        """Restore an accuracy model from its pickle artifact into the in-memory cache.

        Args:
            sub_exp: Accuracy model sub-experiment config.
            pkl_path: Explicit path to the ``.pkl`` artifact (content-addressed;
                computed by the caller so we don't recompute the path here).
            scheme: Active compression scheme; used as part of the cache key.
        """
        from framework.optimizer.accuracy_model import (
            load_accuracy_model,  # noqa: PLC0415
        )

        try:
            model = load_accuracy_model(pkl_path)
        except Exception as exc:
            logger.warning(
                "[%s] Could not load accuracy model artifact %s: %s",
                sub_exp.name,
                pkl_path,
                exc,
            )
            return
        self._accuracy_models.setdefault((sub_exp.name, scheme), {})[
            sub_exp.pipeline_id
        ] = model

    def _resolve_accuracy_models(
        self, accuracy_model_refs: list[str] | None, scheme: str = "topk"
    ) -> dict[str, AccuracyModel]:
        """Merge accuracy models from multiple AccuracyModelSubExperiment references.

        Each reference covers one pipeline (determined by the sub-experiment's
        ``pipeline_id``).  Falls back to ``ConstantAccuracyModel(1.0)`` for any
        pipeline not covered by the provided references.

        Args:
            accuracy_model_refs: Names of AccuracyModelSubExperiments to merge.
                ``None`` or empty list → all pipelines get the constant fallback.
            scheme: Active compression scheme; used to look up the correct
                per-scheme entry in ``_accuracy_models``.

        Returns:
            Dict of ``pipeline_id → AccuracyModel``.
        """
        from framework.optimizer.accuracy_model import (  # noqa: PLC0415
            ConstantAccuracyModel,
        )

        merged: dict[str, AccuracyModel] = {}
        for ref in accuracy_model_refs or []:
            merged.update(self._accuracy_models.get((ref, scheme), {}))

        return {
            pid: merged.get(pid, ConstantAccuracyModel()) for pid in self._pipeline_ids
        }

    # ------------------------------------------------------------------
    # Phase: adapter-based optimizer (all opt sub-experiments)
    # ------------------------------------------------------------------

    async def _run_opt_with_adapter(
        self,
        sub_exp: OptSubExperiment,
        mu: float | None = None,
        pre_built_simulations: dict[str, Any] | None = None,
        skip_stein: bool = False,
        scheme: str = "topk",
    ) -> None:
        """Run a slot loop using the external optimizer adapter interface.

        Builds InferenceTask objects from profiling artifacts, constructs the
        appropriate BaseOptimizerAdapter via build_adapter, then delegates to
        _run_slot_loop_adapter.

        Args:
            sub_exp: Sub-experiment config (any optimization sub-experiment type).
            mu: Override mu for NoCsiSubExperiment runs (one call per mu value).
            pre_built_simulations: Pre-built simulation dict from the caller
                (avoids reloading for each μ in a mu_sweep). When provided,
                skips the internal build_simulations call.
            skip_stein: When True, skip simulation building even if
                stein_config is set (used for CSI-aware, which resolves η*
                analytically and never invokes accuracy callables).
            scheme: Active compression scheme for this sweep pass.
        """
        global_order = build_global_order(self._exp)

        stein_cfg = getattr(sub_exp, "stein_config", None)
        simulations: dict[str, Any] | None = None
        owns_simulations = False
        if pre_built_simulations is not None:
            # Caller already built simulations; reuse them without taking ownership.
            simulations = pre_built_simulations
        elif stein_cfg is not None and not skip_stein:
            from framework.optimizer.simulation_factory import (  # noqa: PLC0415
                build_simulations,
            )

            simulations = build_simulations(
                self._exp,
                stein_cfg,
                self._mapper,
                dataset_override=self._exp.stein_datasets,
            )
            owns_simulations = True
            if not simulations:
                logger.warning(
                    "[%s] stein_config is set but no pipelines have simulation_path; "
                    "falling back to dummy accuracy callables",
                    sub_exp.name,
                )

        # Resolve surrogate accuracy models for sub-experiments that reference them.
        accuracy_model_refs = getattr(sub_exp, "accuracy_model_refs", None)
        accuracy_models = self._resolve_accuracy_models(
            accuracy_model_refs, scheme=scheme
        )

        inference_tasks, task_id_to_pipeline, pipeline_to_task_id = (
            build_inference_tasks(
                exp=self._exp,
                tau_per_node=self._tau_per_node,
                a_per_link_bytes=self._a_per_link_bytes,
                global_order=global_order,
                simulations=simulations,
                stein_cfg=stein_cfg,
                accuracy_models=accuracy_models,
            )
        )

        adapter = build_adapter(
            sub_exp=sub_exp,
            inference_tasks=inference_tasks,
            task_id_to_pipeline=task_id_to_pipeline,
            pipeline_to_task_id=pipeline_to_task_id,
            global_order=global_order,
            exp=self._exp,
            mu=mu,
        )

        # Build a descriptive run name (scoped by scheme).
        if mu is not None:
            run_name = f"{sub_exp.name}__{scheme}_mu{mu}"
        else:
            run_name = f"{sub_exp.name}__{scheme}"

        run_id = f"{self._exp.name}_{run_name}_{uuid.uuid4().hex[:6]}"
        logger.info("[%s] Starting optimization run via adapter", run_name)

        try:
            await self._run_slot_loop_adapter(
                adapter=adapter,
                sub_exp_name=run_name,
                run_id=run_id,
                global_order=global_order,
                task_id_to_pipeline=task_id_to_pipeline,
                pipeline_to_task_id=pipeline_to_task_id,
                inference_tasks=inference_tasks,
                scheme=scheme,
            )
        finally:
            if owns_simulations and simulations:
                for sim in simulations.values():
                    try:
                        sim.remove_hooks()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Shared slot loop (adapter-based)
    # ------------------------------------------------------------------

    async def _run_slot_loop_adapter(
        self,
        adapter: BaseOptimizerAdapter,
        sub_exp_name: str,
        run_id: str,
        global_order: list[str],
        task_id_to_pipeline: dict[int, str],
        pipeline_to_task_id: dict[str, int],
        inference_tasks: Any,
        scheme: str = "topk",
    ) -> None:
        """Execute the optimization slot loop using a BaseOptimizerAdapter.

        Each slot:
          1. Probe links.  CSI-aware adapters probe every slot; estimated
             adapters probe every link_probe_interval_slots slots.  On real
             probe slots, adapter.observe_capacity(c_t) updates the internal
             channel estimator.  On non-probe slots, the estimator is not
             updated and c_t is reused from the last probe (ignored by
             EstimatedAdapter.step() which uses its own estimate internally).
          2. Call adapter.step(t, c_t) to get eta_per_pipeline_per_link and s_comp_per_node.
          3. Push compression config and WFQ weights to nodes concurrently.
          4. Submit batches_per_slot tasks and collect SlotResult.
          5. Emit OptSlotEvent, ThroughputConstraintEvent, TaskAccuracyEvent.
          6. Call adapter.update_dual(t, actual_delays).

        Args:
            adapter: Instantiated adapter for this run.
            sub_exp_name: Human-readable name for logs and metric events.
            run_id: Run identifier for metric events.
            global_order: Ordered list of all node names.
            task_id_to_pipeline: task_id -> pipeline_id mapping.
            pipeline_to_task_id: pipeline_id -> task_id mapping.
            inference_tasks: List of InferenceTask objects (for task_id indexing).
            scheme: Active compression scheme; recorded on emitted events.
        """
        loop_cfg = self._exp.optimization_loop
        tasks_cfg = self._exp.tasks
        pipeline_ids = self._pipeline_ids

        cumulative_violations: dict[str, int] = {pid: 0 for pid in pipeline_ids}

        # CSI-aware adapters probe every slot (probe_every_slot=True); all other
        # adapters probe every link_probe_interval_slots slots.
        last_c_t: np.ndarray | None = None

        for slot_id in range(loop_cfg.n_slots):
            # --- 1. Link probe ---
            should_probe = adapter.probe_every_slot or (
                slot_id % loop_cfg.link_probe_interval_slots == 0
            )

            if should_probe:
                probe_bps = await self._link_prober.probe_all(
                    slot_id=slot_id,
                    experiment_id=self._exp.name,
                    run_id=run_id,
                    sub_experiment_name=sub_exp_name,
                )
                c_t = probe_dict_to_c_t_vector(probe_bps, global_order)
                # Intra-pipeline window concurrency correction: under FILL,
                # W tasks per pipeline are in-flight simultaneously and
                # contend on each link, so per-task effective capacity is
                # link_bps / W.  Inter-pipeline sharing is handled separately
                # by the optimizer via s_comm, so we do NOT divide by the
                # pipeline count.
                if self._exp.workload.pattern == WorkloadPattern.FILL:
                    c_t = c_t / float(self._exp.workload.window_per_pipeline)
                last_c_t = c_t
                # Update the adapter's channel estimator with the real measurement.
                # No-op for DirectCsiAdapter (no internal estimator).
                adapter.observe_capacity(c_t)
            else:
                # Non-probe slot for estimated adapters: reuse last probe value.
                # adapter.step() ignores c_t for EstimatedAdapter (uses internal estimate).
                c_t = last_c_t  # type: ignore[assignment]

            # --- 3. Optimizer step ---
            t_solve_start = time.perf_counter()
            eta_per_pipeline_per_link, s_comp_per_node, infeasible = adapter.step(
                slot_id, c_t
            )
            solve_time_ms = (time.perf_counter() - t_solve_start) * 1000.0
            # Capture dual variables immediately after step(), before update_dual()
            # advances them — these are the λ values that drove this slot's decision.
            lambda_per_task = adapter.get_dual_variables()

            # --- 4. Push compression config and WFQ compute-share weights ---
            await asyncio.gather(
                push_opt_slot_config(
                    eta_per_pipeline_per_link,
                    self._exp,
                    self._mapper,
                    self._node_host,
                    _CONFIG_DRAIN_TIMEOUT_S,
                ),
                push_wfq_weights(s_comp_per_node, self._exp, self._node_host),
            )

            # --- 5. Run slot ---
            slot_run_id = f"{run_id}_s{slot_id}"
            slot_result = await self._data_client.run_slot(
                n_batches=loop_cfg.batches_per_slot,
                run_id=slot_run_id,
                node_host=self._node_host,
                emitter=self._emitter,
                slot_id=slot_id,
                sub_experiment_name=sub_exp_name,
                perplexity_baselines=self._perplexity_baselines,
            )

            # --- 6. Emit events ---
            achieved_rps: dict[str, float] = {}
            d_excess: dict[str, float] = {}
            throughput_shortfall: dict[str, float] = {}
            actual_delays: dict[int, float] = {}

            for pid in pipeline_ids:
                task_cfg = tasks_cfg.get(pid)
                target_rps = task_cfg.throughput_target if task_cfg else 0.0
                rps = slot_result.achieved_rps(pid)
                achieved_rps[pid] = rps
                # Throughput shortfall: deficit in tasks/second (used for ThroughputConstraintEvent).
                excess = max(0.0, target_rps - rps)
                throughput_shortfall[pid] = excess
                # Delay excess: max(0, 1/achieved - 1/target) in seconds (dual update domain).
                if rps > 0 and target_rps > 0:
                    d_excess[pid] = max(0.0, 1.0 / rps - 1.0 / target_rps)
                else:
                    d_excess[pid] = 0.0
                satisfied = rps >= target_rps
                if not satisfied:
                    cumulative_violations[pid] += 1

                # Actual delay approximation: 1/achieved_rps (seconds per inference).
                task_id = pipeline_to_task_id.get(pid)
                if task_id is not None and rps > 0:
                    actual_delays[task_id] = 1.0 / rps

                self._emitter.emit(
                    ThroughputConstraintEvent(
                        experiment_id=self._exp.name,
                        run_id=run_id,
                        slot_id=slot_id,
                        pipeline_id=pid,
                        target_rps=target_rps,
                        achieved_rps=rps,
                        satisfied=satisfied,
                        violation_magnitude=throughput_shortfall[pid],
                        cumulative_violations=cumulative_violations[pid],
                        sub_experiment_name=sub_exp_name,
                    )
                )

                acc = slot_result.quality_metric(pid)
                # Use first link's η for this pipeline as representative compression rate.
                first_link = self._exp.links[0] if self._exp.links else None
                comp_rate = (
                    eta_per_pipeline_per_link.get(pid, {}).get(first_link.link_id, 1.0)
                    if first_link
                    else 1.0
                )
                comp_method = (
                    self._mapper.map(first_link.link_id, pid, comp_rate).method
                    if first_link
                    else "none"
                )
                self._emitter.emit(
                    TaskAccuracyEvent(
                        experiment_id=self._exp.name,
                        run_id=run_id,
                        pipeline_id=pid,
                        task_id=slot_run_id,
                        compression_method=comp_method,
                        compression_rate=comp_rate,
                        accuracy=acc,
                        n_samples=len(slot_result.per_pipeline_latency_ms.get(pid, [])),
                        eta_per_link=eta_per_pipeline_per_link.get(pid),
                        slot_id=slot_id,
                        sub_experiment_name=sub_exp_name,
                        compression_scheme=scheme,
                    )
                )

            self._emitter.emit(
                OptSlotEvent(
                    experiment_id=self._exp.name,
                    run_id=run_id,
                    slot_id=slot_id,
                    eta_per_pipeline_per_link=eta_per_pipeline_per_link,
                    lambda_per_task=lambda_per_task,
                    d_excess_per_task=d_excess,
                    throughput_shortfall_per_pipeline=throughput_shortfall,
                    c_hat_per_link=probe_bps,
                    optimizer_type=sub_exp_name,
                    solve_time_ms=solve_time_ms,
                    sub_experiment_name=sub_exp_name,
                    infeasible=infeasible,
                    compression_scheme=scheme,
                )
            )

            # --- 7. Dual update ---
            adapter.update_dual(slot_id, actual_delays)

            if slot_id % 10 == 0:
                logger.info(
                    "[%s] slot=%d  rps=%s  infeasible=%s",
                    sub_exp_name,
                    slot_id,
                    {k: f"{v:.2f}" for k, v in achieved_rps.items()},
                    infeasible,
                )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


async def run_opt_experiment(
    experiment_dir: Path,
    callback_host: str,
    callback_port: int,
    result_timeout_s: float,
    dry_run: bool,
    node_host: str | None = None,
    metrics_host: str | None = None,
) -> None:
    """Load and execute a full optimization experiment.

    Instantiates all shared infrastructure (data client, link prober, artifact
    store, metrics emitter) and delegates to ``OptRunner.run()``.

    Args:
        experiment_dir: Directory containing the generated experiment.yaml.
        callback_host: Hostname pipeline nodes use to reach the result callback.
        callback_port: Port for the orchestrator's callback server.
        result_timeout_s: Per-task result wait timeout in seconds.
        dry_run: If True, log the plan without executing.
        node_host: Override hostname used to reach all nodes.
        metrics_host: Override hostname for the metrics server.
    """
    exp_yaml = experiment_dir / "experiment.yaml"
    exp = load_opt_experiment_config(exp_yaml)

    logger.info(
        "Optimization experiment: %s  pipelines=%d  sub-experiments=%d",
        exp.name,
        len(exp.pipelines),
        len(exp.sub_experiments),
    )

    if dry_run:
        for sub_exp in exp.sub_experiments:
            logger.info("  [%s] type=%s", sub_exp.name, sub_exp.type)
        logger.info("Dry run — skipping execution")
        return

    resolved_metrics_host = metrics_host or exp.metrics_server.host
    metrics_url = f"http://{resolved_metrics_host}:{exp.metrics_server.port}"

    emitter = MetricsEmitter(server_url=metrics_url)
    await emitter.start()

    await wait_for_multi_nodes_ready(exp, node_host=node_host)

    artifact_store = ArtifactStore(exp.artifacts_dir, exp.shared_artifacts_dir)

    link_prober = LinkProber(
        links=exp.links,
        nodes=exp.nodes,
        emitter=emitter,
    )

    data_client = MultiDataClient(
        exp=exp,
        callback_host=callback_host,
        callback_port=callback_port,
        result_timeout_s=result_timeout_s,
    )

    async with data_client.session():
        runner = OptRunner(
            exp=exp,
            data_client=data_client,
            link_prober=link_prober,
            artifact_store=artifact_store,
            emitter=emitter,
            metrics_url=metrics_url,
            node_host=node_host,
        )
        await runner.run()

    await emitter.stop()
    logger.info("Optimization experiment '%s' finished", exp.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the optimization experiment runner."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Run a DNN compression optimization experiment."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to the generated experiment directory (containing experiment.yaml)",
    )
    parser.add_argument(
        "--callback-host",
        default=os.getenv("CALLBACK_HOST", "localhost"),
    )
    parser.add_argument(
        "--callback-port",
        type=int,
        default=int(os.getenv("CALLBACK_PORT", "8080")),
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--node-host",
        default=os.getenv("NODE_HOST"),
    )
    parser.add_argument(
        "--metrics-host",
        default=os.getenv("METRICS_HOST"),
    )
    args = parser.parse_args()

    asyncio.run(
        run_opt_experiment(
            experiment_dir=args.experiment_dir,
            callback_host=args.callback_host,
            callback_port=args.callback_port,
            result_timeout_s=args.result_timeout,
            dry_run=args.dry_run,
            node_host=args.node_host,
            metrics_host=args.metrics_host,
        )
    )


if __name__ == "__main__":
    main()
