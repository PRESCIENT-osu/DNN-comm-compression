from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from framework.datamodels.events import SubExperimentEvent
from framework.datamodels.experiment import ExperimentConfig
from framework.datamodels.spec import GeneratedExperimentConfig, ResolvedSubExperiment
from framework.nodes.metrics.emitter import MetricsEmitter
from framework.nodes.orchestrator.controller import (
    push_run_config,
    wait_for_nodes_ready,
)
from framework.nodes.orchestrator.data_client import DataClient
from framework.utils.loader import (
    is_generated_experiment,
    load_experiment_dir,
    load_generated_experiment_config,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


async def run_experiment(
    experiment_dir: Path,
    callback_host: str,
    callback_port: int,
    result_timeout_s: float,
    dry_run: bool,
    node_host: str | None = None,
    metrics_host: str | None = None,
) -> None:
    """Load and execute a full experiment, handling both legacy and generated formats.

    For generated experiments (produced by tools/generate.py) each
    sub-experiment is executed sequentially.  For legacy experiments a single
    sweep is executed.

    For each resolved sweep run:
      1. Push compression configs to all relevant nodes.
      2. Send dataset batches and collect results via the data client.
      3. Emit result metrics.

    Args:
        experiment_dir: Directory containing experiment.yaml and infra.yaml.
        callback_host: Hostname pipeline nodes use to reach the result callback.
        callback_port: Port for the orchestrator's callback server.
        result_timeout_s: Per-batch result wait timeout in seconds.
        dry_run: If True, print the sweep plan without executing.
        node_host: Override hostname used to reach all nodes (e.g. ``"localhost"``
            when running against Docker with published ports).  Defaults to each
            node's configured host from the experiment config.
        metrics_host: Override hostname used to reach the metrics server.
            Defaults to ``exp.metrics_server.host``.
    """
    exp_yaml = experiment_dir / "experiment.yaml"

    if is_generated_experiment(exp_yaml):
        generated = load_generated_experiment_config(exp_yaml)
        logger.info(
            "Experiment: %s  model: %s  sub-experiments: %d",
            generated.name,
            generated.model,
            len(generated.sub_experiments),
        )

        resolved_metrics_host = metrics_host or generated.metrics_server.host
        metrics_url = f"http://{resolved_metrics_host}:{generated.metrics_server.port}"
        emitter = MetricsEmitter(server_url=metrics_url)
        await emitter.start()

        # Build the data client once — dataset loading (tokenization, HF download)
        # happens here and is reused across all sub-experiments and runs.
        first_exp = _sub_experiment_to_exp_config(
            generated, generated.sub_experiments[0]
        )
        shared_client = _make_data_client(
            first_exp, callback_host, callback_port, result_timeout_s
        )

        async with shared_client.session():
            for sub_exp in generated.sub_experiments:
                exp = _sub_experiment_to_exp_config(generated, sub_exp)
                _warn_missing_baselines(sub_exp.baselines)
                await _run_sweep(
                    exp=exp,
                    sub_experiment_name=sub_exp.name,
                    client=shared_client,
                    dry_run=dry_run,
                    node_host=node_host,
                    emitter=emitter,
                )

        await emitter.stop()
        logger.info("Experiment '%s' complete", generated.name)

    else:
        # Legacy format — single sweep
        exp, _ = load_experiment_dir(experiment_dir)

        resolved_metrics_host = metrics_host or exp.metrics_server.host
        metrics_url = f"http://{resolved_metrics_host}:{exp.metrics_server.port}"
        emitter = MetricsEmitter(server_url=metrics_url)
        await emitter.start()

        client = _make_data_client(exp, callback_host, callback_port, result_timeout_s)
        async with client.session():
            await _run_sweep(
                exp=exp,
                sub_experiment_name=None,
                client=client,
                dry_run=dry_run,
                node_host=node_host,
                emitter=emitter,
            )

        await emitter.stop()
        logger.info("Experiment '%s' complete", exp.name)


def _sub_experiment_to_exp_config(
    generated: GeneratedExperimentConfig,
    sub_exp: ResolvedSubExperiment,
) -> ExperimentConfig:
    """Build an ExperimentConfig from shared generated fields + one sub-experiment.

    The sub-experiment's dataset overrides the shared dataset when present
    (used by Llama experiments that vary dataset across sub-experiments).

    Args:
        generated: The top-level generated experiment config.
        sub_exp: The resolved sub-experiment entry to execute.

    Returns:
        A valid ExperimentConfig ready for sweep resolution.
    """
    return ExperimentConfig(
        name=generated.name,
        model=generated.model,
        nodes=generated.nodes,
        links=sub_exp.links,
        sweep=sub_exp.sweep,
        sweep_mode=sub_exp.sweep_mode,
        dataset=sub_exp.dataset or generated.dataset,
        metrics_server=generated.metrics_server,
        baselines=sub_exp.baselines,
    )


def _warn_missing_baselines(baselines: list[str]) -> None:
    """Log a warning for each baseline that has no metrics data on disk.

    Args:
        baselines: List of 'experiment_name/sub_experiment_name' strings.
    """
    metrics_root = Path("metrics_data")
    for ref in baselines:
        exp_name = ref.split("/")[0]
        result_path = metrics_root / exp_name / "result.ndjson"
        if not result_path.exists():
            logger.warning(
                "Baseline '%s' has no metrics data at %s — "
                "run the baseline experiment first for accurate analysis",
                ref,
                result_path,
            )


async def _run_sweep(
    exp: ExperimentConfig,
    sub_experiment_name: str | None,
    client: DataClient,
    dry_run: bool,
    node_host: str | None,
    emitter: MetricsEmitter,
) -> None:
    """Execute the sweep for one ExperimentConfig.

    Args:
        exp: Fully resolved experiment config for this sweep.
        sub_experiment_name: Sub-experiment label for logging, or None for legacy.
        client: Data client (already in an active session).
        dry_run: If True, log the sweep plan without executing.
        node_host: Optional node hostname override.
        emitter: Shared metrics emitter (already started).
    """
    label = f"[{sub_experiment_name}] " if sub_experiment_name else ""

    runs = exp.resolve_sweep()
    logger.info("%sSweep: %d run(s)", label, len(runs))
    for run in runs:
        link_summary = ", ".join(
            f"{lk.from_node}→{lk.to_node} {lk.compression.value}"
            + (f"@{lk.rate:.2f}" if lk.compression.value != "none" else "")
            for lk in run.links
        )
        logger.info("  %s[%s] %s", label, run.run_id, link_summary or "no links")

    if dry_run:
        logger.info("%sDry run — skipping execution", label)
        return

    await wait_for_nodes_ready(exp, node_host=node_host)

    order = exp.node_order()
    first_node = next(n for n in exp.nodes if n.name == order[0])
    resolved_host = node_host or first_node.host
    first_node_url = f"http://{resolved_host}:{first_node.port}/infer"

    t_sweep_start = time.perf_counter()

    for run in runs:
        logger.info("%sStarting run: %s", label, run.run_id)

        if run.links:
            await push_run_config(exp, run, node_host=node_host)
        else:
            logger.info("%sNo links to configure for run '%s'", label, run.run_id)

        records = await client.run(
            exp=exp,
            run_id=run.run_id,
            first_node_url=first_node_url,
            emitter=emitter,
        )

        if records:
            _log_run_summary(run.run_id, records)
        else:
            logger.warning("%sRun '%s' produced no results", label, run.run_id)

    duration_s = time.perf_counter() - t_sweep_start
    logger.info("%sSub-experiment complete in %.1fs", label, duration_s)
    emitter.emit(
        SubExperimentEvent(
            experiment_id=exp.name,
            run_id=sub_experiment_name or "default",
            sub_experiment_name=sub_experiment_name,
            duration_s=duration_s,
            n_runs=len(runs),
        )
    )


def _make_data_client(
    exp: ExperimentConfig,
    callback_host: str,
    callback_port: int,
    result_timeout_s: float,
) -> DataClient:
    """Instantiate the appropriate data client based on the experiment model.

    Args:
        exp: Experiment config used to detect model type and tokenizer path.
        callback_host: Hostname pipeline nodes use to reach the result callback.
        callback_port: Port for the callback server.
        result_timeout_s: Per-batch result wait timeout in seconds.

    Returns:
        A DataClient or LlamaDataClient instance.
    """
    if exp.model.lower().startswith("llama"):
        from framework.nodes.orchestrator.llama_data_client import LlamaDataClient

        tokenizer_path = exp.dataset.tokenizer_path or exp.dataset.path
        return LlamaDataClient(  # type: ignore[return-value]
            dataset_config=exp.dataset,
            tokenizer_path=tokenizer_path,
            callback_host=callback_host,
            callback_port=callback_port,
            result_timeout_s=result_timeout_s,
        )
    return DataClient(
        callback_host=callback_host,
        callback_port=callback_port,
        result_timeout_s=result_timeout_s,
    )


def _log_run_summary(run_id: str, records: list[Any]) -> None:
    """Log a human-readable metric summary for one completed sweep run.

    Handles both LlamaRunRecord (perplexity/accuracy) and RunRecord
    (classification accuracy) without importing either type directly.

    Args:
        run_id: Run identifier for log prefix.
        records: Completed records returned by the data client.
    """
    if hasattr(records[0], "metric_type"):
        metric_type = records[0].metric_type
        if metric_type == "perplexity":
            total_nll = sum(r.nll_sum for r in records)
            total_tok = sum(r.token_count for r in records)
            ppl = math.exp(total_nll / total_tok) if total_tok > 0 else float("inf")
            logger.info("Run '%s' perplexity: %.2f", run_id, ppl)
        else:  # accuracy
            correct = sum(
                1
                for r in records
                for gt, pred in zip(r.ground_truth, r.predicted, strict=False)
                if gt == pred
            )
            total = sum(len(r.ground_truth) for r in records)
            accuracy = correct / total if total > 0 else 0.0
            logger.info(
                "Run '%s' accuracy: %.2f%% (%d/%d)",
                run_id,
                accuracy * 100,
                correct,
                total,
            )
    else:
        correct = sum(
            1
            for r in records
            for gt, pred in zip(r.ground_truth, r.predicted, strict=False)
            if gt == pred
        )
        total = sum(len(r.ground_truth) for r in records)
        accuracy = correct / total if total > 0 else 0.0
        logger.info(
            "Run '%s' accuracy: %.2f%% (%d/%d)",
            run_id,
            accuracy * 100,
            correct,
            total,
        )


def main() -> None:
    """CLI entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run a DNN compression experiment sweep."
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
        help="Per-batch result wait timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep plan without executing",
    )
    parser.add_argument(
        "--node-host",
        default=os.getenv("NODE_HOST"),
        help="Override hostname used to reach all nodes (e.g. 'localhost' when "
        "running against Docker with published ports). Defaults to each node's "
        "configured host. Can also be set via NODE_HOST env var.",
    )
    parser.add_argument(
        "--metrics-host",
        default=os.getenv("METRICS_HOST"),
        help="Override hostname used to reach the metrics server (e.g. 'localhost' "
        "when running against Docker with published ports). Defaults to the "
        "metrics_server.host in the experiment config. Can also be set via "
        "METRICS_HOST env var.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_experiment(
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
