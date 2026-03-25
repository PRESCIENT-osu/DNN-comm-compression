from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any

from framework.config.experiment_schema import ExperimentConfig
from framework.config.loader import load_experiment_dir
from framework.node.metrics import MetricsEmitter
from framework.orchestrator.controller import push_run_config
from framework.orchestrator.data_client import DataClient

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
    """Load and execute a full experiment sweep.

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
    exp, _ = load_experiment_dir(experiment_dir)

    logger.info("Experiment: %s  model: %s", exp.name, exp.model)

    runs = exp.resolve_sweep()
    logger.info("Sweep: %d run(s)", len(runs))
    for run in runs:
        link_summary = ", ".join(
            f"{lk.from_node}→{lk.to_node} {lk.compression.value}"
            + (f"@{lk.rate:.2f}" if lk.compression.value != "none" else "")
            for lk in run.links
        )
        logger.info("  [%s] %s", run.run_id, link_summary or "no links")

    if dry_run:
        logger.info("Dry run — exiting without executing")
        return

    order = exp.node_order()
    first_node = next(n for n in exp.nodes if n.name == order[0])
    resolved_host = node_host or first_node.host
    first_node_url = f"http://{resolved_host}:{first_node.port}/infer"

    resolved_metrics_host = metrics_host or exp.metrics_server.host
    metrics_url = f"http://{resolved_metrics_host}:{exp.metrics_server.port}"
    emitter = MetricsEmitter(server_url=metrics_url)
    await emitter.start()

    client = _make_data_client(exp, callback_host, callback_port, result_timeout_s)

    async with client.session():
        for run in runs:
            logger.info("Starting run: %s", run.run_id)

            if run.links:
                await push_run_config(exp, run, node_host=node_host)
            else:
                logger.info("No links to configure for run '%s'", run.run_id)

            records = await client.run(
                exp=exp,
                run_id=run.run_id,
                first_node_url=first_node_url,
                emitter=emitter,
            )

            if records:
                _log_run_summary(run.run_id, records)
            else:
                logger.warning("Run '%s' produced no results", run.run_id)

    await emitter.stop()
    logger.info("Experiment '%s' complete", exp.name)


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
        from framework.orchestrator.llama_data_client import LlamaDataClient

        tokenizer_path = exp.dataset.tokenizer_path or exp.dataset.path
        return LlamaDataClient(  # type: ignore[return-value]
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
