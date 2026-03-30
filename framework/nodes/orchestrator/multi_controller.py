"""Controller functions for multi-model experiment orchestration.

Provides health-check polling and per-pipeline run-config push for
multi-model experiments, mirroring controller.py for the single-model case.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from framework.datamodels.multi_experiment import (
    MultiExperimentConfig,
    MultiResolvedRun,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0
_DRAIN_TIMEOUT_S = 120.0
_CONFIG_TIMEOUT_S = 90.0


def _node_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


async def wait_for_multi_nodes_ready(
    exp: MultiExperimentConfig,
    timeout_s: float = 300.0,
    node_host: str | None = None,
) -> None:
    """Poll all nodes until every node responds to GET /health.

    Args:
        exp: Multi-model experiment config carrying node host/port information.
        timeout_s: Maximum time to wait before raising an error.
        node_host: Override hostname used to reach all nodes.

    Raises:
        RuntimeError: If any node is not reachable within timeout.
    """
    node_urls = {
        n.name: _node_url(node_host or n.host, n.port, "/health") for n in exp.nodes
    }
    pending: set[str] = set(node_urls)
    deadline = asyncio.get_event_loop().time() + timeout_s
    logger.info("Waiting for %d node(s) to become ready...", len(pending))

    async with httpx.AsyncClient(timeout=5.0) as client:
        while pending:
            if asyncio.get_event_loop().time() > deadline:
                raise RuntimeError(
                    f"Timed out waiting for nodes to become ready after {timeout_s}s. "
                    f"Still not reachable: {pending}"
                )
            await asyncio.sleep(_POLL_INTERVAL_S)
            still_pending: set[str] = set()
            for name in list(pending):
                try:
                    resp = await client.get(node_urls[name])
                    if resp.status_code == 200:
                        logger.info("Node '%s' is ready", name)
                    else:
                        still_pending.add(name)
                except Exception:
                    still_pending.add(name)
            pending = still_pending

    logger.info("All nodes ready")


async def wait_for_multi_nodes_idle(
    exp: MultiExperimentConfig,
    timeout_s: float = _DRAIN_TIMEOUT_S,
    node_host: str | None = None,
) -> None:
    """Poll all nodes until every node reports an empty queue and no active task.

    Args:
        exp: Multi-model experiment config.
        timeout_s: Maximum time to wait before raising an error.
        node_host: Override hostname used to reach all nodes.

    Raises:
        RuntimeError: If any node is still busy after timeout.
    """
    node_urls = {
        n.name: _node_url(node_host or n.host, n.port, "/status") for n in exp.nodes
    }
    deadline = asyncio.get_event_loop().time() + timeout_s
    busy: set[str] = set(node_urls)

    async with httpx.AsyncClient(timeout=5.0) as client:
        while busy:
            if asyncio.get_event_loop().time() > deadline:
                raise RuntimeError(
                    f"Timed out waiting for nodes to drain after {timeout_s}s. "
                    f"Still busy: {busy}"
                )
            await asyncio.sleep(_POLL_INTERVAL_S)
            still_busy: set[str] = set()
            for name in list(busy):
                try:
                    resp = await client.get(node_urls[name])
                    data = resp.json()
                    if data.get("queue_length", 1) > 0 or data.get("processing", True):
                        still_busy.add(name)
                    else:
                        logger.debug("Node '%s' is idle", name)
                except Exception as exc:
                    logger.warning("Could not reach node '%s': %s", name, exc)
                    still_busy.add(name)
            busy = still_busy

    logger.info("All nodes idle")


async def push_multi_run_config(
    exp: MultiExperimentConfig,
    run: MultiResolvedRun,
    drain_timeout_s: float = _CONFIG_TIMEOUT_S,
    node_host: str | None = None,
) -> None:
    """Push per-pipeline compression configs for a resolved run to all relevant nodes.

    For each (pipeline, link) pair in the run, sends the outgoing config to
    the sending node and the incoming config to the receiving node concurrently.
    Each node drains its queue internally before applying the new config.

    Args:
        exp: Multi-model experiment config.
        run: Resolved run with per-pipeline link configs to apply.
        drain_timeout_s: Drain timeout forwarded to each node's POST /config.
        node_host: Override hostname used to reach all nodes.
    """
    node_map = {n.name: n for n in exp.nodes}
    tasks = []

    async with httpx.AsyncClient(timeout=drain_timeout_s + 5.0) as client:
        for link in run.links:
            sending = node_map[link.from_node]
            receiving = node_map[link.to_node]

            tasks.append(
                _push_single_config(
                    client=client,
                    url=_node_url(node_host or sending.host, sending.port, "/config"),
                    pipeline_id=link.pipeline_id,
                    direction="outgoing",
                    method=link.compression.value,
                    rate=link.rate,
                    drain_timeout_s=drain_timeout_s,
                    node_name=sending.name,
                    outlier_precision=link.outlier_precision,
                    regular_precision=link.regular_precision,
                )
            )
            tasks.append(
                _push_single_config(
                    client=client,
                    url=_node_url(
                        node_host or receiving.host, receiving.port, "/config"
                    ),
                    pipeline_id=link.pipeline_id,
                    direction="incoming",
                    method=link.compression.value,
                    rate=link.rate,
                    drain_timeout_s=drain_timeout_s,
                    node_name=receiving.name,
                    outlier_precision=link.outlier_precision,
                    regular_precision=link.regular_precision,
                )
            )

        await asyncio.gather(*tasks)

    logger.info(
        "Config applied for run '%s': %s",
        run.run_id,
        [
            f"{lk.pipeline_id}:{lk.from_node}→{lk.to_node} "
            f"{lk.compression.value}"
            + (f"@{lk.rate:.2f}" if lk.compression.value != "none" else "")
            for lk in run.links
        ],
    )


async def _push_single_config(
    client: httpx.AsyncClient,
    url: str,
    pipeline_id: str,
    direction: str,
    method: str,
    rate: float,
    drain_timeout_s: float,
    node_name: str,
    outlier_precision: str = "fp16",
    regular_precision: str = "int8",
) -> None:
    """Send a single POST /config request to one multi-model node.

    Args:
        client: Shared httpx client.
        url: Full POST /config URL for the node.
        pipeline_id: Pipeline to configure on this node.
        direction: ``"incoming"`` or ``"outgoing"``.
        method: Compression method value string.
        rate: Compression rate.
        drain_timeout_s: Drain timeout to pass to the node.
        node_name: Node name for logging.
        outlier_precision: Outlier precision for llmint8.
        regular_precision: Regular-value precision for llmint8.
    """
    payload = {
        "pipeline_id": pipeline_id,
        "direction": direction,
        "method": method,
        "rate": rate,
        "drain_timeout_s": drain_timeout_s,
        "outlier_precision": outlier_precision,
        "regular_precision": regular_precision,
    }
    try:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        logger.debug(
            "Config pushed: node=%s pipeline=%s %s method=%s rate=%s",
            node_name,
            pipeline_id,
            direction,
            method,
            rate,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to push {direction} config for pipeline '{pipeline_id}' "
            f"to node '{node_name}' at {url}: {exc}"
        ) from exc
