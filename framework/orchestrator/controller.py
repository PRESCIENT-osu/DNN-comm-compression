from __future__ import annotations

import asyncio
import logging

import httpx

from framework.config.experiment_schema import ExperimentConfig, ResolvedRun

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0
_DRAIN_TIMEOUT_S = 120.0
_CONFIG_TIMEOUT_S = 90.0


def _node_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


async def wait_for_all_idle(
    exp: ExperimentConfig,
    timeout_s: float = _DRAIN_TIMEOUT_S,
    node_host: str | None = None,
) -> None:
    """Poll all nodes until every node reports zero in-flight requests.

    Args:
        exp: Experiment config carrying node host/port information.
        timeout_s: Maximum time to wait before raising an error.
        node_host: Override hostname used to reach all nodes (e.g. ``"localhost"``
            when the orchestrator runs on the Docker host).  Defaults to each
            node's configured host.

    Raises:
        RuntimeError: If any node still has in-flight requests after timeout.
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
                    if data.get("in_flight", 1) > 0:
                        still_busy.add(name)
                    else:
                        logger.debug("Node '%s' is idle", name)
                except Exception as exc:
                    logger.warning("Could not reach node '%s': %s", name, exc)
                    still_busy.add(name)
            busy = still_busy

    logger.info("All nodes idle")


async def push_run_config(
    exp: ExperimentConfig,
    run: ResolvedRun,
    drain_timeout_s: float = _CONFIG_TIMEOUT_S,
    node_host: str | None = None,
) -> None:
    """Push compression configs for a resolved run to all relevant nodes.

    For each link in the run, sends the outgoing config to the sending node
    and the incoming config to the receiving node concurrently.  Each node
    handles draining internally before applying the new config.

    Args:
        exp: Experiment config carrying node host/port information.
        run: The resolved run whose link configs should be applied.
        drain_timeout_s: Drain timeout forwarded to each node's POST /config.
        node_host: Override hostname used to reach all nodes (e.g. ``"localhost"``
            when the orchestrator runs on the Docker host).  Defaults to each
            node's configured host.
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
            f"{lk.from_node}→{lk.to_node} {lk.compression.value}"
            + (f"@{lk.rate:.2f}" if lk.compression.value != "none" else "")
            for lk in run.links
        ],
    )


async def _push_single_config(
    client: httpx.AsyncClient,
    url: str,
    direction: str,
    method: str,
    rate: float,
    drain_timeout_s: float,
    node_name: str,
    outlier_precision: str = "fp16",
    regular_precision: str = "int8",
) -> None:
    """Send a single POST /config request to one node.

    Args:
        client: Shared httpx client.
        url: Full POST /config URL for the node.
        direction: ``"incoming"`` or ``"outgoing"``.
        method: Compression method value string.
        rate: Compression rate.
        drain_timeout_s: Drain timeout to pass to the node.
        node_name: Node name for logging.
        outlier_precision: Outlier precision for llmint8 (ignored otherwise).
        regular_precision: Regular-value precision for llmint8 (ignored otherwise).
    """
    payload = {
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
            "Config pushed to node '%s' (%s): method=%s rate=%s",
            node_name,
            direction,
            method,
            rate,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to push {direction} config to node '{node_name}' at {url}: {exc}"
        ) from exc
