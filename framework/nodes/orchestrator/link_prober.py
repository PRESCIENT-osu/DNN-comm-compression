"""Orchestrator-side link prober for optimization experiments.

The ``LinkProber`` is called by the optimization runner at controlled slot
boundaries to measure actual link throughput and update channel estimators.
It calls ``POST /probe/measure`` on each sending node; the node probes its
downstream neighbour and returns ``{rtt_ms, throughput_mbps}`` synchronously.

This is distinct from the per-node background probe loop (which runs
continuously during idle windows and emits ``LinkProbeEvent`` autonomously).
The orchestrator-triggered prober runs at a predictable cadence tied to the
optimization slot counter and feeds the channel estimators that drive the
optimizer decisions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import httpx

from framework.datamodels.events import ChannelEstimateQualityEvent, LinkProbeEvent
from framework.datamodels.multi_experiment import MultiNodeConfig
from framework.datamodels.opt_experiment import OptLinkConfig
from framework.nodes.metrics.emitter import MetricsEmitter

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT = httpx.Timeout(connect=10.0, write=120.0, read=60.0, pool=5.0)
_DEFAULT_PAYLOAD_BYTES = 1_048_576  # 1 MiB throughput probe payload


# ---------------------------------------------------------------------------
# Channel estimator protocol
# ---------------------------------------------------------------------------


class ChannelEstimator(Protocol):
    """Minimal interface expected of a channel capacity estimator.

    Concrete implementations live in ``framework.optimizer.channel_estimators``.
    Defined here as a Protocol so ``LinkProber`` does not depend on that module.
    """

    def update(self, bps: float) -> None:
        """Record a new capacity observation.

        Args:
            bps: Measured throughput in bits per second.
        """
        ...

    def estimate(self) -> float:
        """Return the current capacity estimate in bits per second.

        Returns:
            Estimated link capacity in bps.
        """
        ...

    @property
    def estimator_type(self) -> str:
        """Short name of the estimator algorithm (e.g. ``"moving_average"``).

        Returns:
            Estimator type string for metric tagging.
        """
        ...

    @property
    def n_observations(self) -> int:
        """Number of observations accumulated so far.

        Returns:
            Count of ``update()`` calls made.
        """
        ...


# ---------------------------------------------------------------------------
# LinkProber
# ---------------------------------------------------------------------------


class LinkProber:
    """Triggers on-demand link probes from the orchestrator during opt slots.

    For each link, calls ``POST /probe/measure`` on the sending node; the node
    probes the receiving node and returns ``{rtt_ms, throughput_mbps}``.  The
    measured throughput is used to update the corresponding channel estimator
    and emitted as a ``LinkProbeEvent`` and ``ChannelEstimateQualityEvent``.

    Args:
        links: Optimization link configs (defines which links to probe).
        nodes: Resolved node configs with host and port.
        estimators: Per-link channel estimators keyed by ``link_id``
            (``"{from_node}-{to_node}"``).
        emitter: Metrics emitter for event emission.
        payload_bytes: Size of throughput probe payload in bytes.
    """

    def __init__(
        self,
        links: list[OptLinkConfig],
        nodes: list[MultiNodeConfig],
        estimators: dict[str, ChannelEstimator],
        emitter: MetricsEmitter,
        payload_bytes: int = _DEFAULT_PAYLOAD_BYTES,
    ) -> None:
        self.links = links
        self.emitter = emitter
        self.payload_bytes = payload_bytes

        self._node_map: dict[str, MultiNodeConfig] = {n.name: n for n in nodes}
        self._estimators = estimators

    def _node_base_url(self, node_name: str) -> str:
        node = self._node_map[node_name]
        return f"http://{node.host}:{node.port}"

    async def probe_all(
        self,
        slot_id: int | None,
        experiment_id: str,
        run_id: str,
        sub_experiment_name: str | None = None,
    ) -> dict[str, float]:
        """Probe all configured links and update channel estimators.

        For each link A→B:
          1. Snapshot ``c_hat = estimator.estimate()`` before the probe.
          2. POST to ``http://node-A/probe/measure`` with ``target_base_url``
             of node B.
          3. Compute ``c_actual_bps = throughput_mbps * 1e6``.
          4. Update the estimator and emit ``LinkProbeEvent`` +
             ``ChannelEstimateQualityEvent``.

        Probe failures are logged as warnings and the link is skipped so that
        a single unreachable node does not abort the optimization loop.

        Args:
            slot_id: Current optimization slot index (for event tagging).
            experiment_id: Experiment identifier for emitted events.
            run_id: Run identifier for emitted events.
            sub_experiment_name: Active sub-experiment name for event tagging.

        Returns:
            Dict mapping ``link_id`` to the measured throughput in bps.
            Links that failed to probe are absent from the result.
        """
        results: dict[str, float] = {}

        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            for link in self.links:
                link_id = link.link_id
                estimator = self._estimators.get(link_id)

                from_url = self._node_base_url(link.from_node)
                to_base = self._node_base_url(link.to_node)

                # Snapshot estimator prediction before the probe so the
                # ChannelEstimateQualityEvent records the pre-probe estimate.
                c_hat_bps = estimator.estimate() if estimator is not None else 0.0

                try:
                    resp = await client.post(
                        f"{from_url}/probe/measure",
                        json={
                            "target_base_url": to_base,
                            "payload_bytes": self.payload_bytes,
                        },
                    )
                    resp.raise_for_status()
                    data: dict[str, Any] = resp.json()
                    rtt_ms: float = float(data["rtt_ms"])
                    throughput_mbps: float = float(data["throughput_mbps"])
                except Exception as exc:
                    logger.warning(
                        "Probe failed for link %s→%s: %s",
                        link.from_node,
                        link.to_node,
                        exc,
                    )
                    continue

                c_actual_bps = throughput_mbps * 1e6

                # Update estimator with the new observation.
                if estimator is not None:
                    estimator.update(c_actual_bps)

                results[link_id] = c_actual_bps

                # Emit raw probe event (consistent with the node background prober).
                self.emitter.emit(
                    LinkProbeEvent(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        from_node=link.from_node,
                        to_node=link.to_node,
                        rtt_ms=rtt_ms,
                        throughput_mbps=throughput_mbps,
                        slot_id=slot_id,
                        sub_experiment_name=sub_experiment_name,
                    )
                )

                # Emit quality event comparing pre-probe estimate to actual.
                if estimator is not None:
                    abs_err = abs(c_hat_bps - c_actual_bps)
                    rel_err = abs_err / c_actual_bps if c_actual_bps > 0 else 0.0
                    self.emitter.emit(
                        ChannelEstimateQualityEvent(
                            experiment_id=experiment_id,
                            run_id=run_id,
                            slot_id=slot_id,
                            from_node=link.from_node,
                            to_node=link.to_node,
                            c_hat_bps=c_hat_bps,
                            c_actual_bps=c_actual_bps,
                            absolute_error_bps=abs_err,
                            relative_error=rel_err,
                            estimator_type=estimator.estimator_type,
                            n_observations=estimator.n_observations,
                            sub_experiment_name=sub_experiment_name,
                        )
                    )

                logger.debug(
                    "Probe %s→%s: rtt=%.1fms throughput=%.1fMbps c_hat=%.1fMbps err=%.1f%%",
                    link.from_node,
                    link.to_node,
                    rtt_ms,
                    throughput_mbps,
                    c_hat_bps / 1e6,
                    (abs(c_hat_bps - c_actual_bps) / c_actual_bps * 100)
                    if c_actual_bps > 0
                    else 0.0,
                )

        return results

    def latest_estimates(self) -> dict[str, float]:
        """Return the current channel capacity estimate for each link.

        Args: (none)

        Returns:
            Dict mapping ``link_id`` to estimated capacity in bps.
        """
        return {
            link.link_id: (
                self._estimators[link.link_id].estimate()
                if link.link_id in self._estimators
                else 0.0
            )
            for link in self.links
        }


# ---------------------------------------------------------------------------
# Estimator warm-up helpers
# ---------------------------------------------------------------------------


async def warmup_from_metrics_server(
    estimator: ChannelEstimator,
    metrics_url: str,
    from_node: str,
    to_node: str,
    experiment_name_contains: str | None = None,
    limit: int = 500,
) -> int:
    """Warm up a channel estimator with historical probe data from the metrics server.

    Queries ``GET /metrics/query?event_type=link_probe`` with optional
    ``experiment_name_contains`` to scope results to the same hardware profile.
    Replays observations in timestamp order so the estimator's internal state
    (e.g. moving average window) reflects recency correctly.

    Args:
        estimator: The estimator instance to warm up.
        metrics_url: Base URL of the metrics server (e.g. ``http://metrics:9100``).
        from_node: Source node name to filter probe events.
        to_node: Destination node name to filter probe events.
        experiment_name_contains: Substring filter for experiment names.
        limit: Maximum number of probe records to load.

    Returns:
        Number of observations loaded into the estimator.
    """
    params: dict[str, str] = {
        "event_type": "link_probe",
        "from_node": from_node,
        "to_node": to_node,
        "limit": str(limit),
    }
    if experiment_name_contains:
        params["experiment_name_contains"] = experiment_name_contains

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{metrics_url}/metrics/query", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning(
            "Could not warm up estimator %s→%s from metrics server: %s",
            from_node,
            to_node,
            exc,
        )
        return 0

    events = data.get("events", [])
    # Sort by timestamp so moving-window estimators process observations in order.
    events.sort(key=lambda e: e.get("timestamp", 0.0))

    loaded = 0
    for event in events:
        mbps = event.get("throughput_mbps")
        if mbps is not None:
            estimator.update(float(mbps) * 1e6)
            loaded += 1

    logger.info(
        "Warmed up estimator %s→%s with %d observations from metrics server%s",
        from_node,
        to_node,
        loaded,
        f" (filter: {experiment_name_contains!r})" if experiment_name_contains else "",
    )
    return loaded


def warmup_from_prebuilt(
    estimator: ChannelEstimator,
    state_path: str,
    max_age_days: int = 30,
) -> int:
    """Warm up a channel estimator from a persisted state JSON file.

    Loads the ``observations`` list and replays entries newer than
    ``max_age_days`` into the estimator.

    Args:
        estimator: The estimator instance to warm up.
        state_path: Path to the JSON state file written by ``save_estimator_state``.
        max_age_days: Observations older than this many days are discarded.

    Returns:
        Number of observations loaded.
    """
    import json
    from pathlib import Path

    path = Path(state_path)
    if not path.exists():
        logger.warning("Prebuilt estimator state not found: %s", path)
        return 0

    try:
        with open(path) as f:
            state = json.load(f)
    except Exception as exc:
        logger.warning("Could not load prebuilt estimator state %s: %s", path, exc)
        return 0

    cutoff = time.time() - max_age_days * 86400
    observations: list[dict] = state.get("observations", [])
    loaded = 0
    for obs in observations:
        if obs.get("timestamp", 0.0) >= cutoff:
            estimator.update(float(obs["bps"]))
            loaded += 1

    logger.info(
        "Warmed up estimator from prebuilt state %s: %d observations loaded "
        "(cutoff: %d days)",
        path,
        loaded,
        max_age_days,
    )
    return loaded


def save_estimator_state(
    estimator: ChannelEstimator,
    state_path: str,
    observations: list[tuple[float, float]],
) -> None:
    """Persist an estimator's observation history to a JSON file.

    Args:
        estimator: The estimator (used for type metadata only).
        state_path: Destination file path.
        observations: List of ``(timestamp, bps)`` tuples to persist.
    """
    import json
    from pathlib import Path

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "estimator_type": estimator.estimator_type,
        "saved_at": time.time(),
        "observations": [{"timestamp": ts, "bps": bps} for ts, bps in observations],
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    logger.debug(
        "Saved estimator state → %s (%d observations)", path, len(observations)
    )
