"""Maps optimiser η decisions to concrete compression method + parameters.

The optimizer works with a continuous compression rate η ∈ [0, 1].  This
module translates (link_id, pipeline_id, η) into the ``(method, params)``
pair that the node controller pushes to nodes.

Supported methods
-----------------
- ``topk``    — η is the fraction of activation values to transmit.  Params:
                ``{"k": eta}``.
- ``llmint8`` — quantisation with mixed precision controlled by two parameters
                ``(feature_k, outlier_fraction)``.  The closest entry in the
                link's ``llmint8_mapping`` table is selected using a nearest-η
                look-up on the ``feature_k_values`` column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from framework.datamodels.opt_experiment import LLMInt8PipelineMapping, OptLinkConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CompressionDecision:
    """Concrete compression method and parameters for one (link, pipeline) pair.

    Attributes:
        method: Compression method name (``"topk"``, ``"llmint8"``, ``"none"``).
        params: Method-specific parameters dict (passed verbatim to node config).
        effective_eta: Effective η value this decision represents.
    """

    method: str
    params: dict[str, Any]
    effective_eta: float


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


class CompressionMapper:
    """Translates optimizer η values into node compression configs.

    One mapper instance covers all links in an experiment.  It is initialised
    once with the link configs and reused across all optimization slots.

    Args:
        links: All link configs from ``GeneratedOptExperimentConfig``.
    """

    def __init__(self, links: list[OptLinkConfig]) -> None:
        self._links: dict[str, OptLinkConfig] = {lk.link_id: lk for lk in links}

    def map(
        self,
        link_id: str,
        pipeline_id: str,
        eta: float,
    ) -> CompressionDecision:
        """Map η to a concrete compression decision for a specific link/pipeline.

        The method is selected by iterating ``allowed_methods`` in order and
        returning the first match.  If ``llmint8`` is allowed and an
        ``llmint8_mapping`` entry exists for ``pipeline_id``, the closest table
        entry is used.  Otherwise ``topk`` is used.

        Args:
            link_id: Link identifier (``"{from_node}-{to_node}"``).
            pipeline_id: Pipeline whose activations traverse this link.
            eta: Optimizer-selected compression rate in [0, 1].

        Returns:
            ``CompressionDecision`` with method, params, and effective η.

        Raises:
            KeyError: If ``link_id`` is not found in the mapper's link list.
        """
        link = self._links[link_id]
        allowed = link.allowed_methods or ["topk"]

        # Try llmint8 first if allowed and mapping is available.
        if "llmint8" in allowed and link.llmint8_mapping:
            pipeline_mapping = link.llmint8_mapping.get(pipeline_id)
            if pipeline_mapping is not None:
                return self._map_llmint8(pipeline_mapping, eta)

        # Default to topk.
        return CompressionDecision(
            method="topk",
            params={"k": eta},
            effective_eta=eta,
        )

    def clamp_eta(self, link_id: str, eta: float) -> float:
        """Clamp η to the link's [eta_min, eta_max] range.

        Args:
            link_id: Link identifier.
            eta: Raw η value from the optimizer.

        Returns:
            η clamped to [eta_min, eta_max].
        """
        link = self._links[link_id]
        return max(link.eta_min, min(link.eta_max, eta))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_llmint8(
        self,
        mapping: LLMInt8PipelineMapping,
        eta: float,
    ) -> CompressionDecision:
        """Select the llmint8 table entry closest to η.

        The mapping table is sorted by decreasing ``feature_k_values[0]``
        (higher feature_k = less compression = higher η).  The entry with
        the closest ``feature_k_values[0]`` to ``eta`` is selected.

        Args:
            mapping: Pipeline-level llmint8 mapping config.
            eta: Target compression rate.

        Returns:
            CompressionDecision for the best-matching table entry.
        """
        if not mapping.entries:
            logger.warning("llmint8 mapping has no entries; falling back to topk")
            return CompressionDecision(
                method="topk", params={"k": eta}, effective_eta=eta
            )

        # Build sorted list of (feature_k, entry_index) for binary search.
        fk_values = [e.feature_k_values[0] for e in mapping.entries]

        # Find the closest feature_k to eta.
        best_idx = 0
        best_dist = abs(fk_values[0] - eta)
        for i, fk in enumerate(fk_values[1:], 1):
            dist = abs(fk - eta)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        entry = mapping.entries[best_idx]
        effective_eta = entry.feature_k_values[0]

        params: dict[str, Any] = {
            "feature_k_values": entry.feature_k_values,
            "outlier_values": entry.outlier_values,
            "outlier_precision": mapping.outlier_precision,
            "regular_precision": mapping.regular_precision,
        }
        return CompressionDecision(
            method="llmint8",
            params=params,
            effective_eta=effective_eta,
        )
