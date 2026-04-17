"""Maps optimiser η decisions to concrete compression method + parameters.

The optimizer works with a continuous compression rate η ∈ [0, 1].  This
module translates (link_id, pipeline_id, η) into the ``(method, params)``
pair that the node controller pushes to nodes.

Supported methods
-----------------
- ``topk``         — η is the fraction of activation values to transmit.
                     Params: ``{"k": eta}``.
- ``quantization`` — η is snapped to the nearest valid discrete rate
                     ``{0.5, 0.25, 0.125, 0.0625}`` (fp16/int8/int4/int2).
                     Params: ``{"rate": snapped_rate}``.
- ``llmint8``      — η is used directly as the outlier fraction: the top-η
                     fraction of elements by magnitude are stored at
                     ``outlier_precision`` (default: fp16) and the rest at
                     ``regular_precision`` (default: int8).
                     Params: ``{"rate": eta, "outlier_precision": ..., "regular_precision": ...}``.

Quantization discretisation
---------------------------
The quantization scheme has only four effective levels.  The nearest-neighbour
snapping means the optimizer's continuous η collapses to these bands::

    η ∈ [0.375,  1.0   ) → rate = 0.5    (fp16,  2× compression)
    η ∈ [0.1875, 0.375 ) → rate = 0.25   (int8,  4× compression)
    η ∈ [0.09375,0.1875) → rate = 0.125  (int4,  8× compression)
    η ∈ [0.0,    0.09375) → rate = 0.0625 (int2, 16× compression)

Any fine-grained η variation within a band has zero effect on the physical
compression.  The external optimizer is unaware of this discretisation — it
treats η as continuous.  Use ``snap_eta`` to query the effective η.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from framework.datamodels.opt_experiment import OptLinkConfig

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
        self._active_scheme: str | None = None

    def set_active_scheme(self, scheme: str) -> None:
        """Set the active compression scheme for all subsequent ``map()`` calls.

        When set, ``map()`` dispatches directly to the named scheme rather than
        using the priority-chain over ``allowed_methods``.  Call this at the
        start of each scheme iteration in the scheme sweep loop.

        Args:
            scheme: Compression scheme name (e.g. ``"topk"``, ``"quantization"``,
                ``"llmint8"``).
        """
        self._active_scheme = scheme

    def map(
        self,
        link_id: str,
        pipeline_id: str,
        eta: float,
    ) -> CompressionDecision:
        """Map η to a concrete compression decision for a specific link/pipeline.

        When ``_active_scheme`` is set (via ``set_active_scheme``), dispatches
        directly to that scheme.  Otherwise falls back to ``topk``.

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

        if self._active_scheme == "llmint8":
            if link.llmint8_mapping:
                pipeline_mapping = link.llmint8_mapping.get(pipeline_id)
                if pipeline_mapping is not None:
                    return self._map_llmint8(pipeline_mapping, eta)
            # Fall through to topk if no llmint8 mapping for this pipeline.
            return CompressionDecision(
                method="topk",
                params={"k": eta},
                effective_eta=eta,
            )

        if self._active_scheme == "quantization":
            return self._map_quantization(eta)

        # Default (topk or None active scheme).
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

    _QUAN_RATES: list[float] = [0.5, 0.25, 0.125, 0.0625]

    def _map_quantization(self, eta: float) -> CompressionDecision:
        """Snap η to the nearest valid quantization rate.

        Args:
            eta: Target compression rate.

        Returns:
            CompressionDecision with method ``"quantization"`` and the snapped rate.
        """
        snapped = min(self._QUAN_RATES, key=lambda r: abs(r - eta))
        return CompressionDecision(
            method="quantization",
            params={"rate": snapped},
            effective_eta=snapped,
        )

    def _map_llmint8(
        self,
        mapping: Any,
        eta: float,
    ) -> CompressionDecision:
        """Map η directly to an LLMInt8 compression decision.

        η is used as the outlier fraction: the top-η fraction of activation
        elements by magnitude are stored at ``outlier_precision``; the rest
        are quantized to ``regular_precision``.  This matches the library's
        convention where η is passed directly as the outlier ratio.

        Args:
            mapping: Pipeline-level LLMInt8PipelineMapping config.
            eta: Optimizer-selected compression rate in [0, 1].

        Returns:
            CompressionDecision with method ``"llmint8"`` and the precision params.
        """
        return CompressionDecision(
            method="llmint8",
            params={
                "rate": eta,
                "outlier_precision": mapping.outlier_precision,
                "regular_precision": mapping.regular_precision,
            },
            effective_eta=eta,
        )
