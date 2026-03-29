from __future__ import annotations

import itertools
import logging
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

_VALID_OUTLIER_PRECISION = {"fp16", "int8"}
_VALID_REGULAR_PRECISION = {"fp16", "int8", "int4", "int2"}


class CompressionMethod(str, Enum):
    """Supported activation compression methods."""

    NONE = "none"
    TOPK = "topk"
    RANDOMK = "randomk"
    QUANTIZATION = "quantization"
    LLMINT8 = "llmint8"


class SweepMode(str, Enum):
    """How rates are expanded across links within a sweep entry.

    PAIRED applies the same rate index to all links (zip behaviour).
    PRODUCT generates the cartesian product of rates across links.
    """

    PAIRED = "paired"
    PRODUCT = "product"


class BaselineType(str, Enum):
    """Tags an experiment as a specific type of baseline."""

    SINGLE_NODE = "single_node"
    DISTRIBUTED_NO_COMPRESSION = "distributed_no_compression"


class NodeConfig(BaseModel):
    """Configuration for a single pipeline node."""

    name: str
    host: str
    port: int
    partitions: list[str]


class LinkConfig(BaseModel):
    """Default compression configuration for an inter-node link."""

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    compression: CompressionMethod
    rate: float | None = None
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"

    @model_validator(mode="after")
    def validate_link(self) -> LinkConfig:
        """Validate rate and LLMInt8 precision fields."""
        if self.compression != CompressionMethod.NONE and self.rate is None:
            raise ValueError(
                f"rate is required for compression method '{self.compression}'"
            )
        if self.compression == CompressionMethod.LLMINT8:
            if self.outlier_precision not in _VALID_OUTLIER_PRECISION:
                raise ValueError(
                    f"outlier_precision must be one of {_VALID_OUTLIER_PRECISION}, "
                    f"got '{self.outlier_precision}'"
                )
            if self.regular_precision not in _VALID_REGULAR_PRECISION:
                raise ValueError(
                    f"regular_precision must be one of {_VALID_REGULAR_PRECISION}, "
                    f"got '{self.regular_precision}'"
                )
        return self


class SweepLinkConfig(BaseModel):
    """Link compression config within a sweep entry, supporting multiple rates."""

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    compression: CompressionMethod
    rate: float | None = None
    rates: list[float] | None = None
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"

    @model_validator(mode="after")
    def validate_rates(self) -> SweepLinkConfig:
        """Validate rate fields and LLMInt8 precision values."""
        if self.compression != CompressionMethod.NONE:
            if self.rate is None and not self.rates:
                raise ValueError(
                    f"rate or rates required for compression method '{self.compression}'"
                )
            if self.rate is not None and self.rates:
                raise ValueError("Specify rate or rates, not both")
        if self.compression == CompressionMethod.LLMINT8:
            if self.outlier_precision not in _VALID_OUTLIER_PRECISION:
                raise ValueError(
                    f"outlier_precision must be one of {_VALID_OUTLIER_PRECISION}, "
                    f"got '{self.outlier_precision}'"
                )
            if self.regular_precision not in _VALID_REGULAR_PRECISION:
                raise ValueError(
                    f"regular_precision must be one of {_VALID_REGULAR_PRECISION}, "
                    f"got '{self.regular_precision}'"
                )
        return self

    def effective_rates(self) -> list[float]:
        """Return the list of rates to sweep over for this link."""
        if self.compression == CompressionMethod.NONE:
            return [0.0]
        if self.rates:
            return self.rates
        return [self.rate]  # type: ignore[list-item]


class SweepEntry(BaseModel):
    """A group of link configs forming one or more sweep runs."""

    links: list[SweepLinkConfig]


class ResolvedLinkConfig(BaseModel):
    """A fully resolved link config with a single concrete compression rate."""

    from_node: str
    to_node: str
    compression: CompressionMethod
    rate: float
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"


class ResolvedRun(BaseModel):
    """A fully expanded sweep run with concrete compression configs for all links."""

    run_id: str
    links: list[ResolvedLinkConfig]


class DatasetConfig(BaseModel):
    """Dataset configuration consumed by the orchestrator's data client."""

    name: str
    path: str
    batch_size: int = 100
    max_in_flight: int = 10
    max_seq_len: int = 512
    subjects: list[str] | None = None
    samples_per_subject: int | None = None
    tokenizer_path: str | None = None
    max_samples: int | None = None
    seed: int | None = None


class MetricsServerConfig(BaseModel):
    """Address of the central metrics ingestion server."""

    host: str
    port: int


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str
    model: str
    nodes: list[NodeConfig]
    links: list[LinkConfig] = Field(default_factory=list)
    sweep: list[SweepEntry] = Field(default_factory=list)
    sweep_mode: SweepMode = SweepMode.PAIRED
    dataset: DatasetConfig
    metrics_server: MetricsServerConfig
    baselines: list[str] = Field(default_factory=list)
    baseline: BaselineType | None = None

    @model_validator(mode="after")
    def validate_link_nodes(self) -> ExperimentConfig:
        """Validate that all link node references exist in the nodes list."""
        node_names = {n.name for n in self.nodes}
        for link in self.links:
            if link.from_node not in node_names:
                raise ValueError(f"Link references unknown node '{link.from_node}'")
            if link.to_node not in node_names:
                raise ValueError(f"Link references unknown node '{link.to_node}'")
        for entry in self.sweep:
            for link in entry.links:
                if link.from_node not in node_names:
                    raise ValueError(
                        f"Sweep link references unknown node '{link.from_node}'"
                    )
                if link.to_node not in node_names:
                    raise ValueError(
                        f"Sweep link references unknown node '{link.to_node}'"
                    )
        return self

    def node_order(self) -> list[str]:
        """Return node names in pipeline order derived from link topology.

        Returns:
            Node names ordered from first to last in the pipeline.

        Raises:
            ValueError: If the links do not form a simple linear chain.
        """
        if not self.links:
            return [n.name for n in self.nodes]
        sources = {link.from_node for link in self.links}
        targets = {link.to_node for link in self.links}
        heads = sources - targets
        if len(heads) != 1:
            raise ValueError("Links do not form a simple linear chain")
        link_map = {link.from_node: link.to_node for link in self.links}
        order: list[str] = []
        current: str | None = next(iter(heads))
        while current:
            order.append(current)
            current = link_map.get(current)
        return order

    def resolve_sweep(self) -> list[ResolvedRun]:
        """Expand the sweep definition into a flat list of resolved runs.

        If no sweep entries are defined, returns a single run using the
        default link configs.

        Returns:
            List of ResolvedRun instances, one per concrete experiment run.
        """
        if not self.sweep:
            return [self._run_from_links(self.links)]
        resolved: list[ResolvedRun] = []
        for idx, entry in enumerate(self.sweep):
            if self.sweep_mode == SweepMode.PAIRED:
                resolved.extend(self._expand_paired(idx, entry))
            else:
                resolved.extend(self._expand_product(entry))
        return resolved

    @staticmethod
    def _make_run_id(links: list[ResolvedLinkConfig]) -> str:
        """Build a self-describing run_id from a list of resolved link configs.

        Each link contributes one descriptor of the form
        ``{method}-{from}-{to}_{param}`` where param is:

        - ``{rate:.2f}`` for topk / randomk / quantization
        - ``{outlier_precision}-{regular_precision}`` for llmint8
        - omitted for none

        Descriptors are sorted by (from_node, to_node) and joined with ``--``
        so links are unambiguously delimited regardless of topology shape.

        Examples::

            topk-A-B_0.10--topk-B-C_0.30
            topk-A-B_0.10--quantization-B-C_0.25
            llmint8-A-B_fp16-int8@0.01--none-B-C
            none-A-B--none-B-C
        """
        parts: list[str] = []
        for lk in sorted(links, key=lambda a: (a.from_node, a.to_node)):
            if lk.compression == CompressionMethod.NONE:
                parts.append(f"none-{lk.from_node}-{lk.to_node}")
            elif lk.compression == CompressionMethod.LLMINT8:
                parts.append(
                    f"llmint8-{lk.from_node}-{lk.to_node}"
                    f"_{lk.outlier_precision}-{lk.regular_precision}@{lk.rate:.2f}"
                )
            else:
                parts.append(
                    f"{lk.compression.value}-{lk.from_node}-{lk.to_node}_{lk.rate:.2f}"
                )
        return "--".join(parts) if parts else "single-node"

    def _run_from_links(self, links: list[LinkConfig]) -> ResolvedRun:
        resolved_links = [
            ResolvedLinkConfig(
                from_node=lk.from_node,
                to_node=lk.to_node,
                compression=lk.compression,
                rate=lk.rate or 0.0,
                outlier_precision=lk.outlier_precision,
                regular_precision=lk.regular_precision,
            )
            for lk in links
        ]
        return ResolvedRun(
            run_id=self._make_run_id(resolved_links), links=resolved_links
        )

    def _expand_paired(self, entry_idx: int, entry: SweepEntry) -> list[ResolvedRun]:
        rates_per_link = [lk.effective_rates() for lk in entry.links]
        lengths = {len(r) for r in rates_per_link}
        if len(lengths) > 1:
            raise ValueError(
                f"Sweep entry {entry_idx}: all links must have the same number of "
                f"rates in paired mode, got lengths {[len(r) for r in rates_per_link]}"
            )
        n = next(iter(lengths)) if lengths else 1
        runs = []
        for rate_idx in range(n):
            resolved_links = [
                ResolvedLinkConfig(
                    from_node=lk.from_node,
                    to_node=lk.to_node,
                    compression=lk.compression,
                    rate=rates_per_link[i][rate_idx],
                    outlier_precision=lk.outlier_precision,
                    regular_precision=lk.regular_precision,
                )
                for i, lk in enumerate(entry.links)
            ]
            runs.append(
                ResolvedRun(
                    run_id=self._make_run_id(resolved_links), links=resolved_links
                )
            )
        return runs

    def _expand_product(self, entry: SweepEntry) -> list[ResolvedRun]:
        rates_per_link = [lk.effective_rates() for lk in entry.links]
        runs = []
        for rate_combo in itertools.product(*rates_per_link):
            resolved_links = [
                ResolvedLinkConfig(
                    from_node=lk.from_node,
                    to_node=lk.to_node,
                    compression=lk.compression,
                    rate=rate_combo[i],
                    outlier_precision=lk.outlier_precision,
                    regular_precision=lk.regular_precision,
                )
                for i, lk in enumerate(entry.links)
            ]
            runs.append(
                ResolvedRun(
                    run_id=self._make_run_id(resolved_links), links=resolved_links
                )
            )
        return runs
