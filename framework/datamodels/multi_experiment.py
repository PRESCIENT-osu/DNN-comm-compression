from __future__ import annotations

import itertools
import logging
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from framework.datamodels.experiment import (
    CompressionMethod,
    DatasetConfig,
    MetricsServerConfig,
    SweepMode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline and node
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """A named pipeline instance: model type, partition assignment, execution flow.

    Args:
        name: Unique pipeline identifier within the experiment (e.g. ``"resnet-a"``).
        model: Model type string (e.g. ``"resnet56"``, ``"llama-3.1-8b"``).
        partitions: Mapping from node name to the list of partition IDs loaded on
            that node for this pipeline.
        flow: Node names in execution order, from input to output.
    """

    name: str
    model: str
    partitions: dict[str, list[str]]
    flow: list[str]

    @model_validator(mode="after")
    def validate_flow(self) -> PipelineConfig:
        """Validate that flow nodes match partition keys."""
        partition_nodes = set(self.partitions.keys())
        flow_nodes = set(self.flow)
        if flow_nodes != partition_nodes:
            raise ValueError(
                f"Pipeline '{self.name}': flow nodes {sorted(flow_nodes)} do not match "
                f"partition nodes {sorted(partition_nodes)}"
            )
        return self


class MultiNodeConfig(BaseModel):
    """A physical node in the multi-model deployment.

    Args:
        name: Node identifier, must match names used in pipeline flows.
        host: Hostname or IP address used to reach this node.
        port: HTTP port the node server listens on.
    """

    name: str
    host: str
    port: int


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


class WorkloadPattern(StrEnum):
    """Task submission pattern for the multi-model orchestrator.

    FILL maintains a fixed number of in-flight tasks per pipeline at all times,
    replenishing immediately on completion.  ``window_per_pipeline: 1`` is
    equivalent to closed-loop.

    FIXED_RATE submits tasks at a fixed inter-arrival interval, sampling the
    pipeline by mix ratio.

    POISSON submits tasks with exponentially distributed inter-arrival times
    (mean = 1 / arrival_rate), sampling the pipeline by mix ratio.
    """

    FILL = "fill"
    FIXED_RATE = "fixed_rate"
    POISSON = "poisson"


class WorkloadConfig(BaseModel):
    """Workload submission configuration for the multi-model orchestrator.

    Args:
        pattern: Submission pattern (fill, fixed_rate, or poisson).
        window_per_pipeline: Number of tasks kept in-flight per pipeline.
            Required for ``fill`` pattern.
        arrival_rate: Mean task submission rate in tasks/second.
            Required for ``fixed_rate`` and ``poisson`` patterns.
        mix: Mapping from pipeline name to fraction of tasks assigned to that
            pipeline.  Fractions must sum to 1.0.
    """

    pattern: WorkloadPattern
    window_per_pipeline: int | None = None
    arrival_rate: float | None = None
    mix: dict[str, float]

    @model_validator(mode="after")
    def validate_workload(self) -> WorkloadConfig:
        """Validate pattern-specific fields and mix fractions."""
        if self.pattern == WorkloadPattern.FILL and self.window_per_pipeline is None:
            raise ValueError("window_per_pipeline is required for the fill pattern")
        if self.pattern in (WorkloadPattern.FIXED_RATE, WorkloadPattern.POISSON):
            if self.arrival_rate is None:
                raise ValueError(
                    f"arrival_rate is required for the {self.pattern.value} pattern"
                )
        total = sum(self.mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mix fractions must sum to 1.0, got {total:.6f}")
        return self


# ---------------------------------------------------------------------------
# Per-pipeline compression on a link
# ---------------------------------------------------------------------------


class PipelineLinkCompression(BaseModel):
    """Resolved compression config for one pipeline on one link.

    Args:
        compression: Compression method to apply.
        rate: Compression rate (required for all methods except ``none``).
        outlier_precision: Outlier precision for llmint8 (default: ``"fp16"``).
        regular_precision: Regular precision for llmint8 (default: ``"int8"``).
    """

    compression: CompressionMethod
    rate: float | None = None
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"

    @model_validator(mode="after")
    def validate_rate(self) -> PipelineLinkCompression:
        """Validate that rate is present for non-none compression methods."""
        if self.compression != CompressionMethod.NONE and self.rate is None:
            raise ValueError(
                f"rate is required for compression method '{self.compression}'"
            )
        return self


class MultiLinkConfig(BaseModel):
    """Default compression config for an inter-node link, keyed by pipeline ID.

    Args:
        from_node: Source node name.
        to_node: Destination node name.
        compression: Mapping from pipeline ID to its compression config on this link.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    compression: dict[str, PipelineLinkCompression]


# ---------------------------------------------------------------------------
# Sweep types
# ---------------------------------------------------------------------------


class SweepPipelineLinkConfig(BaseModel):
    """Compression config for one pipeline on one link within a sweep entry.

    Supports either a single ``rate`` or a list of ``rates`` to sweep over.

    Args:
        compression: Compression method to apply.
        rate: Single compression rate (mutually exclusive with ``rates``).
        rates: List of rates to sweep over (mutually exclusive with ``rate``).
        outlier_precision: Outlier precision for llmint8 (default: ``"fp16"``).
        regular_precision: Regular precision for llmint8 (default: ``"int8"``).
    """

    compression: CompressionMethod
    rate: float | None = None
    rates: list[float] | None = None
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"

    @model_validator(mode="after")
    def validate_rates(self) -> SweepPipelineLinkConfig:
        """Validate that exactly one of rate/rates is specified for non-none methods."""
        if self.compression != CompressionMethod.NONE:
            if self.rate is None and not self.rates:
                raise ValueError(
                    f"rate or rates is required for compression method '{self.compression}'"
                )
            if self.rate is not None and self.rates:
                raise ValueError("Specify rate or rates, not both")
        return self

    def effective_rates(self) -> list[float]:
        """Return the list of rates to sweep over for this (pipeline, link) pair."""
        if self.compression == CompressionMethod.NONE:
            return [0.0]
        if self.rates:
            return self.rates
        return [self.rate]  # type: ignore[list-item]


class MultiSweepLinkConfig(BaseModel):
    """Link config within a sweep entry — compression config is keyed by pipeline ID.

    Args:
        from_node: Source node name.
        to_node: Destination node name.
        compression: Mapping from pipeline ID to its sweep compression config.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    compression: dict[str, SweepPipelineLinkConfig]


class MultiSweepEntry(BaseModel):
    """A group of per-pipeline link configs forming one or more sweep runs.

    Args:
        links: Links with per-pipeline compression configs to sweep over.
    """

    links: list[MultiSweepLinkConfig]


# ---------------------------------------------------------------------------
# Resolved types
# ---------------------------------------------------------------------------


class ResolvedPipelineLinkConfig(BaseModel):
    """Fully resolved compression config for one pipeline on one link.

    Args:
        pipeline_id: Pipeline identifier.
        from_node: Source node name.
        to_node: Destination node name.
        compression: Compression method.
        rate: Concrete compression rate.
        outlier_precision: Outlier precision for llmint8.
        regular_precision: Regular precision for llmint8.
    """

    pipeline_id: str
    from_node: str
    to_node: str
    compression: CompressionMethod
    rate: float
    outlier_precision: str = "fp16"
    regular_precision: str = "int8"


class MultiResolvedRun(BaseModel):
    """A fully expanded sweep run with concrete configs for all (pipeline, link) pairs.

    Args:
        run_id: Self-describing identifier encoding all (pipeline, link, rate) configs.
        links: Resolved compression configs for every (pipeline, link) pair in the run.
    """

    run_id: str
    links: list[ResolvedPipelineLinkConfig]


# ---------------------------------------------------------------------------
# Spec types (source — read from multispecs/)
# ---------------------------------------------------------------------------


class MultiSubExperimentEntry(BaseModel):
    """One named entry in ``multispecs/<name>/sub_experiments.yaml``.

    Args:
        links: Default per-pipeline compression configs for all links.
        sweep_mode: How rates are expanded across (link, pipeline) dimensions.
        sweep: List of sweep entries defining compression rate combinations.
        baselines: References to previously run experiments for comparison,
            as ``"experiment_name/sub_experiment_name"`` strings.
    """

    model_config = ConfigDict(populate_by_name=True)

    links: list[MultiLinkConfig] = Field(default_factory=list)
    sweep_mode: SweepMode = SweepMode.PAIRED
    sweep: list[MultiSweepEntry] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)


class MultiSubExperimentsConfig(BaseModel):
    """Top-level schema for ``multispecs/<name>/sub_experiments.yaml``.

    Args:
        sub_experiments: Named sub-experiment entries keyed by sub-experiment name.
    """

    sub_experiments: dict[str, MultiSubExperimentEntry]


class MultiSpecConfig(BaseModel):
    """Top-level schema for ``multispecs/<name>/experiment.yaml``.

    Consumed by the experiment generator (``tools/generate.py``) together with a
    profile to produce a fully resolved ``experiments/multi/<name>/experiment.yaml``.

    Args:
        nodes: Physical nodes (name only, no host/port — filled in from profile).
        pipelines: Named pipeline instances with partition assignments and flows.
        datasets: Dataset config keyed by model type string.
        workload: Task submission pattern and mix ratio.
        metrics_server: Central metrics server address.
    """

    nodes: list[str]
    pipelines: list[PipelineConfig]
    datasets: dict[str, DatasetConfig]
    workload: WorkloadConfig
    metrics_server: MetricsServerConfig


# ---------------------------------------------------------------------------
# Generated types (consumed by multi-model runner — written to experiments/multi/)
# ---------------------------------------------------------------------------


class MultiResolvedSubExperiment(BaseModel):
    """A fully resolved sub-experiment in the generated multi-model experiment.yaml.

    Args:
        name: Sub-experiment name.
        links: Default per-pipeline compression configs for all links.
        sweep_mode: How rates are expanded.
        sweep: Sweep entries for this sub-experiment.
        baselines: Resolved baseline references as experiment_name/sub_experiment_name.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    links: list[MultiLinkConfig] = Field(default_factory=list)
    sweep_mode: SweepMode = SweepMode.PAIRED
    sweep: list[MultiSweepEntry] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)


class MultiExperimentConfig(BaseModel):
    """Top-level schema for a generated ``experiments/multi/<name>/experiment.yaml``.

    Consumed by the multi-model orchestrator runner.

    Args:
        name: Experiment name.
        nodes: Physical nodes with host and port resolved from the profile.
        pipelines: Named pipeline instances.
        datasets: Dataset config keyed by model type string.
        workload: Task submission pattern and mix ratio.
        metrics_server: Central metrics server address.
        sub_experiments: Resolved sub-experiments to execute sequentially.
    """

    name: str
    nodes: list[MultiNodeConfig]
    pipelines: list[PipelineConfig]
    datasets: dict[str, DatasetConfig]
    workload: WorkloadConfig
    metrics_server: MetricsServerConfig
    sub_experiments: list[MultiResolvedSubExperiment]

    @model_validator(mode="after")
    def validate_pipelines(self) -> MultiExperimentConfig:
        """Validate pipeline flow nodes exist in nodes list and mix covers all pipelines."""
        node_names = {n.name for n in self.nodes}
        pipeline_names = {p.name for p in self.pipelines}
        for pipeline in self.pipelines:
            for node in pipeline.flow:
                if node not in node_names:
                    raise ValueError(
                        f"Pipeline '{pipeline.name}' flow references unknown node '{node}'"
                    )
        mix_names = set(self.workload.mix.keys())
        if mix_names != pipeline_names:
            raise ValueError(
                f"Workload mix pipelines {sorted(mix_names)} do not match "
                f"defined pipelines {sorted(pipeline_names)}"
            )
        return self

    def node_for(self, name: str) -> MultiNodeConfig:
        """Return the node config for the given node name.

        Args:
            name: Node name to look up.

        Returns:
            Matching MultiNodeConfig.

        Raises:
            KeyError: If no node with the given name exists.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"Node '{name}' not found in experiment config")

    def pipeline_for(self, name: str) -> PipelineConfig:
        """Return the pipeline config for the given pipeline name.

        Args:
            name: Pipeline name to look up.

        Returns:
            Matching PipelineConfig.

        Raises:
            KeyError: If no pipeline with the given name exists.
        """
        for pipeline in self.pipelines:
            if pipeline.name == name:
                return pipeline
        raise KeyError(f"Pipeline '{name}' not found in experiment config")

    def resolve_sweep(
        self, sub_exp: MultiResolvedSubExperiment
    ) -> list[MultiResolvedRun]:
        """Expand a sub-experiment's sweep into a flat list of resolved runs.

        If no sweep entries are defined, returns a single run using the default
        link configs.

        Args:
            sub_exp: The sub-experiment whose sweep to resolve.

        Returns:
            List of MultiResolvedRun instances, one per concrete run.
        """
        if not sub_exp.sweep:
            return [self._run_from_links(sub_exp.links)]
        resolved: list[MultiResolvedRun] = []
        for entry in sub_exp.sweep:
            if sub_exp.sweep_mode == SweepMode.PAIRED:
                resolved.extend(self._expand_paired(entry))
            else:
                resolved.extend(self._expand_product(entry))
        return resolved

    def _run_from_links(self, links: list[MultiLinkConfig]) -> MultiResolvedRun:
        resolved = [
            ResolvedPipelineLinkConfig(
                pipeline_id=pipeline_id,
                from_node=lk.from_node,
                to_node=lk.to_node,
                compression=cfg.compression,
                rate=cfg.rate or 0.0,
                outlier_precision=cfg.outlier_precision,
                regular_precision=cfg.regular_precision,
            )
            for lk in links
            for pipeline_id, cfg in lk.compression.items()
        ]
        return MultiResolvedRun(run_id=self._make_run_id(resolved), links=resolved)

    def _expand_paired(self, entry: MultiSweepEntry) -> list[MultiResolvedRun]:
        # Each (link, pipeline) pair is one dimension; zip rates across all dimensions.
        dimensions = [
            (lk.from_node, lk.to_node, pipeline_id, cfg)
            for lk in entry.links
            for pipeline_id, cfg in lk.compression.items()
        ]
        rates_per_dim = [dim[3].effective_rates() for dim in dimensions]
        lengths = {len(r) for r in rates_per_dim}
        if len(lengths) > 1:
            raise ValueError(
                "Paired sweep requires all (link, pipeline) pairs to have the same "
                f"number of rates, got lengths {[len(r) for r in rates_per_dim]}"
            )
        n = next(iter(lengths)) if lengths else 1
        runs = []
        for rate_idx in range(n):
            resolved = [
                ResolvedPipelineLinkConfig(
                    pipeline_id=pipeline_id,
                    from_node=from_node,
                    to_node=to_node,
                    compression=cfg.compression,
                    rate=rates_per_dim[i][rate_idx],
                    outlier_precision=cfg.outlier_precision,
                    regular_precision=cfg.regular_precision,
                )
                for i, (from_node, to_node, pipeline_id, cfg) in enumerate(dimensions)
            ]
            runs.append(
                MultiResolvedRun(run_id=self._make_run_id(resolved), links=resolved)
            )
        return runs

    def _expand_product(self, entry: MultiSweepEntry) -> list[MultiResolvedRun]:
        # Each (link, pipeline) pair is one dimension; take the cartesian product.
        dimensions = [
            (lk.from_node, lk.to_node, pipeline_id, cfg)
            for lk in entry.links
            for pipeline_id, cfg in lk.compression.items()
        ]
        rates_per_dim = [dim[3].effective_rates() for dim in dimensions]
        runs = []
        for rate_combo in itertools.product(*rates_per_dim):
            resolved = [
                ResolvedPipelineLinkConfig(
                    pipeline_id=pipeline_id,
                    from_node=from_node,
                    to_node=to_node,
                    compression=cfg.compression,
                    rate=rate_combo[i],
                    outlier_precision=cfg.outlier_precision,
                    regular_precision=cfg.regular_precision,
                )
                for i, (from_node, to_node, pipeline_id, cfg) in enumerate(dimensions)
            ]
            runs.append(
                MultiResolvedRun(run_id=self._make_run_id(resolved), links=resolved)
            )
        return runs

    @staticmethod
    def _make_run_id(links: list[ResolvedPipelineLinkConfig]) -> str:
        """Build a self-describing run_id from resolved (pipeline, link) configs.

        Each entry contributes a descriptor of the form
        ``{method}-{pipeline}-{from}-{to}_{param}`` where param is:

        - ``{rate:.2f}`` for topk / randomk / quantization
        - ``{outlier}-{regular}@{rate:.2f}`` for llmint8
        - omitted for none

        Descriptors are sorted by (pipeline_id, from_node, to_node) and joined
        with ``--``.

        Args:
            links: Resolved (pipeline, link) configs for this run.

        Returns:
            Self-describing run ID string.
        """
        if not links:
            return "empty"
        parts: list[str] = []
        for lk in sorted(links, key=lambda a: (a.pipeline_id, a.from_node, a.to_node)):
            prefix = (
                f"{lk.compression.value}-{lk.pipeline_id}-{lk.from_node}-{lk.to_node}"
            )
            if lk.compression == CompressionMethod.NONE:
                parts.append(prefix)
            elif lk.compression == CompressionMethod.LLMINT8:
                parts.append(
                    f"{prefix}_{lk.outlier_precision}-{lk.regular_precision}@{lk.rate:.2f}"
                )
            else:
                parts.append(f"{prefix}_{lk.rate:.2f}")
        return "--".join(parts)
