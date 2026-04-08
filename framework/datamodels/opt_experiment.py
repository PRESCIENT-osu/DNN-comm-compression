"""Data models for optimization experiments.

Covers both the spec-side config read from optspecs/ and the fully resolved
config written to experiments/opt/ by the generator.

Spec-side (consumed by tools/generate.py --opt):
    OptSpecConfig             -- optspecs/<name>/experiment.yaml
    OptSubExperimentsConfig   -- optspecs/<name>/sub_experiments.yaml

Generated-side (consumed by opt_runner):
    GeneratedOptExperimentConfig -- experiments/opt/<name>/experiment.yaml
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from framework.datamodels.experiment import DatasetConfig, MetricsServerConfig
from framework.datamodels.multi_experiment import (
    MultiNodeConfig,
    PipelineConfig,
    WorkloadConfig,
)

# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------


class OptTaskConfig(BaseModel):
    """Throughput target and compute share weight for one pipeline.

    Args:
        throughput_target: Required tasks per second (R_k) for this pipeline.
        task_weight: Proportional WFQ scheduling weight (s_comp share).
            Weights across all pipelines need not sum to 1 — the scheduler
            normalises them internally.
    """

    throughput_target: float
    task_weight: float = 1.0


# ---------------------------------------------------------------------------
# Link config (optimization search space)
# ---------------------------------------------------------------------------


class LLMInt8PipelineMapping(BaseModel):
    """LLMInt8 precision config for one pipeline on one link.

    The optimizer's η is used directly as the outlier fraction — the top-η
    fraction of activation elements by magnitude are kept at ``outlier_precision``
    and the remaining elements are quantized to ``regular_precision``.

    Args:
        outlier_precision: Float dtype for the outlier group (default: ``"fp16"``).
        regular_precision: Integer dtype for the regular group (default: ``"int8"``).
    """

    outlier_precision: str = "fp16"
    regular_precision: str = "int8"


class OptLinkConfig(BaseModel):
    """Optimization search space constraints for one inter-node link.

    Args:
        from_node: Source node name.
        to_node: Destination node name.
        eta_min: Minimum compression ratio (most compressed).
        eta_max: Maximum compression ratio (least compressed; 1.0 = no compression).
        allowed_methods: Compression methods the optimizer may select for this link.
        llmint8_mapping: Per-pipeline LLMInt8 precision config.  Required when
            ``"llmint8"`` is in allowed_methods.  The optimizer's η is used
            directly as the outlier fraction for each pipeline.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    eta_min: float = 0.05
    eta_max: float = 1.0
    allowed_methods: list[str] = Field(default_factory=lambda: ["topk"])
    llmint8_mapping: dict[str, LLMInt8PipelineMapping] = Field(default_factory=dict)

    @property
    def link_id(self) -> str:
        """Canonical string identifier for this link."""
        return f"{self.from_node}-{self.to_node}"

    @model_validator(mode="after")
    def validate_eta_range(self) -> OptLinkConfig:
        """Validate that eta_min < eta_max."""
        if self.eta_min >= self.eta_max:
            raise ValueError(
                f"Link {self.from_node}→{self.to_node}: "
                f"eta_min ({self.eta_min}) must be < eta_max ({self.eta_max})"
            )
        return self


# ---------------------------------------------------------------------------
# Optimization loop config
# ---------------------------------------------------------------------------


class OptimizationLoopConfig(BaseModel):
    """Hyperparameters shared across all optimization sub-experiments.

    Args:
        n_slots: Number of optimization slots to run per sub-experiment.
        batches_per_slot: Inference batches submitted per slot.
        profiling_batches: Batches used during the profiling phase (η=1.0).
        link_probe_interval_slots: Probe links every N slots.
        epsilon: Floor and initial value for the Lagrangian dual variable λ (No-CSI only).
            Prevents λ from reaching zero, ensuring the delay penalty remains active.
    """

    n_slots: int = 100
    batches_per_slot: int = 10
    profiling_batches: int = 20
    link_probe_interval_slots: int = 1
    epsilon: float = 0.01


# ---------------------------------------------------------------------------
# Channel estimator config
# ---------------------------------------------------------------------------


class ChannelEstimatorType(StrEnum):
    """Estimator algorithm for link capacity c_i."""

    LAST_OBS = "last_observation"
    MEAN = "mean"
    RUNNING_MIN = "running_min"
    MOVING_AVG = "moving_average"
    LCB = "lcb"
    WINDOWED_LCB = "windowed_lcb"


class ChannelEstimatorHistorySource(StrEnum):
    """Where to load historical probe observations for warm-up."""

    METRICS_SERVER = "metrics_server"
    PREBUILT = "prebuilt"
    NONE = "none"


class ChannelEstimatorConfig(BaseModel):
    """Configuration for a per-link channel capacity estimator.

    Args:
        type: Estimator algorithm.
        window_size: Number of recent observations used by moving_average and lcb.
        z: LCB confidence width: estimate = mean - z * std.
        history_source: Where to load past probe observations for warm-up.
        experiment_name_contains: Substring filter applied to experiment_id when
            querying the metrics server for probe history.  Injected by the
            generator from the profile filename stem (e.g. ``"100mbps"``).
        prebuilt_path: Directory of persisted estimator state JSON files.
            Required when history_source is ``prebuilt``.
        max_age_days: Maximum age of loaded observations in days.
        warmup_value_bps: Fallback capacity used when no history is available.
    """

    type: ChannelEstimatorType = ChannelEstimatorType.MOVING_AVG
    window_size: int = 10
    z: float = 1.5
    history_source: ChannelEstimatorHistorySource = ChannelEstimatorHistorySource.NONE
    experiment_name_contains: str | None = None
    prebuilt_path: str | None = None
    max_age_days: int = 30
    warmup_value_bps: float = 1.0e8


class SteinOracleConfig(BaseModel):
    """Hyperparameters for the Stein gradient oracle used by stein_simulated backends.

    Args:
        sigma: Smoothing scale for antithetic perturbations (η ± σz).
        N: Number of antithetic sample pairs per gradient estimate (2N evals).
        n_fast_samples: Random subset size for accuracy_callable (fast path).
            Pass None to use the full test/evaluation set on every oracle call.
    """

    sigma: float = 0.05
    N: int = 50
    n_fast_samples: int = 512


# ---------------------------------------------------------------------------
# Sub-experiment models (discriminated union on `type`)
# ---------------------------------------------------------------------------


class ProfilingSubExperiment(BaseModel):
    """Runs profiling_batches slots at η=1.0 to measure τ_i and a_i.

    τ_i (compute latency per node) and a_i (activation tensor size) are stored
    as artifacts and reused by subsequent sub-experiments.

    Args:
        name: Sub-experiment name, used as artifact namespace.
    """

    type: Literal["profiling"] = "profiling"
    name: str


class AccuracyModelSubExperiment(BaseModel):
    """Trains a surrogate accuracy model A_k(η) for one pipeline.

    Runs a simulation-based sweep over per-link η vectors drawn from the
    configured sweep design, evaluates accuracy on each sample using the
    pipeline's simulation (``simulation_path``), then fits a sklearn surrogate
    to the resulting (η_vector, accuracy) pairs.  The fitted model is persisted
    as a pickle artifact and loaded by subsequent optimization sub-experiments.

    The sweep dataset is independent of the main task dataset and the Stein
    oracle dataset to prevent data leakage across evaluation roles.

    Args:
        name: Sub-experiment name, used as artifact key.
        pipeline_id: Which pipeline to model.
        model_type: sklearn model family.  One of ``"linear_monotonic"``,
            ``"poly2"``, ``"poly3"``, ``"gbm"``, ``"rf"``, ``"mlp"``,
            ``"mlp_small"``.
        sweep_design: How to sample the η space.  ``"random"`` draws
            ``n_sweep_samples`` points uniformly at random in
            ``[eta_min, eta_max]^n_links``; ``"diagonal"`` uses a 1-D grid
            with uniform η across all links.
        n_sweep_samples: Number of distinct η vectors to evaluate during the
            sweep.  Each vector triggers one simulation pass over the dataset
            batch configured in ``dataset``.
        dataset: Dataset slice used exclusively for accuracy evaluation during
            the sweep.  Should use a different seed than the main task dataset
            and the Stein oracle dataset.
        force_retrain: When ``True`` (default), always run the simulation sweep
            and fit a new model even if a valid cached artifact exists.  Set to
            ``False`` to reuse a previously fitted artifact when the config hash
            matches.
    """

    type: Literal["accuracy_model"] = "accuracy_model"
    name: str
    pipeline_id: str
    model_type: str = "gbm"
    sweep_design: Literal["diagonal", "random"] = "random"
    n_sweep_samples: int = 60
    dataset: DatasetConfig
    force_retrain: bool = True


class NoCsiSubExperiment(BaseModel):
    """No-CSI optimizer: SLSQP with surrogate/oracle A_k(η) and Lagrangian dual.

    Expands into one independent optimization run per mu value in mu_sweep.
    Each run uses the same accuracy model but its own dual variable state.

    Args:
        name: Sub-experiment name prefix; each mu run is named ``{name}_mu{mu}``.
        mu_sweep: Penalty weight μ values to try (one run each).
        accuracy_model_refs: Names of AccuracyModelSubExperiment whose artifacts
            to load as A_k(η) per pipeline.  Each entry is the ``name`` of an
            AccuracyModelSubExperiment; its ``pipeline_id`` determines which
            pipeline the model applies to.  May be None when a Stein oracle
            is used instead.
        stein_config: Stein gradient oracle hyperparameters.  Used when the
            accuracy model backend is stein_simulated.
        channel_estimator: Channel capacity estimator config.
        bcd_iterations: Number of Block Coordinate Descent iterations J per slot
            for the multi-task optimizer.  More iterations improve joint η/resource
            convergence at the cost of additional SLSQP solves (and Stein oracle
            calls when using stein_simulated backend).  Ignored for single-task
            (NoCSISingleTaskOptimizer has no BCD loop).
    """

    type: Literal["no_csi"] = "no_csi"
    name: str
    mu_sweep: list[float] = Field(default_factory=lambda: [1.0])
    accuracy_model_refs: list[str] | None = None
    stein_config: SteinOracleConfig | None = None
    channel_estimator: ChannelEstimatorConfig = Field(
        default_factory=ChannelEstimatorConfig
    )
    bcd_iterations: int = 10


class CsiAwareSubExperiment(BaseModel):
    """CSI-aware optimizer: closed-form η* = clip(c_i / (R_k × a_i), eta_min, eta_max).

    Requires no accuracy model — optimal η is derived directly from the measured
    link capacity and the throughput target.  Receives the raw probed c_t via
    ``DirectCsiAdapter``; no channel estimator is used.

    Args:
        name: Sub-experiment name.
        stein_config: Stein gradient oracle hyperparameters.  Used when the
            accuracy model backend is stein_simulated.
    """

    type: Literal["csi_aware"] = "csi_aware"
    name: str
    stein_config: SteinOracleConfig | None = None


# ---------------------------------------------------------------------------
# Baseline sub-experiments (single-task)
# ---------------------------------------------------------------------------


class MaxCompressionSingleSubExperiment(BaseModel):
    """Single-task baseline: maximum compression (η = eta_min) on every link.

    Corresponds to ``MaxCompressionSingleTaskBaseline`` in the external optimizer.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["max_compression_single"] = "max_compression_single"
    name: str


class NoCompressionSingleSubExperiment(BaseModel):
    """Single-task baseline: no compression (η = 1.0) on every link.

    Corresponds to ``NoCompressionSingleTaskBaseline`` in the external optimizer.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["no_compression_single"] = "no_compression_single"
    name: str


class UniformCompressionSingleSubExperiment(BaseModel):
    """Single-task baseline: uniform η derived from instantaneous channel capacity.

    Sets the same η on every link: η = min(1, min_i c_i / (R * a_i)).
    Receives the raw probed c_t via ``DirectCsiAdapter``; no channel estimator
    is used.
    Corresponds to ``UniformCompressionSingleTaskBaseline``.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["uniform_compression_single"] = "uniform_compression_single"
    name: str


class EstimatedCsiSingleSubExperiment(BaseModel):
    """Single-task baseline: per-link η from estimated channel capacity.

    η_i = min(1, c_hat_i / (R * a_i)) per link.  The estimator type in
    channel_estimator selects the variant: last_observation (Myopic),
    running_min (Conservative), or moving_average.
    Corresponds to ``EstimatedCSICompressionSingleTaskBaseline`` and its
    Myopic/Conservative/MovingAverage subclasses.

    Args:
        name: Sub-experiment name.
        channel_estimator: Channel capacity estimator config.
    """

    type: Literal["estimated_csi_single"] = "estimated_csi_single"
    name: str
    channel_estimator: ChannelEstimatorConfig = Field(
        default_factory=ChannelEstimatorConfig
    )


# ---------------------------------------------------------------------------
# Baseline sub-experiments (multi-task)
# ---------------------------------------------------------------------------


class MaxCompressionMultiSubExperiment(BaseModel):
    """Multi-task baseline: maximum compression with static equal resource shares.

    Corresponds to ``MaxCompressionMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["max_compression_multi"] = "max_compression_multi"
    name: str


class NoCompressionMultiSubExperiment(BaseModel):
    """Multi-task baseline: no compression with static equal resource shares.

    Corresponds to ``NoCompressionMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["no_compression_multi"] = "no_compression_multi"
    name: str


class StaticEqualShareSubExperiment(BaseModel):
    """Multi-task baseline: static equal-share allocation with CSI-derived η.

    Splits s_comp and s_comm equally across co-located tasks.  η derived
    from channel capacity and equal share.  Receives the raw probed c_t via
    ``DirectCsiAdapter``; no channel estimator is used.
    Corresponds to ``StaticEqualShareMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["static_equal_share"] = "static_equal_share"
    name: str


class ProportionalResourceSubExperiment(BaseModel):
    """Multi-task baseline: proportional resource allocation (τ_k / a_k weighted).

    Receives the raw probed c_t via ``DirectCsiAdapter``; no channel estimator
    is used.
    Corresponds to ``ProportionalResourceAllocationMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["proportional_resource"] = "proportional_resource"
    name: str


class StrictPriorityGreedySubExperiment(BaseModel):
    """Multi-task baseline: strict priority greedy allocation by task weight w_k.

    Receives the raw probed c_t via ``DirectCsiAdapter``; no channel estimator
    is used.
    Corresponds to ``StrictPriorityGreedyMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
    """

    type: Literal["strict_priority_greedy"] = "strict_priority_greedy"
    name: str


class DecoupledDescentSubExperiment(BaseModel):
    """Multi-task no-CSI baseline: decoupled equal-split stochastic descent.

    Maintains per-task dual queues λ_k and solves per-task SLSQP subproblems.
    Corresponds to ``DecoupledEqualSplitStochasticDescentMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
        mu: Penalty weight for the Lagrangian regularisation term.
        epsilon: Initial and minimum dual variable value.
        channel_estimator: Channel capacity estimator config (provides c_hat).
    """

    type: Literal["decoupled_descent"] = "decoupled_descent"
    name: str
    mu: float = 1.0
    epsilon: float = 0.1
    channel_estimator: ChannelEstimatorConfig = Field(
        default_factory=ChannelEstimatorConfig
    )


class QueueProportionalSubExperiment(BaseModel):
    """Multi-task no-CSI baseline: queue-proportional heuristic allocation.

    Allocates resources proportionally to dual queue depths λ_k; solves per-task
    SLSQP subproblems.
    Corresponds to ``QueueProportionalHeuristicMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
        mu: Penalty weight for the Lagrangian regularisation term.
        epsilon: Initial and minimum dual variable value.
        channel_estimator: Channel capacity estimator config (provides c_hat).
    """

    type: Literal["queue_proportional"] = "queue_proportional"
    name: str
    mu: float = 1.0
    epsilon: float = 0.1
    channel_estimator: ChannelEstimatorConfig = Field(
        default_factory=ChannelEstimatorConfig
    )


class HistoricalAverageCESubExperiment(BaseModel):
    """Multi-task no-CSI baseline: historical average certainty-equivalence.

    Feeds the historical mean channel estimate into the CSI-aware multi-task
    convex solver at each slot.
    Corresponds to ``HistoricalAverageCertaintyEquivalenceMultiTaskBaseline``.

    Args:
        name: Sub-experiment name.
        channel_estimator: Channel capacity estimator config (provides c_hat mean).
    """

    type: Literal["historical_average_ce"] = "historical_average_ce"
    name: str
    channel_estimator: ChannelEstimatorConfig = Field(
        default_factory=ChannelEstimatorConfig
    )


OptSubExperiment = Annotated[
    ProfilingSubExperiment
    | AccuracyModelSubExperiment
    | NoCsiSubExperiment
    | CsiAwareSubExperiment
    | MaxCompressionSingleSubExperiment
    | NoCompressionSingleSubExperiment
    | UniformCompressionSingleSubExperiment
    | EstimatedCsiSingleSubExperiment
    | MaxCompressionMultiSubExperiment
    | NoCompressionMultiSubExperiment
    | StaticEqualShareSubExperiment
    | ProportionalResourceSubExperiment
    | StrictPriorityGreedySubExperiment
    | DecoupledDescentSubExperiment
    | QueueProportionalSubExperiment
    | HistoricalAverageCESubExperiment,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Spec-side config (consumed by generator from optspecs/)
# ---------------------------------------------------------------------------


class OptSpecConfig(BaseModel):
    """Top-level schema for optspecs/<name>/experiment.yaml.

    Consumed by the experiment generator (tools/generate.py --opt) together
    with a profile to produce a fully resolved experiments/opt/<name>/experiment.yaml.

    Nodes list contains names only; host/port are filled in from the profile.

    Args:
        nodes: Node names in the experiment topology.
        pipelines: Named pipeline instances with partition assignments and flows.
        datasets: Dataset config keyed by model type string.
        stein_datasets: Dataset configs used exclusively by the Stein oracle,
            keyed by model type string.  Should use different seeds than
            ``datasets`` to prevent data overlap with actual inference tasks.
            Required when any sub-experiment uses a ``stein_config``.
        workload: Task submission pattern and mix ratio.
        tasks: Per-pipeline throughput targets and WFQ weights.
        links: Per-link optimization search space constraints.
        optimization_loop: Slot loop hyperparameters shared across sub-experiments.
        metrics_server: Central metrics server address.
    """

    nodes: list[str]
    pipelines: list[PipelineConfig]
    datasets: dict[str, DatasetConfig]
    stein_datasets: dict[str, DatasetConfig] | None = None
    workload: WorkloadConfig
    tasks: dict[str, OptTaskConfig]
    links: list[OptLinkConfig]
    optimization_loop: OptimizationLoopConfig
    metrics_server: MetricsServerConfig


class OptSubExperimentsConfig(BaseModel):
    """Top-level schema for optspecs/<name>/sub_experiments.yaml.

    Sub-experiments are an ordered list; the runner executes them sequentially.
    Order matters: profiling must precede accuracy_model which must precede
    optimization loops that reference it.

    Args:
        sub_experiments: Ordered list of typed sub-experiment configs.
    """

    sub_experiments: list[OptSubExperiment]


# ---------------------------------------------------------------------------
# Generated config (consumed by opt_runner from experiments/opt/)
# ---------------------------------------------------------------------------


class GeneratedOptExperimentConfig(BaseModel):
    """Top-level schema for a generated experiments/opt/<name>/experiment.yaml.

    Produced by tools/generate.py --opt; consumed by the optimization runner.
    Differs from OptSpecConfig in that nodes carry host/port from the profile,
    artifacts_dir is set, and channel_estimator configs have experiment_name_contains
    injected by the generator.

    Args:
        name: Canonical experiment name (derived from spec + profile).
        nodes: Physical nodes with host and port resolved from the profile.
        pipelines: Named pipeline instances.
        datasets: Dataset config keyed by model type string.
        stein_datasets: Dataset configs used exclusively by the Stein oracle,
            keyed by model type string.  Should use different seeds than
            ``datasets`` to prevent data overlap with actual inference tasks.
        workload: Task submission pattern and mix ratio.
        tasks: Per-pipeline throughput targets and WFQ weights.
        links: Per-link optimization search space constraints.
        optimization_loop: Slot loop hyperparameters.
        metrics_server: Central metrics server address.
        artifacts_dir: Root directory for profiling, accuracy model, and estimator
            state artifacts.
        sub_experiments: Ordered list of resolved sub-experiment configs.
    """

    name: str
    nodes: list[MultiNodeConfig]
    pipelines: list[PipelineConfig]
    datasets: dict[str, DatasetConfig]
    stein_datasets: dict[str, DatasetConfig] | None = None
    workload: WorkloadConfig
    tasks: dict[str, OptTaskConfig]
    links: list[OptLinkConfig]
    optimization_loop: OptimizationLoopConfig
    metrics_server: MetricsServerConfig
    artifacts_dir: str
    sub_experiments: list[OptSubExperiment]

    @model_validator(mode="after")
    def validate_tasks_and_links(self) -> GeneratedOptExperimentConfig:
        """Validate task ids match pipelines and eta bounds are consistent."""
        pipeline_names = {p.name for p in self.pipelines}
        for task_id in self.tasks:
            if task_id not in pipeline_names:
                raise ValueError(
                    f"Task '{task_id}' does not match any defined pipeline. "
                    f"Defined pipelines: {sorted(pipeline_names)}"
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

    def link_for(self, from_node: str, to_node: str) -> OptLinkConfig:
        """Return the link config for the given node pair.

        Args:
            from_node: Source node name.
            to_node: Destination node name.

        Returns:
            Matching OptLinkConfig.

        Raises:
            KeyError: If no link between the given nodes exists.
        """
        for lk in self.links:
            if lk.from_node == from_node and lk.to_node == to_node:
                return lk
        raise KeyError(f"Link {from_node}→{to_node} not found in experiment config")
