from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from framework.datamodels.experiment import (
    DatasetConfig,
    LinkConfig,
    MetricsServerConfig,
    NodeConfig,
    SweepEntry,
    SweepMode,
)


class BaselineRef(BaseModel):
    """Reference to a baseline sub-experiment.

    A same-spec reference omits ``spec`` and ``profile`` — the tool fills
    them in from the current generation context.  A cross-spec reference
    specifies all three fields and is fully self-contained.
    """

    model_config = ConfigDict(populate_by_name=True)

    sub_experiment: str
    spec: str | None = None  # if None, same spec as current
    profile: str | None = None  # if None, same profile as current


class SubExperimentEntry(BaseModel):
    """One named entry in sub_experiments.yaml."""

    model_config = ConfigDict(populate_by_name=True)

    links: list[LinkConfig] = Field(default_factory=list)
    sweep_mode: SweepMode = SweepMode.PAIRED
    sweep: list[SweepEntry] = Field(default_factory=list)
    baselines: list[BaselineRef] = Field(default_factory=list)
    dataset: DatasetConfig | None = None  # overrides model-level dataset when set


class SubExperimentsConfig(BaseModel):
    """Top-level schema for sub_experiments.yaml."""

    sub_experiments: dict[str, SubExperimentEntry]


class ResolvedSubExperiment(BaseModel):
    """A fully resolved sub-experiment entry in the generated experiment.yaml.

    Baselines are stored as ``experiment_name/sub_experiment_name`` strings
    pointing to already-materialised experiments.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    links: list[LinkConfig] = Field(default_factory=list)
    sweep_mode: SweepMode = SweepMode.PAIRED
    sweep: list[SweepEntry] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    dataset: DatasetConfig | None = None


class GeneratedExperimentConfig(BaseModel):
    """Top-level schema for the experiment.yaml produced by tools/generate.py.

    Consumed by the orchestrator. Shared
    fields (model, nodes, dataset, metrics_server) apply to all
    sub-experiments unless overridden at the sub-experiment level.
    """

    name: str
    model: str
    nodes: list[NodeConfig]
    dataset: DatasetConfig
    metrics_server: MetricsServerConfig
    sub_experiments: list[ResolvedSubExperiment]
