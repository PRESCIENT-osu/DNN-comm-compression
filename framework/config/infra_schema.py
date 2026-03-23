from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NodeResources(BaseModel):
    """Compute resources allocated to a node.

    ``cpu`` and ``memory`` default to ``None`` (no limit).  Set them only when
    you want to enforce a specific resource ceiling in the generated manifests.
    """

    cpu: float | None = None
    memory: str | None = None
    gpu: int = 0


class InfraNodeConfig(BaseModel):
    """Infrastructure configuration for a single node."""

    name: str
    resources: NodeResources = Field(default_factory=NodeResources)


class InfraLinkConfig(BaseModel):
    """Infrastructure configuration for an inter-node link."""

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    bandwidth_mbps: float | None = None


class InfraConfig(BaseModel):
    """Infrastructure configuration for a deployment.

    Inheritance is resolved at load time by the config loader; the
    inherits field is not present on a fully resolved InfraConfig.
    """

    nodes: list[InfraNodeConfig]
    links: list[InfraLinkConfig] = Field(default_factory=list)
