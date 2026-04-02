"""Linear topology validation and global node ordering for optimization experiments.

Optimization experiments using Inference_Optimizer algorithms require that the
union of all pipeline node flows forms a single linear chain — no branching, no
merging, no skips.  Each pipeline must span a contiguous range [b_k, e_k] in
that chain.

This module provides:

- ``validate_linear_topology`` — raises ``ValueError`` on any violation.
- ``resolve_global_node_order`` — returns the canonical node list in chain order.
- ``build_task_topology`` — computes (b_k, L_k) per pipeline from the chain.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.datamodels.multi_experiment import PipelineConfig


def validate_linear_topology(pipelines: list[PipelineConfig]) -> None:
    """Validate that all pipeline flows are consistent with a single linear chain.

    Collects the directed edges implied by every pipeline's ``flow`` list and
    checks that:

    1. No node has more than one predecessor (no merges).
    2. No node has more than one successor (no branches).
    3. The resulting directed graph has no cycles.
    4. Each pipeline's flow is contiguous in the global linear ordering (no
       skips — every consecutive pair in the flow must be adjacent in the
       chain).

    Args:
        pipelines: Pipeline configs to validate.

    Raises:
        ValueError: If any of the above conditions are violated.  The message
            names all offending nodes or pipelines.
    """
    global_order = resolve_global_node_order(pipelines)
    global_index = {node: i for i, node in enumerate(global_order)}

    errors: list[str] = []
    for pipeline in pipelines:
        flow = pipeline.flow
        for i in range(len(flow) - 1):
            u, v = flow[i], flow[i + 1]
            if global_index[v] != global_index[u] + 1:
                errors.append(
                    f"Pipeline '{pipeline.name}': hop {u}→{v} is not adjacent in "
                    f"global chain {global_order} "
                    f"(indices {global_index[u]} and {global_index[v]})"
                )

    if errors:
        raise ValueError("Non-contiguous pipeline flows:\n" + "\n".join(errors))


def resolve_global_node_order(pipelines: list[PipelineConfig]) -> list[str]:
    """Derive the global linear node ordering from all pipeline flows.

    Collects directed edges (u → v) from every consecutive pair in every
    pipeline's ``flow`` list, then performs a topological sort.  Validates
    linearity (no branching, no merging) and absence of cycles before
    returning.

    Args:
        pipelines: Pipeline configs whose flows define the topology.

    Returns:
        Node names in linear chain order (index 0 = entry node).

    Raises:
        ValueError: If the union of pipeline flows is not a linear chain or
            contains a cycle.
    """
    edges: set[tuple[str, str]] = set()
    in_degree: dict[str, int] = defaultdict(int)
    out_degree: dict[str, int] = defaultdict(int)
    all_nodes: set[str] = set()

    for pipeline in pipelines:
        all_nodes.update(pipeline.flow)
        for i in range(len(pipeline.flow) - 1):
            u, v = pipeline.flow[i], pipeline.flow[i + 1]
            if (u, v) not in edges:
                edges.add((u, v))
                out_degree[u] += 1
                in_degree[v] += 1

    # Check linearity: each node has at most one predecessor and one successor.
    branch_errors: list[str] = []
    for node in sorted(all_nodes):
        if in_degree[node] > 1:
            preds = [u for u, v in edges if v == node]
            branch_errors.append(
                f"Node '{node}' has multiple predecessors: {preds} "
                f"(merge — topology is not linear)"
            )
        if out_degree[node] > 1:
            succs = [v for u, v in edges if u == node]
            branch_errors.append(
                f"Node '{node}' has multiple successors: {succs} "
                f"(branch — topology is not linear)"
            )
    if branch_errors:
        raise ValueError(
            "Pipeline flows do not form a linear chain:\n" + "\n".join(branch_errors)
        )

    # Topological sort: walk from the unique start node (in-degree 0).
    start_nodes = sorted(n for n in all_nodes if in_degree[n] == 0)
    if len(start_nodes) != 1:
        raise ValueError(
            f"Expected exactly one entry node (in-degree 0), found: {start_nodes}. "
            "Pipeline flows may contain a cycle or disconnected subgraph."
        )

    successors: dict[str, str] = {u: v for u, v in edges}
    chain: list[str] = []
    current: str | None = start_nodes[0]
    visited: set[str] = set()

    while current is not None:
        if current in visited:
            raise ValueError(f"Cycle detected in pipeline flows at node '{current}'.")
        chain.append(current)
        visited.add(current)
        current = successors.get(current)

    if len(chain) != len(all_nodes):
        unreachable = sorted(all_nodes - visited)
        raise ValueError(
            f"Nodes {unreachable} are not reachable from the entry node '{start_nodes[0]}'. "
            "Pipeline flows may contain a cycle or disconnected subgraph."
        )

    return chain


def build_task_topology(
    pipelines: list[PipelineConfig],
    global_order: list[str],
) -> dict[str, tuple[int, int]]:
    """Compute (b_k, L_k) per pipeline from the global node chain.

    Args:
        pipelines: Pipeline configs.
        global_order: Node names in linear chain order, as returned by
            ``resolve_global_node_order``.

    Returns:
        Dict mapping pipeline name to ``(b_k, L_k)`` where ``b_k`` is the
        zero-based index of the pipeline's first node in the chain and ``L_k``
        is the number of nodes it spans.
    """
    global_index = {node: i for i, node in enumerate(global_order)}
    result: dict[str, tuple[int, int]] = {}
    for pipeline in pipelines:
        b_k = global_index[pipeline.flow[0]]
        L_k = len(pipeline.flow)
        result[pipeline.name] = (b_k, L_k)
    return result
