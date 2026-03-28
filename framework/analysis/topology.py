from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from framework.datamodels.infra import InfraConfig
from framework.utils.loader import load_infra_config

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _layered_layout(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    """Assign (x, y) positions using a BFS layered (Sugiyama-style) layout.

    Nodes are placed in horizontal layers determined by their longest path
    from any source node.  Within each layer nodes are spread evenly on the
    y-axis.

    Args:
        nodes: Node names.
        edges: Directed edges as (from, to) pairs.

    Returns:
        Mapping of node name to (x, y) in data coordinates.
    """
    # Build adjacency and in-degree maps
    in_deg: dict[str, int] = {n: 0 for n in nodes}
    successors: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src in in_deg and dst in in_deg:
            successors[src].append(dst)
            in_deg[dst] += 1

    # Longest path from sources (BFS-based topological layer assignment)
    layer: dict[str, int] = {}
    queue: deque[str] = deque(n for n in nodes if in_deg[n] == 0)
    remaining_in: dict[str, int] = dict(in_deg)
    while queue:
        node = queue.popleft()
        cur_layer = layer.get(node, 0)
        for succ in successors[node]:
            layer[succ] = max(layer.get(succ, 0), cur_layer + 1)
            remaining_in[succ] -= 1
            if remaining_in[succ] == 0:
                queue.append(succ)
    # Any node not yet assigned (isolated or cycle) gets layer 0
    for n in nodes:
        if n not in layer:
            layer[n] = 0

    # Group nodes by layer
    layer_groups: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        layer_groups[layer[n]].append(n)

    positions: dict[str, tuple[float, float]] = {}
    for x_idx, group in layer_groups.items():
        for y_idx, node in enumerate(group):
            # Centre the group vertically around 0
            y = y_idx - (len(group) - 1) / 2.0
            positions[node] = (float(x_idx), y)
    return positions


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_arrow(
    ax: Any,
    src_pos: tuple[float, float],
    dst_pos: tuple[float, float],
    node_radius: float,
    label: str,
    color: str,
) -> None:
    """Draw an annotated curved arrow between two node centres."""
    x1, y1 = src_pos
    x2, y2 = dst_pos

    dx, dy = x2 - x1, y2 - y1
    dist = (dx**2 + dy**2) ** 0.5
    if dist == 0:
        return

    # Shorten arrow so it doesn't overlap node circles
    ux, uy = dx / dist, dy / dist
    sx = x1 + ux * node_radius
    sy = y1 + uy * node_radius
    ex = x2 - ux * node_radius
    ey = y2 - uy * node_radius

    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.8,
            connectionstyle="arc3,rad=0.08",
        ),
    )

    # Edge label at midpoint, slightly offset perpendicular to the edge
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    perp_x, perp_y = -uy * 0.12, ux * 0.12
    ax.text(
        mx + perp_x,
        my + perp_y,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )


def visualize_infra(
    infra: InfraConfig,
    title: str = "Infrastructure Topology",
    output_path: Path | None = None,
) -> None:
    """Render an InfraConfig as a directed graph and save or show it.

    Args:
        infra: Resolved InfraConfig to draw.
        title: Plot title.
        output_path: If given, save to this path; otherwise display interactively.
    """
    node_names = [n.name for n in infra.nodes]
    edges = [(lk.from_node, lk.to_node) for lk in infra.links]
    positions = _layered_layout(node_names, edges)

    fig, ax = plt.subplots(
        figsize=(max(6, len(set(p[0] for p in positions.values())) * 2.5), 5)
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    node_radius = 0.28
    node_color = "#4C72B0"
    edge_color = "#555555"

    # Draw edges first so nodes sit on top
    for link in infra.links:
        if link.from_node not in positions or link.to_node not in positions:
            continue
        parts: list[str] = []
        if link.bandwidth_mbps is not None:
            parts.append(f"{link.bandwidth_mbps:.0f} Mbps")
        if link.delay_ms is not None:
            parts.append(f"{link.delay_ms:.0f} ms")
        if link.loss_pct is not None:
            parts.append(f"{link.loss_pct:.1f}% loss")
        label = "\n".join(parts) if parts else ""
        _draw_arrow(
            ax,
            positions[link.from_node],
            positions[link.to_node],
            node_radius,
            label,
            edge_color,
        )

    # Draw nodes
    for node in infra.nodes:
        if node.name not in positions:
            continue
        x, y = positions[node.name]
        circle = plt.Circle((x, y), node_radius, color=node_color, zorder=3)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            node.name,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        # Resource annotation below node
        r = node.resources
        res_parts: list[str] = []
        if r.cpu is not None:
            res_parts.append(f"CPU:{r.cpu}")
        if r.memory is not None:
            res_parts.append(f"Mem:{r.memory}")
        if r.gpu > 0:
            res_parts.append(f"GPU:{r.gpu}")
        if res_parts:
            ax.text(
                x,
                y - node_radius - 0.12,
                " ".join(res_parts),
                ha="center",
                va="top",
                fontsize=7,
                color="#444444",
                zorder=4,
            )

    # Auto-scale axes with padding
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    pad = 0.7
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # Legend patch
    legend_handles = [
        mpatches.Patch(color=node_color, label="Compute node"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved topology graph to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualise an infra topology YAML as a directed graph."
    )
    parser.add_argument("--infra", required=True, type=Path, help="Path to infra.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (PNG/PDF/SVG). If omitted, display interactively.",
    )
    parser.add_argument("--title", default=None, help="Plot title override")
    args = parser.parse_args()

    if not args.infra.exists():
        print(f"ERROR: infra file not found: {args.infra}", file=sys.stderr)
        sys.exit(1)

    infra = load_infra_config(args.infra)
    title = args.title or f"Topology — {args.infra.stem}"
    visualize_infra(infra, title=title, output_path=args.output)


if __name__ == "__main__":
    main()
