"""Shared style constants and helpers for optimizer experiment plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paper-quality global rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,  # screen preview; save_fig overrides for print
    }
)

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# AM surrogate model types
SURROGATE_PALETTE: dict[str, str] = {
    "gbm": "#1f77b4",
    "poly2": "#ff7f0e",
    "poly3": "#2ca02c",
    "rf": "#d62728",
    "mlp": "#9467bd",
}

# Stein channel estimator types
ESTIMATOR_PALETTE: dict[str, str] = {
    "ma": "#8c564b",
    "lcb": "#e377c2",
    "mean": "#7f7f7f",
}

# Baseline variants
BASELINE_PALETTE: dict[str, str] = {
    "max_compression": "#bcbd22",
    "no_compression": "#17becf",
    "uniform_compression": "#aec7e8",
    "myopic": "#ffbb78",
    "conservative": "#98df8a",
    "movingavg": "#ff9896",
}

CSI_AWARE_COLOR: str = "#FFD700"

# ---------------------------------------------------------------------------
# Line / marker styles
# ---------------------------------------------------------------------------

AM_LINESTYLE: str = "-"
STEIN_LINESTYLE: str = "--"
BASELINE_LINESTYLE: str = ":"
CSI_AWARE_LINESTYLE: str = "-."

AM_MARKER: str = "o"
STEIN_MARKER: str = "^"
BASELINE_MARKER: str = "s"
CSI_AWARE_MARKER: str = "*"

MU_ALPHA: dict[float, float] = {0.5: 0.45, 4.0: 0.75, 10.0: 1.0}

# ---------------------------------------------------------------------------
# Figure defaults
# ---------------------------------------------------------------------------

FIGURE_SIZE: tuple[int, int] = (10, 6)
DPI: int = 300

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def surrogate_color(name: str) -> str:
    """Return the color for a surrogate type, falling back to gray."""
    return SURROGATE_PALETTE.get(name, "#999999")


def estimator_color(name: str) -> str:
    """Return the color for an estimator type, falling back to gray."""
    return ESTIMATOR_PALETTE.get(name, "#999999")


def baseline_color(name: str) -> str:
    """Return the color for a baseline variant, falling back to gray."""
    return BASELINE_PALETTE.get(name, "#cccccc")


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as PNG and close it.

    Args:
        fig: Matplotlib figure to save.
        output_dir: Directory to write the PNG into.
        name: Filename without extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def baseline_hlines(
    ax: plt.Axes,
    baselines_df,
    metric: str,
    *,
    alpha: float = 0.6,
    linewidth: float = 1.2,
) -> None:
    """Draw horizontal reference lines for each baseline on an Axes.

    Args:
        ax: Target axes.
        baselines_df: Slice of summary_df for baseline sub-experiments.
        metric: Column name to use as the Y value.
        alpha: Line alpha.
        linewidth: Line width.
    """
    if baselines_df.empty or metric not in baselines_df.columns:
        return
    for _, row in baselines_df.iterrows():
        variant = row.get("baseline_variant", "baseline")
        val = row.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.axhline(
            val,
            color=baseline_color(variant),
            linestyle=BASELINE_LINESTYLE,
            linewidth=linewidth,
            alpha=alpha,
            label=variant,
        )


def csi_aware_hline(
    ax: plt.Axes,
    csi_aware_df,
    metric: str,
    *,
    alpha: float = 0.8,
    linewidth: float = 1.5,
) -> None:
    """Draw a horizontal reference line for the CSI-aware oracle.

    Args:
        ax: Target axes.
        csi_aware_df: Slice of summary_df for csi_aware sub-experiment.
        metric: Column name to use as the Y value.
        alpha: Line alpha.
        linewidth: Line width.
    """
    if csi_aware_df.empty or metric not in csi_aware_df.columns:
        return
    val = csi_aware_df[metric].iloc[0]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return
    ax.axhline(
        val,
        color=CSI_AWARE_COLOR,
        linestyle=CSI_AWARE_LINESTYLE,
        linewidth=linewidth,
        alpha=alpha,
        label="csi_aware",
    )


def pareto_front_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points.

    Args:
        x: X values array.
        y: Y values array.
        maximize_x: Whether higher X is better.
        maximize_y: Whether higher Y is better.

    Returns:
        Boolean mask of shape (n,) where True = Pareto-optimal.
    """
    points = np.column_stack([x, y])
    n = len(points)
    dominated = np.zeros(n, dtype=bool)

    sx = 1.0 if maximize_x else -1.0
    sy = 1.0 if maximize_y else -1.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                sx * points[j, 0] >= sx * points[i, 0]
                and sy * points[j, 1] >= sy * points[i, 1]
                and (
                    sx * points[j, 0] > sx * points[i, 0]
                    or sy * points[j, 1] > sy * points[i, 1]
                )
            ):
                dominated[i] = True
                break

    return ~dominated
