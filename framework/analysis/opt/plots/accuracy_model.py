"""Accuracy model sweep visualizations."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from framework.analysis.opt.plots._style import save_fig

# Dense grid resolution for contour evaluation
_GRID_N = 80


def _link_labels(col_x: str, col_y: str) -> tuple[str, str]:
    def _fmt(c: str) -> str:
        nodes = c.removeprefix("eta_").split("_")
        return r"$\eta_{" + r" \to ".join(nodes) + r"}$"

    return _fmt(col_x), _fmt(col_y)


def _load_model(pkl_path: Path):
    """Load a saved accuracy model dict from a pkl file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _predict_grid(model_dict: dict, grid_xy: np.ndarray) -> np.ndarray:
    """Run scaler → poly (optional) → model on an (N, 2) grid."""
    X = model_dict["scaler"].transform(grid_xy)
    if model_dict.get("poly") is not None:
        X = model_dict["poly"].transform(X)
    return np.clip(model_dict["model"].predict(X), 0.0, 1.0)


def _discover_models(model_dir: Path) -> dict[str, Path]:
    """Return {surrogate_type: pkl_path} by scanning model_dir for known types."""
    known = {"gbm", "poly2", "poly3", "rf", "mlp"}
    found: dict[str, Path] = {}
    for pkl in model_dir.glob("*.pkl"):
        for stype in known:
            if f"__{stype}__" in pkl.name and stype not in found:
                found[stype] = pkl
    return found


def plot_accuracy_sweep_surface(
    sweep_df: pd.DataFrame,
    output_dir: Path,
    model_dir: Path | None = None,
) -> None:
    """Accuracy model sweep: scatter of training data + per-surrogate contour plots.

    Generates:
    - ``am_sweep_scatter.png``  — raw 100-point training set, coloured by
      accuracy.  All surrogates share these points; the scatter shows the
      actual sparsity of the search space.
    - ``am_sweep_contour_{surrogate}.png``  — one figure per surrogate found
      in *model_dir*, showing the fitted accuracy surface as filled contours
      with the training scatter overlaid.  Requires *model_dir*.

    Args:
        sweep_df: Output of ``build_accuracy_sweep_table()``.
        output_dir: Directory to write PNGs.
        model_dir: Directory containing saved ``.pkl`` accuracy model
            artifacts.  If ``None`` or the directory does not exist, contour
            plots are skipped.
    """
    if sweep_df.empty:
        return

    eta_link_cols = sorted(
        [c for c in sweep_df.columns if c.startswith("eta_") and c != "mean_eta"]
    )
    if len(eta_link_cols) != 2:
        return

    col_x, col_y = eta_link_cols
    label_x, label_y = _link_labels(col_x, col_y)

    data = sweep_df[[col_x, col_y, "accuracy"]].dropna().drop_duplicates()
    x = data[col_x].values
    y = data[col_y].values
    acc = data["accuracy"].values

    # ------------------------------------------------------------------
    # Scatter: raw training points
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(
        x,
        y,
        c=acc,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=70,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, label="Accuracy")
    cb.ax.tick_params(labelsize=10)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    save_fig(fig, output_dir, "am_sweep_scatter")

    # ------------------------------------------------------------------
    # Per-surrogate contour plots
    # ------------------------------------------------------------------
    if model_dir is None or not model_dir.exists():
        return

    models = _discover_models(model_dir)
    if not models:
        return

    # Build evaluation grid once
    g = np.linspace(0, 1, _GRID_N)
    gx, gy = np.meshgrid(g, g)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    levels = np.linspace(0, 1, 21)

    for surrogate, pkl_path in sorted(models.items()):
        try:
            model_dict = _load_model(pkl_path)
        except Exception:
            continue

        z = _predict_grid(model_dict, grid_xy).reshape(_GRID_N, _GRID_N)

        fig, ax = plt.subplots(figsize=(5, 4.5))

        # Filled contour surface
        cf = ax.contourf(gx, gy, z, levels=levels, cmap="RdYlGn", vmin=0, vmax=1)
        # Contour lines for readability
        ax.contour(
            gx, gy, z, levels=levels[::4], colors="white", linewidths=0.6, alpha=0.5
        )

        # Training scatter overlaid
        ax.scatter(
            x,
            y,
            c=acc,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            s=55,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )

        cb = fig.colorbar(cf, ax=ax, label="Accuracy")
        cb.ax.tick_params(labelsize=10)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(surrogate)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.tight_layout()
        save_fig(fig, output_dir, f"am_sweep_contour_{surrogate}")
