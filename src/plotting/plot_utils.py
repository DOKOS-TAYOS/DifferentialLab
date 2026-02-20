"""Matplotlib plotting utilities for ODE solutions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from config import get_env_from_schema
from utils import get_logger

logger = get_logger(__name__)

_MAX_ELEMENTS_PLOT = 50

def _apply_plot_style() -> None:
    """Configure matplotlib rcParams from environment variables."""
    matplotlib.rcParams.update({
        "font.family": get_env_from_schema("FONT_FAMILY"),
        "font.size": get_env_from_schema("FONT_TICK_SIZE"),
        "axes.titlesize": get_env_from_schema("FONT_TITLE_SIZE"),
        "axes.titleweight": get_env_from_schema("FONT_TITLE_WEIGHT"),
        "axes.labelsize": get_env_from_schema("FONT_AXIS_SIZE"),
        "figure.dpi": get_env_from_schema("DPI"),
    })


def create_solution_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "y(x)",
    xlabel: str = "x",
    ylabel: str = "y",
    show_markers: bool = False,
    selected_derivatives: list[int] | None = None,
) -> Figure:
    """Create a publication-ready plot of the ODE solution.

    Args:
        x: Independent variable values.
        y: Solution values — shape ``(n_vars, n_points)`` or ``(n_points,)``.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        show_markers: Whether to overlay data-point markers.

    Returns:
        A matplotlib :class:`Figure`.
    """
    _apply_plot_style()

    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    line_color: str = get_env_from_schema("PLOT_LINE_COLOR")
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")
    line_style: str = get_env_from_schema("PLOT_LINE_STYLE")
    color_scheme: str = get_env_from_schema("PLOT_COLOR_SCHEME")

    y_2d = np.atleast_2d(y)
    if y_2d.shape[1] != len(x):
        y_2d = y_2d.T

    if selected_derivatives is None:
        selected_derivatives = list(range(y_2d.shape[0]))

    labels = ["y"] if y_2d.shape[0] == 1 else [f"y[{i}]" for i in range(y_2d.shape[0])]

    try:
        cmap = plt.cm.get_cmap(color_scheme)
        n_colors = max(1, len(selected_derivatives) - 1)
        colors = [line_color] + list(cmap(np.linspace(0, 1, n_colors)))
    except (ValueError, AttributeError):
        n_fallback = max(1, len(selected_derivatives) - 1)
        colors = [line_color] + list(plt.cm.Set1(np.linspace(0, 1, n_fallback)))

    for plot_idx, deriv_idx in enumerate(selected_derivatives):
        if deriv_idx >= y_2d.shape[0]:
            continue
        color = colors[plot_idx] if plot_idx < len(colors) else None
        ax.plot(x, y_2d[deriv_idx], color=color, linewidth=line_width,
                linestyle=line_style, label=labels[deriv_idx])

    if show_markers:
        marker: str = get_env_from_schema("PLOT_MARKER_FORMAT")
        msize: int = get_env_from_schema("PLOT_MARKER_SIZE")
        mfc: str = get_env_from_schema("PLOT_MARKER_FACE_COLOR")
        mec: str = get_env_from_schema("PLOT_MARKER_EDGE_COLOR")
        step = max(1, len(x) // _MAX_ELEMENTS_PLOT)
        for plot_idx, deriv_idx in enumerate(selected_derivatives):
            if deriv_idx >= y_2d.shape[0]:
                continue
            ax.plot(x[::step], y_2d[deriv_idx, ::step], marker=marker, markersize=msize,
                    markerfacecolor=mfc, markeredgecolor=mec, linestyle="none")

    show_title: bool = get_env_from_schema("PLOT_SHOW_TITLE")
    show_grid: bool = get_env_from_schema("PLOT_SHOW_GRID")

    if show_title and title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(True, alpha=0.3)
    if len(selected_derivatives) > 1:
        ax.legend()

    fig.tight_layout()
    return fig


def create_phase_plot(
    y: np.ndarray,
    title: str = "Phase Portrait",
    xlabel: str = "y",
    ylabel: str = "y'",
) -> Figure:
    """Create a phase portrait for a second-order ODE.

    Args:
        y: Solution array — shape ``(n_vars, n_points)`` with at least 2 rows.
        title: Plot title.
        xlabel: Label for horizontal axis.
        ylabel: Label for vertical axis.

    Returns:
        A matplotlib :class:`Figure`.
    """
    _apply_plot_style()

    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")
    line_color: str = get_env_from_schema("PLOT_LINE_COLOR")
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    y_2d = np.atleast_2d(y)
    ax.plot(y_2d[0], y_2d[1], color=line_color, linewidth=line_width)

    ax.plot(y_2d[0, 0], y_2d[1, 0], "o", color="green", markersize=8, label="Start")
    ax.plot(y_2d[0, -1], y_2d[1, -1], "s", color="red", markersize=8, label="End")

    show_title: bool = get_env_from_schema("PLOT_SHOW_TITLE")
    show_grid: bool = get_env_from_schema("PLOT_SHOW_GRID")

    if show_title and title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def save_plot(fig: Figure, filepath: Path) -> Path:
    """Save a matplotlib figure to disk.

    Args:
        fig: The figure to save.
        filepath: Destination file path.

    Returns:
        The path that was written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dpi: int = get_env_from_schema("DPI")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info("Plot saved: %s", filepath)
    return filepath


