"""Matplotlib plotting utilities for ODE solutions."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from config.env import get_env_from_schema
from utils.logger import get_logger

logger = get_logger(__name__)


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

    y_2d = np.atleast_2d(y)
    if y_2d.shape[1] != len(x):
        y_2d = y_2d.T

    labels = ["y"] if y_2d.shape[0] == 1 else [f"y[{i}]" for i in range(y_2d.shape[0])]
    colors = [line_color] + list(plt.cm.Set1(np.linspace(0, 1, max(1, y_2d.shape[0] - 1))))

    for i in range(y_2d.shape[0]):
        color = colors[i] if i < len(colors) else None
        ax.plot(x, y_2d[i], color=color, linewidth=line_width,
                linestyle=line_style, label=labels[i])

        if show_markers:
            marker: str = get_env_from_schema("PLOT_MARKER_FORMAT")
            msize: int = get_env_from_schema("PLOT_MARKER_SIZE")
            mfc: str = get_env_from_schema("PLOT_MARKER_FACE_COLOR")
            mec: str = get_env_from_schema("PLOT_MARKER_EDGE_COLOR")
            step = max(1, len(x) // 50)
            ax.plot(x[::step], y_2d[i, ::step], marker=marker, markersize=msize,
                    markerfacecolor=mfc, markeredgecolor=mec, linestyle="none")

    show_title: bool = get_env_from_schema("PLOT_SHOW_TITLE")
    show_grid: bool = get_env_from_schema("PLOT_SHOW_GRID")

    if show_title and title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(True, alpha=0.3)
    if y_2d.shape[0] > 1:
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


def embed_plot_in_tk(
    fig: Figure,
    parent: tk.Widget,
    toolbar: bool = True,
) -> FigureCanvasTkAgg:
    """Embed a matplotlib figure in a Tkinter parent widget.

    Args:
        fig: Figure to embed.
        parent: Tkinter parent (Frame, Toplevel, etc.).
        toolbar: Whether to add the navigation toolbar.

    Returns:
        The canvas object.
    """
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent)
        tb.update()
        tb.pack(fill=tk.X)

    return canvas
