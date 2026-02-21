"""Matplotlib plotting utilities for ODE solutions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from config import get_env_from_schema
from utils import get_logger

logger = get_logger(__name__)

_MAX_ELEMENTS_PLOT = 50


def _apply_plot_style() -> None:
    """Configure matplotlib rcParams from environment variables."""
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": get_env_from_schema("FONT_FAMILY"),
        "font.size": get_env_from_schema("FONT_TICK_SIZE"),
        "axes.titlesize": get_env_from_schema("FONT_TITLE_SIZE"),
        "axes.titleweight": get_env_from_schema("FONT_TITLE_WEIGHT"),
        "axes.labelsize": get_env_from_schema("FONT_AXIS_SIZE"),
        "figure.dpi": get_env_from_schema("DPI"),
    })


def _new_figure() -> tuple[Any, Any]:
    """Create a configured figure and axes from env settings.

    Returns:
        Tuple of ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    _apply_plot_style()
    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")
    return plt.subplots(figsize=(width, height), dpi=dpi)


def _finalize_plot(
    ax: Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    legend: bool = False,
) -> None:
    """Apply title, grid, axis labels, and optional legend from env settings.

    Args:
        ax: Matplotlib axes.
        title: Plot title (shown only if ``PLOT_SHOW_TITLE`` is enabled).
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        legend: Whether to display the legend.
    """
    axis_style: str = get_env_from_schema("FONT_AXIS_STYLE")

    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    ax.set_xlabel(xlabel, fontstyle=axis_style)
    ax.set_ylabel(ylabel, fontstyle=axis_style)
    if get_env_from_schema("PLOT_SHOW_GRID"):
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()


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
        selected_derivatives: Indices of solution components to plot.

    Returns:
        A matplotlib :class:`Figure`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = _new_figure()

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
        cmap = plt.colormaps.get_cmap(color_scheme)
        n_colors = max(1, len(selected_derivatives) - 1)
        colors = [line_color] + list(cmap(np.linspace(0, 1, n_colors)))
    except (ValueError, AttributeError):
        n_fallback = max(1, len(selected_derivatives) - 1)
        colors = [line_color] + list(plt.colormaps.get_cmap("Set1")(np.linspace(0, 1, n_fallback)))

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

    _finalize_plot(ax, title, xlabel, ylabel, legend=len(selected_derivatives) > 1)
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
    import numpy as np

    fig, ax = _new_figure()

    line_color: str = get_env_from_schema("PLOT_LINE_COLOR")
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")

    y_2d = np.atleast_2d(y)
    ax.plot(y_2d[0], y_2d[1], color=line_color, linewidth=line_width)

    ax.plot(y_2d[0, 0], y_2d[1, 0], "o", color="green", markersize=8, label="Start")
    ax.plot(y_2d[0, -1], y_2d[1, -1], "s", color="red", markersize=8, label="End")

    _finalize_plot(ax, title, xlabel, ylabel, legend=True)
    fig.tight_layout()
    return fig


def create_surface_plot(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "f(x, y)",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "f",
) -> Figure:
    """Create a 3D surface plot for 2D scalar field data.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        z: 2D array of values, shape (len(y), len(x)).
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        zlabel: Label for z-axis.

    Returns:
        A matplotlib :class:`Figure`.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _apply_plot_style()
    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(x, y)
    if z.shape != X.shape:
        z = np.asarray(z)
        if z.shape != X.shape:
            raise ValueError(
                f"z shape {z.shape} does not match grid {X.shape}"
            )

    surf = ax.plot_surface(
        X, Y, z,
        cmap="viridis",
        alpha=0.9,
        edgecolor="none",
    )
    fig.colorbar(surf, ax=ax, shrink=0.6)

    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    axis_style: str = get_env_from_schema("FONT_AXIS_STYLE")
    ax.set_xlabel(xlabel, fontstyle=axis_style)
    ax.set_ylabel(ylabel, fontstyle=axis_style)
    ax.set_zlabel(zlabel, fontstyle=axis_style)

    fig.tight_layout()
    return fig


def create_contour_plot(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "f(x, y)",
    xlabel: str = "x",
    ylabel: str = "y",
) -> Figure:
    """Create a 2D contour plot for 2D scalar field data.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        z: 2D array of values, shape (len(y), len(x)).
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.

    Returns:
        A matplotlib :class:`Figure`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = _new_figure()

    X, Y = np.meshgrid(x, y)
    if z.shape != X.shape:
        z = np.asarray(z)
        if z.shape != X.shape:
            raise ValueError(f"z shape {z.shape} does not match grid {X.shape}")

    contour = ax.contourf(X, Y, z, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax)

    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    axis_style: str = get_env_from_schema("FONT_AXIS_STYLE")
    ax.set_xlabel(xlabel, fontstyle=axis_style)
    ax.set_ylabel(ylabel, fontstyle=axis_style)
    if get_env_from_schema("PLOT_SHOW_GRID"):
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def create_vector_animation_plot(
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    vector_components: int,
    title: str = "f_i(x) vs component",
) -> Figure:
    """Create an interactive plot: x-axis = component index i, y-axis = f_i(x).

    The figure stores _animation_update(idx) and _animation_n_points for use
    with a Tkinter Scale (matplotlib Slider is unreliable when embedded in Tk).
    For vector ODE: y has shape (n_state, n_points), with f_i at y[i*order].

    Args:
        x: Independent variable values (1D).
        y: Solution array, shape (n_state, n_points).
        order: Order per component.
        vector_components: Number of components.
        title: Plot title.

    Returns:
        A matplotlib Figure (use with embed_animation_plot_in_tk).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax_main = fig.add_axes([0.12, 0.15, 0.78, 0.78])

    y_2d = np.atleast_2d(y)
    if y_2d.shape[1] != len(x):
        y_2d = y_2d.T if y_2d.shape[0] == len(x) else y_2d

    f_values = np.array([y_2d[i * order] for i in range(vector_components)])
    n_points = len(x)
    time_index = min(n_points // 2, n_points - 1) if n_points else 0

    color_scheme: str = get_env_from_schema("PLOT_COLOR_SCHEME")
    try:
        cmap = plt.colormaps.get_cmap(color_scheme)
        colors = [cmap(i / max(1, vector_components - 1)) for i in range(vector_components)]
    except (ValueError, AttributeError):
        colors = list(plt.colormaps.get_cmap("Set1")(np.linspace(0, 1, max(1, vector_components))))

    y_min_global = float(np.min(f_values)) - 0.1
    y_max_global = float(np.max(f_values)) + 0.1
    ax_main.set_ylim(y_min_global, y_max_global)

    indices = np.arange(vector_components)
    vals = np.array([f_values[j][time_index] for j in range(vector_components)])
    (line_chain,) = ax_main.plot(
        indices, vals, "o-", color=colors[0], markersize=8, linewidth=2,
    )
    vlines_coll = ax_main.vlines(
        indices, 0, vals, colors=colors, linewidth=1.5, alpha=0.6
    )
    ax_main.set_xticks(indices)
    ax_main.set_xticklabels([f"f_{i}" for i in range(vector_components)])

    def update(idx: int) -> None:
        i = max(0, min(idx, n_points - 1))
        new_vals = np.array([f_values[j][i] for j in range(vector_components)])
        line_chain.set_ydata(new_vals)
        vlines_coll.set_segments(
            [np.array([[j, 0], [j, new_vals[j]]]) for j in range(vector_components)]
        )
        fig.canvas.draw_idle()

    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax_main.set_title(title)
    axis_style: str = get_env_from_schema("FONT_AXIS_STYLE")
    ax_main.set_xlabel("Component index i", fontstyle=axis_style)
    ax_main.set_ylabel("f_i(x)", fontstyle=axis_style)
    if get_env_from_schema("PLOT_SHOW_GRID"):
        ax_main.grid(True, alpha=0.3)
    fig.subplots_adjust(bottom=0.12, left=0.12, right=0.95, top=0.92)

    fig._animation_update = update
    fig._animation_n_points = n_points
    fig._animation_initial_index = time_index
    fig._animation_x = x
    fig._animation_f_values = f_values
    fig._animation_vector_components = vector_components
    return fig


def create_vector_animation_3d(
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    vector_components: int,
    title: str = "f_i(x) — 3D",
) -> Figure:
    """Create a 3D plot: x (independent), component index i, f_i(x).

    Args:
        x: Independent variable values.
        y: Solution array, shape (n_state, n_points).
        order: Order per component.
        vector_components: Number of components.
        title: Plot title.

    Returns:
        A matplotlib Figure with 3D surface.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _apply_plot_style()
    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    y_2d = np.atleast_2d(y)
    if y_2d.shape[1] != len(x):
        y_2d = y_2d.T if y_2d.shape[0] == len(x) else y_2d

    X_grid, I_grid = np.meshgrid(x, np.arange(vector_components))
    Z_grid = np.array([y_2d[i * order] for i in range(vector_components)])
    ax.plot_surface(X_grid, I_grid, Z_grid, cmap="viridis", alpha=0.9, edgecolor="none")

    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    axis_style: str = get_env_from_schema("FONT_AXIS_STYLE")
    ax.set_xlabel("x", fontstyle=axis_style)
    ax.set_ylabel("Component i", fontstyle=axis_style)
    ax.set_zlabel("f_i(x)", fontstyle=axis_style)
    fig.tight_layout()
    return fig


_MAX_MP4_FRAMES = 500


def export_animation_to_mp4(
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    vector_components: int,
    filepath: Path,
    *,
    title: str = "f_i(x) vs component",
    duration_seconds: float = 10.0,
) -> Path:
    """Export vector animation as MP4 video.

    Frames are downsampled to at most _MAX_MP4_FRAMES to avoid memory exhaustion.
    Requires ffmpeg to be installed on the system.

    Args:
        x: Independent variable values.
        y: Solution array, shape (n_state, n_points).
        order: Order per component.
        vector_components: Number of components.
        filepath: Output path for the MP4 file.
        title: Plot title.
        duration_seconds: Desired video duration in seconds. FPS is computed.

    Returns:
        The path that was written.

    Raises:
        RuntimeError: If ffmpeg is not available.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    _apply_plot_style()
    dpi: int = get_env_from_schema("DPI")

    y_2d = np.atleast_2d(y)
    if y_2d.shape[1] != len(x):
        y_2d = y_2d.T if y_2d.shape[0] == len(x) else y_2d

    f_values = np.array([y_2d[i * order] for i in range(vector_components)])
    n_points = len(x)

    frame_indices = np.linspace(0, n_points - 1, min(n_points, _MAX_MP4_FRAMES), dtype=int)
    num_frames = len(frame_indices)
    fps = max(1, num_frames / max(0.5, duration_seconds))

    color_scheme: str = get_env_from_schema("PLOT_COLOR_SCHEME")
    try:
        cmap = plt.colormaps.get_cmap(color_scheme)
        colors = [cmap(i / max(1, vector_components - 1)) for i in range(vector_components)]
    except (ValueError, AttributeError):
        colors = list(plt.colormaps.get_cmap("Set1")(np.linspace(0, 1, max(1, vector_components))))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    y_min = float(np.min(f_values)) - 0.1
    y_max = float(np.max(f_values)) + 0.1
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Component index i")
    ax.set_ylabel("f_i(x)")
    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    if get_env_from_schema("PLOT_SHOW_GRID"):
        ax.grid(True, alpha=0.3)

    indices = np.arange(vector_components)
    vals = np.array([f_values[j][frame_indices[0]] for j in range(vector_components)])
    (line_chain,) = ax.plot(indices, vals, "o-", color=colors[0], markersize=8, linewidth=2)
    vlines_coll = ax.vlines(indices, 0, vals, colors=colors, linewidth=1.5, alpha=0.6)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"f_{i}" for i in range(vector_components)])

    def _frame(idx: int) -> None:
        new_vals = np.array([f_values[j][idx] for j in range(vector_components)])
        line_chain.set_ydata(new_vals)
        vlines_coll.set_segments(
            [np.array([[j, 0], [j, new_vals[j]]]) for j in range(vector_components)]
        )

    anim = FuncAnimation(
        fig,
        lambda i: _frame(frame_indices[i]),
        frames=num_frames,
        interval=int(1000 / fps),
        blit=False,
    )

    from matplotlib.animation import writers

    if not writers.is_available("ffmpeg"):
        plt.close(fig)
        raise RuntimeError(
            "FFMpeg is not available. Install ffmpeg and ensure it is in your PATH."
        )

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(filepath), writer="ffmpeg", fps=fps)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            f"Failed to export MP4. Is ffmpeg installed? {exc}"
        ) from exc
    plt.close(fig)
    logger.info("Animation exported: %s (%d frames)", filepath, len(frame_indices))
    return filepath


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


