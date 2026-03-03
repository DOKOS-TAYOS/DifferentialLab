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
_SUB_DIGS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"


def _get_colors(color_scheme: str, n: int) -> list:
    """Get a list of n colors from the specified colormap with fallback.

    Args:
        color_scheme: Matplotlib colormap name (e.g., 'Set1', 'tab10').
        n: Number of colors to generate.

    Returns:
        List of matplotlib color objects.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        cmap = plt.colormaps.get_cmap(color_scheme)
        return [cmap(i / max(1, n - 1)) for i in range(n)]
    except (ValueError, AttributeError):
        logger.debug("Colormap '%s' invalid, using Set1 fallback", color_scheme)
        cmap_fallback = plt.colormaps.get_cmap("Set1")
        return list(cmap_fallback(np.linspace(0, 1, max(1, n))))


def _apply_plot_style() -> None:
    """Configure matplotlib rcParams from environment variables."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": get_env_from_schema("FONT_FAMILY"),
            "font.size": get_env_from_schema("FONT_TICK_SIZE"),
            "axes.titlesize": get_env_from_schema("FONT_TITLE_SIZE"),
            "axes.titleweight": get_env_from_schema("FONT_TITLE_WEIGHT"),
            "axes.labelsize": get_env_from_schema("FONT_AXIS_SIZE"),
            "figure.dpi": get_env_from_schema("DPI"),
        }
    )


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
        grid_alpha: float = get_env_from_schema("PLOT_GRID_ALPHA")
        ax.grid(True, alpha=grid_alpha)
    if legend:
        ax.legend()


def _new_3d_figure() -> tuple[Any, Any]:
    """Create a configured 3D figure and axes from env settings.

    Returns:
        Tuple of ``(fig, ax)`` with 3D projection.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _apply_plot_style()
    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def _finalize_3d_plot(
    ax: Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    *,
    legend: bool = False,
) -> None:
    """Apply title, axis labels, and optional legend for 3D plots."""
    axis_style: str = get_env_from_schema("FONT_AXIS_STYLE")
    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    ax.set_xlabel(xlabel, fontstyle=axis_style)
    ax.set_ylabel(ylabel, fontstyle=axis_style)
    ax.set_zlabel(zlabel, fontstyle=axis_style)
    if legend:
        ax.legend()


def _component_labels(n: int) -> list[str]:
    """Generate f₀, f₁, ... labels for component indices."""
    return [
        f"f{_SUB_DIGS[i]}" if i < len(_SUB_DIGS) else f"f_{i}"
        for i in range(n)
    ]


def create_solution_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "f(x)",
    xlabel: str = "x",
    ylabel: str = "f",
    show_markers: bool = False,
    selected_derivatives: list[int] | None = None,
    labels: list[str] | None = None,
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
        labels: Custom legend labels for each derivative (f-notation).

    Returns:
        A matplotlib :class:`Figure`.
    """
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

    if labels is None:
        labels = ["f"] if y_2d.shape[0] == 1 else [f"f[{i}]" for i in range(y_2d.shape[0])]

    n_colors = max(1, len(selected_derivatives) - 1)
    colors = [line_color] + _get_colors(color_scheme, n_colors)

    for plot_idx, deriv_idx in enumerate(selected_derivatives):
        if deriv_idx >= y_2d.shape[0]:
            continue
        color = colors[plot_idx] if plot_idx < len(colors) else None
        ax.plot(
            x,
            y_2d[deriv_idx],
            color=color,
            linewidth=line_width,
            linestyle=line_style,
            label=labels[deriv_idx],
        )

    if show_markers:
        marker: str = get_env_from_schema("PLOT_MARKER_FORMAT")
        msize: int = get_env_from_schema("PLOT_MARKER_SIZE")
        mfc: str = get_env_from_schema("PLOT_MARKER_FACE_COLOR")
        mec: str = get_env_from_schema("PLOT_MARKER_EDGE_COLOR")
        step = max(1, len(x) // _MAX_ELEMENTS_PLOT)
        for plot_idx, deriv_idx in enumerate(selected_derivatives):
            if deriv_idx >= y_2d.shape[0]:
                continue
            ax.plot(
                x[::step],
                y_2d[deriv_idx, ::step],
                marker=marker,
                markersize=msize,
                markerfacecolor=mfc,
                markeredgecolor=mec,
                linestyle="none",
            )

    _finalize_plot(ax, title, xlabel, ylabel, legend=len(selected_derivatives) > 1)
    fig.tight_layout()
    return fig


def create_phase_plot(
    y: np.ndarray,
    title: str = "Phase Portrait",
    xlabel: str = "f",
    ylabel: str = "f'",
    x: np.ndarray | None = None,
) -> Figure:
    """Create a phase portrait for an ODE.

    For second-order (or higher): plots y vs y' (position vs velocity).
    For first-order: plots y vs dy/dx using numerical derivative (requires x).

    Args:
        y: Solution array — shape ``(n_vars, n_points)``.
        title: Plot title.
        xlabel: Label for horizontal axis.
        ylabel: Label for vertical axis.
        x: Independent variable (required for first-order to compute dy/dx).

    Returns:
        A matplotlib :class:`Figure`.
    """
    import numpy as np

    fig, ax = _new_figure()

    line_color: str = get_env_from_schema("PLOT_LINE_COLOR")
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")

    y_2d = np.atleast_2d(y)
    if y_2d.shape[0] >= 2:
        ax.plot(y_2d[0], y_2d[1], color=line_color, linewidth=line_width)
        horiz, vert = y_2d[0], y_2d[1]
    else:
        if x is None:
            raise ValueError("x is required for first-order phase portrait")
        horiz = y_2d[0]
        vert = np.gradient(y_2d[0], x)
        ax.plot(horiz, vert, color=line_color, linewidth=line_width)

    phase_start_color: str = get_env_from_schema("PLOT_PHASE_START_COLOR")
    phase_end_color: str = get_env_from_schema("PLOT_PHASE_END_COLOR")
    phase_marker_size: int = get_env_from_schema("PLOT_PHASE_MARKER_SIZE")
    ax.plot(
        horiz[0], vert[0], "o",
        color=phase_start_color, markersize=phase_marker_size, label="Start",
    )
    ax.plot(
        horiz[-1], vert[-1], "s",
        color=phase_end_color, markersize=phase_marker_size, label="End",
    )

    _finalize_plot(ax, title, xlabel, ylabel, legend=True)
    fig.tight_layout()
    return fig


def create_phase_3d_plot(
    data_x: np.ndarray,
    data_y: np.ndarray,
    data_z: np.ndarray,
    title: str = "Phase Space 3D",
    xlabel: str = "f\u2080",
    ylabel: str = "f\u2081",
    zlabel: str = "f\u2082",
) -> Figure:
    """Create a 3D phase-space trajectory plot.

    Args:
        data_x: Values for the x-axis.
        data_y: Values for the y-axis.
        data_z: Values for the z-axis.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        zlabel: Label for z-axis.

    Returns:
        A matplotlib :class:`Figure`.
    """
    fig, ax = _new_3d_figure()
    line_color: str = get_env_from_schema("PLOT_LINE_COLOR")
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")

    phase_start_color: str = get_env_from_schema("PLOT_PHASE_START_COLOR")
    phase_end_color: str = get_env_from_schema("PLOT_PHASE_END_COLOR")
    phase_marker_size: int = get_env_from_schema("PLOT_PHASE_MARKER_SIZE")
    ax.plot(data_x, data_y, data_z, color=line_color, linewidth=line_width)
    ax.plot(
        [data_x[0]], [data_y[0]], [data_z[0]], "o",
        color=phase_start_color, markersize=phase_marker_size, label="Start",
    )
    ax.plot(
        [data_x[-1]], [data_y[-1]], [data_z[-1]], "s",
        color=phase_end_color, markersize=phase_marker_size, label="End",
    )
    _finalize_3d_plot(ax, title, xlabel, ylabel, zlabel, legend=True)
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
    import numpy as np

    fig, ax = _new_3d_figure()
    X, Y = np.meshgrid(x, y)
    if z.shape != X.shape:
        z = np.asarray(z)
        if z.shape != X.shape:
            raise ValueError(f"z shape {z.shape} does not match grid {X.shape}")

    surface_cmap: str = get_env_from_schema("PLOT_SURFACE_CMAP")
    surface_alpha: float = get_env_from_schema("PLOT_SURFACE_ALPHA")
    colorbar_shrink: float = get_env_from_schema("PLOT_COLORBAR_SHRINK")
    surf = ax.plot_surface(
        X,
        Y,
        z,
        cmap=surface_cmap,
        alpha=surface_alpha,
        edgecolor="none",
    )
    fig.colorbar(surf, ax=ax, shrink=colorbar_shrink)
    _finalize_3d_plot(ax, title, xlabel, ylabel, zlabel)
    fig.tight_layout()
    return fig


def create_contour_plot(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "f(x, y)",
    xlabel: str = "x",
    ylabel: str = "f",
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
    import numpy as np

    fig, ax = _new_figure()

    X, Y = np.meshgrid(x, y)
    if z.shape != X.shape:
        z = np.asarray(z)
        if z.shape != X.shape:
            raise ValueError(f"z shape {z.shape} does not match grid {X.shape}")

    contour_levels: int = get_env_from_schema("PLOT_CONTOUR_LEVELS")
    surface_cmap: str = get_env_from_schema("PLOT_SURFACE_CMAP")
    contour = ax.contourf(X, Y, z, levels=contour_levels, cmap=surface_cmap)
    fig.colorbar(contour, ax=ax)
    _finalize_plot(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def create_vector_animation_plot(
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    vector_components: int,
    title: str = "f_i(x) vs component",
    deriv_offset: int = 0,
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
        deriv_offset: Derivative order to display (0=value, 1=first derivative, etc.).

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

    row_indices = np.arange(vector_components) * order + deriv_offset
    f_values = y_2d[row_indices]
    n_points = len(x)
    time_index = 0

    color_scheme: str = get_env_from_schema("PLOT_COLOR_SCHEME")
    colors = _get_colors(color_scheme, vector_components)
    marker_size: int = get_env_from_schema("PLOT_PHASE_MARKER_SIZE")
    anim_line_width: float = get_env_from_schema("PLOT_ANIMATION_LINE_WIDTH")
    vlines_line_width: float = get_env_from_schema("PLOT_VLINES_LINE_WIDTH")
    vlines_alpha: float = get_env_from_schema("PLOT_VLINES_ALPHA")
    y_margin: float = get_env_from_schema("PLOT_ANIMATION_Y_MARGIN")

    y_min_global = float(np.min(f_values)) - y_margin
    y_max_global = float(np.max(f_values)) + y_margin
    ax_main.set_ylim(y_min_global, y_max_global)

    indices = np.arange(vector_components)
    vals = f_values[:, time_index]
    (line_chain,) = ax_main.plot(
        indices,
        vals,
        "o-",
        color=colors[0],
        markersize=marker_size,
        linewidth=anim_line_width,
    )
    vlines_coll = ax_main.vlines(
        indices, 0, vals, colors=colors, linewidth=vlines_line_width, alpha=vlines_alpha
    )
    ax_main.set_xticks(indices)
    ax_main.set_xticklabels(_component_labels(vector_components))
    j_vals = indices  # Reuse for segment construction in update

    def update(idx: int) -> None:
        i = max(0, min(idx, n_points - 1))
        new_vals = f_values[:, i]
        line_chain.set_ydata(new_vals)
        segments = np.stack([
            np.column_stack([j_vals, np.zeros(vector_components)]),
            np.column_stack([j_vals, new_vals]),
        ], axis=1)
        vlines_coll.set_segments(segments)
        fig.canvas.draw_idle()

    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax_main.set_title(title)
    axis_style: str = get_env_from_schema("FONT_AXIS_STYLE")
    ax_main.set_xlabel("Component index i", fontstyle=axis_style)
    ax_main.set_ylabel("f_i(x)", fontstyle=axis_style)
    if get_env_from_schema("PLOT_SHOW_GRID"):
        grid_alpha: float = get_env_from_schema("PLOT_GRID_ALPHA")
        ax_main.grid(True, alpha=grid_alpha)
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
    deriv_offset: int = 0,
) -> Figure:
    """Create a 3D plot: x (independent), component index i, f_i(x).

    Args:
        x: Independent variable values.
        y: Solution array, shape (n_state, n_points).
        order: Order per component.
        vector_components: Number of components.
        title: Plot title.
        deriv_offset: Derivative order to display (0=value, 1=first derivative, etc.).

    Returns:
        A matplotlib Figure with 3D surface.
    """
    import numpy as np

    fig, ax = _new_3d_figure()
    y_2d = np.atleast_2d(y)
    if y_2d.shape[1] != len(x):
        y_2d = y_2d.T if y_2d.shape[0] == len(x) else y_2d

    surface_cmap: str = get_env_from_schema("PLOT_SURFACE_CMAP")
    surface_alpha: float = get_env_from_schema("PLOT_SURFACE_ALPHA")
    row_indices = np.arange(vector_components) * order + deriv_offset
    Z_grid = y_2d[row_indices]
    X_grid, I_grid = np.meshgrid(x, np.arange(vector_components))
    ax.plot_surface(
        X_grid, I_grid, Z_grid, cmap=surface_cmap, alpha=surface_alpha, edgecolor="none"
    )
    _finalize_3d_plot(ax, title, "x", "Component i", "f_i(x)")
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
    deriv_offset: int = 0,
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

    row_indices = np.arange(vector_components) * order + deriv_offset
    f_values = y_2d[row_indices]
    n_points = len(x)

    frame_indices = np.linspace(0, n_points - 1, min(n_points, _MAX_MP4_FRAMES), dtype=int)
    num_frames = len(frame_indices)
    fps = max(1, num_frames / max(0.5, duration_seconds))

    color_scheme: str = get_env_from_schema("PLOT_COLOR_SCHEME")
    colors = _get_colors(color_scheme, vector_components)
    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    marker_size: int = get_env_from_schema("PLOT_PHASE_MARKER_SIZE")
    anim_line_width: float = get_env_from_schema("PLOT_ANIMATION_LINE_WIDTH")
    vlines_line_width: float = get_env_from_schema("PLOT_VLINES_LINE_WIDTH")
    vlines_alpha: float = get_env_from_schema("PLOT_VLINES_ALPHA")
    y_margin: float = get_env_from_schema("PLOT_ANIMATION_Y_MARGIN")

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    y_min = float(np.min(f_values)) - y_margin
    y_max = float(np.max(f_values)) + y_margin
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Component index i")
    ax.set_ylabel("f_i(x)")
    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    if get_env_from_schema("PLOT_SHOW_GRID"):
        grid_alpha: float = get_env_from_schema("PLOT_GRID_ALPHA")
        ax.grid(True, alpha=grid_alpha)

    indices = np.arange(vector_components)
    vals = f_values[:, frame_indices[0]]
    (line_chain,) = ax.plot(
        indices, vals, "o-",
        color=colors[0], markersize=marker_size, linewidth=anim_line_width,
    )
    vlines_coll = ax.vlines(
        indices, 0, vals, colors=colors, linewidth=vlines_line_width, alpha=vlines_alpha
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(_component_labels(vector_components))
    j_vals = indices

    def _frame(idx: int) -> None:
        new_vals = f_values[:, idx]
        line_chain.set_ydata(new_vals)
        segments = np.stack([
            np.column_stack([j_vals, np.zeros(vector_components)]),
            np.column_stack([j_vals, new_vals]),
        ], axis=1)
        vlines_coll.set_segments(segments)

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
        raise RuntimeError("FFMpeg is not available. Install ffmpeg and ensure it is in your PATH.")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(filepath), writer="ffmpeg", fps=fps)
    except Exception as exc:
        logger.error("MP4 export failed: %s", exc, exc_info=True)
        plt.close(fig)
        raise RuntimeError(f"Failed to export MP4. Is ffmpeg installed? {exc}") from exc
    plt.close(fig)
    logger.info("Animation exported: %s (%d frames)", filepath, len(frame_indices))
    return filepath
