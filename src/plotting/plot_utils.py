"""Matplotlib plotting utilities for ODE solutions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D

from config import get_env_from_schema
from plotting.animation_metadata import attach_animation_metadata
from utils import get_logger

logger = get_logger(__name__)

_MAX_ELEMENTS_PLOT = 50
_MAX_COMPONENT_TICKS = 10
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
    ax: Axes3D,
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
    return [f"f{_SUB_DIGS[i]}" if i < len(_SUB_DIGS) else f"f_{i}" for i in range(n)]


def _set_component_ticks(
    ax: "Axes",
    vector_components: int,
    component_labels: list[str] | None = None,
) -> None:
    """Apply readable, range-preserving ticks for a component animation."""
    labels = (
        component_labels if component_labels is not None else _component_labels(vector_components)
    )
    if vector_components <= _MAX_COMPONENT_TICKS:
        tick_indices = list(range(vector_components))
    else:
        tick_indices = [
            round(index * (vector_components - 1) / (_MAX_COMPONENT_TICKS - 1))
            for index in range(_MAX_COMPONENT_TICKS)
        ]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels([labels[index] for index in tick_indices])


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


def create_energy_evolution_plot(
    t: np.ndarray,
    E_kin: np.ndarray,
    E_pot: np.ndarray,
    E_tot: np.ndarray,
    title: str = "Energy vs time",
    xlabel: str = "t",
) -> Figure:
    """Create a plot of kinetic, potential, and total energy vs time.

    Args:
        t: Time values (1D).
        E_kin: Kinetic energy at each time step.
        E_pot: Potential energy at each time step.
        E_tot: Total energy at each time step.
        title: Plot title.
        xlabel: Label for x-axis.

    Returns:
        A matplotlib :class:`Figure`.
    """

    fig, ax = _new_figure()
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")
    line_style: str = get_env_from_schema("PLOT_LINE_STYLE")

    ax.plot(t, E_kin, linewidth=line_width, linestyle=line_style, label="Kinetic")
    ax.plot(t, E_pot, linewidth=line_width, linestyle=line_style, label="Potential")
    ax.plot(t, E_tot, linewidth=line_width, linestyle=line_style, label="Total")

    _finalize_plot(ax, title, xlabel, "Energy", legend=True)
    fig.tight_layout()
    return fig


def create_energy_per_mode_plot(
    t: np.ndarray,
    E_modes: np.ndarray,
    selected_indices: list[int],
    labels: list[str],
    title: str = "Energy per mode",
    xlabel: str = "t",
) -> Figure:
    """Create a multi-line plot of energy per mode (or oscillator) vs time.

    Args:
        t: Time values (1D).
        E_modes: Energy array shape (n_modes, n_points).
        selected_indices: Indices of modes/oscillators to plot.
        labels: Legend labels for each selected index.
        title: Plot title.
        xlabel: Label for x-axis.

    Returns:
        A matplotlib :class:`Figure`.
    """

    fig, ax = _new_figure()
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")
    line_style: str = get_env_from_schema("PLOT_LINE_STYLE")
    color_scheme: str = get_env_from_schema("PLOT_COLOR_SCHEME")

    colors = _get_colors(color_scheme, len(selected_indices))
    for idx, (mode_idx, lbl) in enumerate(zip(selected_indices, labels)):
        if mode_idx < E_modes.shape[0]:
            ax.plot(
                t,
                E_modes[mode_idx],
                color=colors[idx],
                linewidth=line_width,
                linestyle=line_style,
                label=lbl,
            )

    _finalize_plot(ax, title, xlabel, "Energy", legend=True)
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
        horiz[0],
        vert[0],
        "o",
        color=phase_start_color,
        markersize=phase_marker_size,
        label="Start",
    )
    ax.plot(
        horiz[-1],
        vert[-1],
        "s",
        color=phase_end_color,
        markersize=phase_marker_size,
        label="End",
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
        [data_x[0]],
        [data_y[0]],
        [data_z[0]],
        "o",
        color=phase_start_color,
        markersize=phase_marker_size,
        label="Start",
    )
    ax.plot(
        [data_x[-1]],
        [data_y[-1]],
        [data_z[-1]],
        "s",
        color=phase_end_color,
        markersize=phase_marker_size,
        label="End",
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

    # Mask NaN values (e.g. exterior points in non-rectangular PDE domains)
    z_masked = np.ma.masked_invalid(z)

    surface_cmap: str = get_env_from_schema("PLOT_SURFACE_CMAP")
    surface_alpha: float = get_env_from_schema("PLOT_SURFACE_ALPHA")
    colorbar_shrink: float = get_env_from_schema("PLOT_COLORBAR_SHRINK")
    surf = ax.plot_surface(
        X,
        Y,
        z_masked,
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

    # Mask NaN values (e.g. exterior points in non-rectangular PDE domains)
    z_masked = np.ma.masked_invalid(z)

    contour_levels: int = get_env_from_schema("PLOT_CONTOUR_LEVELS")
    surface_cmap: str = get_env_from_schema("PLOT_SURFACE_CMAP")
    contour = ax.contourf(X, Y, z_masked, levels=contour_levels, cmap=surface_cmap)
    fig.colorbar(contour, ax=ax)
    _finalize_plot(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def create_polar_contour_plot(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "f(r, θ)",
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    radial_points: int = 160,
    angular_points: int = 180,
) -> Figure:
    """Create a polar re-sampled contour plot from Cartesian scalar data."""
    import matplotlib.pyplot as plt
    import numpy as np

    from plotting.coordinates import resample_scalar_to_polar

    _apply_plot_style()
    width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
    height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
    dpi: int = get_env_from_schema("DPI")
    sampled = resample_scalar_to_polar(
        x,
        y,
        z,
        origin=origin,
        radial_points=radial_points,
        angular_points=angular_points,
    )
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi, subplot_kw={"projection": "polar"})
    angle_grid, radius_grid = np.meshgrid(sampled.angle, sampled.radius, indexing="ij")
    mesh = ax.pcolormesh(
        angle_grid,
        radius_grid,
        np.ma.masked_invalid(sampled.values),
        cmap=get_env_from_schema("PLOT_SURFACE_CMAP"),
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, shrink=get_env_from_schema("PLOT_COLORBAR_SHRINK"))
    if get_env_from_schema("PLOT_SHOW_TITLE") and title:
        ax.set_title(title)
    ax.set_xlabel("azimuth θ (rad)")
    ax.set_ylabel("radius r")
    fig.tight_layout()
    return fig


def create_vector_field_plot(
    x: np.ndarray,
    y: np.ndarray,
    components: np.ndarray,
    *,
    view: str = "magnitude",
    origin: tuple[float, float] = (0.0, 0.0),
    title: str = "Vector field",
) -> Figure:
    """Create component, magnitude, quiver, stream, or radial field views.

    Quiver, stream, and radial/tangential views require exactly two components;
    component and magnitude views support any positive component count.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from plotting.coordinates import cartesian_vector_to_polar, vector_field_data

    field = np.asarray(components, dtype=float)
    if field.ndim != 3 or field.shape[1:] != (len(y), len(x)) or field.shape[0] < 1:
        raise ValueError("components must have shape (components, len(y), len(x))")
    _apply_plot_style()
    cmap = get_env_from_schema("PLOT_SURFACE_CMAP")
    if view == "components":
        fig, axes = plt.subplots(1, field.shape[0], squeeze=False)
        for index, axis in enumerate(axes[0]):
            image = axis.pcolormesh(
                x, y, np.ma.masked_invalid(field[index]), cmap=cmap, shading="auto"
            )
            fig.colorbar(image, ax=axis)
            _finalize_plot(axis, f"f[{index}]", "x", "y")
        fig.suptitle(title)
        fig.tight_layout()
        return fig
    if view == "magnitude":
        magnitude = np.linalg.norm(field, axis=0)
        fig, ax = _new_figure()
        image = ax.pcolormesh(x, y, np.ma.masked_invalid(magnitude), cmap=cmap, shading="auto")
        fig.colorbar(image, ax=ax)
        _finalize_plot(ax, title, "x", "y")
        fig.tight_layout()
        return fig

    if field.shape[0] != 2:
        raise ValueError(f"{view} view requires exactly two vector components")

    data = vector_field_data(x, y, field)
    x_grid, y_grid = np.meshgrid(data.x, data.y)
    fig, ax = _new_figure()
    if view == "quiver":
        stride = max(1, int(np.ceil(max(len(x), len(y)) / 25)))
        ax.quiver(
            x_grid[::stride, ::stride],
            y_grid[::stride, ::stride],
            data.u[::stride, ::stride],
            data.v[::stride, ::stride],
            data.magnitude[::stride, ::stride],
            cmap=cmap,
        )
    elif view == "stream":
        ax.streamplot(data.x, data.y, data.u, data.v, color=data.magnitude, cmap=cmap)
    elif view == "radial_tangential":
        radial, tangential = cartesian_vector_to_polar(
            data.u, data.v, x_grid - origin[0], y_grid - origin[1]
        )
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, squeeze=False)
        for axis, component, label in zip(axes[0], (radial, tangential), ("radial", "tangential")):
            image = axis.pcolormesh(
                x, y, np.ma.masked_invalid(component), cmap=cmap, shading="auto"
            )
            fig.colorbar(image, ax=axis, label=f"{label} component")
            axis.plot(origin[0], origin[1], "ko", markersize=3)
            _finalize_plot(axis, label.capitalize(), "x", "y")
        fig.suptitle(title)
        fig.tight_layout()
        return fig
    else:
        raise ValueError("view must be components, magnitude, quiver, stream, or radial_tangential")
    _finalize_plot(ax, title, "x", "y")
    fig.tight_layout()
    return fig


def create_vector_animation_plot(
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    vector_components: int,
    title: str = "f_i(x) vs component",
    deriv_offset: int = 0,
    component_labels: list[str] | None = None,
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
    ax_main = fig.add_axes((0.12, 0.15, 0.78, 0.78))

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
    _set_component_ticks(ax_main, vector_components, component_labels)
    j_vals = indices  # Reuse for segment construction in update

    def update(idx: int) -> None:
        i = max(0, min(idx, n_points - 1))
        new_vals = f_values[:, i]
        line_chain.set_ydata(new_vals)
        segments = [
            np.array([[float(j), 0.0], [float(j), float(val)]]) for j, val in zip(j_vals, new_vals)
        ]
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

    return attach_animation_metadata(
        fig,
        update=update,
        n_points=n_points,
        initial_index=time_index,
        extras={
            "_animation_x": x,
            "_animation_f_values": f_values,
            "_animation_vector_components": vector_components,
        },
    )


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


def create_line_animation_plot(
    frame_coordinates: np.ndarray,
    x: np.ndarray,
    frames: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    frame_label: str = "t",
    symmetric_y_range: bool = False,
) -> Figure:
    """Create a stable-scale animation of one-dimensional frames."""
    import numpy as np

    coordinates = np.asarray(frame_coordinates, dtype=float)
    x_values = np.asarray(x, dtype=float)
    frame_data = np.asarray(frames, dtype=float)
    if coordinates.ndim != 1 or x_values.ndim != 1:
        raise ValueError("frame_coordinates and x must be one-dimensional")
    if frame_data.shape != (len(coordinates), len(x_values)):
        raise ValueError("frames must have shape (len(frame_coordinates), len(x))")
    if len(coordinates) == 0:
        raise ValueError("frame_coordinates must not be empty")

    finite_values = frame_data[np.isfinite(frame_data)]
    if finite_values.size == 0:
        data_min = data_max = 0.0
    else:
        data_min = float(np.min(finite_values))
        data_max = float(np.max(finite_values))
    if symmetric_y_range:
        bound = max(abs(data_min), abs(data_max), 1.0 if data_min == data_max == 0.0 else 0.0)
        y_min, y_max = -bound, bound
    elif data_min == data_max:
        margin = max(abs(data_min) * 0.05, 1.0)
        y_min, y_max = data_min - margin, data_max + margin
    else:
        y_min, y_max = data_min, data_max

    fig, ax = _new_figure()
    line_color: str = get_env_from_schema("PLOT_LINE_COLOR")
    line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")
    line_style: str = get_env_from_schema("PLOT_LINE_STYLE")
    (line,) = ax.plot(
        x_values,
        frame_data[0],
        color=line_color,
        linewidth=line_width,
        linestyle=line_style,
    )
    ax.set_ylim(y_min, y_max)
    _finalize_plot(ax, f"{title} ({frame_label}={coordinates[0]:.3g})", xlabel, ylabel)
    fig.tight_layout()

    def update(index: int) -> None:
        """Draw one bounded animation frame."""
        idx = max(0, min(index, len(coordinates) - 1))
        line.set_ydata(frame_data[idx])
        if get_env_from_schema("PLOT_SHOW_TITLE"):
            ax.set_title(f"{title} ({frame_label}={coordinates[idx]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(
        fig,
        update=update,
        n_points=len(coordinates),
        frame_label=frame_label,
        frame_coordinates=coordinates,
        extras={"_animation_x": x_values, "_animation_frames": frame_data},
    )


def create_image_animation_plot(
    t: np.ndarray,
    frames: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis",
    x_coordinates: np.ndarray | None = None,
    y_coordinates: np.ndarray | None = None,
    symmetric_color_range: bool = False,
    frame_label: str = "t",
) -> Figure:
    """Create a stable-scale animated 2D image sequence.

    The returned figure uses the shared animation metadata consumed by the Tk
    embedding controls and :func:`export_animated_figure_to_mp4`.
    """
    import numpy as np

    frame_data = np.asarray(frames)
    if frame_data.ndim != 3 or frame_data.shape[0] != len(t):
        raise ValueError("frames must have shape (len(t), ny, nx).")

    ny, nx = frame_data.shape[1:]
    x_values = np.arange(nx) if x_coordinates is None else np.asarray(x_coordinates)
    y_values = np.arange(ny) if y_coordinates is None else np.asarray(y_coordinates)
    if len(x_values) != nx or len(y_values) != ny:
        raise ValueError("Coordinate lengths must match frame dimensions.")

    data_min = float(np.min(frame_data))
    data_max = float(np.max(frame_data))
    if symmetric_color_range:
        bound = max(abs(data_min), abs(data_max))
        if bound == 0.0:
            bound = 1.0
        vmin, vmax = -bound, bound
    else:
        vmin, vmax = data_min, data_max
        if vmin == vmax:
            vmax = vmin + 1.0

    fig, ax = _new_figure()
    image = ax.imshow(
        frame_data[0],
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
    )
    _finalize_plot(ax, f"{title} ({frame_label}={t[0]:.3g})", xlabel, ylabel)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()

    def update(index: int) -> None:
        """Draw one bounded animation frame."""
        idx = max(0, min(index, len(t) - 1))
        image.set_data(frame_data[idx])
        ax.set_title(f"{title} ({frame_label}={t[idx]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(
        fig,
        update=update,
        n_points=len(t),
        frame_label=frame_label,
        frame_coordinates=t,
    )


def create_surface_animation_plot(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    frames: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    max_render_resolution: int = 80,
    xlabel: str = "x index",
    ylabel: str = "y index",
    zlabel: str = "u",
    colorbar_label: str = "Displacement u",
    symmetric_z_range: bool = True,
    frame_label: str = "t",
) -> Figure:
    """Create an animated 3D surface with stable axes and colors across time.

    Surface rendering is locally downsampled when needed, while the supplied
    frame history remains unchanged for other result views and exports.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    frame_data = np.asarray(frames)
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if frame_data.ndim != 3 or frame_data.shape[0] != len(t):
        raise ValueError("frames must have shape (len(t), ny, nx).")
    if frame_data.shape[1:] != (len(y_values), len(x_values)):
        raise ValueError("Coordinate lengths must match frame dimensions.")
    if max_render_resolution < 2:
        raise ValueError("max_render_resolution must be at least 2.")

    x_step = max(1, int(np.ceil(len(x_values) / max_render_resolution)))
    y_step = max(1, int(np.ceil(len(y_values) / max_render_resolution)))
    render_x = x_values[::x_step]
    render_y = y_values[::y_step]
    render_frames = frame_data[:, ::y_step, ::x_step]
    x_grid, y_grid = np.meshgrid(render_x, render_y)
    if symmetric_z_range:
        z_bound = float(np.max(np.abs(frame_data)))
        if z_bound == 0.0:
            z_bound = 1.0
        z_min, z_max = -z_bound, z_bound
    else:
        z_min = float(np.min(frame_data))
        z_max = float(np.max(frame_data))
        if z_min == z_max:
            z_max = z_min + 1.0

    fig, ax = _new_3d_figure()
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        render_frames[0],
        cmap=cmap,
        vmin=z_min,
        vmax=z_max,
        edgecolor="none",
    )
    color_map = plt.cm.ScalarMappable(norm=Normalize(z_min, z_max), cmap=cmap)
    color_map.set_array([])
    fig.colorbar(color_map, ax=ax, shrink=0.7, label=colorbar_label)
    ax.set_zlim(z_min, z_max)
    _finalize_3d_plot(ax, f"{title} ({frame_label}={t[0]:.3g})", xlabel, ylabel, zlabel)
    fig.tight_layout()

    def update(index: int) -> None:
        """Replace the surface artist with the requested time frame."""
        nonlocal surface
        idx = max(0, min(index, len(t) - 1))
        surface.remove()
        surface = ax.plot_surface(
            x_grid,
            y_grid,
            render_frames[idx],
            cmap=cmap,
            vmin=z_min,
            vmax=z_max,
            edgecolor="none",
        )
        ax.set_title(f"{title} ({frame_label}={t[idx]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(
        fig,
        update=update,
        n_points=len(t),
        frame_label=frame_label,
        frame_coordinates=t,
    )


def export_animated_figure_to_mp4(
    fig: Figure,
    filepath: Path,
    *,
    duration_seconds: float = 10.0,
) -> Path:
    """Export an animation-metadata figure through the shared MP4 path.

    Callers create the export figure with the same plot factory used for Tk,
    ensuring that display and video use identical data transformations.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, writers

    update = getattr(fig, "_animation_update", None)
    n_points = int(getattr(fig, "_animation_n_points", 0))
    if not callable(update) or n_points < 1:
        plt.close(fig)
        raise ValueError("Figure does not contain valid animation metadata.")

    frame_indices = np.linspace(0, n_points - 1, min(n_points, _MAX_MP4_FRAMES), dtype=int)
    fps = max(1, int(round(len(frame_indices) / max(0.5, duration_seconds))))

    def animate(frame_number: int) -> None:
        """Render one sampled export frame."""
        update(int(frame_indices[frame_number]))

    animation = FuncAnimation(
        fig,
        animate,
        frames=len(frame_indices),
        interval=int(1000 / fps),
        blit=False,
    )
    if not writers.is_available("ffmpeg"):
        plt.close(fig)
        raise RuntimeError("FFMpeg is not available. Install ffmpeg and ensure it is in your PATH.")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(filepath), writer="ffmpeg", fps=fps)
    except Exception as exc:
        logger.error("MP4 export failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Failed to export MP4. Is ffmpeg installed? {exc}") from exc
    finally:
        plt.close(fig)
    logger.info("Animation exported: %s (%d frames)", filepath, len(frame_indices))
    return filepath


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
    component_labels: list[str] | None = None,
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
        component_labels: Optional labels for the displayed components.

    Returns:
        The path that was written.

    Raises:
        RuntimeError: If ffmpeg is not available.
    """
    fig = create_vector_animation_plot(
        x,
        y,
        order=order,
        vector_components=vector_components,
        title=title,
        deriv_offset=deriv_offset,
        component_labels=component_labels,
    )
    return export_animated_figure_to_mp4(fig, filepath, duration_seconds=duration_seconds)
