"""Advanced result views for the 3D aerodynamics solver."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from complex_problems.aerodynamics_3d.model import obstacle_surface_mesh, vorticity_periodic
from complex_problems.aerodynamics_3d.solver import Aerodynamics3DResult
from complex_problems.common.result_dialog_ui import (
    AdvancedResultShell,
    AdvancedResultSize,
    close_embedded_figures,
    format_result_summary,
    make_view_controls,
    reset_embedded_animation,
)
from config import generate_output_basename, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import make_modal
from plotting import (
    create_image_animation_plot,
    create_solution_plot,
    export_animated_figure_to_mp4,
)
from plotting.animation_metadata import attach_animation_metadata
from utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)

_FLOW_DISPLAYS = ("Velocity perturbation", "Velocity vectors", "Vorticity vectors")
_SLICE_FIELDS = ("Vorticity magnitude", "Speed", "Pressure", "u", "v", "w")


def derived_speed(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute speed without retaining a separate history."""
    return np.sqrt(u * u + v * v + w * w)


def derived_vorticity(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, *, dx: float, dy: float, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute vorticity components and magnitude on demand."""
    omega = vorticity_periodic(u, v, w, dx, dy, dz)
    magnitude = np.sqrt(sum(component * component for component in omega))
    return (*omega, magnitude)


def _interpolators(
    result: Aerodynamics3DResult, frame: int
) -> tuple[RegularGridInterpolator, RegularGridInterpolator, RegularGridInterpolator]:
    """Create linear interpolators for one saved velocity frame."""
    axes = (result.z, result.y, result.x)
    return tuple(
        RegularGridInterpolator(
            axes,
            field[frame],
            bounds_error=False,
            fill_value=np.nan,
        )
        for field in (result.u, result.v, result.w)
    )  # type: ignore[return-value]


def trace_streamline_3d(
    seed: tuple[float, float, float],
    *,
    result: Aerodynamics3DResult,
    frame: int = 0,
    step_size: float | None = None,
    max_steps: int = 300,
    speed_epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Trace one streamline with RK4 using a normalized tangent field.

    The integration step is a physical arclength, so multiplying a positive
    velocity field by a constant does not change the streamline geometry.
    """
    if not 0 <= frame < len(result.t):
        raise ValueError("frame is outside the saved result history.")
    if max_steps < 1:
        return np.asarray([seed], dtype=float)
    step = (
        step_size
        or min(
            result.x[1] - result.x[0],
            result.y[1] - result.y[0],
            result.z[1] - result.z[0],
        )
        * 0.45
    )
    if step <= 0:
        raise ValueError("step_size must be positive.")
    interpolators = _interpolators(result, frame)
    lower = np.array([result.x[0], result.y[0], result.z[0]], dtype=float)
    upper = np.array([result.x[-1], result.y[-1], result.z[-1]], dtype=float)

    def tangent(point: np.ndarray) -> np.ndarray | None:
        """Interpolate, validate, and normalize one RK stage."""
        if np.any(point < lower) or np.any(point > upper):
            return None
        values = np.array(
            [interpolator((point[2], point[1], point[0]))[()] for interpolator in interpolators],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return None
        magnitude = float(np.linalg.norm(values))
        if not np.isfinite(magnitude) or magnitude <= speed_epsilon:
            return None
        return values / magnitude

    points = [np.asarray(seed, dtype=float)]
    for _ in range(max_steps):
        point = points[-1]
        if _point_is_obstacle(result, point):
            break
        k1 = tangent(point)
        if k1 is None:
            break
        k2 = tangent(point + 0.5 * step * k1)
        if k2 is None:
            break
        k3 = tangent(point + 0.5 * step * k2)
        if k3 is None:
            break
        k4 = tangent(point + step * k3)
        if k4 is None:
            break
        next_point = point + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if tangent(next_point) is None or _point_is_obstacle(result, next_point):
            break
        points.append(next_point)
    return np.asarray(points, dtype=float)


def _point_is_obstacle(result: Aerodynamics3DResult, point: np.ndarray) -> bool:
    """Check a point against the nearest stored obstacle cell."""
    ix = int(np.argmin(np.abs(result.x - point[0])))
    iy = int(np.argmin(np.abs(result.y - point[1])))
    iz = int(np.argmin(np.abs(result.z - point[2])))
    return bool(result.obstacle_mask[iz, iy, ix])


def make_streamline_seeds(
    result: Aerodynamics3DResult, density: int = 4
) -> list[tuple[float, float, float]]:
    """Seed a modest y-z grid near the upstream x boundary."""
    count = max(2, min(9, int(density)))
    x_seed = float(result.x[0] + 1.5 * (result.x[1] - result.x[0]))
    ys = np.linspace(result.y[0], result.y[-1], count + 2)[1:-1]
    zs = np.linspace(result.z[0], result.z[-1], count + 2)[1:-1]
    return [(x_seed, float(y_value), float(z_value)) for z_value in zs for y_value in ys]


@dataclass(frozen=True)
class StreamlineAnimationCache:
    """Cached streamline polylines for every saved solver frame."""

    seed_density: int
    lines_by_frame: tuple[tuple[np.ndarray, ...], ...]

    @property
    def frames(self) -> tuple[tuple[np.ndarray, ...], ...]:
        """Expose frame lines with a concise name for callers and tests."""
        return self.lines_by_frame

    @property
    def frame_count(self) -> int:
        """Return the deterministic number of cached animation frames."""
        return len(self.lines_by_frame)


def build_streamline_cache(
    result: Aerodynamics3DResult, density: int = 4
) -> StreamlineAnimationCache:
    """Trace and cache all streamline frames from an existing result only."""
    seeds = make_streamline_seeds(result, density)
    frames = tuple(
        tuple(trace_streamline_3d(seed, result=result, frame=frame) for seed in seeds)
        for frame in range(len(result.t))
    )
    return StreamlineAnimationCache(seed_density=int(density), lines_by_frame=frames)


def slice_index_count(result: Aerodynamics3DResult, plane: str) -> int:
    """Return the valid number of slice indexes for one plane."""
    counts = {"XY": len(result.z), "XZ": len(result.y), "YZ": len(result.x)}
    try:
        return counts[plane.upper()]
    except KeyError as exc:
        raise ValueError("plane must be XY, XZ, or YZ") from exc


def slice_center_index(result: Aerodynamics3DResult, plane: str) -> int:
    """Return the central valid index for one slice plane."""
    return slice_index_count(result, plane) // 2


def _slice_data(
    result: Aerodynamics3DResult, data: np.ndarray, *, plane: str, index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Select one physical-coordinate 2D slice."""
    plane_name = plane.upper()
    selected = max(0, min(int(index), slice_index_count(result, plane_name) - 1))
    if plane_name == "XY":
        return (
            result.x,
            result.y,
            data[selected],
            float(result.z[selected]),
            f"z = {result.z[selected]:.4g}",
        )
    if plane_name == "XZ":
        return (
            result.x,
            result.z,
            data[:, selected, :],
            float(result.y[selected]),
            f"y = {result.y[selected]:.4g}",
        )
    return (
        result.y,
        result.z,
        data[:, :, selected],
        float(result.x[selected]),
        f"x = {result.x[selected]:.4g}",
    )


def slice_field(
    result: Aerodynamics3DResult, *, frame: int, plane: str, index: int, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return one physical-coordinate slice for the selected plane and field."""
    if not 0 <= frame < len(result.t):
        raise ValueError("frame is outside the saved result history.")
    if field == "Speed":
        data = derived_speed(result.u[frame], result.v[frame], result.w[frame])
    elif field == "Vorticity magnitude":
        _, _, _, data = derived_vorticity(
            result.u[frame],
            result.v[frame],
            result.w[frame],
            dx=float(result.x[1] - result.x[0]),
            dy=float(result.y[1] - result.y[0]),
            dz=float(result.z[1] - result.z[0]),
        )
    elif field in {"Pressure", "u", "v", "w"}:
        data = {"Pressure": result.pressure, "u": result.u, "v": result.v, "w": result.w}[field][
            frame
        ]
    else:
        raise ValueError(f"Unknown slice field '{field}'.")
    x_axis, y_axis, values, _, coordinate = _slice_data(result, data, plane=plane, index=index)
    return x_axis, y_axis, values, coordinate


@dataclass(frozen=True)
class SliceAnimationPayload:
    """One cached 2D time history prepared for a single slice selector."""

    t: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    frames: np.ndarray
    plane: str
    index: int
    field: str
    coordinate: float
    coordinate_label: str


def prepare_slice_history(
    result: Aerodynamics3DResult, *, plane: str, index: int, field: str
) -> SliceAnimationPayload:
    """Prepare only the requested 2D history, deriving fields frame by frame."""
    plane_name = plane.upper()
    selected = max(0, min(int(index), slice_index_count(result, plane_name) - 1))
    slices: list[np.ndarray] = []
    x_axis: np.ndarray | None = None
    y_axis: np.ndarray | None = None
    coordinate = 0.0
    coordinate_label = ""
    for frame in range(len(result.t)):
        x_axis, y_axis, values, coordinate_label = slice_field(
            result, frame=frame, plane=plane_name, index=selected, field=field
        )
        slices.append(np.asarray(values, dtype=float))
        if plane_name == "XY":
            coordinate = float(result.z[selected])
        elif plane_name == "XZ":
            coordinate = float(result.y[selected])
        else:
            coordinate = float(result.x[selected])
    if x_axis is None or y_axis is None:
        raise ValueError("The result history must contain at least one frame.")
    return SliceAnimationPayload(
        t=result.t,
        x_axis=x_axis,
        y_axis=y_axis,
        frames=np.stack(slices),
        plane=plane_name,
        index=selected,
        field=field,
        coordinate=coordinate,
        coordinate_label=coordinate_label,
    )


@dataclass(frozen=True)
class _FlowPayload:
    """Cached data used by both the embedded flow animation and MP4 export."""

    result: Aerodynamics3DResult
    display: str
    arrow_density: int
    magnitude_scale: float


@dataclass(frozen=True)
class _StreamlinePayload:
    """Cached streamlines shared by the embedded view and MP4 export."""

    result: Aerodynamics3DResult
    cache: StreamlineAnimationCache


def _obstacle_kwargs(result: Aerodynamics3DResult) -> dict[str, Any]:
    """Read continuous obstacle parameters from solver metadata."""
    metadata = result.metadata
    return {
        "shape": metadata.get("obstacle_shape", "sphere"),
        "center_x": float(metadata.get("obstacle_center_x", result.x[len(result.x) // 2])),
        "center_y": float(metadata.get("obstacle_center_y", result.y[len(result.y) // 2])),
        "center_z": float(metadata.get("obstacle_center_z", result.z[len(result.z) // 2])),
        "diameter": float(metadata.get("obstacle_diameter", 0.4)),
        "size_x": float(metadata.get("obstacle_size_x", 0.7)),
        "size_y": float(metadata.get("obstacle_size_y", 0.4)),
        "size_z": float(metadata.get("obstacle_size_z", 0.4)),
        "chord": float(metadata.get("obstacle_chord", 0.8)),
        "span": float(metadata.get("obstacle_span", 0.8)),
        "thickness_ratio": float(metadata.get("obstacle_thickness_ratio", 0.12)),
        "attack_deg": float(metadata.get("obstacle_attack_deg", 0.0)),
    }


def _draw_obstacle_surface(axis: Any, result: Aerodynamics3DResult) -> None:
    """Draw the analytical continuous obstacle surface."""
    for surface_x, surface_y, surface_z in obstacle_surface_mesh(
        **_obstacle_kwargs(result), resolution=28
    ):
        axis.plot_surface(
            surface_x,
            surface_y,
            surface_z,
            color="#65717f",
            alpha=0.68,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )


def _decimated_indices(
    shape: tuple[int, int, int], target: int = 900
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a regular subsampling whose product is close to the arrow target."""
    stride = max(1, int(np.ceil((np.prod(shape) / target) ** (1.0 / 3.0))))
    indices = np.indices(shape)
    return (
        indices[0, ::stride, ::stride, ::stride],
        indices[1, ::stride, ::stride, ::stride],
        indices[2, ::stride, ::stride, ::stride],
    )


def flow_vector_components(
    result: Aerodynamics3DResult, *, frame: int, display: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the selected visualization-only vector field for one frame."""
    if display == "Velocity perturbation":
        u_inf = float(result.metadata.get("u_inf", 0.0))
        return result.u[frame] - u_inf, result.v[frame], result.w[frame]
    if display == "Velocity vectors":
        return result.u[frame], result.v[frame], result.w[frame]
    if display == "Vorticity vectors":
        return derived_vorticity(
            result.u[frame],
            result.v[frame],
            result.w[frame],
            dx=float(result.x[1] - result.x[0]),
            dy=float(result.y[1] - result.y[0]),
            dz=float(result.z[1] - result.z[0]),
        )[:3]
    raise ValueError(f"Unknown flow display '{display}'.")


def _flow_scale(result: Aerodynamics3DResult, display: str) -> float:
    """Compute one stable magnitude scale for a flow display mode."""
    if display == "Velocity vectors":
        return max(float(np.max(result.max_speed)), 1.0e-12)
    maximum = 0.0
    for frame in range(len(result.t)):
        components = flow_vector_components(result, frame=frame, display=display)
        maximum = max(maximum, float(np.max(derived_speed(*components))))
    return max(maximum, 1.0e-12)


def _create_flow_figure(payload: _FlowPayload) -> Figure:
    """Build an animated 3D vector view from cached result arrays."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    result = payload.result
    figure = plt.figure()
    axis: Any = figure.add_subplot(111, projection="3d")
    indices = _decimated_indices(
        result.obstacle_mask.shape, target=max(300, payload.arrow_density * 250)
    )
    zz, yy, xx = indices
    fluid = ~result.obstacle_mask[zz, yy, xx]
    xs, ys, zs = result.x[xx][fluid], result.y[yy][fluid], result.z[zz][fluid]
    cmap = LinearSegmentedColormap.from_list(
        "aero_vectors", ("#14253d", "#1f4e79", "#2c7fb8", "#66c2a5")
    )

    def draw(frame: int) -> None:
        elevation, azimuth = axis.elev, axis.azim
        axis.clear()
        _draw_obstacle_surface(axis, result)
        u, v, w = flow_vector_components(result, frame=frame, display=payload.display)
        u, v, w = u[zz, yy, xx][fluid], v[zz, yy, xx][fluid], w[zz, yy, xx][fluid]
        magnitude = derived_speed(u, v, w)
        valid = np.isfinite(magnitude) & (magnitude > 1.0e-12)
        directions = [np.zeros_like(component) for component in (u, v, w)]
        for direction, component in zip(directions, (u, v, w)):
            np.divide(component, magnitude, out=direction, where=valid)
        colors = cmap(0.05 + 0.75 * np.clip(magnitude / payload.magnitude_scale, 0.0, 1.0))
        axis.quiver(
            xs[valid],
            ys[valid],
            zs[valid],
            directions[0][valid],
            directions[1][valid],
            directions[2][valid],
            colors=colors[valid],
            length=0.14,
            normalize=False,
            linewidth=0.9,
            alpha=0.86,
        )
        axis.set_xlim(result.x[0], result.x[-1])
        axis.set_ylim(result.y[0], result.y[-1])
        axis.set_zlim(result.z[0], result.z[-1])
        axis.set_box_aspect(
            (result.x[-1] - result.x[0], result.y[-1] - result.y[0], result.z[-1] - result.z[0])
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.grid(False)
        axis.set_title(f"{payload.display} (t={result.t[frame]:.3g})")
        axis.view_init(elev=elevation, azim=azimuth)

    draw(0)

    def update(frame: int) -> None:
        draw(max(0, min(frame, len(result.t) - 1)))
        figure.canvas.draw_idle()

    return attach_animation_metadata(
        figure,
        update=update,
        n_points=len(result.t),
        frame_label="t",
        frame_coordinates=result.t,
    )


def _create_streamline_figure(payload: _StreamlinePayload) -> Figure:
    """Build a 3D animation from already cached streamline polylines."""
    import matplotlib.pyplot as plt

    result = payload.result
    cache = payload.cache
    figure = plt.figure()
    axis: Any = figure.add_subplot(111, projection="3d")
    _draw_obstacle_surface(axis, result)
    line_artists: list[Any] = []

    def draw_lines(frame: int) -> None:
        line_artists.clear()
        for line in cache.lines_by_frame[frame]:
            if len(line) > 1:
                (artist,) = axis.plot(
                    line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color="#1f4e79",
                    linewidth=1.8,
                    alpha=0.92,
                )
                line_artists.append(artist)

    draw_lines(0)
    axis.set_xlim(result.x[0], result.x[-1])
    axis.set_ylim(result.y[0], result.y[-1])
    axis.set_zlim(result.z[0], result.z[-1])
    axis.set_box_aspect(
        (result.x[-1] - result.x[0], result.y[-1] - result.y[0], result.z[-1] - result.z[0])
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.grid(False)
    axis.set_title(f"3D streamlines (t={result.t[0]:.3g})")
    figure.tight_layout()

    def update(frame: int) -> None:
        index = max(0, min(frame, cache.frame_count - 1))
        for artist in line_artists:
            artist.remove()
        draw_lines(index)
        axis.set_title(f"3D streamlines (t={result.t[index]:.3g})")
        figure.canvas.draw_idle()

    return attach_animation_metadata(
        figure,
        update=update,
        n_points=cache.frame_count,
        frame_label="t",
        frame_coordinates=result.t,
    )


def _line_limits(values: np.ndarray) -> tuple[float, float]:
    """Return readable limits for a possibly constant diagnostic."""
    lower, upper = float(np.min(values)), float(np.max(values))
    if lower == upper:
        margin = max(1.0, abs(lower) * 0.1)
        return lower - margin, upper + margin
    margin = 0.05 * (upper - lower)
    return lower - margin, upper + margin


class Aerodynamics3DResultDialog:
    """Tabbed visual workspace for a completed 3D aerodynamics solve."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: Aerodynamics3DResult) -> None:
        self.parent = parent
        self._result = result
        self._flow_scale_cache: dict[str, float] = {}
        self._streamline_cache: dict[int, StreamlineAnimationCache] = {}
        self._slice_cache: dict[tuple[str, int, str], SliceAnimationPayload] = {}
        self._canvases: dict[str, object | None] = {}
        self._stream_initialized = False
        self.win = tk.Toplevel(parent)
        self.win.title("Aerodynamics 3D Results")
        self._build_ui()
        self._shell.finish(AdvancedResultSize(1460, 940, 980, 650))
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        close_embedded_figures(self, tuple(f"_canvas_{name}" for name in self._canvases))
        self.win.destroy()

    def _build_ui(self) -> None:
        magnitudes = self._result.magnitudes
        summary = format_result_summary(
            (
                ("Reynolds number", f"{magnitudes['reynolds']:.1f}"),
                ("Mean tail Cd", f"{magnitudes['mean_cd_tail']:+.3e}"),
                ("RMS Cl", f"{magnitudes['rms_cl']:.3e}"),
                ("RMS Cs", f"{magnitudes['rms_cs']:.3e}"),
                ("Maximum L2 divergence", f"{magnitudes['max_divergence_l2']:.3e}"),
            )
        )
        self._shell = AdvancedResultShell(
            self.win, title="Aerodynamics 3D Results", summary=summary, close_command=self._on_close
        )
        notebook = self._shell.notebook
        tabs: list[tuple[str, Callable[[ttk.Frame], None]]] = [
            ("3D Flow", self._build_flow_tab),
            ("3D Streamlines", self._build_streamlines_tab),
            ("Slices", self._build_slices_tab),
            ("Forces", self._build_forces_tab),
            ("Diagnostics", self._build_diagnostics_tab),
        ]
        for label, builder in tabs:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=label)
            if label == "3D Streamlines":
                self._stream_tab = tab
            else:
                builder(tab)
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, _event: object) -> None:
        if (
            not self._stream_initialized
            and self._shell.notebook.tab("current", "text") == "3D Streamlines"
        ):
            self._stream_initialized = True
            self._build_streamlines_tab(self._stream_tab)

    def _build_flow_tab(self, parent: ttk.Frame) -> None:
        controls = make_view_controls(parent)
        group = controls.add_group(requested_width=260)
        self._flow_display = tk.StringVar(value="Velocity perturbation")
        ttk.Label(group, text="Display:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            group,
            textvariable=self._flow_display,
            values=_FLOW_DISPLAYS,
            state="readonly",
            width=22,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT)
        self._flow_density = tk.StringVar(value="4")
        group = controls.add_group(requested_width=150)
        ttk.Label(group, text="Arrow density:", style="Small.TLabel").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        density_combo = ttk.Combobox(
            group,
            textvariable=self._flow_density,
            values=("2", "3", "4", "5", "6"),
            state="readonly",
            width=5,
            font=get_font(),
        )
        density_combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._update_flow())
        density_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_flow())
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        self._flow_frame = frame
        self._update_flow()

    def _update_flow(self) -> None:
        reset_embedded_animation(self._flow_frame, self._canvas("flow"))
        display = self._flow_display.get()
        if display not in self._flow_scale_cache:
            self._flow_scale_cache[display] = _flow_scale(self._result, display)
        payload = _FlowPayload(
            self._result,
            display,
            int(self._flow_density.get()),
            self._flow_scale_cache[display],
        )
        canvas = embed_animation_plot_in_tk(
            _create_flow_figure(payload),
            self._flow_frame,
            on_export_mp4=lambda duration: self._export_flow(payload, duration),
        )
        self._set_canvas("flow", canvas)

    def _build_streamlines_tab(self, parent: ttk.Frame) -> None:
        controls = make_view_controls(parent)
        density_var = tk.StringVar(value="4")
        group = controls.add_group(requested_width=205)
        ttk.Label(group, text="Seed density:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        density_combo = ttk.Combobox(
            group,
            textvariable=density_var,
            values=("2", "3", "4", "5", "6"),
            state="readonly",
            width=5,
            font=get_font(),
        )
        density_combo.pack(side=tk.LEFT)
        self._stream_density = density_var
        self._stream_build_button = ttk.Button(
            controls.add_group(requested_width=240),
            text="Build streamline animation",
            command=lambda: self._request_streamline_build(int(density_var.get()), ask=False),
        )
        self._stream_build_button.pack(side=tk.LEFT)
        self._stream_status = ttk.Label(parent, text="", style="Small.TLabel")
        self._stream_status.pack(fill=tk.X, padx=8, pady=(0, 4))
        target = ttk.Frame(parent)
        target.pack(fill=tk.BOTH, expand=True)
        self._stream_frame = target

        def density_changed(_event: object) -> None:
            self._request_streamline_build(int(density_var.get()), ask=True)

        density_combo.bind("<<ComboboxSelected>>", density_changed)
        self._request_streamline_build(4, ask=True)

    def _request_streamline_build(self, density: int, *, ask: bool) -> None:
        if density in self._streamline_cache:
            self._show_streamline_cache(self._streamline_cache[density])
            return
        if ask and not messagebox.askyesno(
            "Build streamline animation?",
            "DifferentialLab needs to trace the streamline set for all saved frames.\n\n"
            "This may take some time. The result will be cached for this Results window.\n\n"
            "Build it now?",
            parent=self.win,
        ):
            self._stream_status.configure(
                text="Streamline animation not built. Use Build streamline animation when ready."
            )
            return
        self._build_streamline_cache(density)

    def _build_streamline_cache(self, density: int) -> None:
        self._stream_build_button.configure(state=tk.DISABLED)
        self._stream_status.configure(text="Tracing streamlines for all saved frames…")
        self.win.configure(cursor="watch")
        self.win.update_idletasks()
        try:
            cache = build_streamline_cache(self._result, density)
            self._streamline_cache[density] = cache
            self._show_streamline_cache(cache)
        except Exception as exc:
            logger.error("Streamline animation cache failed: %s", exc, exc_info=True)
            self._stream_status.configure(text="Streamline animation could not be built.")
            messagebox.showerror("Streamline animation", str(exc), parent=self.win)
        finally:
            self.win.configure(cursor="")
            self._stream_build_button.configure(state=tk.NORMAL)

    def _show_streamline_cache(self, cache: StreamlineAnimationCache) -> None:
        reset_embedded_animation(self._stream_frame, self._canvas("streamlines"))
        self._stream_status.configure(
            text=f"Cached {cache.frame_count} frames at seed density {cache.seed_density}."
        )
        payload = _StreamlinePayload(self._result, cache)
        canvas = embed_animation_plot_in_tk(
            _create_streamline_figure(payload),
            self._stream_frame,
            on_export_mp4=lambda duration: self._export_streamlines(payload, duration),
        )
        self._set_canvas("streamlines", canvas)

    def _build_slices_tab(self, parent: ttk.Frame) -> None:
        controls = make_view_controls(parent)
        self._slice_plane = tk.StringVar(value="XY")
        self._slice_field = tk.StringVar(value="Vorticity magnitude")
        self._slice_index = tk.StringVar(value=str(slice_center_index(self._result, "XY")))
        group = controls.add_group(requested_width=150)
        ttk.Label(group, text="Plane:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        plane_combo = ttk.Combobox(
            group,
            textvariable=self._slice_plane,
            values=("XY", "XZ", "YZ"),
            state="readonly",
            width=5,
            font=get_font(),
        )
        plane_combo.pack(side=tk.LEFT)
        group = controls.add_group(requested_width=210)
        ttk.Label(group, text="Coordinate/index:", style="Small.TLabel").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        index_combo = ttk.Combobox(
            group,
            textvariable=self._slice_index,
            values=tuple(str(i) for i in range(slice_index_count(self._result, "XY"))),
            state="readonly",
            width=6,
            font=get_font(),
        )
        index_combo.pack(side=tk.LEFT)
        self._slice_index_combo = index_combo
        group = controls.add_group(requested_width=250)
        ttk.Label(group, text="Field:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        field_combo = ttk.Combobox(
            group,
            textvariable=self._slice_field,
            values=_SLICE_FIELDS,
            state="readonly",
            width=18,
            font=get_font(),
        )
        field_combo.pack(side=tk.LEFT)
        self._slice_coordinate = ttk.Label(parent, text="", style="Small.TLabel")
        self._slice_coordinate.pack(fill=tk.X, padx=8, pady=(0, 4))
        target = ttk.Frame(parent)
        target.pack(fill=tk.BOTH, expand=True)
        self._slice_frame = target

        def redraw(_event: object | None = None) -> None:
            plane = self._slice_plane.get()
            index = int(self._slice_index.get())
            field = self._slice_field.get()
            key = (plane, index, field)
            payload = self._slice_cache.get(key)
            if payload is None:
                payload = prepare_slice_history(self._result, plane=plane, index=index, field=field)
                self._slice_cache[key] = payload
            self._slice_coordinate.configure(
                text=f"Physical slice coordinate: {payload.coordinate_label}"
            )
            reset_embedded_animation(target, self._canvas("slices"))
            figure = create_image_animation_plot(
                payload.t,
                payload.frames,
                title=f"{payload.field} ({payload.coordinate_label})",
                xlabel=plane[0].lower(),
                ylabel=plane[1].lower(),
                x_coordinates=payload.x_axis,
                y_coordinates=payload.y_axis,
                symmetric_color_range=payload.field not in {"Speed", "Pressure"},
            )
            self._set_canvas("slices", embed_animation_plot_in_tk(figure, target))

        def plane_changed(_event: object) -> None:
            plane = self._slice_plane.get()
            index_combo.configure(
                values=tuple(str(i) for i in range(slice_index_count(self._result, plane)))
            )
            self._slice_index.set(str(slice_center_index(self._result, plane)))
            redraw()

        plane_combo.bind("<<ComboboxSelected>>", plane_changed)
        index_combo.bind("<<ComboboxSelected>>", redraw)
        field_combo.bind("<<ComboboxSelected>>", redraw)
        redraw()

    def _build_forces_tab(self, parent: ttk.Frame) -> None:
        values = np.vstack(
            [self._result.drag_coeff, self._result.lift_coeff, self._result.side_force_coeff]
        )
        figure = create_solution_plot(
            self._result.t,
            values,
            title="Aerodynamic force coefficients",
            xlabel="t",
            ylabel="coefficient",
            selected_derivatives=[0, 1, 2],
            labels=["Cd", "Cl", "Cs"],
        )
        self._set_canvas("forces", embed_plot_in_tk(figure, parent))

    def _build_diagnostics_tab(self, parent: ttk.Frame) -> None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(self._result.t, self._result.divergence_l2, label="divergence L2")
        axes[0].set_ylabel("L2")
        axes[0].legend()
        axes[1].plot(self._result.t, self._result.max_speed, label="maximum speed")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("speed")
        axes[1].legend()
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.set_ylim(*_line_limits(axis.lines[0].get_ydata()))
        figure.tight_layout()
        self._set_canvas("diagnostics", embed_plot_in_tk(figure, parent))

    def _canvas(self, name: str) -> object | None:
        return self._canvases.get(name)

    def _set_canvas(self, name: str, canvas: object) -> None:
        self._canvases[name] = canvas
        setattr(self, f"_canvas_{name}", canvas)

    def _export_flow(self, payload: _FlowPayload, duration_seconds: float) -> None:
        self._export_figure(
            _create_flow_figure(payload), duration_seconds, prefix="aerodynamics_3d_flow"
        )

    def _export_streamlines(self, payload: _StreamlinePayload, duration_seconds: float) -> None:
        self._export_figure(
            _create_streamline_figure(payload),
            duration_seconds,
            prefix="aerodynamics_3d_streamlines",
        )

    def _export_figure(self, figure: Figure, duration_seconds: float, *, prefix: str) -> None:
        default_path = get_output_dir() / f"{generate_output_basename(prefix=prefix)}.mp4"
        filepath_str = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".mp4",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if not filepath_str:
            return
        try:
            export_animated_figure_to_mp4(
                figure, Path(filepath_str), duration_seconds=duration_seconds
            )
            messagebox.showinfo(
                "Animation export saved",
                f"Animation was saved to:\n{filepath_str}",
                parent=self.win,
            )
        except RuntimeError as exc:
            logger.warning("MP4 export failed (ffmpeg): %s", exc)
            messagebox.showerror(
                "Animation was not saved",
                str(exc) + "\n\nInstall ffmpeg and ensure it is in your PATH.",
                parent=self.win,
            )
        except Exception as exc:
            logger.error("MP4 export failed: %s", exc, exc_info=True)
            messagebox.showerror("Animation was not saved", str(exc), parent=self.win)
