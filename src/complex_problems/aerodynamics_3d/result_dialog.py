"""Advanced result views for the 3D aerodynamics solver."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from complex_problems.aerodynamics_3d.model import vorticity_periodic
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
from plotting import create_solution_plot, export_animated_figure_to_mp4
from plotting.animation_metadata import attach_animation_metadata
from utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)


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
    """Trace one deterministic 3D streamline with fixed-step RK4.

    Coordinates are represented as ``(x, y, z)`` for callers while the
    interpolators use the stored ``(z, y, x)`` array order.
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

    def velocity(point: np.ndarray) -> np.ndarray | None:
        if np.any(point < lower) or np.any(point > upper):
            return None
        values = np.array(
            [interpolator((point[2], point[1], point[0]))[()] for interpolator in interpolators],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return None
        return values

    points = [np.asarray(seed, dtype=float)]
    for _ in range(max_steps):
        point = points[-1]
        if velocity(point) is None or _point_is_obstacle(result, point):
            break
        k1 = velocity(point)
        if k1 is None or np.linalg.norm(k1) <= speed_epsilon:
            break
        k2 = velocity(point + 0.5 * step * k1)
        k3 = velocity(point + 0.5 * step * (k2 if k2 is not None else k1))
        k4 = velocity(point + step * (k3 if k3 is not None else k1))
        if k2 is None or k3 is None or k4 is None:
            break
        next_point = point + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if velocity(next_point) is None or _point_is_obstacle(result, next_point):
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


def slice_field(
    result: Aerodynamics3DResult, *, frame: int, plane: str, index: int, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return one physical-coordinate slice for the selected plane and field."""
    plane_name = plane.upper()
    max_index = {"XY": len(result.z), "XZ": len(result.y), "YZ": len(result.x)}.get(plane_name)
    if max_index is None:
        raise ValueError("plane must be XY, XZ, or YZ")
    selected = max(0, min(int(index), max_index - 1))
    speed = derived_speed(result.u[frame], result.v[frame], result.w[frame])
    _, _, _, vorticity_magnitude = derived_vorticity(
        result.u[frame],
        result.v[frame],
        result.w[frame],
        dx=float(result.x[1] - result.x[0]),
        dy=float(result.y[1] - result.y[0]),
        dz=float(result.z[1] - result.z[0]),
    )
    values = {
        "Speed": speed,
        "Vorticity magnitude": vorticity_magnitude,
        "Pressure": result.pressure[frame],
        "u": result.u[frame],
        "v": result.v[frame],
        "w": result.w[frame],
    }
    if field not in values:
        raise ValueError(f"Unknown slice field '{field}'.")
    data = values[field]
    if plane_name == "XY":
        return result.x, result.y, data[selected], f"z = {result.z[selected]:.4g}"
    if plane_name == "XZ":
        return result.x, result.z, data[:, selected, :], f"y = {result.y[selected]:.4g}"
    return result.y, result.z, data[:, :, selected], f"x = {result.x[selected]:.4g}"


@dataclass(frozen=True)
class _FlowPayload:
    """Cached data used by both the embedded flow animation and MP4 export."""

    result: Aerodynamics3DResult
    display: str
    arrow_density: int


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return obstacle cells adjacent to at least one fluid cell."""
    return mask & (
        ~(
            np.roll(mask, 1, axis=0)
            & np.roll(mask, -1, axis=0)
            & np.roll(mask, 1, axis=1)
            & np.roll(mask, -1, axis=1)
            & np.roll(mask, 1, axis=2)
            & np.roll(mask, -1, axis=2)
        )
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


def _create_flow_figure(payload: _FlowPayload) -> Figure:
    """Build an animated 3D vector view from cached result arrays."""
    import matplotlib.pyplot as plt

    result = payload.result
    figure = plt.figure()
    axis: Any = figure.add_subplot(111, projection="3d")
    boundary = _boundary_mask(result.obstacle_mask)
    indices = _decimated_indices(
        result.obstacle_mask.shape, target=max(300, payload.arrow_density * 250)
    )
    zz, yy, xx = indices
    fluid = ~result.obstacle_mask[zz, yy, xx]
    xs, ys, zs = result.x[xx][fluid], result.y[yy][fluid], result.z[zz][fluid]
    norm_max = max(float(np.max(result.max_speed)), 1.0e-12)

    def draw(frame: int) -> None:
        elevation, azimuth = axis.elev, axis.azim
        axis.clear()
        if np.any(boundary):
            bz, by, bx = np.where(boundary)
            axis.scatter(result.x[bx], result.y[by], result.z[bz], c="dimgray", s=8, alpha=0.7)
        u = result.u[frame][zz, yy, xx][fluid]
        v = result.v[frame][zz, yy, xx][fluid]
        w = result.w[frame][zz, yy, xx][fluid]
        if payload.display == "Vorticity vectors":
            wx, wy, wz, _ = derived_vorticity(
                result.u[frame],
                result.v[frame],
                result.w[frame],
                dx=float(result.x[1] - result.x[0]),
                dy=float(result.y[1] - result.y[0]),
                dz=float(result.z[1] - result.z[0]),
            )
            u, v, w = wx[zz, yy, xx][fluid], wy[zz, yy, xx][fluid], wz[zz, yy, xx][fluid]
        colors = plt.cm.viridis(np.clip(derived_speed(u, v, w) / norm_max, 0.0, 1.0))
        axis.quiver(
            xs, ys, zs, u, v, w, colors=colors, length=0.12, normalize=False, linewidth=0.55
        )
        axis.set_xlim(result.x[0], result.x[-1])
        axis.set_ylim(result.y[0], result.y[-1])
        axis.set_zlim(result.z[0], result.z[-1])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(f"{payload.display} (t={result.t[frame]:.3g})")
        axis.view_init(elev=elevation, azim=azimuth)

    draw(0)

    def update(frame: int) -> None:
        draw(max(0, min(frame, len(result.t) - 1)))
        figure.canvas.draw_idle()

    return attach_animation_metadata(figure, update=update, n_points=len(result.t))


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
        self.win = tk.Toplevel(parent)
        self.win.title("Aerodynamics 3D Results")
        self._canvases: dict[str, object | None] = {}
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
            builder(tab)

    def _build_flow_tab(self, parent: ttk.Frame) -> None:
        controls = make_view_controls(parent)
        group = controls.add_group(requested_width=260)
        self._flow_display = tk.StringVar(value="Velocity vectors")
        ttk.Label(group, text="Display:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            group,
            textvariable=self._flow_display,
            values=("Velocity vectors", "Vorticity vectors"),
            state="readonly",
            width=19,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT)
        self._flow_density = tk.StringVar(value="4")
        group = controls.add_group(requested_width=150)
        ttk.Label(group, text="Arrow density:", style="Small.TLabel").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Combobox(
            group,
            textvariable=self._flow_density,
            values=("2", "3", "4", "5", "6"),
            state="readonly",
            width=5,
            font=get_font(),
        ).pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._update_flow())
        self._flow_density.trace_add("write", lambda *_args: self._update_flow())
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        self._flow_frame = frame
        self._update_flow()

    def _update_flow(self) -> None:
        reset_embedded_animation(self._flow_frame, self._canvas("flow"))
        payload = _FlowPayload(
            self._result, self._flow_display.get(), int(self._flow_density.get())
        )
        canvas = embed_animation_plot_in_tk(
            _create_flow_figure(payload),
            self._flow_frame,
            on_export_mp4=lambda duration: self._export(payload, duration),
        )
        self._set_canvas("flow", canvas)

    def _build_streamlines_tab(self, parent: ttk.Frame) -> None:
        controls = make_view_controls(parent)
        frame_var = tk.StringVar(value="0")
        density_var = tk.StringVar(value="4")
        group = controls.add_group(requested_width=200)
        ttk.Label(group, text="Frame:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        frame_combo = ttk.Combobox(
            group,
            textvariable=frame_var,
            values=tuple(str(i) for i in range(len(self._result.t))),
            state="readonly",
            width=7,
            font=get_font(),
        )
        frame_combo.pack(side=tk.LEFT)
        group = controls.add_group(requested_width=180)
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
        target = ttk.Frame(parent)
        target.pack(fill=tk.BOTH, expand=True)
        self._stream_frame = target

        def redraw(_event: object | None = None) -> None:
            reset_embedded_animation(target, self._canvas("streamlines"))
            figure = self._streamline_figure(int(frame_var.get()), int(density_var.get()))
            self._set_canvas("streamlines", embed_plot_in_tk(figure, target))

        frame_combo.bind("<<ComboboxSelected>>", redraw)
        density_combo.bind("<<ComboboxSelected>>", redraw)
        redraw()

    def _streamline_figure(self, frame: int, density: int) -> Figure:
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis: Any = figure.add_subplot(111, projection="3d")
        mask = _boundary_mask(self._result.obstacle_mask)
        if np.any(mask):
            iz, iy, ix = np.where(mask)
            axis.scatter(
                self._result.x[ix], self._result.y[iy], self._result.z[iz], c="dimgray", s=8
            )
        for seed in make_streamline_seeds(self._result, density):
            line = trace_streamline_3d(seed, result=self._result, frame=frame)
            if len(line) > 1:
                axis.plot(line[:, 0], line[:, 1], line[:, 2], linewidth=1.2)
        axis.set_xlim(self._result.x[0], self._result.x[-1])
        axis.set_ylim(self._result.y[0], self._result.y[-1])
        axis.set_zlim(self._result.z[0], self._result.z[-1])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(f"3D streamlines (t={self._result.t[frame]:.3g})")
        figure.tight_layout()
        return figure

    def _build_slices_tab(self, parent: ttk.Frame) -> None:
        controls = make_view_controls(parent)
        self._slice_plane = tk.StringVar(value="XY")
        self._slice_field = tk.StringVar(value="Vorticity magnitude")
        self._slice_index = tk.StringVar(value=str(len(self._result.z) // 2))
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
            values=tuple(
                str(i)
                for i in range(max(len(self._result.x), len(self._result.y), len(self._result.z)))
            ),
            state="readonly",
            width=6,
            font=get_font(),
        )
        index_combo.pack(side=tk.LEFT)
        group = controls.add_group(requested_width=250)
        ttk.Label(group, text="Field:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        field_combo = ttk.Combobox(
            group,
            textvariable=self._slice_field,
            values=("Vorticity magnitude", "Speed", "Pressure", "u", "v", "w"),
            state="readonly",
            width=18,
            font=get_font(),
        )
        field_combo.pack(side=tk.LEFT)
        target = ttk.Frame(parent)
        target.pack(fill=tk.BOTH, expand=True)
        self._slice_frame = target

        def redraw(_event: object | None = None) -> None:
            reset_embedded_animation(target, self._canvas("slices"))
            x_axis, y_axis, values, coordinate = slice_field(
                self._result,
                frame=0,
                plane=self._slice_plane.get(),
                index=int(self._slice_index.get()),
                field=self._slice_field.get(),
            )
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots()
            image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
            )
            figure.colorbar(image, ax=axis, label=self._slice_field.get())
            axis.set_xlabel(self._slice_plane.get()[0].lower())
            axis.set_ylabel(self._slice_plane.get()[1].lower())
            axis.set_title(f"{self._slice_field.get()} ({coordinate})")
            figure.tight_layout()
            self._set_canvas("slices", embed_plot_in_tk(figure, target))

        for combo in (plane_combo, index_combo, field_combo):
            combo.bind("<<ComboboxSelected>>", redraw)
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

    def _export(self, payload: _FlowPayload, duration_seconds: float) -> None:
        default_path = (
            get_output_dir() / f"{generate_output_basename(prefix='aerodynamics_3d')}.mp4"
        )
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
                _create_flow_figure(payload), Path(filepath_str), duration_seconds=duration_seconds
            )
            messagebox.showinfo(
                "Animation export saved",
                f"Animation was saved to:\n{filepath_str}",
                parent=self.win,
            )
        except RuntimeError as exc:
            logger.warning("MP4 export failed (ffmpeg): %s", exc)
            messagebox.showerror(
                "Animation export was not saved",
                str(exc) + "\n\nInstall ffmpeg and ensure it is in your PATH.",
                parent=self.win,
            )
        except Exception as exc:
            logger.error("MP4 export failed: %s", exc, exc_info=True)
            messagebox.showerror("Animation export was not saved", str(exc), parent=self.win)
