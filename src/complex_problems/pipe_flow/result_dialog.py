"""Result dialog for pipe-flow simulations."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np

from complex_problems.common.result_dialog_ui import (
    AdvancedResultShell,
    AdvancedResultSize,
    close_embedded_figures,
    format_result_summary,
    make_view_controls,
    reset_embedded_animation,
)
from complex_problems.pipe_flow.solver import PipeFlowResult
from config import generate_output_basename, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import make_modal
from plotting import create_contour_plot, export_animated_figure_to_mp4
from plotting.animation_metadata import attach_animation_metadata
from utils import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = get_logger(__name__)


def _set_visible_ylim(ax: Axes, values: np.ndarray, references: tuple[float, ...] = ()) -> None:
    """Keep flat profiles visible without distorting non-flat data."""
    data = np.asarray(values, dtype=float).ravel()
    if references:
        data = np.concatenate((data, np.asarray(references, dtype=float)))
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        ax.set_ylim(-1.0, 1.0)
        return

    low = float(np.min(finite))
    high = float(np.max(finite))
    span = high - low
    if np.isclose(low, high, rtol=1e-9, atol=1e-12):
        margin = max(abs(low) * 0.08, 1e-6)
    else:
        margin = 0.08 * span
    ax.set_ylim(low - margin, high + margin)


def _style_profile_axis(ax: Axes, *, ylabel: str, xlabel: str | None = None) -> None:
    """Apply the common styling used by pipe-flow profile subplots."""
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


def _create_geometry_figure(result: PipeFlowResult) -> Figure:
    """Create separate diameter and area profiles for the pipe geometry."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(result.x, result.diameter, linewidth=2.0)
    _style_profile_axis(axes[0], ylabel="Diameter [m]")
    axes[0].set_title("Diameter profile")

    axes[1].plot(result.x, result.area, linewidth=2.0)
    _style_profile_axis(axes[1], ylabel="Area [m²]", xlabel="x [m]")
    axes[1].set_title("Cross-sectional area")

    fig.suptitle("Pipe geometry")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _create_steady_pressure_figure(result: PipeFlowResult) -> Figure:
    """Create the steady pressure profile in kPa."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    pressure_kpa = result.pressure[0] / 1000.0
    ax.plot(result.x, pressure_kpa, linewidth=2.0, label="p(x)")
    ax.set_title("Steady pressure profile")
    _style_profile_axis(ax, ylabel="Pressure [kPa]", xlabel="x [m]")
    ax.legend()
    fig.tight_layout()
    return fig


def _create_steady_velocity_figure(result: PipeFlowResult) -> Figure:
    """Create independent spatial profiles for velocity, Reynolds, and friction."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axes[0].plot(result.x, result.velocity[0], linewidth=2.0)
    axes[0].set_title("Velocity")
    _style_profile_axis(axes[0], ylabel="Velocity [m/s]")

    axes[1].plot(result.x, result.reynolds[0], linewidth=2.0)
    axes[1].set_title("Reynolds number")
    _style_profile_axis(axes[1], ylabel="Re")

    axes[2].plot(result.x, result.friction[0], linewidth=2.0)
    axes[2].set_title("Darcy friction factor")
    _style_profile_axis(axes[2], ylabel="f_D", xlabel="x [m]")

    fig.suptitle("Steady flow profiles")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _create_steady_flow_diagnostics_figure(result: PipeFlowResult) -> Figure:
    """Create spatial flow-rate and mass-conservation residual diagnostics."""
    import matplotlib.pyplot as plt

    q_profile = result.velocity[0] * result.area
    q_mean = float(result.flow_rate_mean[0])
    residual = q_profile - q_mean

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(result.x, q_profile, linewidth=2.0, label="Q(x)")
    axes[0].axhline(q_mean, color="tab:red", linestyle="--", label="Q mean")
    axes[0].set_title("Volumetric flow rate")
    _style_profile_axis(axes[0], ylabel="Q [m³/s]")
    _set_visible_ylim(axes[0], q_profile, (q_mean,))
    axes[0].legend()

    axes[1].plot(result.x, residual, linewidth=2.0, label="Q - Q mean")
    axes[1].axhline(0.0, color="tab:red", linestyle="--", label="zero")
    axes[1].set_title("Flow-rate residual")
    _style_profile_axis(axes[1], ylabel="Q - Q_mean [m³/s]", xlabel="x [m]")
    _set_visible_ylim(axes[1], residual, (0.0,))
    axes[1].legend()

    fig.suptitle("Steady flow-rate diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _create_transient_flow_diagnostics_figure(result: PipeFlowResult) -> Figure:
    """Create separate pressure and flow-rate time-series diagnostics."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    p_in = result.pressure[:, 0] / 1000.0
    p_out = result.pressure[:, -1] / 1000.0
    axes[0].plot(result.t, p_in, linewidth=2.0, label="p_in")
    axes[0].plot(result.t, p_out, linewidth=2.0, label="p_out")
    axes[0].set_title("Boundary pressures")
    _style_profile_axis(axes[0], ylabel="Pressure [kPa]")
    axes[0].legend()

    axes[1].plot(result.t, result.flow_rate_mean, linewidth=2.0, label="Q mean")
    axes[1].plot(result.t, result.flow_rate_std, linewidth=2.0, label="Q std")
    axes[1].set_title("Flow-rate statistics")
    _style_profile_axis(axes[1], ylabel="Q [m³/s]", xlabel="t")
    axes[1].legend()

    fig.suptitle("Transient flow diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _format_result_summary(result: PipeFlowResult) -> str:
    """Format a compact, human-readable result summary for the dialog header."""
    if result.model_type == "steady":
        q_mean = float(result.flow_rate_mean[0])
        mean_velocity = float(np.mean(result.velocity[0]))
        re_max = float(np.max(result.reynolds[0]))
        delta_p_kpa = float((result.pressure[0, 0] - result.pressure[0, -1]) / 1000.0)
        return (
            f"Q: {q_mean:.4g} m³/s    mean u: {mean_velocity:.4g} m/s    "
            f"Re max: {re_max:.4g}    Δp: {delta_p_kpa:.4g} kPa"
        )

    max_pressure_kpa = float(np.max(np.abs(result.pressure)) / 1000.0)
    max_velocity = float(np.max(np.abs(result.velocity)))
    re_max = float(np.max(result.reynolds))
    cfl = float(result.magnitudes.get("cfl", float("nan")))
    mean_q_std = float(np.mean(result.flow_rate_std))
    return (
        f"max |p|: {max_pressure_kpa:.4g} kPa    max |u|: {max_velocity:.4g} m/s    "
        f"Re max: {re_max:.4g}    CFL: {cfl:.4g}    mean Q std: {mean_q_std:.4g} m³/s"
    )


def _create_line_animation_figure(
    x: np.ndarray,
    t: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ymin = float(np.min(field))
    ymax = float(np.max(field))
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    margin = 0.08 * (ymax - ymin)
    (line,) = ax.plot(x, field[0], linewidth=2.0)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (t={t[0]:.3g})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    def _update(idx: int) -> None:
        i = max(0, min(idx, len(t) - 1))
        line.set_ydata(field[i])
        ax.set_title(f"{title} (t={t[i]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(fig, update=_update, n_points=len(t))


class PipeFlowResultDialog:
    """Result window for steady/transient pipe flow."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: PipeFlowResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Pipe Flow Results")

        self._anim_canvas = None
        self._geometry_canvas = None
        self._pressure_canvas = None
        self._velocity_canvas = None
        self._quality_canvas = None

        self._build_ui()
        self._shell.finish(AdvancedResultSize(1360, 900))
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        close_embedded_figures(
            self,
            (
                "_anim_canvas",
                "_geometry_canvas",
                "_pressure_canvas",
                "_velocity_canvas",
                "_quality_canvas",
            ),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        summary = format_result_summary(
            (
                ("Mode", self._result.model_type.capitalize()),
                ("Run summary", _format_result_summary(self._result)),
            )
        )
        self._shell = AdvancedResultShell(
            self.win,
            title="Pipe Flow Results",
            summary=summary,
            close_command=self._on_close,
        )
        nb = self._shell.notebook

        if self._result.model_type == "transient":
            tab_anim = ttk.Frame(nb)
            nb.add(tab_anim, text="Animation")
            self._build_anim_tab(tab_anim)

        tab_geom = ttk.Frame(nb)
        nb.add(tab_geom, text="Geometry")
        self._build_geometry_tab(tab_geom)

        tab_p = ttk.Frame(nb)
        nb.add(tab_p, text="Pressure")
        self._build_pressure_tab(tab_p)

        tab_u = ttk.Frame(nb)
        nb.add(tab_u, text="Velocity")
        self._build_velocity_tab(tab_u)

        tab_q = ttk.Frame(nb)
        nb.add(tab_q, text="Flow Diagnostics")
        self._build_quality_tab(tab_q)

    def _build_anim_tab(self, parent: ttk.Frame) -> None:
        ctrl = make_view_controls(parent)
        self._anim_view_var = tk.StringVar(value="pressure")
        ttk.Label(ctrl, text="Display:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            ctrl,
            textvariable=self._anim_view_var,
            values=("pressure", "velocity", "reynolds"),
            state="readonly",
            width=12,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._update_animation())
        self._anim_frame = ttk.Frame(parent)
        self._anim_frame.pack(fill=tk.BOTH, expand=True)
        self._update_animation()

    def _update_animation(self) -> None:
        reset_embedded_animation(self._anim_frame, self._anim_canvas)

        view = self._anim_view_var.get()
        if view == "velocity":
            arr = self._result.velocity
            ylabel = "u"
            title = "Velocity profile"
        elif view == "reynolds":
            arr = self._result.reynolds
            ylabel = "Re"
            title = "Reynolds profile"
        else:
            arr = self._result.pressure
            ylabel = "p"
            title = "Pressure profile"
        fig = _create_line_animation_figure(
            self._result.x,
            self._result.t,
            arr,
            title=title,
            ylabel=ylabel,
        )
        self._anim_canvas = embed_animation_plot_in_tk(
            fig,
            self._anim_frame,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(
                arr, title, ylabel, duration
            ),
        )

    def _on_export_animation_mp4(
        self,
        field: np.ndarray,
        title: str,
        ylabel: str,
        duration_seconds: float,
    ) -> None:
        """Export the selected transient field through the shared MP4 path."""
        default_path = get_output_dir() / (f"{generate_output_basename(prefix='pipe_flow')}.mp4")
        filepath_str = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".mp4",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if not filepath_str:
            return

        filepath = Path(filepath_str)
        try:
            export_animated_figure_to_mp4(
                _create_line_animation_figure(
                    self._result.x,
                    self._result.t,
                    field,
                    title=title,
                    ylabel=ylabel,
                ),
                filepath,
                duration_seconds=duration_seconds,
            )
            messagebox.showinfo(
                "Animation export saved",
                f"Animation was saved to:\n{filepath}",
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

    def _build_geometry_tab(self, parent: ttk.Frame) -> None:
        fig = _create_geometry_figure(self._result)
        self._geometry_canvas = embed_plot_in_tk(fig, parent)

    def _build_pressure_tab(self, parent: ttk.Frame) -> None:
        if self._result.model_type == "steady":
            fig = _create_steady_pressure_figure(self._result)
        else:
            fig = create_contour_plot(
                self._result.x,
                self._result.t,
                self._result.pressure,
                title="Pressure space-time map",
                xlabel="x",
                ylabel="t",
            )
        self._pressure_canvas = embed_plot_in_tk(fig, parent)

    def _build_velocity_tab(self, parent: ttk.Frame) -> None:
        if self._result.model_type == "steady":
            fig = _create_steady_velocity_figure(self._result)
        else:
            fig = create_contour_plot(
                self._result.x,
                self._result.t,
                self._result.velocity,
                title="Velocity space-time map",
                xlabel="x",
                ylabel="t",
            )
        self._velocity_canvas = embed_plot_in_tk(fig, parent)

    def _build_quality_tab(self, parent: ttk.Frame) -> None:
        if self._result.model_type == "steady":
            fig = _create_steady_flow_diagnostics_figure(self._result)
        else:
            fig = _create_transient_flow_diagnostics_figure(self._result)
        self._quality_canvas = embed_plot_in_tk(fig, parent)
