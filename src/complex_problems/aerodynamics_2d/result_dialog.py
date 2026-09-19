"""Result dialog for 2D aerodynamics simulations."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np

from complex_problems.aerodynamics_2d.solver import Aerodynamics2DResult
from complex_problems.common.result_dialog_ui import (
    close_embedded_figures,
    reset_embedded_animation,
)
from config import generate_output_basename, get_env_from_schema, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import (
    create_image_animation_plot,
    create_solution_plot,
    export_animated_figure_to_mp4,
)
from plotting.animation_metadata import attach_animation_metadata
from utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.streamplot import StreamplotSet

logger = get_logger(__name__)


@dataclass(frozen=True)
class _FieldAnimationPayload:
    """Prepared field data shared by the embedded view and MP4 export."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frames: np.ndarray
    title: str
    symmetric_color_range: bool


@dataclass(frozen=True)
class _StreamlineAnimationPayload:
    """Prepared velocity data shared by the embedded view and MP4 export."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    speed: np.ndarray
    obstacle_mask: np.ndarray


@dataclass(frozen=True)
class _CenterlineAnimationPayload:
    """Prepared centerline histories shared by the embedded view and export."""

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    vorticity: np.ndarray
    mid: int


_AnimationPayload = (
    _FieldAnimationPayload | _StreamlineAnimationPayload | _CenterlineAnimationPayload
)


def _create_field_animation_figure(payload: _FieldAnimationPayload) -> Figure:
    """Build a physical-coordinate field animation from one prepared payload."""
    return create_image_animation_plot(
        payload.t,
        payload.frames,
        title=payload.title,
        xlabel="x",
        ylabel="y",
        x_coordinates=payload.x,
        y_coordinates=payload.y,
        symmetric_color_range=payload.symmetric_color_range,
    )


def _create_streamplot_figure(payload: _StreamlineAnimationPayload) -> Figure:
    """Build a streamline animation while keeping the obstacle contour static."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch

    speed_min = 0.0
    speed_max = float(np.max(payload.speed))
    if speed_max <= speed_min:
        speed_max = speed_min + 1.0
    norm = Normalize(vmin=speed_min, vmax=speed_max)

    fig, ax = plt.subplots()
    ax.contour(
        payload.x,
        payload.y,
        payload.obstacle_mask.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.5,
    )

    def _draw_streamlines(index: int) -> tuple[StreamplotSet, tuple[FancyArrowPatch, ...]]:
        u_frame = np.array(payload.u[index], copy=True)
        v_frame = np.array(payload.v[index], copy=True)
        u_frame[payload.obstacle_mask] = np.nan
        v_frame[payload.obstacle_mask] = np.nan
        existing_patch_ids = {id(patch) for patch in ax.patches}
        stream = ax.streamplot(
            payload.x,
            payload.y,
            u_frame,
            v_frame,
            color=payload.speed[index],
            cmap="viridis",
            norm=norm,
            density=1.4,
            linewidth=1.0,
        )
        arrow_patches = tuple(
            patch
            for patch in ax.patches
            if id(patch) not in existing_patch_ids and isinstance(patch, FancyArrowPatch)
        )
        return stream, arrow_patches

    stream, arrow_patches = _draw_streamlines(0)
    fig.colorbar(stream.lines, ax=ax, shrink=0.8, label="speed")
    ax.set_title(f"Streamlines and obstacle (t={payload.t[0]:.3g})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    def _update(index: int) -> None:
        nonlocal arrow_patches, stream
        i = max(0, min(index, len(payload.t) - 1))
        stream.lines.remove()
        for arrow_patch in arrow_patches:
            arrow_patch.remove()
        stream, arrow_patches = _draw_streamlines(i)
        ax.set_title(f"Streamlines and obstacle (t={payload.t[i]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(fig, update=_update, n_points=len(payload.t))


def _line_limits(values: np.ndarray) -> tuple[float, float]:
    """Return stable readable limits, including a fallback for constant histories."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        margin = max(abs(lower) * 0.1, 1.0)
        return lower - margin, upper + margin
    margin = 0.05 * (upper - lower)
    return lower - margin, upper + margin


def _create_centerline_figure(payload: _CenterlineAnimationPayload) -> Figure:
    """Build the two-curve centerline animation from all stored time frames."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    y_min, y_max = _line_limits(np.concatenate([payload.u.ravel(), payload.vorticity.ravel()]))
    (u_line,) = ax.plot(payload.x, payload.u[0], linewidth=2.0, label="u centerline")
    (vorticity_line,) = ax.plot(
        payload.x, payload.vorticity[0], linewidth=2.0, label="ω centerline"
    )
    ax.set_xlim(float(payload.x[0]), float(payload.x[-1]))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title(f"Centerline profiles (y index={payload.mid}, t={payload.t[0]:.3g})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    def _update(index: int) -> None:
        i = max(0, min(index, len(payload.t) - 1))
        u_line.set_ydata(payload.u[i])
        vorticity_line.set_ydata(payload.vorticity[i])
        ax.set_title(f"Centerline profiles (y index={payload.mid}, t={payload.t[i]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(fig, update=_update, n_points=len(payload.t))


def _create_animation_figure(payload: _AnimationPayload) -> Figure:
    """Build the exact figure used by both Tk display and MP4 export."""
    if isinstance(payload, _FieldAnimationPayload):
        return _create_field_animation_figure(payload)
    if isinstance(payload, _StreamlineAnimationPayload):
        return _create_streamplot_figure(payload)
    return _create_centerline_figure(payload)


def _create_field_payload(result: Aerodynamics2DResult, view: str) -> _FieldAnimationPayload:
    """Prepare one of the existing field histories for animated display."""
    if view == "vorticity":
        return _FieldAnimationPayload(
            result.t, result.x, result.y, result.vorticity, "Vorticity", True
        )
    if view == "pressure":
        return _FieldAnimationPayload(
            result.t, result.x, result.y, result.pressure, "Pressure", True
        )
    return _FieldAnimationPayload(
        result.t, result.x, result.y, result.speed, "Speed magnitude", False
    )


def _create_streamline_payload(result: Aerodynamics2DResult) -> _StreamlineAnimationPayload:
    """Prepare the complete sampled velocity history without solving again."""
    return _StreamlineAnimationPayload(
        result.t, result.x, result.y, result.u, result.v, result.speed, result.obstacle_mask
    )


def _create_centerline_payload(result: Aerodynamics2DResult) -> _CenterlineAnimationPayload:
    """Prepare both centerline histories from the sampled solver result."""
    mid = len(result.y) // 2
    return _CenterlineAnimationPayload(
        result.t, result.x, result.u[:, mid, :], result.vorticity[:, mid, :], mid
    )


class Aerodynamics2DResultDialog:
    """Result window for 2D aerodynamics."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: Aerodynamics2DResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("2D Aerodynamics Results")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))
        self._anim_canvas = None
        self._map_canvas = None
        self._coef_canvas = None
        self._stream_canvas = None
        self._profile_canvas = None
        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1420, height=930, max_width_ratio=0.96, resizable=True)
        self.win.minsize(1120, 740)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        close_embedded_figures(
            self,
            ("_anim_canvas", "_map_canvas", "_coef_canvas", "_stream_canvas", "_profile_canvas"),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        top = ttk.Frame(self.win, padding=pad)
        top.pack(fill=tk.BOTH, expand=True)
        mag = self._result.magnitudes
        info = (
            f"Re: {mag['reynolds']:.1f}   "
            f"Cd(mean tail): {mag['mean_cd_tail']:+.3e}   "
            f"Cl(rms): {mag['rms_cl']:.3e}   "
            f"max|div|: {mag['max_divergence_l2']:.3e}"
        )
        ttk.Label(top, text=info, style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))
        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True)
        tab_anim = ttk.Frame(nb)
        nb.add(tab_anim, text="  Animation  ")
        self._build_anim_tab(tab_anim)
        tab_map = ttk.Frame(nb)
        nb.add(tab_map, text="  Field Map  ")
        self._build_map_tab(tab_map)
        tab_coef = ttk.Frame(nb)
        nb.add(tab_coef, text="  Drag / Lift  ")
        self._build_coeff_tab(tab_coef)
        tab_stream = ttk.Frame(nb)
        nb.add(tab_stream, text="  Streamlines  ")
        self._build_stream_tab(tab_stream)
        tab_profile = ttk.Frame(nb)
        nb.add(tab_profile, text="  Centerline Profiles  ")
        self._build_profile_tab(tab_profile)
        btn_frame = ttk.Frame(self.win, padding=(pad, 0, pad, pad))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", style="Cancel.TButton", command=self._on_close).pack(
            side=tk.RIGHT
        )

    def _build_anim_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        self._view_var = tk.StringVar(value="speed")
        ttk.Label(ctrl, text="Display:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            ctrl,
            textvariable=self._view_var,
            values=("speed", "vorticity", "pressure"),
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
        self._anim_canvas = None
        payload = _create_field_payload(self._result, self._view_var.get())
        self._anim_canvas = embed_animation_plot_in_tk(
            _create_animation_figure(payload),
            self._anim_frame,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _build_map_tab(self, parent: ttk.Frame) -> None:
        payload = _FieldAnimationPayload(
            self._result.t,
            self._result.x,
            self._result.y,
            self._result.speed,
            "Speed magnitude",
            False,
        )
        self._map_canvas = embed_animation_plot_in_tk(
            _create_animation_figure(payload),
            parent,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _build_coeff_tab(self, parent: ttk.Frame) -> None:
        arr = np.vstack([self._result.drag_coeff, self._result.lift_coeff])
        fig = create_solution_plot(
            self._result.t,
            arr,
            title="Aerodynamic coefficients vs time",
            xlabel="t",
            ylabel="coefficient",
            selected_derivatives=[0, 1],
            labels=["Cd", "Cl"],
        )
        self._coef_canvas = embed_plot_in_tk(fig, parent)

    def _build_stream_tab(self, parent: ttk.Frame) -> None:
        payload = _create_streamline_payload(self._result)
        self._stream_canvas = embed_animation_plot_in_tk(
            _create_animation_figure(payload),
            parent,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _build_profile_tab(self, parent: ttk.Frame) -> None:
        payload = _create_centerline_payload(self._result)
        self._profile_canvas = embed_animation_plot_in_tk(
            _create_animation_figure(payload),
            parent,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _on_export_animation_mp4(self, payload: _AnimationPayload, duration_seconds: float) -> None:
        """Export an Aerodynamics animation through the shared MP4 path."""
        default_path = (
            get_output_dir() / f"{generate_output_basename(prefix='aerodynamics_2d')}.mp4"
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
        filepath = Path(filepath_str)
        try:
            export_animated_figure_to_mp4(
                _create_animation_figure(payload), filepath, duration_seconds=duration_seconds
            )
            messagebox.showinfo(
                "Animation export saved", f"Animation was saved to:\n{filepath}", parent=self.win
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
