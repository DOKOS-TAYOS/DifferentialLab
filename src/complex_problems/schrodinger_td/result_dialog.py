"""Result dialog for time-dependent Schrodinger simulations."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from complex_problems.common.result_dialog_ui import (
    close_embedded_figures,
    reset_embedded_animation,
)
from complex_problems.schrodinger_td.solver import SchrodingerTDResult
from config import generate_output_basename, get_env_from_schema, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import (
    create_contour_plot,
    create_image_animation_plot,
    create_solution_plot,
    create_surface_animation_plot,
    export_animated_figure_to_mp4,
)
from plotting.animation_metadata import attach_animation_metadata
from utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)


@dataclass(frozen=True)
class _SchrodingerAnimationViewPayload:
    """Prepared data shared by a Schrodinger animation and its MP4 export."""

    kind: Literal["image", "surface"]
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frames: np.ndarray
    title: str
    cmap: str
    xlabel: str
    ylabel: str
    zlabel: str = ""
    colorbar_label: str = ""
    symmetric_z_range: bool = True


@dataclass(frozen=True)
class _SchrodingerLineAnimationPayload:
    """Prepared 1D data shared by the embedded view and MP4 export."""

    x: np.ndarray
    t: np.ndarray
    frames: np.ndarray
    title: str
    ylabel: str


@dataclass(frozen=True)
class _SchrodingerMainImageAnimationPayload:
    """Prepared 2D main-tab data shared by the embedded view and MP4 export."""

    t: np.ndarray
    frames: np.ndarray
    title: str
    symmetric: bool


_SchrodingerAnimationPayload = (
    _SchrodingerAnimationViewPayload
    | _SchrodingerLineAnimationPayload
    | _SchrodingerMainImageAnimationPayload
)


def _create_animation_figure(payload: _SchrodingerAnimationPayload) -> Figure:
    """Build the figure used both by Tk and by MP4 export."""
    if isinstance(payload, _SchrodingerLineAnimationPayload):
        return _create_line_anim_figure(
            payload.x,
            payload.t,
            payload.frames,
            title=payload.title,
            ylabel=payload.ylabel,
        )
    if isinstance(payload, _SchrodingerMainImageAnimationPayload):
        return _create_image_anim_figure(
            payload.t,
            payload.frames,
            title=payload.title,
            symmetric=payload.symmetric,
        )
    if payload.kind == "surface":
        return create_surface_animation_plot(
            payload.t,
            payload.x,
            payload.y,
            payload.frames,
            title=payload.title,
            cmap=payload.cmap,
            xlabel=payload.xlabel,
            ylabel=payload.ylabel,
            zlabel=payload.zlabel,
            colorbar_label=payload.colorbar_label,
            symmetric_z_range=payload.symmetric_z_range,
        )
    return create_image_animation_plot(
        payload.t,
        payload.frames,
        title=payload.title,
        xlabel=payload.xlabel,
        ylabel=payload.ylabel,
        cmap=payload.cmap,
        x_coordinates=payload.x,
        y_coordinates=payload.y,
        symmetric_color_range=payload.symmetric_z_range,
    )


def _create_line_anim_figure(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    y_abs = float(np.max(np.abs(y)))
    y_lim = 1.1 * (y_abs if y_abs > 0 else 1.0)
    (line,) = ax.plot(x, y[0], linewidth=2.0)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (t={t[0]:.3g})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    def _update(idx: int) -> None:
        i = max(0, min(idx, len(t) - 1))
        line.set_ydata(y[i])
        ax.set_title(f"{title} (t={t[i]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(fig, update=_update, n_points=len(t))


def _create_image_anim_figure(
    t: np.ndarray,
    frames: np.ndarray,
    *,
    title: str,
    symmetric: bool = False,
) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if symmetric:
        v = float(np.max(np.abs(frames)))
        if v <= 0:
            v = 1.0
        vmin, vmax = -v, v
        cmap = "coolwarm"
    else:
        vmin, vmax = float(np.min(frames)), float(np.max(frames))
        if abs(vmax - vmin) < 1e-15:
            vmax = vmin + 1.0
        cmap = "viridis"
    im = ax.imshow(frames[0], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title} (t={t[0]:.3g})")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    def _update(idx: int) -> None:
        i = max(0, min(idx, len(t) - 1))
        im.set_data(frames[i])
        ax.set_title(f"{title} (t={t[i]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(fig, update=_update, n_points=len(t))


class SchrodingerTDResultDialog:
    """Result window for TDSE 1D/2D."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: SchrodingerTDResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Schrodinger Time Evolution Results")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))

        self._anim_canvas = None
        self._st_canvas = None
        self._spec_canvas = None
        self._inv_canvas = None
        self._extra_canvas = None

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1400, height=920, max_width_ratio=0.96, resizable=True)
        self.win.minsize(1120, 720)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        close_embedded_figures(
            self,
            ("_anim_canvas", "_st_canvas", "_spec_canvas", "_inv_canvas", "_extra_canvas"),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        top = ttk.Frame(self.win, padding=pad)
        top.pack(fill=tk.BOTH, expand=True)

        drift_text = ", ".join(f"{k}: {v:+.3e}" for k, v in self._result.magnitudes.items())
        ttk.Label(top, text=drift_text, style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))

        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_anim = ttk.Frame(nb)
        nb.add(tab_anim, text="  Animation  ")
        self._build_animation_tab(tab_anim)

        tab_st = ttk.Frame(nb)
        nb.add(tab_st, text="  Density Maps  ")
        self._build_space_tab(tab_st)

        tab_spec = ttk.Frame(nb)
        nb.add(tab_spec, text="  Spectrum  ")
        if self._result.dimension == 1:
            self._build_spectrum_tab(tab_spec)
        else:
            self._spectrum_tab = tab_spec
            self._spectrum_tab_initialized = False
            nb.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        tab_inv = ttk.Frame(nb)
        nb.add(tab_inv, text="  Expectations  ")
        self._build_invariants_tab(tab_inv)

        tab_extra = ttk.Frame(nb)
        nb.add(tab_extra, text="  Potential / Surface  ")
        self._build_extra_tab(tab_extra)

        btn_frame = ttk.Frame(self.win, padding=(pad, 0, pad, pad))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", style="Cancel.TButton", command=self._on_close).pack(
            side=tk.RIGHT
        )

    def _on_notebook_tab_changed(self, event: tk.Event[tk.Misc]) -> None:
        """Initialize the deferred 2D Spectrum tab on its first selection."""
        if self._spectrum_tab_initialized:
            return
        notebook = cast(ttk.Notebook, event.widget)
        selected_tab = notebook.nametowidget(notebook.select())
        if selected_tab is self._spectrum_tab:
            self._build_spectrum_tab(self._spectrum_tab)
            self._spectrum_tab_initialized = True

    def _build_animation_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        if self._result.dimension == 1:
            options = ("Density", "Real", "Imag")
        else:
            options = ("Density", "Phase")
        self._anim_view_var = tk.StringVar(value=options[0])
        ttk.Label(ctrl, text="Display:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            ctrl,
            textvariable=self._anim_view_var,
            values=list(options),
            state="readonly",
            width=12,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._update_anim())
        self._anim_frame = ttk.Frame(parent)
        self._anim_frame.pack(fill=tk.BOTH, expand=True)
        self._update_anim()

    def _update_anim(self) -> None:
        reset_embedded_animation(self._anim_frame, self._anim_canvas)

        payload = self._get_main_animation_payload()
        self._anim_canvas = embed_animation_plot_in_tk(
            _create_animation_figure(payload),
            self._anim_frame,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _get_main_animation_payload(self) -> _SchrodingerAnimationPayload:
        """Return the currently selected main-animation representation."""
        view = self._anim_view_var.get()
        result = self._result
        if result.dimension == 1:
            if view == "Real":
                frames, ylabel = np.real(result.psi), "Re(ψ)"
            elif view == "Imag":
                frames, ylabel = np.imag(result.psi), "Im(ψ)"
            else:
                frames, ylabel = result.magnitude, "|ψ|²"
            return _SchrodingerLineAnimationPayload(
                x=result.x,
                t=result.t,
                frames=frames,
                title="TDSE 1D profile",
                ylabel=ylabel,
            )

        if view == "Phase":
            frames, symmetric, title = result.phase, True, "TDSE 2D phase"
        else:
            frames, symmetric, title = result.magnitude, False, "TDSE 2D density"
        return _SchrodingerMainImageAnimationPayload(
            t=result.t,
            frames=frames,
            title=title,
            symmetric=symmetric,
        )

    def _build_space_tab(self, parent: ttk.Frame) -> None:
        r = self._result
        if r.dimension == 1:
            z = r.magnitude
            fig = create_contour_plot(
                r.x,
                r.t,
                z,
                title="|ψ(x,t)|²",
                xlabel="x",
                ylabel="t",
            )
        else:
            center_y = r.magnitude.shape[1] // 2
            z = r.magnitude[:, center_y, :]
            fig = create_contour_plot(
                r.x,
                r.t,
                z,
                title=f"Center-line density map (y index={center_y})",
                xlabel="x",
                ylabel="t",
            )
        self._st_canvas = embed_plot_in_tk(fig, parent)

    def _build_spectrum_tab(self, parent: ttk.Frame) -> None:
        r = self._result
        if r.dimension == 1:
            fig = create_solution_plot(
                r.kx,
                np.atleast_2d(r.spectrum_power),
                title="Final momentum spectrum",
                xlabel="k",
                ylabel="Power",
                selected_derivatives=[0],
                labels=["|ψ(k)|²"],
            )
            self._spec_canvas = embed_plot_in_tk(fig, parent)
        else:
            payload = self._get_spectrum_animation_payload()
            self._spec_canvas = embed_animation_plot_in_tk(
                _create_animation_figure(payload),
                parent,
                on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
            )

    def _build_invariants_tab(self, parent: ttk.Frame) -> None:
        keys = list(self._result.invariants.keys())
        arr = np.vstack([self._result.invariants[k] for k in keys])
        fig = create_solution_plot(
            self._result.t,
            arr,
            title="Expectation values and invariants",
            xlabel="t",
            ylabel="value",
            selected_derivatives=list(range(len(keys))),
            labels=keys,
        )
        self._inv_canvas = embed_plot_in_tk(fig, parent)

    def _build_extra_tab(self, parent: ttk.Frame) -> None:
        r = self._result
        if r.dimension == 1:
            fig = create_solution_plot(
                r.x,
                np.atleast_2d(r.potential),
                title="Potential V(x)",
                xlabel="x",
                ylabel="V",
                selected_derivatives=[0],
                labels=["V"],
            )
            self._extra_canvas = embed_plot_in_tk(fig, parent)
        else:
            payload = self._get_density_surface_animation_payload()
            self._extra_canvas = embed_animation_plot_in_tk(
                _create_animation_figure(payload),
                parent,
                on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
            )

    def _get_spectrum_animation_payload(self) -> _SchrodingerAnimationViewPayload:
        """Return the prepared 2D k-space history for display or export."""
        result = self._result
        if result.dimension != 2 or result.ky is None:
            raise ValueError("2D spectrum animation requires a 2D Schrodinger result.")
        return _SchrodingerAnimationViewPayload(
            kind="image",
            t=result.t,
            x=result.kx,
            y=result.ky,
            frames=result.spectrum_power_history,
            title="2D k-space power",
            cmap="magma",
            xlabel="kₓ",
            ylabel="kᵧ",
            symmetric_z_range=False,
        )

    def _get_density_surface_animation_payload(self) -> _SchrodingerAnimationViewPayload:
        """Return the prepared 2D density history for display or export."""
        result = self._result
        if result.dimension != 2 or result.y is None:
            raise ValueError("2D density-surface animation requires a 2D Schrodinger result.")
        return _SchrodingerAnimationViewPayload(
            kind="surface",
            t=result.t,
            x=result.x,
            y=result.y,
            frames=result.magnitude,
            title="2D density surface",
            cmap="viridis",
            xlabel="x",
            ylabel="y",
            zlabel="|ψ|²",
            colorbar_label="|ψ|²",
            symmetric_z_range=False,
        )

    def _on_export_animation_mp4(
        self,
        payload: _SchrodingerAnimationPayload,
        duration_seconds: float,
    ) -> None:
        """Export a selected 2D animation through the shared MP4 path."""
        default_path = get_output_dir() / (
            f"{generate_output_basename(prefix='schrodinger_td')}.mp4"
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
