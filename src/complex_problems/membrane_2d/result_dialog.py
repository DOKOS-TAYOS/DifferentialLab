"""Result dialog for the 2D nonlinear membrane."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Literal

import numpy as np

from complex_problems.common.result_dialog_ui import (
    close_embedded_figures,
    reset_embedded_animation,
)
from complex_problems.membrane_2d.model import compute_fft_power_history_2d
from complex_problems.membrane_2d.solver import Membrane2DResult
from config import generate_output_basename, get_env_from_schema, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import (
    create_contour_plot,
    create_energy_evolution_plot,
    create_image_animation_plot,
    create_surface_animation_plot,
    create_surface_plot,
    export_animated_figure_to_mp4,
)
from utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)


@dataclass(frozen=True)
class _MembraneAnimationViewPayload:
    """Prepared data and metadata shared by a membrane view and its MP4 export."""

    kind: Literal["image", "surface"]
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frames: np.ndarray
    title: str
    cmap: str
    xlabel: str
    ylabel: str
    symmetric_color_range: bool


def _create_animation_figure(payload: _MembraneAnimationViewPayload) -> Figure:
    """Build the embedded or export figure from one prepared animation payload."""
    if payload.kind == "surface":
        return create_surface_animation_plot(
            payload.t,
            payload.x,
            payload.y,
            payload.frames,
            title=payload.title,
            cmap=payload.cmap,
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
        symmetric_color_range=payload.symmetric_color_range,
    )


class Membrane2DResultDialog:
    """Result window for membrane simulations."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: Membrane2DResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("2D Nonlinear Membrane Results")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))

        self._anim_canvas = None
        self._st_canvas = None
        self._surface_canvas = None
        self._energy_canvas = None
        self._spec_canvas = None
        self._spectrum_power_history = result.spectrum_power_history

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1400, height=900, max_width_ratio=0.96, resizable=True)
        self.win.minsize(1100, 700)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        close_embedded_figures(
            self,
            (
                "_anim_canvas",
                "_st_canvas",
                "_surface_canvas",
                "_energy_canvas",
                "_spec_canvas",
            ),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        top = ttk.Frame(self.win, padding=pad)
        top.pack(fill=tk.BOTH, expand=True)

        info = ttk.Frame(top)
        info.pack(fill=tk.X, pady=(0, pad))
        drift = self._result.magnitudes.get("energy_drift_rel", 0.0)
        max_u = self._result.magnitudes.get("max_displacement", 0.0)
        ttk.Label(
            info,
            text=f"Energy drift: {drift:+.3e}   |   Max |u|: {max_u:.4g}",
            style="Small.TLabel",
        ).pack(side=tk.LEFT)

        notebook = ttk.Notebook(top)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_anim = ttk.Frame(notebook)
        notebook.add(tab_anim, text="  Animation  ")
        self._build_animation_tab(tab_anim)

        tab_st = ttk.Frame(notebook)
        notebook.add(tab_st, text="  Centerline Map  ")
        self._build_space_time_tab(tab_st)

        tab_surface = ttk.Frame(notebook)
        notebook.add(tab_surface, text="  Surface 3D  ")
        self._build_surface_tab(tab_surface)

        tab_energy = ttk.Frame(notebook)
        notebook.add(tab_energy, text="  Energy  ")
        self._build_energy_tab(tab_energy)

        tab_spec = ttk.Frame(notebook)
        notebook.add(tab_spec, text="  Spectrum  ")
        self._build_spectrum_tab(tab_spec)

        btn_frame = ttk.Frame(self.win, padding=(pad, 0, pad, pad))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", style="Cancel.TButton", command=self._on_close).pack(
            side=tk.RIGHT
        )

    def _build_animation_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(ctrl, text="Display:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        self._anim_field_var = tk.StringVar(value="2D Field")
        combo = ttk.Combobox(
            ctrl,
            textvariable=self._anim_field_var,
            values=("2D Field", "2D Velocity", "3D Surface", "Spectrum"),
            state="readonly",
            width=14,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._update_animation())

        self._anim_frame = ttk.Frame(parent)
        self._anim_frame.pack(fill=tk.BOTH, expand=True)
        self._update_animation()

    def _update_animation(self) -> None:
        """Replace the embedded animation with the selected membrane representation."""
        reset_embedded_animation(self._anim_frame, self._anim_canvas)
        self._anim_canvas = None
        payload = self._get_animation_view_payload()
        fig = _create_animation_figure(payload)
        self._anim_canvas = embed_animation_plot_in_tk(
            fig,
            self._anim_frame,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _get_animation_view_payload(self) -> _MembraneAnimationViewPayload:
        """Return the exact prepared representation selected in the animation tab."""
        result = self._result
        x = np.arange(result.displacement.shape[2])
        y = np.arange(result.displacement.shape[1])
        mode = self._anim_field_var.get()

        if mode == "3D Surface":
            return _MembraneAnimationViewPayload(
                kind="surface",
                t=result.t,
                x=x,
                y=y,
                frames=result.displacement,
                title="Membrane displacement surface",
                cmap="viridis",
                xlabel="x index",
                ylabel="y index",
                symmetric_color_range=True,
            )
        if mode == "Spectrum":
            spectrum_history = self._spectrum_power_history
            if spectrum_history is None:
                _kx, _ky, spectrum_history = compute_fft_power_history_2d(result.displacement)
                self._spectrum_power_history = spectrum_history
            return _MembraneAnimationViewPayload(
                kind="image",
                t=result.t,
                x=result.kx,
                y=result.ky,
                frames=spectrum_history,
                title="2D FFT power spectrum",
                cmap="magma",
                xlabel="kₓ",
                ylabel="kᵧ",
                symmetric_color_range=False,
            )
        if mode == "2D Velocity":
            return _MembraneAnimationViewPayload(
                kind="image",
                t=result.t,
                x=x,
                y=y,
                frames=result.velocity,
                title="Membrane velocity field",
                cmap="coolwarm",
                xlabel="x index",
                ylabel="y index",
                symmetric_color_range=True,
            )
        return _MembraneAnimationViewPayload(
            kind="image",
            t=result.t,
            x=x,
            y=y,
            frames=result.displacement,
            title="Membrane displacement field",
            cmap="viridis",
            xlabel="x index",
            ylabel="y index",
            symmetric_color_range=True,
        )

    def _on_export_animation_mp4(
        self,
        payload: _MembraneAnimationViewPayload,
        duration_seconds: float,
    ) -> None:
        """Export the selected animation through the shared MP4 infrastructure."""
        default_path = get_output_dir() / f"{generate_output_basename(prefix='membrane')}.mp4"
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

    def _build_space_time_tab(self, parent: ttk.Frame) -> None:
        ny, _nx = self._result.displacement.shape[1:]
        center_y = ny // 2
        z = self._result.displacement[:, center_y, :]
        fig = create_contour_plot(
            np.arange(z.shape[1]),
            self._result.t,
            z,
            title=f"Center-line space-time map (y={center_y})",
            xlabel="x index",
            ylabel="t",
        )
        self._st_canvas = embed_plot_in_tk(fig, parent)

    def _build_surface_tab(self, parent: ttk.Frame) -> None:
        final_u = self._result.displacement[-1]
        fig = create_surface_plot(
            np.arange(final_u.shape[1]),
            np.arange(final_u.shape[0]),
            final_u,
            title="Final membrane shape",
            xlabel="x index",
            ylabel="y index",
            zlabel="u",
        )
        self._surface_canvas = embed_plot_in_tk(fig, parent)

    def _build_energy_tab(self, parent: ttk.Frame) -> None:
        fig = create_energy_evolution_plot(
            self._result.t,
            self._result.kinetic_energy,
            self._result.potential_energy,
            self._result.total_energy,
            title="Energy evolution",
            xlabel="t",
        )
        self._energy_canvas = embed_plot_in_tk(fig, parent)

    def _build_spectrum_tab(self, parent: ttk.Frame) -> None:
        fig = create_contour_plot(
            self._result.kx,
            self._result.ky,
            self._result.spectrum_power,
            title="2D FFT power spectrum (final field)",
            xlabel="kₓ",
            ylabel="kᵧ",
        )
        self._spec_canvas = embed_plot_in_tk(fig, parent)
