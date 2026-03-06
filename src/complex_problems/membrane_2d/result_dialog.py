"""Result dialog for the 2D nonlinear membrane."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from complex_problems.membrane_2d.solver import Membrane2DResult
from config import get_env_from_schema
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import create_contour_plot, create_energy_evolution_plot, create_surface_plot
from utils import get_logger

logger = get_logger(__name__)


def _create_field_animation_figure(
    t: np.ndarray,
    frames: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
) -> "object":
    """Create an imshow figure compatible with embed_animation_plot_in_tk."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    v_abs = float(np.max(np.abs(frames)))
    if v_abs <= 0:
        v_abs = 1.0
    im = ax.imshow(
        frames[0],
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=-v_abs,
        vmax=v_abs,
    )
    ax.set_title(f"{title}  (t={t[0]:.3g})")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    def _update(idx: int) -> None:
        i = max(0, min(idx, len(t) - 1))
        im.set_data(frames[i])
        ax.set_title(f"{title}  (t={t[i]:.3g})")
        fig.canvas.draw_idle()

    fig._animation_update = _update
    fig._animation_n_points = len(t)
    fig._animation_initial_index = 0
    return fig


class Membrane2DResultDialog:
    """Result window for membrane simulations."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: Membrane2DResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Results - 2D Nonlinear Membrane")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))

        self._anim_canvas = None
        self._st_canvas = None
        self._surface_canvas = None
        self._energy_canvas = None
        self._spec_canvas = None

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1400, height=900, max_width_ratio=0.96, resizable=True)
        self.win.minsize(1100, 700)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        import matplotlib.pyplot as plt

        for attr in (
            "_anim_canvas",
            "_st_canvas",
            "_surface_canvas",
            "_energy_canvas",
            "_spec_canvas",
        ):
            canvas = getattr(self, attr, None)
            if canvas is not None and hasattr(canvas, "figure"):
                try:
                    plt.close(canvas.figure)
                except Exception:
                    pass
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
        notebook.add(tab_st, text="  Space-Time  ")
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
        ttk.Label(ctrl, text="Field:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        self._anim_field_var = tk.StringVar(value="Displacement")
        combo = ttk.Combobox(
            ctrl,
            textvariable=self._anim_field_var,
            values=("Displacement", "Velocity"),
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
        import matplotlib.pyplot as plt

        if self._anim_canvas is not None and hasattr(self._anim_canvas, "figure"):
            try:
                plt.close(self._anim_canvas.figure)
            except Exception:
                pass
        for w in self._anim_frame.winfo_children():
            w.destroy()

        if self._anim_field_var.get() == "Velocity":
            frames = self._result.velocity
            title = "Membrane velocity field"
            cmap = "coolwarm"
        else:
            frames = self._result.displacement
            title = "Membrane displacement field"
            cmap = "viridis"
        fig = _create_field_animation_figure(self._result.t, frames, title=title, cmap=cmap)
        self._anim_canvas = embed_animation_plot_in_tk(fig, self._anim_frame)

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
