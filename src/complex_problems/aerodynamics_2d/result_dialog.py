"""Result dialog for 2D aerodynamics simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from complex_problems.aerodynamics_2d.solver import Aerodynamics2DResult
from config import get_env_from_schema
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import create_contour_plot, create_solution_plot


def _create_field_animation_figure(
    t: np.ndarray,
    frames: np.ndarray,
    *,
    title: str,
    symmetric: bool = False,
) -> "object":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if symmetric:
        vmax = float(np.max(np.abs(frames)))
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmin = float(np.min(frames))
        vmax = float(np.max(frames))
        if abs(vmax - vmin) < 1e-12:
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

    fig._animation_update = _update
    fig._animation_n_points = len(t)
    fig._animation_initial_index = 0
    return fig


def _create_streamplot_figure(result: Aerodynamics2DResult) -> "object":
    import matplotlib.pyplot as plt

    x = result.x
    y = result.y
    u = result.u[-1].copy()
    v = result.v[-1].copy()
    speed = result.speed[-1]
    mask = result.obstacle_mask
    u[mask] = np.nan
    v[mask] = np.nan

    fig, ax = plt.subplots()
    ax.streamplot(x, y, u, v, color=speed, cmap="viridis", density=1.4, linewidth=1.0)
    ax.contour(x, y, mask.astype(float), levels=[0.5], colors="black", linewidths=1.5)
    ax.set_title("Final streamlines and obstacle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


class Aerodynamics2DResultDialog:
    """Result window for 2D aerodynamics."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: Aerodynamics2DResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Results - Aerodynamics 2D")
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
        import matplotlib.pyplot as plt

        for attr in (
            "_anim_canvas",
            "_map_canvas",
            "_coef_canvas",
            "_stream_canvas",
            "_profile_canvas",
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
        nb.add(tab_map, text="  Final Map  ")
        self._build_map_tab(tab_map)

        tab_coef = ttk.Frame(nb)
        nb.add(tab_coef, text="  Coefficients  ")
        self._build_coeff_tab(tab_coef)

        tab_stream = ttk.Frame(nb)
        nb.add(tab_stream, text="  Streamlines  ")
        self._build_stream_tab(tab_stream)

        tab_profile = ttk.Frame(nb)
        nb.add(tab_profile, text="  Centerline  ")
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
        ttk.Label(ctrl, text="Field:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
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
        import matplotlib.pyplot as plt

        if self._anim_canvas is not None and hasattr(self._anim_canvas, "figure"):
            try:
                plt.close(self._anim_canvas.figure)
            except Exception:
                pass
        for w in self._anim_frame.winfo_children():
            w.destroy()

        view = self._view_var.get()
        if view == "vorticity":
            frames = self._result.vorticity
            title = "Vorticity"
            symmetric = True
        elif view == "pressure":
            frames = self._result.pressure
            title = "Pressure"
            symmetric = True
        else:
            frames = self._result.speed
            title = "Speed magnitude"
            symmetric = False

        fig = _create_field_animation_figure(
            self._result.t,
            frames,
            title=title,
            symmetric=symmetric,
        )
        self._anim_canvas = embed_animation_plot_in_tk(fig, self._anim_frame)

    def _build_map_tab(self, parent: ttk.Frame) -> None:
        fig = create_contour_plot(
            self._result.x,
            self._result.y,
            self._result.speed[-1],
            title="Final speed magnitude",
            xlabel="x",
            ylabel="y",
        )
        self._map_canvas = embed_plot_in_tk(fig, parent)

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
        fig = _create_streamplot_figure(self._result)
        self._stream_canvas = embed_plot_in_tk(fig, parent)

    def _build_profile_tab(self, parent: ttk.Frame) -> None:
        mid = len(self._result.y) // 2
        arr = np.vstack([self._result.u[-1, mid], self._result.vorticity[-1, mid]])
        fig = create_solution_plot(
            self._result.x,
            arr,
            title=f"Final centerline (y index={mid})",
            xlabel="x",
            ylabel="value",
            selected_derivatives=[0, 1],
            labels=["u centerline", "ω centerline"],
        )
        self._profile_canvas = embed_plot_in_tk(fig, parent)
