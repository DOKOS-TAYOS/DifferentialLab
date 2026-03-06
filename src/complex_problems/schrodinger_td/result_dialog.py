"""Result dialog for time-dependent Schrodinger simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from complex_problems.schrodinger_td.solver import SchrodingerTDResult
from config import get_env_from_schema
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import create_contour_plot, create_solution_plot, create_surface_plot


def _create_line_anim_figure(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> "object":
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

    fig._animation_update = _update
    fig._animation_n_points = len(t)
    fig._animation_initial_index = 0
    return fig


def _create_image_anim_figure(
    t: np.ndarray,
    frames: np.ndarray,
    *,
    title: str,
    symmetric: bool = False,
) -> "object":
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

    fig._animation_update = _update
    fig._animation_n_points = len(t)
    fig._animation_initial_index = 0
    return fig


class SchrodingerTDResultDialog:
    """Result window for TDSE 1D/2D."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: SchrodingerTDResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Results - Schrodinger TD")
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
        import matplotlib.pyplot as plt

        for attr in ("_anim_canvas", "_st_canvas", "_spec_canvas", "_inv_canvas", "_extra_canvas"):
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

        drift_text = ", ".join(f"{k}: {v:+.3e}" for k, v in self._result.magnitudes.items())
        ttk.Label(top, text=drift_text, style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))

        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_anim = ttk.Frame(nb)
        nb.add(tab_anim, text="  Animation  ")
        self._build_animation_tab(tab_anim)

        tab_st = ttk.Frame(nb)
        nb.add(tab_st, text="  Space-Time / Density  ")
        self._build_space_tab(tab_st)

        tab_spec = ttk.Frame(nb)
        nb.add(tab_spec, text="  Spectrum  ")
        self._build_spectrum_tab(tab_spec)

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

    def _build_animation_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        if self._result.dimension == 1:
            options = ("Density", "Real", "Imag")
        else:
            options = ("Density", "Phase")
        self._anim_view_var = tk.StringVar(value=options[0])
        ttk.Label(ctrl, text="View:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
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
        import matplotlib.pyplot as plt

        if self._anim_canvas is not None and hasattr(self._anim_canvas, "figure"):
            try:
                plt.close(self._anim_canvas.figure)
            except Exception:
                pass
        for w in self._anim_frame.winfo_children():
            w.destroy()

        view = self._anim_view_var.get()
        r = self._result
        if r.dimension == 1:
            if view == "Real":
                y = r.psi.real
                ylabel = "Re(ψ)"
            elif view == "Imag":
                y = r.psi.imag
                ylabel = "Im(ψ)"
            else:
                y = r.magnitude
                ylabel = "|ψ|²"
            fig = _create_line_anim_figure(r.x, r.t, y, title="TDSE 1D profile", ylabel=ylabel)
        else:
            if view == "Phase":
                frames = r.phase
                symmetric = True
                title = "TDSE 2D phase"
            else:
                frames = r.magnitude
                symmetric = False
                title = "TDSE 2D density"
            fig = _create_image_anim_figure(r.t, frames, title=title, symmetric=symmetric)
        self._anim_canvas = embed_animation_plot_in_tk(fig, self._anim_frame)

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
        else:
            assert r.ky is not None
            fig = create_contour_plot(
                r.kx,
                r.ky,
                r.spectrum_power,
                title="Final 2D k-space power",
                xlabel="kₓ",
                ylabel="kᵧ",
            )
        self._spec_canvas = embed_plot_in_tk(fig, parent)

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
        else:
            fig = create_surface_plot(
                np.arange(r.magnitude.shape[2]),
                np.arange(r.magnitude.shape[1]),
                r.magnitude[-1],
                title="Final density surface",
                xlabel="x index",
                ylabel="y index",
                zlabel="|ψ|²",
            )
        self._extra_canvas = embed_plot_in_tk(fig, parent)
