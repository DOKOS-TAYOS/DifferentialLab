"""Result dialog for nonlinear wave simulations (NLSE/KdV)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from complex_problems.nonlinear_waves.solver import NonlinearWavesResult
from config import get_env_from_schema
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import create_contour_plot, create_solution_plot


def _create_line_animation_figure(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> "object":
    """Create a line animation figure compatible with embed_animation_plot_in_tk."""
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


class NonlinearWavesResultDialog:
    """Result window for nonlinear waves."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: NonlinearWavesResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Results - Nonlinear Waves")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))

        self._anim_canvas = None
        self._st_canvas = None
        self._phase_canvas = None
        self._spec_canvas = None
        self._inv_canvas = None

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1300, height=880, max_width_ratio=0.95, resizable=True)
        self.win.minsize(1100, 700)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        import matplotlib.pyplot as plt

        for attr in ("_anim_canvas", "_st_canvas", "_phase_canvas", "_spec_canvas", "_inv_canvas"):
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
        nb.add(tab_anim, text="  Profile Animation  ")
        self._build_anim_tab(tab_anim)

        tab_st = ttk.Frame(nb)
        nb.add(tab_st, text="  Space-Time  ")
        self._build_spacetime_tab(tab_st)

        if self._result.phase is not None:
            tab_phase = ttk.Frame(nb)
            nb.add(tab_phase, text="  Phase  ")
            self._build_phase_tab(tab_phase)

        tab_spec = ttk.Frame(nb)
        nb.add(tab_spec, text="  Spectrum  ")
        self._build_spectrum_tab(tab_spec)

        tab_inv = ttk.Frame(nb)
        nb.add(tab_inv, text="  Invariants  ")
        self._build_invariants_tab(tab_inv)

        btn_frame = ttk.Frame(self.win, padding=(pad, 0, pad, pad))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", style="Cancel.TButton", command=self._on_close).pack(
            side=tk.RIGHT
        )

    def _build_anim_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        options = ["Field"] if self._result.model_type == "kdv" else ["Intensity", "Real", "Imag"]
        self._anim_view_var = tk.StringVar(value=options[0])
        ttk.Label(ctrl, text="View:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            ctrl,
            textvariable=self._anim_view_var,
            values=options,
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

        if self._result.model_type == "kdv":
            y = self._result.field.real
            title = "KdV profile"
            ylabel = "u"
        else:
            view = self._anim_view_var.get()
            if view == "Real":
                y = self._result.field.real
                ylabel = "Re(ψ)"
            elif view == "Imag":
                y = self._result.field.imag
                ylabel = "Im(ψ)"
            else:
                y = self._result.magnitude
                ylabel = "|ψ|²"
            title = f"NLSE profile - {view}"

        fig = _create_line_animation_figure(
            self._result.x,
            self._result.t,
            y,
            title=title,
            ylabel=ylabel,
        )
        self._anim_canvas = embed_animation_plot_in_tk(fig, self._anim_frame)

    def _build_spacetime_tab(self, parent: ttk.Frame) -> None:
        ylabel = "t"
        title = "Space-time map"
        fig = create_contour_plot(
            self._result.x,
            self._result.t,
            self._result.magnitude,
            title=title,
            xlabel="x",
            ylabel=ylabel,
        )
        self._st_canvas = embed_plot_in_tk(fig, parent)

    def _build_phase_tab(self, parent: ttk.Frame) -> None:
        assert self._result.phase is not None
        fig = create_contour_plot(
            self._result.x,
            self._result.t,
            self._result.phase,
            title="NLSE phase map",
            xlabel="x",
            ylabel="t",
        )
        self._phase_canvas = embed_plot_in_tk(fig, parent)

    def _build_spectrum_tab(self, parent: ttk.Frame) -> None:
        fig = create_solution_plot(
            self._result.k,
            np.atleast_2d(self._result.spectrum_power),
            title="Final spectrum",
            xlabel="k",
            ylabel="Power",
            selected_derivatives=[0],
            labels=["|F(k)|^2"],
        )
        self._spec_canvas = embed_plot_in_tk(fig, parent)

    def _build_invariants_tab(self, parent: ttk.Frame) -> None:
        keys = list(self._result.invariants.keys())
        arr = np.vstack([self._result.invariants[k] for k in keys])
        fig = create_solution_plot(
            self._result.t,
            arr,
            title="Invariants vs time",
            xlabel="t",
            ylabel="value",
            selected_derivatives=list(range(len(keys))),
            labels=keys,
        )
        self._inv_canvas = embed_plot_in_tk(fig, parent)
