"""Result dialog for pipe-flow simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from complex_problems.pipe_flow.solver import PipeFlowResult
from config import get_env_from_schema
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import create_contour_plot, create_solution_plot


def _create_line_animation_figure(
    x: np.ndarray,
    t: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> "object":
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

    fig._animation_update = _update
    fig._animation_n_points = len(t)
    fig._animation_initial_index = 0
    return fig


class PipeFlowResultDialog:
    """Result window for steady/transient pipe flow."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: PipeFlowResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Results - Pipe Flow")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))

        self._anim_canvas = None
        self._geometry_canvas = None
        self._pressure_canvas = None
        self._velocity_canvas = None
        self._quality_canvas = None

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1360, height=900, max_width_ratio=0.95, resizable=True)
        self.win.minsize(1060, 700)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        import matplotlib.pyplot as plt

        for attr in (
            "_anim_canvas",
            "_geometry_canvas",
            "_pressure_canvas",
            "_velocity_canvas",
            "_quality_canvas",
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

        info = ", ".join(f"{k}: {v:.4g}" for k, v in self._result.magnitudes.items())
        ttk.Label(top, text=info, style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))

        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True)

        if self._result.model_type == "transient":
            tab_anim = ttk.Frame(nb)
            nb.add(tab_anim, text="  Animation  ")
            self._build_anim_tab(tab_anim)

        tab_geom = ttk.Frame(nb)
        nb.add(tab_geom, text="  Geometry  ")
        self._build_geometry_tab(tab_geom)

        tab_p = ttk.Frame(nb)
        nb.add(tab_p, text="  Pressure  ")
        self._build_pressure_tab(tab_p)

        tab_u = ttk.Frame(nb)
        nb.add(tab_u, text="  Velocity  ")
        self._build_velocity_tab(tab_u)

        tab_q = ttk.Frame(nb)
        nb.add(tab_q, text="  Flow / Quality  ")
        self._build_quality_tab(tab_q)

        btn_frame = ttk.Frame(self.win, padding=(pad, 0, pad, pad))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", style="Cancel.TButton", command=self._on_close).pack(
            side=tk.RIGHT
        )

    def _build_anim_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        self._anim_view_var = tk.StringVar(value="pressure")
        ttk.Label(ctrl, text="Field:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
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
        import matplotlib.pyplot as plt

        if self._anim_canvas is not None and hasattr(self._anim_canvas, "figure"):
            try:
                plt.close(self._anim_canvas.figure)
            except Exception:
                pass
        for w in self._anim_frame.winfo_children():
            w.destroy()

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
        self._anim_canvas = embed_animation_plot_in_tk(fig, self._anim_frame)

    def _build_geometry_tab(self, parent: ttk.Frame) -> None:
        arr = np.vstack([self._result.diameter, self._result.area])
        fig = create_solution_plot(
            self._result.x,
            arr,
            title="Pipe geometry",
            xlabel="x",
            ylabel="value",
            selected_derivatives=[0, 1],
            labels=["Diameter", "Area"],
        )
        self._geometry_canvas = embed_plot_in_tk(fig, parent)

    def _build_pressure_tab(self, parent: ttk.Frame) -> None:
        if self._result.model_type == "steady":
            fig = create_solution_plot(
                self._result.x,
                np.atleast_2d(self._result.pressure[0]),
                title="Steady pressure profile",
                xlabel="x",
                ylabel="p",
                selected_derivatives=[0],
                labels=["p(x)"],
            )
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
            arr = np.vstack(
                [
                    self._result.velocity[0],
                    self._result.reynolds[0],
                    self._result.friction[0],
                ]
            )
            fig = create_solution_plot(
                self._result.x,
                arr,
                title="Steady velocity, Reynolds and friction",
                xlabel="x",
                ylabel="value",
                selected_derivatives=[0, 1, 2],
                labels=["u(x)", "Re(x)", "f(x)"],
            )
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
            arr = np.vstack([self._result.flow_rate_mean, self._result.flow_rate_std])
            fig = create_solution_plot(
                self._result.t,
                arr,
                title="Steady flow-rate diagnostics",
                xlabel="t",
                ylabel="Q",
                selected_derivatives=[0, 1],
                labels=["Q mean", "Q std"],
            )
        else:
            p_in = self._result.pressure[:, 0]
            p_out = self._result.pressure[:, -1]
            arr = np.vstack([p_in, p_out, self._result.flow_rate_mean, self._result.flow_rate_std])
            fig = create_solution_plot(
                self._result.t,
                arr,
                title="Boundary signals and flow statistics",
                xlabel="t",
                ylabel="value",
                selected_derivatives=[0, 1, 2, 3],
                labels=["pᵢₙ", "pₒᵤₜ", "Q mean", "Q std"],
            )
        self._quality_canvas = embed_plot_in_tk(fig, parent)
