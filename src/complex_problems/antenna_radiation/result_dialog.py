"""Result dialog for antenna radiation simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from complex_problems.antenna_radiation.solver import AntennaRadiationResult
from config import get_env_from_schema
from frontend.plot_embed import embed_plot_in_tk
from frontend.window_utils import center_window, make_modal
from plotting import create_contour_plot, create_solution_plot


def _create_polar_cut_figure(theta_deg: np.ndarray, cut_db: np.ndarray, *, title: str) -> "object":
    import matplotlib.pyplot as plt

    theta_rad = np.deg2rad(theta_deg)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta_rad, cut_db, linewidth=2.0)
    ax.set_title(title)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def _create_3d_pattern_figure(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    gain_db: np.ndarray,
) -> "object":
    import matplotlib.pyplot as plt

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    g_lin = np.power(10.0, np.maximum(gain_db, -40.0) / 10.0)
    r = g_lin / (float(np.max(g_lin)) + 1e-15)
    x = r * np.sin(TH) * np.cos(PH)
    y = r * np.sin(TH) * np.sin(PH)
    z = r * np.cos(TH)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", linewidth=0.0, antialiased=True, alpha=0.95)
    ax.set_title("Normalized 3D radiation pattern")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    return fig


class AntennaRadiationResultDialog:
    """Result window for antenna radiation problem."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: AntennaRadiationResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Results - Antenna Radiation")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))

        self._map_canvas = None
        self._cut_canvas = None
        self._phi_canvas = None
        self._surface_canvas = None
        self._field_canvas = None

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1380, height=920, max_width_ratio=0.96, resizable=True)
        self.win.minsize(1080, 720)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        import matplotlib.pyplot as plt

        for attr in (
            "_map_canvas",
            "_cut_canvas",
            "_phi_canvas",
            "_surface_canvas",
            "_field_canvas",
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
        info_txt = (
            f"Dmax: {mag['directivity_max_db']:+.2f} dBi   "
            f"Gmax: {mag['gain_max_db']:+.2f} dBi   "
            f"BW(-3dB): {mag['beamwidth_deg']:.2f} deg   "
            f"Max E(rms): {mag['max_e_rms_vpm']:.3g} V/m"
        )
        far_field_note = (
            "Far field: OK"
            if self._result.metadata.get("is_far_field", False)
            else f"Far field: NOT OK (Rmin={mag['far_field_min_m']:.3g} m)"
        )
        ttk.Label(top, text=info_txt, style="Small.TLabel").pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(top, text=far_field_note, style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))

        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_map = ttk.Frame(nb)
        nb.add(tab_map, text="  Angular Map  ")
        self._build_map_tab(tab_map)

        tab_cut = ttk.Frame(nb)
        nb.add(tab_cut, text="  Theta Cut  ")
        self._build_theta_cut_tab(tab_cut)

        tab_phi = ttk.Frame(nb)
        nb.add(tab_phi, text="  Phi Cut  ")
        self._build_phi_cut_tab(tab_phi)

        tab_3d = ttk.Frame(nb)
        nb.add(tab_3d, text="  3D Pattern  ")
        self._build_3d_tab(tab_3d)

        tab_field = ttk.Frame(nb)
        nb.add(tab_field, text="  Field Map  ")
        self._build_field_tab(tab_field)

        btn_frame = ttk.Frame(self.win, padding=(pad, 0, pad, pad))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", style="Cancel.TButton", command=self._on_close).pack(
            side=tk.RIGHT
        )

    def _build_map_tab(self, parent: ttk.Frame) -> None:
        fig = create_contour_plot(
            self._result.phi,
            self._result.theta,
            self._result.gain_db,
            title="Gain pattern (dBi) vs angles",
            xlabel="φ (deg)",
            ylabel="θ (deg)",
        )
        self._map_canvas = embed_plot_in_tk(fig, parent)

    def _build_theta_cut_tab(self, parent: ttk.Frame) -> None:
        fig = _create_polar_cut_figure(
            self._result.theta,
            self._result.theta_cut_db,
            title="θ cut (φ=0) in dBi",
        )
        self._cut_canvas = embed_plot_in_tk(fig, parent)

    def _build_phi_cut_tab(self, parent: ttk.Frame) -> None:
        fig = create_solution_plot(
            self._result.phi,
            np.atleast_2d(self._result.phi_cut_db),
            title="φ cut at θ=90 deg",
            xlabel="φ (deg)",
            ylabel="Gain (dBi)",
            selected_derivatives=[0],
            labels=["Gain"],
        )
        self._phi_canvas = embed_plot_in_tk(fig, parent)

    def _build_3d_tab(self, parent: ttk.Frame) -> None:
        fig = _create_3d_pattern_figure(self._result.theta, self._result.phi, self._result.gain_db)
        self._surface_canvas = embed_plot_in_tk(fig, parent)

    def _build_field_tab(self, parent: ttk.Frame) -> None:
        fig = create_contour_plot(
            self._result.phi,
            self._result.theta,
            self._result.e_rms,
            title="RMS electric field magnitude",
            xlabel="φ (deg)",
            ylabel="θ (deg)",
        )
        self._field_canvas = embed_plot_in_tk(fig, parent)
