"""Result dialog for antenna radiation simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, cast

import numpy as np

from complex_problems.antenna_radiation.solver import AntennaRadiationResult
from complex_problems.common.result_dialog_ui import (
    AdvancedResultShell,
    AdvancedResultSize,
    close_embedded_figures,
    format_result_summary,
)
from frontend.plot_embed import embed_plot_in_tk
from frontend.window_utils import make_modal
from plotting import create_contour_plot, create_solution_plot

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.projections.polar import PolarAxes
    from mpl_toolkits.mplot3d.axes3d import Axes3D


def _create_polar_cut_figure(theta_deg: np.ndarray, cut_db: np.ndarray, *, title: str) -> Figure:
    import matplotlib.pyplot as plt

    theta_rad = np.deg2rad(theta_deg)
    fig = plt.figure()
    polar_ax = cast("PolarAxes", fig.add_subplot(111, projection="polar"))
    polar_ax.plot(theta_rad, cut_db, linewidth=2.0)
    polar_ax.set_title(title)
    polar_ax.set_theta_zero_location("N")
    polar_ax.set_theta_direction(-1)
    polar_ax.set_rlabel_position(135)
    polar_ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def _create_3d_pattern_figure(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    gain_db: np.ndarray,
) -> Figure:
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
    axes_3d = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    axes_3d.plot_surface(x, y, z, cmap="viridis", linewidth=0.0, antialiased=True, alpha=0.95)
    axes_3d.set_title("Normalized 3D radiation pattern")
    axes_3d.set_xlabel("x")
    axes_3d.set_ylabel("y")
    axes_3d.set_zlabel("z")
    fig.tight_layout()
    return fig


class AntennaRadiationResultDialog:
    """Result window for antenna radiation problem."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: AntennaRadiationResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Antenna Radiation Results")

        self._map_canvas = None
        self._cut_canvas = None
        self._phi_canvas = None
        self._surface_canvas = None
        self._field_canvas = None

        self._build_ui()
        self._shell.finish(AdvancedResultSize(1380, 920))
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        close_embedded_figures(
            self,
            (
                "_map_canvas",
                "_cut_canvas",
                "_phi_canvas",
                "_surface_canvas",
                "_field_canvas",
            ),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        mag = self._result.magnitudes
        far_field = (
            "satisfied"
            if self._result.metadata.get("is_far_field", False)
            else f"not satisfied (minimum R = {mag['far_field_min_m']:.3g} m)"
        )
        summary = format_result_summary(
            (
                ("Dmax", f"{mag['directivity_max_db']:+.2f} dBi"),
                ("Gmax", f"{mag['gain_max_db']:+.2f} dBi"),
                ("−3 dB beamwidth", f"{mag['beamwidth_deg']:.2f}°"),
                ("Maximum E(rms)", f"{mag['max_e_rms_vpm']:.3g} V/m"),
                ("Far-field criterion", far_field),
            )
        )
        self._shell = AdvancedResultShell(
            self.win,
            title="Antenna Radiation Results",
            summary=summary,
            close_command=self._on_close,
        )
        nb = self._shell.notebook

        tab_3d = ttk.Frame(nb)
        nb.add(tab_3d, text="3D Pattern")
        self._build_3d_tab(tab_3d)

        tab_map = ttk.Frame(nb)
        nb.add(tab_map, text="Angular Gain Map")
        self._build_map_tab(tab_map)

        tab_cut = ttk.Frame(nb)
        nb.add(tab_cut, text="Theta Cut")
        self._build_theta_cut_tab(tab_cut)

        tab_phi = ttk.Frame(nb)
        nb.add(tab_phi, text="Phi Cut")
        self._build_phi_cut_tab(tab_phi)

        tab_field = ttk.Frame(nb)
        nb.add(tab_field, text="Field Strength")
        self._build_field_tab(tab_field)

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
