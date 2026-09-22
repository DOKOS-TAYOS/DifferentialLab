"""Result dialog for coupled harmonic oscillators."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from complex_problems.common.result_dialog_ui import (
    AdvancedMultiSelector,
    AdvancedResultShell,
    AdvancedResultSize,
    close_embedded_figures,
    format_result_summary,
    make_view_controls,
    reset_embedded_animation,
)
from complex_problems.coupled_oscillators.solver import CoupledOscillatorsResult
from config import generate_output_basename, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, replace_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import make_modal
from plotting import (
    create_contour_plot,
    create_energy_evolution_plot,
    create_energy_per_mode_plot,
    create_surface_plot,
    create_vector_animation_plot,
    export_animation_to_mp4,
)
from utils import get_logger

logger = get_logger(__name__)


def _compute_energy(
    y: np.ndarray,
    n: int,
    masses: np.ndarray,
    k_arr: np.ndarray,
    boundary: str = "fixed",
    coupling_types: list[str] | None = None,
    nonlinear_coeff: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute kinetic, potential, and total energy at each time step.

    For β-FPUT: V = ½kδ² + (ε/4)δ⁴ per spring.
    FPUT-α uses F = α·(x_{i+1}+x_{i-1}-2x_i)·(x_{i+1}-x_{i-1}); potential form differs.

    Returns:
        Tuple of (E_kin, E_pot, E_tot), each shape (n_points,).
    """
    x = y[:n]
    v = y[n:]
    E_kin = 0.5 * np.sum(masses[:, np.newaxis] * v**2, axis=0)
    E_pot = np.zeros(y.shape[1])
    has_nonlinear = coupling_types and "nonlinear" in coupling_types and nonlinear_coeff != 0

    for i in range(n - 1):
        delta = x[i + 1] - x[i]
        E_pot += 0.5 * k_arr[i] * delta**2
        if has_nonlinear:
            E_pot += 0.25 * nonlinear_coeff * delta**4

    if boundary == "periodic":
        delta = x[0] - x[-1]
        E_pot += 0.5 * k_arr[-1] * delta**2
        if has_nonlinear:
            E_pot += 0.25 * nonlinear_coeff * delta**4
    elif boundary == "fixed" and n >= 2:
        # Wall springs at x_{-1}=x_N=0: 0.5*k[0]*x[0]^2 + 0.5*k[n-2]*x[n-1]^2
        # (no nonlinear term on wall springs)
        E_pot += 0.5 * k_arr[0] * x[0] ** 2
        E_pot += 0.5 * k_arr[-1] * x[-1] ** 2

    E_tot = E_kin + E_pot
    return E_kin, E_pot, E_tot


def _compute_energy_per_mode(
    y: np.ndarray,
    n: int,
    M_modes: np.ndarray,
    omega_modes: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Compute energy per normal mode at each time step.

    Returns:
        Array shape (n, n_points).
    """
    M_diag = np.diag(masses)
    q = M_modes.T @ M_diag @ y[:n]
    dq = M_modes.T @ M_diag @ y[n:]
    E = 0.5 * dq**2 + 0.5 * (omega_modes[:, np.newaxis] ** 2) * q**2
    return E


def _state_to_vector_ode_format(y: np.ndarray, n: int) -> np.ndarray:
    """Convert state [x_0..x_{N-1}, v_0..v_{N-1}] to [x0,v0,x1,v1,...] format.

    create_vector_animation_plot expects order=2, so rows 0,2,4,... are positions.
    """
    n_points = y.shape[1]
    new_y = np.zeros((2 * n, n_points))
    for i in range(n):
        new_y[2 * i] = y[i]  # position
        new_y[2 * i + 1] = y[n + i]  # velocity
    return new_y


@dataclass(frozen=True)
class _AnimationViewPayload:
    """Data and display metadata shared by an animation and its MP4 export."""

    x: np.ndarray
    y: np.ndarray
    vector_components: int
    title: str
    component_labels: list[str] | None


class CoupledOscillatorsResultDialog:
    """Result window for coupled harmonic oscillators.

    Shows visual-first tabs for animation, structure, modal energy, surfaces, and energy.
    Phase 1: Animation tab only. Phase 3 adds the rest.

    Args:
        parent: Parent window.
        result: CoupledOscillatorsResult from the solver.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        result: CoupledOscillatorsResult,
    ) -> None:
        self.parent = parent
        self._result = result

        self.win = tk.Toplevel(parent)
        self.win.title("Coupled Harmonic Oscillator Results")

        self._energy_canvas = None
        self._em_canvas = None
        self._anim_canvas = None
        self._hm_canvas = None
        self._surf_canvas = None

        self._build_ui()
        self._shell.finish(AdvancedResultSize(1420, 900))
        make_modal(self.win, parent)
        logger.info("Coupled oscillators result dialog displayed")

    def _on_close(self) -> None:
        """Close all matplotlib figures and destroy the window."""
        close_embedded_figures(
            self,
            ("_energy_canvas", "_em_canvas", "_anim_canvas", "_hm_canvas", "_surf_canvas"),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        """Construct the dialog layout."""
        metadata = self._result.metadata
        coupling_types = ", ".join(str(value) for value in metadata.get("coupling_types", ()))
        summary = format_result_summary(
            (
                ("Oscillators", self._result.n_oscillators),
                ("Boundary", str(metadata.get("boundary", "unknown")).capitalize()),
                ("Couplings", coupling_types or "unspecified"),
                ("Integrator", metadata.get("method", "unknown")),
            )
        )
        self._shell = AdvancedResultShell(
            self.win,
            title="Coupled Oscillators Results",
            summary=summary,
            close_command=self._on_close,
        )
        nb = self._shell.notebook

        # Tab 1: Animation
        anim_tab = ttk.Frame(nb)
        nb.add(anim_tab, text="Animation")
        anim_ctrl = make_view_controls(anim_tab)
        anim_group = anim_ctrl.add_group(requested_width=190)
        ttk.Label(anim_group, text="Display:").pack(side=tk.LEFT, padx=(0, 4))
        self._anim_view_var = tk.StringVar(value="Oscillators")
        anim_values = ["Oscillators", "Modes"] if self._result.has_modes else ["Oscillators"]
        view_combo = ttk.Combobox(
            anim_group,
            textvariable=self._anim_view_var,
            values=anim_values,
            state="readonly",
            width=12,
            font=get_font(),
        )
        view_combo.pack(side=tk.LEFT, padx=(0, 8))
        view_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_animation())
        self._anim_plot_frame = ttk.Frame(anim_tab)
        self._anim_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_animation()

        # Tab 2: Space-time heatmap
        heatmap_tab = ttk.Frame(nb)
        nb.add(heatmap_tab, text="Space-Time Heatmap")
        hm_ctrl = make_view_controls(heatmap_tab)
        hm_group = hm_ctrl.add_group(requested_width=190)
        ttk.Label(hm_group, text="Display:").pack(side=tk.LEFT, padx=(0, 4))
        self._hm_view_var = tk.StringVar(value="Modes" if self._result.has_modes else "Oscillators")
        hm_values = ["Modes", "Oscillators"] if self._result.has_modes else ["Oscillators"]
        hm_view_combo = ttk.Combobox(
            hm_group,
            textvariable=self._hm_view_var,
            values=hm_values,
            state="readonly",
            width=12,
            font=get_font(),
        )
        hm_view_combo.pack(side=tk.LEFT, padx=(0, 8))
        hm_view_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_heatmap())
        self._hm_plot_frame = ttk.Frame(heatmap_tab)
        self._hm_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_heatmap()

        # Tab 3: Energy per mode or oscillator
        energy_mode_tab = ttk.Frame(nb)
        nb.add(energy_mode_tab, text="Energy by Mode/Oscillator")
        em_ctrl = make_view_controls(energy_mode_tab)
        display_group = em_ctrl.add_group(requested_width=190)
        ttk.Label(display_group, text="Display:").pack(side=tk.LEFT, padx=(0, 4))
        self._em_view_var = tk.StringVar(value="Modes" if self._result.has_modes else "Oscillators")
        em_values = ["Modes", "Oscillators"] if self._result.has_modes else ["Oscillators"]
        em_view_combo = ttk.Combobox(
            display_group,
            textvariable=self._em_view_var,
            values=em_values,
            state="readonly",
            width=12,
            font=get_font(),
        )
        em_view_combo.pack(side=tk.LEFT, padx=(0, 8))
        em_view_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_em_view_change())
        n = self._result.n_oscillators
        select_group = em_ctrl.add_group(requested_width=min(360, max(180, n * 86)))
        ttk.Label(select_group, text="Select:").pack(side=tk.LEFT, padx=(0, 4))
        self._em_selector = AdvancedMultiSelector(
            select_group,
            self._energy_mode_labels(self._em_view_var.get()),
            selected_indexes=range(min(3, n)),
            allow_empty=False,
            command=self._update_energy_mode,
        )
        self._em_selector.pack(side=tk.LEFT)
        self._em_plot_frame = ttk.Frame(energy_mode_tab)
        self._em_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_energy_mode()

        # Tab 4: Surface 3D
        surf_tab = ttk.Frame(nb)
        nb.add(surf_tab, text="Surface 3D")
        surf_ctrl = make_view_controls(surf_tab)
        surf_group = surf_ctrl.add_group(requested_width=190)
        ttk.Label(surf_group, text="Display:").pack(side=tk.LEFT, padx=(0, 4))
        self._surf_view_var = tk.StringVar(value="Oscillators")
        surf_values = ["Oscillators", "Modes"] if self._result.has_modes else ["Oscillators"]
        surf_view_combo = ttk.Combobox(
            surf_group,
            textvariable=self._surf_view_var,
            values=surf_values,
            state="readonly",
            width=12,
            font=get_font(),
        )
        surf_view_combo.pack(side=tk.LEFT, padx=(0, 8))
        surf_view_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_surface())
        self._surf_plot_frame = ttk.Frame(surf_tab)
        self._surf_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_surface()

        # Tab 5: Energy evolution
        energy_tab = ttk.Frame(nb)
        nb.add(energy_tab, text="Energy")
        self._energy_plot_frame = ttk.Frame(energy_tab)
        self._energy_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_energy_plot()

    def _update_animation(self) -> None:
        """Regenerate the animation tab."""
        reset_embedded_animation(self._anim_plot_frame, self._anim_canvas)
        self._anim_canvas = None

        payload = self._get_animation_view_payload()

        fig = create_vector_animation_plot(
            payload.x,
            payload.y,
            order=2,
            vector_components=payload.vector_components,
            title=payload.title,
            deriv_offset=0,
            component_labels=payload.component_labels,
        )
        self._anim_canvas = embed_animation_plot_in_tk(
            fig,
            self._anim_plot_frame,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _get_animation_view_payload(self) -> _AnimationViewPayload:
        """Build the exact component representation currently shown in the animation."""
        r = self._result
        n = r.n_oscillators
        view = self._anim_view_var.get()

        if view == "Modes" and r.has_modes:
            # Transform to mode space: q = M_modes.T @ M @ x.
            M_diag = np.diag(r.masses)
            positions = r.y[:n]
            q = r.M_modes.T @ M_diag @ positions
            dq = r.M_modes.T @ M_diag @ r.y[n:]
            mode_y = np.zeros((2 * n, r.y.shape[1]))
            for i in range(n):
                mode_y[2 * i] = q[i]
                mode_y[2 * i + 1] = dq[i]
            return _AnimationViewPayload(
                x=r.x,
                y=mode_y,
                vector_components=n,
                title="Coupled Oscillators — Mode amplitudes",
                component_labels=[f"Mode {i + 1}" for i in range(n)],
            )

        plot_y = _state_to_vector_ode_format(r.y, n)
        component_labels = None
        n_components = n
        if r.metadata.get("boundary", "fixed") == "fixed":
            # Prepend x_{-1}=0, v_{-1}=0 and append x_N=0, v_N=0.
            extended = np.zeros((2 * (n + 2), plot_y.shape[1]))
            extended[2 : 2 * (n + 1)] = plot_y
            plot_y = extended
            n_components = n + 2
            component_labels = [str(i) for i in range(-1, n + 1)]

        return _AnimationViewPayload(
            x=r.x,
            y=plot_y,
            vector_components=n_components,
            title="Coupled Oscillators — Oscillator positions",
            component_labels=component_labels,
        )

    def _on_export_animation_mp4(
        self,
        payload: _AnimationViewPayload,
        duration_seconds: float,
    ) -> None:
        """Export the currently displayed animation representation as MP4."""
        default_path = get_output_dir() / f"{generate_output_basename(prefix='animation')}.mp4"
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
            export_animation_to_mp4(
                payload.x,
                payload.y,
                order=2,
                vector_components=payload.vector_components,
                filepath=filepath,
                title=payload.title,
                duration_seconds=duration_seconds,
                component_labels=payload.component_labels,
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

    def _replace_plot(
        self,
        frame: ttk.Frame,
        fig: "Figure",
        canvas_attr: str,
    ) -> None:
        """Reuse the existing canvas in frame when the figure changes."""
        old_canvas = getattr(self, canvas_attr, None)
        canvas = replace_plot_in_tk(fig, frame, current_canvas=old_canvas)
        setattr(self, canvas_attr, canvas)

    def _update_energy_plot(self) -> None:
        """Regenerate the energy evolution tab."""
        r = self._result
        boundary = r.metadata.get("boundary", "fixed")
        coupling_types = r.metadata.get("coupling_types", ["linear"])
        nonlinear_coeff = r.metadata.get("nonlinear_coeff", 0.0)
        E_kin, E_pot, E_tot = _compute_energy(
            r.y,
            r.n_oscillators,
            r.masses,
            r.k_coupling,
            boundary,
            coupling_types=coupling_types,
            nonlinear_coeff=nonlinear_coeff,
        )
        fig = create_energy_evolution_plot(
            r.x,
            E_kin,
            E_pot,
            E_tot,
            title="Energy vs time",
            xlabel="t",
        )
        self._replace_plot(self._energy_plot_frame, fig, "_energy_canvas")

    def _update_energy_mode(self) -> None:
        """Regenerate the energy per mode tab."""
        r = self._result
        n = r.n_oscillators
        view = self._em_view_var.get()
        selected = list(self._em_selector.selected_indexes())

        if view == "Modes" and r.has_modes:
            E_modes = _compute_energy_per_mode(r.y, n, r.M_modes, r.omega_modes, r.masses)
            labels = [f"Mode {i + 1}" for i in selected]
        else:
            # Energy per oscillator: kinetic + share of potential
            x, v = r.y[:n], r.y[n:]
            E_osc = 0.5 * r.masses[:, np.newaxis] * v**2
            boundary = r.metadata.get("boundary", "fixed")
            k_arr = r.k_coupling
            coupling_types = r.metadata.get("coupling_types", ["linear"])
            nonlinear_coeff = r.metadata.get("nonlinear_coeff", 0.0)
            has_nonlinear = "nonlinear" in coupling_types and nonlinear_coeff != 0

            for i in range(n - 1):
                delta = x[i + 1] - x[i]
                E_osc[i] += 0.25 * k_arr[i] * delta**2
                E_osc[i + 1] += 0.25 * k_arr[i] * delta**2
                if has_nonlinear:
                    quartic_share = 0.125 * nonlinear_coeff * delta**4
                    E_osc[i] += quartic_share
                    E_osc[i + 1] += quartic_share
            if boundary == "periodic":
                delta = x[0] - x[-1]
                E_osc[0] += 0.25 * k_arr[-1] * delta**2
                E_osc[-1] += 0.25 * k_arr[-1] * delta**2
                if has_nonlinear:
                    quartic_share = 0.125 * nonlinear_coeff * delta**4
                    E_osc[0] += quartic_share
                    E_osc[-1] += quartic_share
            elif boundary == "fixed" and n >= 2:
                # Wall springs at x_{-1}=x_N=0 (no nonlinear on walls)
                E_osc[0] += 0.5 * k_arr[0] * x[0] ** 2
                E_osc[-1] += 0.5 * k_arr[-1] * x[-1] ** 2
            E_modes = E_osc
            labels = [f"Oscillator {i}" for i in selected]

        fig = create_energy_per_mode_plot(
            r.x,
            E_modes,
            selected,
            labels,
            title="Energy per " + ("mode" if view == "Modes" else "oscillator"),
            xlabel="t",
        )
        self._replace_plot(self._em_plot_frame, fig, "_em_canvas")

    def _on_em_view_change(self) -> None:
        """Rebuild energy selector labels when view changes."""
        self._em_selector.destroy()
        self._em_selector = AdvancedMultiSelector(
            self._em_selector.master,
            self._energy_mode_labels(self._em_view_var.get()),
            selected_indexes=range(min(3, self._result.n_oscillators)),
            allow_empty=False,
            command=self._update_energy_mode,
        )
        self._em_selector.pack(side=tk.LEFT)
        self._update_energy_mode()

    def _energy_mode_labels(self, view: str) -> tuple[str, ...]:
        """Return established 1-based mode and 0-based oscillator labels."""
        if view == "Modes":
            return tuple(f"Mode {index + 1}" for index in range(self._result.n_oscillators))
        return tuple(f"Oscillator {index}" for index in range(self._result.n_oscillators))

    def _get_amplitude_data(
        self, view_var: tk.StringVar
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get amplitude data for heatmap/surface. Returns (x_axis, t, Z)."""
        r = self._result
        n = r.n_oscillators
        view_val = view_var.get()
        boundary = r.metadata.get("boundary", "fixed")

        if view_val == "Modes" and r.has_modes:
            M_diag = np.diag(r.masses)
            q = r.M_modes.T @ M_diag @ r.y[:n]
            Z = q
        else:
            Z = r.y[:n]
            if boundary == "fixed":
                n_points = Z.shape[1]
                Z_ext = np.zeros((n + 2, n_points))
                Z_ext[0] = 0.0  # x_{-1} fixed
                Z_ext[1 : n + 1] = Z
                Z_ext[-1] = 0.0  # x_N fixed
                Z = Z_ext

        if view_val == "Modes" and r.has_modes:
            x_axis = np.arange(1, Z.shape[0] + 1)  # Mode 1, 2, 3, ...
        elif boundary == "fixed" and view_val == "Oscillators":
            x_axis = np.arange(-1, n + 1)
        else:
            x_axis = np.arange(Z.shape[0])
        t = r.x
        return x_axis, t, Z

    def _update_heatmap(self) -> None:
        """Regenerate the 2D heatmap tab."""
        x_axis, t, Z = self._get_amplitude_data(self._hm_view_var)
        xlabel = "Mode" if self._hm_view_var.get() == "Modes" else "Oscillator"
        fig = create_contour_plot(
            x_axis,
            t,
            Z.T,
            title="Amplitude heatmap",
            xlabel=xlabel,
            ylabel="t",
        )
        self._replace_plot(self._hm_plot_frame, fig, "_hm_canvas")

    def _update_surface(self) -> None:
        """Regenerate the 3D surface tab."""
        x_axis, t, Z = self._get_amplitude_data(self._surf_view_var)
        xlabel = "Mode" if self._surf_view_var.get() == "Modes" else "Oscillator"
        fig = create_surface_plot(
            x_axis,
            t,
            Z.T,
            title="Amplitude 3D",
            xlabel=xlabel,
            ylabel="t",
            zlabel="Amplitude",
        )
        self._replace_plot(self._surf_plot_frame, fig, "_surf_canvas")
