"""Result dialog for coupled harmonic oscillators."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from complex_problems.coupled_oscillators.solver import CoupledOscillatorsResult
from config import get_env_from_schema
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_contrast_foreground, get_font
from frontend.window_utils import center_window, make_modal
from plotting import (
    create_contour_plot,
    create_energy_evolution_plot,
    create_energy_per_mode_plot,
    create_surface_plot,
    create_vector_animation_plot,
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


class CoupledOscillatorsResultDialog:
    """Result window for coupled harmonic oscillators.

    Shows tabs: Energy, Energy per mode, Animation, Heatmap, 3D Surface.
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
        self.win.title("Results — Coupled Harmonic Oscillators")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._energy_canvas = None
        self._em_canvas = None
        self._anim_canvas = None
        self._hm_canvas = None
        self._surf_canvas = None

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        win_w = int(screen_w * 0.96)
        win_h = min(int(screen_h * 0.92), 1100)
        center_window(self.win, win_w, win_h, max_width_ratio=0.98, resizable=True)
        self.win.minsize(1200, 700)
        make_modal(self.win, parent)
        logger.info("Coupled oscillators result dialog displayed")

    def _on_close(self) -> None:
        """Close all matplotlib figures and destroy the window."""
        import matplotlib.pyplot as plt

        for attr in ("_energy_canvas", "_em_canvas", "_anim_canvas", "_hm_canvas", "_surf_canvas"):
            canvas = getattr(self, attr, None)
            if canvas is not None and hasattr(canvas, "figure"):
                try:
                    plt.close(canvas.figure)
                except Exception:
                    pass
        self.win.destroy()

    def _build_ui(self) -> None:
        """Construct the dialog layout."""
        pad: int = get_env_from_schema("UI_PADDING")

        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)
        ttk.Button(
            btn_frame,
            text="Close",
            style="Cancel.TButton",
            command=self._on_close,
        ).pack()

        content = ttk.Frame(self.win)
        content.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # Right: notebook with tabs
        nb = ttk.Notebook(content)
        nb.grid(row=0, column=1, sticky="nsew")

        # Tab 1: Energy evolution
        energy_tab = ttk.Frame(nb)
        nb.add(energy_tab, text="  Energy  ")
        self._energy_plot_frame = ttk.Frame(energy_tab)
        self._energy_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_energy_plot()

        # Tab 2: Energy per mode
        energy_mode_tab = ttk.Frame(nb)
        nb.add(energy_mode_tab, text="  Energy per mode  ")
        em_ctrl = ttk.Frame(energy_mode_tab)
        em_ctrl.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(em_ctrl, text="View:").pack(side=tk.LEFT, padx=(0, 4))
        self._em_view_var = tk.StringVar(value="Modes" if self._result.has_modes else "Oscillators")
        em_values = ["Modes", "Oscillators"] if self._result.has_modes else ["Oscillators"]
        em_view_combo = ttk.Combobox(
            em_ctrl,
            textvariable=self._em_view_var,
            values=em_values,
            state="readonly",
            width=12,
            font=get_font(),
        )
        em_view_combo.pack(side=tk.LEFT, padx=(0, 8))
        em_view_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_em_view_change()
        )
        ttk.Label(em_ctrl, text="Select:").pack(side=tk.LEFT, padx=(16, 4))
        n = self._result.n_oscillators
        btn_bg = get_env_from_schema("UI_BUTTON_BG")
        fg = get_env_from_schema("UI_FOREGROUND")
        select_bg = get_env_from_schema("UI_BUTTON_FG")
        select_fg = get_contrast_foreground(select_bg)
        self._em_listbox = tk.Listbox(
            em_ctrl,
            selectmode=tk.EXTENDED,
            height=6,
            width=14,
            bg=btn_bg,
            fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=get_font(),
            exportselection=False,
        )
        lbl_prefix = "Mode" if self._result.has_modes else "Oscillator"
        for i in range(n):
            # Physics convention: Mode 1 = fundamental, Mode 2 = second harmonic, etc.
            label = f"{lbl_prefix} {i + 1}" if self._result.has_modes else f"{lbl_prefix} {i}"
            self._em_listbox.insert(tk.END, label)
        self._em_listbox.selection_set(0, min(2, n - 1))
        self._em_listbox.pack(side=tk.LEFT, padx=(0, 4))
        self._em_listbox.bind("<<ListboxSelect>>", lambda _e: self._update_energy_mode())
        self._em_plot_frame = ttk.Frame(energy_mode_tab)
        self._em_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_energy_mode()

        # Tab 3: Animation
        anim_tab = ttk.Frame(nb)
        nb.add(anim_tab, text="  Animation  ")
        anim_ctrl = ttk.Frame(anim_tab)
        anim_ctrl.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(anim_ctrl, text="View:").pack(side=tk.LEFT, padx=(0, 4))
        self._anim_view_var = tk.StringVar(value="Oscillators")
        anim_values = ["Oscillators", "Modes"] if self._result.has_modes else ["Oscillators"]
        view_combo = ttk.Combobox(
            anim_ctrl,
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

        # Tab 4: Heatmap 2D
        heatmap_tab = ttk.Frame(nb)
        nb.add(heatmap_tab, text="  Heatmap 2D  ")
        hm_ctrl = ttk.Frame(heatmap_tab)
        hm_ctrl.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(hm_ctrl, text="View:").pack(side=tk.LEFT, padx=(0, 4))
        self._hm_view_var = tk.StringVar(value="Oscillators")
        hm_values = ["Oscillators", "Modes"] if self._result.has_modes else ["Oscillators"]
        hm_view_combo = ttk.Combobox(
            hm_ctrl,
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

        # Tab 5: Surface 3D
        surf_tab = ttk.Frame(nb)
        nb.add(surf_tab, text="  Surface 3D  ")
        surf_ctrl = ttk.Frame(surf_tab)
        surf_ctrl.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(surf_ctrl, text="View:").pack(side=tk.LEFT, padx=(0, 4))
        self._surf_view_var = tk.StringVar(value="Oscillators")
        surf_values = ["Oscillators", "Modes"] if self._result.has_modes else ["Oscillators"]
        surf_view_combo = ttk.Combobox(
            surf_ctrl,
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

    def _update_animation(self) -> None:
        """Regenerate the animation tab."""
        import matplotlib.pyplot as plt

        if self._anim_canvas is not None and hasattr(self._anim_canvas, "figure"):
            try:
                plt.close(self._anim_canvas.figure)
            except Exception:
                pass
            self._anim_canvas = None
        for w in self._anim_plot_frame.winfo_children():
            w.destroy()

        r = self._result
        n = r.n_oscillators
        x = r.x
        y = r.y
        boundary = r.metadata.get("boundary", "fixed")

        view = self._anim_view_var.get()
        component_labels = None
        if view == "Modes" and r.has_modes:
            # Transform to mode space: q = M_modes.T @ M @ x
            # For orthonormal modes (A^T M A = I): q = A^T M x
            M_diag = np.diag(r.masses)
            positions = y[:n]  # (n, n_points)
            q = r.M_modes.T @ M_diag @ positions  # (n, n_points)
            # Build format for animation: [q0, dq0, q1, dq1, ...]
            dq = r.M_modes.T @ np.diag(r.masses) @ y[n:]
            n_points = y.shape[1]
            mode_y = np.zeros((2 * n, n_points))
            for i in range(n):
                mode_y[2 * i] = q[i]
                mode_y[2 * i + 1] = dq[i]
            plot_y = mode_y
            title = "Coupled Oscillators — Mode amplitudes"
            n_components = n
            # Physics convention: Mode 1 = fundamental
            component_labels = [f"Mode {i + 1}" for i in range(n)]
        else:
            plot_y = _state_to_vector_ode_format(y, n)
            title = "Coupled Oscillators — Oscillator positions"
            if boundary == "fixed":
                # Prepend x_{-1}=0, v_{-1}=0 and append x_N=0, v_N=0
                n_points = plot_y.shape[1]
                extended = np.zeros((2 * (n + 2), n_points))
                extended[0] = 0.0
                extended[1] = 0.0
                extended[2 : 2 * (n + 1)] = plot_y
                extended[-2] = 0.0
                extended[-1] = 0.0
                plot_y = extended
                n_components = n + 2
                component_labels = [str(i) for i in range(-1, n + 1)]
            else:
                n_components = n

        fig = create_vector_animation_plot(
            x,
            plot_y,
            order=2,
            vector_components=n_components,
            title=title,
            deriv_offset=0,
            component_labels=component_labels,
        )
        self._anim_canvas = embed_animation_plot_in_tk(fig, self._anim_plot_frame)

    def _replace_plot(
        self,
        frame: ttk.Frame,
        fig: "Figure",
        canvas_attr: str,
    ) -> None:
        """Destroy the old canvas in frame and embed fig in its place."""
        from matplotlib.pyplot import close as plt_close

        old_canvas = getattr(self, canvas_attr, None)
        if old_canvas is not None:
            old_fig = old_canvas.figure
            old_canvas.get_tk_widget().destroy()
            plt_close(old_fig)

        for w in frame.winfo_children():
            w.destroy()

        canvas = embed_plot_in_tk(fig, frame)
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
        selected = list(self._em_listbox.curselection())
        if not selected:
            selected = [0]

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
            has_nonlinear = (
                "nonlinear" in coupling_types and nonlinear_coeff != 0
            )

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
        """Rebuild energy-mode listbox labels when view changes."""
        n = self._result.n_oscillators
        view = self._em_view_var.get()
        self._em_listbox.delete(0, tk.END)
        prefix = "Mode" if view == "Modes" else "Oscillator"
        for i in range(n):
            label = f"{prefix} {i + 1}" if view == "Modes" else f"{prefix} {i}"
            self._em_listbox.insert(tk.END, label)
        self._em_listbox.selection_set(0, min(2, n - 1))
        self._update_energy_mode()

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
