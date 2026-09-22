"""Plots, animation, and Tk result dialog for gravitational N-body solutions."""
# ruff: noqa: E501

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from complex_problems.common.result_dialog_ui import (
    AdvancedResultShell,
    AdvancedResultSize,
    close_embedded_figures,
    format_result_summary,
    make_view_controls,
    reset_embedded_animation,
)
from complex_problems.gravitational_n_body.solver import NBodyResult
from config import generate_output_basename, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, replace_plot_in_tk
from frontend.window_utils import make_modal
from plotting.animation_metadata import attach_animation_metadata
from plotting.plot_utils import export_animated_figure_to_mp4

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_ReferenceFrame = Literal["Inertial", "Center of mass"]


@dataclass(frozen=True, slots=True)
class OrbitAnimationPayload:
    """Prepared frame data shared by displayed and exported orbit animations."""

    t: np.ndarray
    positions: np.ndarray
    center_of_mass: np.ndarray
    masses: np.ndarray
    dimension: int
    reference_frame: _ReferenceFrame


def orbit_animation_payload(
    result: NBodyResult, reference_frame: _ReferenceFrame = "Inertial"
) -> OrbitAnimationPayload:
    """Prepare current reference-frame data without rerunning the solver."""
    positions = result.positions.copy()
    center_of_mass = result.center_of_mass.copy()
    if reference_frame == "Center of mass":
        positions -= center_of_mass[:, np.newaxis, :]
        center_of_mass = np.zeros_like(center_of_mass)
    return OrbitAnimationPayload(
        result.t, positions, center_of_mass, result.masses, result.dimension, reference_frame
    )


def _orbit_bounds(positions: np.ndarray) -> tuple[float, float]:
    finite = positions[np.isfinite(positions)]
    extent = float(np.max(np.abs(finite))) if finite.size else 1.0
    extent = max(extent, 1.0e-6)
    return -1.12 * extent, 1.12 * extent


def _marker_sizes(masses: np.ndarray) -> np.ndarray:
    """Return bounded mass-sensitive marker sizes for visual consistency."""
    scaled = np.sqrt(masses / np.max(masses))
    return 45.0 + 145.0 * scaled


def create_orbit_animation_figure(payload: OrbitAnimationPayload) -> Figure:
    """Build an animation-metadata orbit figure with stable limits and trails."""
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")(np.arange(payload.masses.size) % 10)
    lower, upper = _orbit_bounds(payload.positions)
    initial = payload.positions[0]
    if payload.dimension == 3:
        fig = plt.figure()
        axis: Any = fig.add_subplot(111, projection="3d")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_zlim(lower, upper)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        trails = [
            axis.plot([], [], [], color=colors[i], alpha=0.65, linewidth=1.2)[0]
            for i in range(len(colors))
        ]
        points: Any = axis.scatter(
            initial[:, 0], initial[:, 1], initial[:, 2], s=_marker_sizes(payload.masses), c=colors
        )
        initial_com = payload.center_of_mass[0]
        com_marker = axis.scatter(
            [initial_com[0]],
            [initial_com[1]],
            [initial_com[2]],
            marker="+",
            s=100,
            c="black",
            label="COM",
        )

        def update(index: int) -> None:
            current = payload.positions[index]
            for body, trail in enumerate(trails):
                trail.set_data(
                    payload.positions[: index + 1, body, 0], payload.positions[: index + 1, body, 1]
                )
                trail.set_3d_properties(payload.positions[: index + 1, body, 2])
            points._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
            current_com = payload.center_of_mass[index]
            com_marker._offsets3d = (
                np.asarray([current_com[0]]),
                np.asarray([current_com[1]]),
                np.asarray([current_com[2]]),
            )
            axis.set_title(f"Gravitational N-body orbit — t={payload.t[index]:.4g}")
            fig.canvas.draw_idle()

    else:
        fig, axis = plt.subplots()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.25)
        trails = [
            axis.plot([], [], color=colors[i], alpha=0.65, linewidth=1.2)[0]
            for i in range(len(colors))
        ]
        points: Any = axis.scatter(
            initial[:, 0], initial[:, 1], s=_marker_sizes(payload.masses), c=colors
        )
        initial_com = payload.center_of_mass[0]
        com_marker = axis.scatter(
            [initial_com[0]], [initial_com[1]], marker="+", s=100, c="black", label="COM"
        )

        def update(index: int) -> None:
            current = payload.positions[index]
            for body, trail in enumerate(trails):
                trail.set_data(
                    payload.positions[: index + 1, body, 0], payload.positions[: index + 1, body, 1]
                )
            points.set_offsets(current[:, :2])
            com_marker.set_offsets(payload.center_of_mass[index, :2])
            axis.set_title(f"Gravitational N-body orbit — t={payload.t[index]:.4g}")
            fig.canvas.draw_idle()

    _ = com_marker
    axis.set_title(f"Gravitational N-body orbit — t={payload.t[0]:.4g}")
    fig.tight_layout()
    return attach_animation_metadata(
        fig,
        update=update,
        n_points=len(payload.t),
        frame_label="time",
        frame_coordinates=payload.t,
        extras={"_n_body_animation_payload": payload},
    )


def selected_pair_distance(result: NBodyResult, body_a: int, body_b: int) -> np.ndarray:
    """Return a selected pair separation history from existing solved positions."""
    if body_a == body_b:
        raise ValueError("Choose two different bodies for a pair separation.")
    if not 0 <= body_a < result.masses.size or not 0 <= body_b < result.masses.size:
        raise ValueError("Selected body index is out of range.")
    return np.linalg.norm(result.positions[:, body_b] - result.positions[:, body_a], axis=1)


def create_trajectory_figure(
    result: NBodyResult, reference_frame: _ReferenceFrame = "Inertial"
) -> Figure:
    """Build static full trajectories in either requested physical reference frame."""
    import matplotlib.pyplot as plt

    payload = orbit_animation_payload(result, reference_frame)
    colors = plt.get_cmap("tab10")(np.arange(result.masses.size) % 10)
    lower, upper = _orbit_bounds(payload.positions)
    if result.dimension == 3:
        fig = plt.figure()
        axis = fig.add_subplot(111, projection="3d")
        for body, color in enumerate(colors):
            axis.plot(*payload.positions[:, body].T, color=color, label=f"Body {body + 1}")
            axis.scatter(*payload.positions[0, body], color=color, marker="o")
            axis.scatter(*payload.positions[-1, body], color=color, marker="x")
        axis.set_zlabel("z")
        axis.set_zlim(lower, upper)
    else:
        fig, axis = plt.subplots()
        for body, color in enumerate(colors):
            axis.plot(
                payload.positions[:, body, 0],
                payload.positions[:, body, 1],
                color=color,
                label=f"Body {body + 1}",
            )
            axis.scatter(*payload.positions[0, body], color=color, marker="o")
            axis.scatter(*payload.positions[-1, body], color=color, marker="x")
        axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"Trajectories ({reference_frame} frame; circle=start, ×=end)")
    axis.legend(loc="best", fontsize="small")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def create_phase_space_figure(result: NBodyResult, body: int) -> Figure:
    """Plot Cartesian coordinate-velocity projections and radial reduced phase space."""
    import matplotlib.pyplot as plt

    if not 0 <= body < result.masses.size:
        raise ValueError("Selected body index is out of range.")
    dimension = result.dimension
    fig, axes = plt.subplots(1, dimension + 1, figsize=(4.2 * (dimension + 1), 3.7))
    position = result.positions[:, body]
    velocity = result.velocities[:, body]
    for component in range(dimension):
        axes[component].plot(position[:, component], velocity[:, component])
        axes[component].set_xlabel(f"{'xyz'[component]}")
        axes[component].set_ylabel(f"v_{'xyz'[component]}")
        axes[component].grid(True, alpha=0.25)
    relative_position = position - result.center_of_mass
    relative_velocity = velocity - result.center_of_mass_velocity
    radius = np.linalg.norm(relative_position, axis=1)
    radial_velocity = np.divide(
        np.sum(relative_position * relative_velocity, axis=1),
        radius,
        out=np.zeros_like(radius),
        where=radius > 0.0,
    )
    axes[-1].plot(radius, radial_velocity)
    axes[-1].set_xlabel("r from COM")
    axes[-1].set_ylabel("v_r")
    axes[-1].grid(True, alpha=0.25)
    fig.suptitle(f"Phase space — body {body + 1}")
    fig.tight_layout()
    return fig


def create_energy_figure(result: NBodyResult) -> Figure:
    """Plot energy terms and safely normalized total-energy error."""
    import matplotlib.pyplot as plt

    fig, (energy_axis, error_axis) = plt.subplots(2, 1, sharex=True)
    energy_axis.plot(result.t, result.kinetic_energy, label="K")
    energy_axis.plot(result.t, result.potential_energy, label="U")
    energy_axis.plot(result.t, result.total_energy, label="E")
    energy_axis.set_ylabel("Energy")
    energy_axis.legend()
    energy_axis.grid(True, alpha=0.25)
    baseline = abs(float(result.total_energy[0]))
    scale = (
        baseline
        if baseline > np.finfo(float).eps
        else max(float(np.max(np.abs(result.total_energy))), 1.0)
    )
    error_axis.plot(result.t, (result.total_energy - result.total_energy[0]) / scale)
    error_axis.set_xlabel("time")
    error_axis.set_ylabel("relative ΔE")
    error_axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


class GravitationalNBodyResultDialog:
    """Resizable notebook with visualization-only selectors over an NBodyResult."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: NBodyResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Gravitational N-Body Dynamics Results")
        self._anim_canvas = self._trajectory_canvas = self._phase_canvas = None
        self._energy_canvas = self._diagnostics_canvas = self._separation_canvas = None
        self._build_ui()
        self._shell.finish(AdvancedResultSize(1400, 900))
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        close_embedded_figures(
            self,
            (
                "_anim_canvas",
                "_trajectory_canvas",
                "_phase_canvas",
                "_energy_canvas",
                "_diagnostics_canvas",
                "_separation_canvas",
            ),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        labels = {
            "max_relative_total_energy_drift": "Maximum relative total-energy drift",
            "max_linear_momentum_drift": "Maximum linear-momentum drift",
            "max_angular_momentum_drift": "Maximum angular-momentum drift",
            "max_com_displacement_from_uniform_motion": (
                "Maximum COM displacement from uniform motion"
            ),
            "minimum_pair_separation": "Minimum pair separation",
        }
        summary = format_result_summary(
            tuple(
                (labels.get(key, key.replace("_", " ").capitalize()), f"{value:.3e}")
                for key, value in self._result.magnitudes.items()
            )
        )
        self._shell = AdvancedResultShell(
            self.win,
            title="Gravitational N-Body Results",
            summary=summary,
            close_command=self._on_close,
        )
        notebook = self._shell.notebook
        for title, build in (
            ("Orbit Animation", self._build_orbit_tab),
            ("Trajectories", self._build_trajectory_tab),
            ("Phase Space", self._build_phase_tab),
            ("Energy", self._build_energy_tab),
            ("Conserved Quantities / Diagnostics", self._build_diagnostics_tab),
            ("Separations", self._build_separations_tab),
        ):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=title)
            build(tab)

    def _replace(self, frame: ttk.Frame, figure: Figure, attribute: str) -> None:
        setattr(
            self,
            attribute,
            replace_plot_in_tk(figure, frame, current_canvas=getattr(self, attribute)),
        )

    def _build_orbit_tab(self, tab: ttk.Frame) -> None:
        controls = make_view_controls(tab)
        group = controls.add_group(requested_width=230)
        ttk.Label(group, text="Reference frame:").pack(side=tk.LEFT)
        self._orbit_frame_var = tk.StringVar(value="Inertial")
        combo = ttk.Combobox(
            group,
            textvariable=self._orbit_frame_var,
            values=("Inertial", "Center of mass"),
            state="readonly",
            width=16,
        )
        combo.pack(side=tk.LEFT, padx=5)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._update_orbit())
        self._orbit_plot_frame = ttk.Frame(tab)
        self._orbit_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_orbit()

    def _update_orbit(self) -> None:
        reset_embedded_animation(self._orbit_plot_frame, self._anim_canvas)
        payload = orbit_animation_payload(
            self._result, cast(_ReferenceFrame, self._orbit_frame_var.get())
        )
        figure = create_orbit_animation_figure(payload)
        self._anim_canvas = embed_animation_plot_in_tk(
            figure,
            self._orbit_plot_frame,
            on_export_mp4=lambda duration: self._export_orbit(payload, duration),
        )

    def _export_orbit(self, payload: OrbitAnimationPayload, duration: float) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".mp4",
            initialdir=str(get_output_dir()),
            initialfile=f"{generate_output_basename(prefix='n_body_orbit')}.mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if not filename:
            return
        try:
            export_animated_figure_to_mp4(
                create_orbit_animation_figure(payload), Path(filename), duration_seconds=duration
            )
            messagebox.showinfo(
                "Animation export saved", f"Animation was saved to:\n{filename}", parent=self.win
            )
        except RuntimeError as exc:
            messagebox.showerror("Animation export was not saved", str(exc), parent=self.win)

    def _build_trajectory_tab(self, tab: ttk.Frame) -> None:
        controls = make_view_controls(tab)
        self._trajectory_frame_var = tk.StringVar(value="Inertial")
        group = controls.add_group(requested_width=230)
        ttk.Label(group, text="Reference frame:").pack(side=tk.LEFT)
        combo = ttk.Combobox(
            group,
            textvariable=self._trajectory_frame_var,
            values=("Inertial", "Center of mass"),
            state="readonly",
            width=16,
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._update_trajectory())
        self._trajectory_plot_frame = ttk.Frame(tab)
        self._trajectory_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_trajectory()

    def _update_trajectory(self) -> None:
        self._replace(
            self._trajectory_plot_frame,
            create_trajectory_figure(
                self._result, cast(_ReferenceFrame, self._trajectory_frame_var.get())
            ),
            "_trajectory_canvas",
        )

    def _build_phase_tab(self, tab: ttk.Frame) -> None:
        controls = make_view_controls(tab)
        self._phase_body_var = tk.StringVar(value="1")
        group = controls.add_group(requested_width=130)
        ttk.Label(group, text="Body:").pack(side=tk.LEFT)
        combo = ttk.Combobox(
            group,
            textvariable=self._phase_body_var,
            values=[str(index + 1) for index in range(self._result.masses.size)],
            state="readonly",
            width=8,
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._update_phase())
        self._phase_plot_frame = ttk.Frame(tab)
        self._phase_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_phase()

    def _update_phase(self) -> None:
        self._replace(
            self._phase_plot_frame,
            create_phase_space_figure(self._result, int(self._phase_body_var.get()) - 1),
            "_phase_canvas",
        )

    def _build_energy_tab(self, tab: ttk.Frame) -> None:
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True)
        self._replace(frame, create_energy_figure(self._result), "_energy_canvas")

    def _build_diagnostics_tab(self, tab: ttk.Frame) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.ravel()
        axes[0].plot(self._result.t, self._result.linear_momentum)
        axes[0].set_title("Linear momentum")
        axes[1].plot(self._result.t, self._result.angular_momentum)
        axes[1].set_title("Angular momentum")
        expected = (
            self._result.center_of_mass[0]
            + (self._result.t - self._result.t[0])[:, None] * self._result.center_of_mass_velocity
        )
        axes[2].plot(
            self._result.t,
            np.linalg.norm(self._result.center_of_mass - expected, axis=1),
            label="COM deviation",
        )
        axes[2].plot(self._result.t, self._result.moment_of_inertia, label="I_COM")
        axes[2].legend()
        axes[3].plot(self._result.t, self._result.virial_ratio)
        axes[3].set_title("Virial ratio 2K/|U|")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.set_xlabel("time")
        fig.tight_layout()
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True)
        self._replace(frame, fig, "_diagnostics_canvas")

    def _build_separations_tab(self, tab: ttk.Frame) -> None:
        controls = make_view_controls(tab)
        bodies = [str(index + 1) for index in range(self._result.masses.size)]
        self._pair_a_var, self._pair_b_var = tk.StringVar(value="1"), tk.StringVar(value="2")
        group = controls.add_group(requested_width=220)
        ttk.Label(group, text="Pair:").pack(side=tk.LEFT)
        pair_a = ttk.Combobox(
            group, textvariable=self._pair_a_var, values=bodies, state="readonly", width=8
        )
        pair_a.pack(side=tk.LEFT)
        pair_b = ttk.Combobox(
            group, textvariable=self._pair_b_var, values=bodies, state="readonly", width=8
        )
        pair_b.pack(side=tk.LEFT, padx=4)
        pair_a.bind("<<ComboboxSelected>>", lambda _event: self._update_separations())
        pair_b.bind("<<ComboboxSelected>>", lambda _event: self._update_separations())
        self._separation_plot_frame = ttk.Frame(tab)
        self._separation_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_separations()

    def _update_separations(self) -> None:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots()
        axis.plot(self._result.t, self._result.min_pair_distance, label="Global minimum")
        a, b = int(self._pair_a_var.get()) - 1, int(self._pair_b_var.get()) - 1
        if a != b:
            axis.plot(
                self._result.t,
                selected_pair_distance(self._result, a, b),
                label=f"Bodies {a + 1}–{b + 1}",
            )
        if self._result.masses.size == 3:
            for a, b in ((0, 1), (0, 2), (1, 2)):
                axis.plot(
                    self._result.t,
                    selected_pair_distance(self._result, a, b),
                    alpha=0.55,
                    label=f"{a + 1}–{b + 1}",
                )
        axis.set_xlabel("time")
        axis.set_ylabel("separation")
        axis.legend()
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        self._replace(self._separation_plot_frame, fig, "_separation_canvas")
