"""Cached result notebook for FPUT simulations and recurrence sweeps."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Callable, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from complex_problems.common.result_dialog_ui import (
    AdvancedResultShell,
    AdvancedResultSize,
    close_embedded_figure,
    make_view_controls,
    reset_embedded_animation,
)
from complex_problems.fput_experiment.model import bond_strain
from complex_problems.fput_experiment.solver import FPUTResult, FPUTSweepResult
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.window_utils import make_modal

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class FPUTAnimationPayload:
    """Prepared cached representation used identically by display and MP4 export."""

    representation: Literal["displacement", "strain"]
    t: np.ndarray
    coordinates: np.ndarray
    frames: np.ndarray
    title: str
    ylabel: str


def prepare_fput_animation_payload(
    result: FPUTResult, representation: Literal["displacement", "strain"]
) -> FPUTAnimationPayload:
    """Prepare displacement or strain frames without rerunning or recomputing a solve."""
    if representation == "displacement":
        frames = np.pad(result.displacement, ((0, 0), (1, 1)))
        return FPUTAnimationPayload(
            representation,
            result.t,
            np.arange(result.displacement.shape[1] + 2, dtype=float),
            frames,
            "FPUT displacement (fixed endpoints)",
            "x_j",
        )
    if representation == "strain":
        frames = bond_strain(result.displacement)
        return FPUTAnimationPayload(
            representation,
            result.t,
            np.arange(frames.shape[1], dtype=float) + 0.5,
            frames,
            "FPUT bond strain",
            "delta_j",
        )
    raise ValueError("representation must be 'displacement' or 'strain'.")


def create_fput_animation_figure(payload: FPUTAnimationPayload) -> Figure:
    """Create one shared-metadata animation figure from prepared cached payload data."""
    from plotting import create_line_animation_plot

    return create_line_animation_plot(
        payload.t,
        payload.coordinates,
        payload.frames,
        title=payload.title,
        xlabel="particle coordinate"
        if payload.representation == "displacement"
        else "bond coordinate",
        ylabel=payload.ylabel,
        symmetric_y_range=True,
    )


def export_fput_animation_to_mp4(
    payload: FPUTAnimationPayload, filepath: Path, *, duration_seconds: float = 5.0
) -> Path:
    """Export a fresh figure from the selected cached representation without resolving."""
    from plotting import export_animated_figure_to_mp4

    return export_animated_figure_to_mp4(
        create_fput_animation_figure(payload), filepath, duration_seconds=duration_seconds
    )


def _line_figure(
    x: np.ndarray, series: list[tuple[np.ndarray, str]], title: str, xlabel: str, ylabel: str
) -> Figure:
    """Create a small independent Matplotlib line figure from cached arrays."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5), tight_layout=True)
    axis = figure.add_subplot(111)
    for y, label in series:
        axis.plot(x, y, label=label)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(True, alpha=0.25)
    if len(series) > 1:
        axis.legend()
    return figure


def fundamental_angular_frequency(n_particles: int) -> float:
    """Return the fixed-end linear angular frequency of the fundamental mode."""
    if n_particles < 2:
        raise ValueError("FPUT requires at least two particles.")
    return float(2.0 * np.sin(np.pi / (2.0 * (n_particles + 1))))


def time_to_fundamental_cycles(time: ArrayLike, n_particles: int) -> np.ndarray | float:
    """Convert physical time to fundamental linear-mode cycles."""
    cycles = np.asarray(time) * fundamental_angular_frequency(n_particles) / (2.0 * np.pi)
    return float(cycles) if np.ndim(time) == 0 else cycles


def fundamental_cycles_to_time(cycles: ArrayLike, n_particles: int) -> np.ndarray | float:
    """Convert fundamental linear-mode cycles to physical time."""
    time = np.asarray(cycles) * 2.0 * np.pi / fundamental_angular_frequency(n_particles)
    return float(time) if np.ndim(cycles) == 0 else time


def _format_value(value: float | int | None) -> str:
    """Format cached scalar diagnostics without exposing Python None values."""
    return "not detected" if value is None else f"{value:.6g}"


def format_fput_summary(result: FPUTResult) -> str:
    """Return the visible single-run summary using cached diagnostics only."""
    summary = result.summary
    metadata = result.metadata
    first_time = _format_value(summary["first_recurrence_time"])
    first_fidelity = _format_value(summary["first_recurrence_fidelity"])
    initial_hamiltonian = _format_value(summary["initial_hamiltonian"])
    final_hamiltonian = _format_value(summary["final_hamiltonian"])
    absolute_drift = _format_value(summary["maximum_absolute_hamiltonian_drift"])
    relative_drift = _format_value(summary["maximum_relative_hamiltonian_drift"])
    identity_error = _format_value(summary["maximum_interaction_energy_identity_error"])
    return "\n".join(
        (
            f"Model: {result.model}; coefficient: {result.coefficient:.6g}; "
            f"integrator: {metadata['integrator']}; requested dt: {metadata['requested_dt']:.6g}",
            f"Integration steps: {metadata['number_of_integration_steps']}; "
            f"saved frames: {result.t.size}; first recurrence: {first_time}; "
            f"fidelity: {first_fidelity}; "
            f"accepted peaks: {summary['number_of_recurrence_peaks']}",
            f"Initial H: {initial_hamiltonian}; final H: {final_hamiltonian}; "
            f"max |ΔH|: {absolute_drift}; "
            f"max relative |ΔH|: {relative_drift}; max |H - ΣEₖ - Vnl|: {identity_error}",
        )
    )


def format_fput_sweep_summary(result: FPUTSweepResult) -> str:
    """Return the compact cached recurrence-scaling summary."""
    fit = "; ".join(
        (
            f"slope: {_format_value(result.slope)}",
            f"intercept: {_format_value(result.intercept)}",
            f"R²: {_format_value(result.r_squared)}",
        )
    )
    return "\n".join(
        (
            (
                f"Sweep variable: {result.sweep_variable}; runs: {result.parameter_values.size}; "
                f"detected: {np.count_nonzero(result.detected)}"
            ),
            fit,
        )
    )


def create_recurrence_figure(result: FPUTResult) -> Figure:
    """Plot cached recurrence fidelity with actual fundamental-cycle coordinates."""
    recurrence = _line_figure(
        result.t, [(result.recurrence_fidelity, "F(t)")], "Recurrence fidelity", "t", "F"
    )
    axis = recurrence.axes[0]
    peaks = result.recurrence_peak_indices
    if peaks.size:
        axis.scatter(
            result.t[peaks],
            result.recurrence_fidelity[peaks],
            color="tab:red",
            label="accepted peaks",
        )
        axis.plot(
            result.t[peaks],
            result.recurrence_fidelity[peaks],
            color="tab:red",
            linewidth=0.8,
            alpha=0.75,
            label="peak sequence",
        )
        late_time = result.summary["highest_late_recurrence_time"]
        late_fidelity = result.summary["highest_late_recurrence_fidelity"]
        if late_time is not None and late_fidelity is not None:
            axis.annotate(
                "superrecurrence candidate",
                (late_time, late_fidelity),
                xytext=(6, 8),
                textcoords="offset points",
            )
        axis.legend()
    n_particles = result.displacement.shape[1]
    axis.secondary_xaxis(
        "top",
        functions=(
            lambda time: time_to_fundamental_cycles(time, n_particles),
            lambda cycles: fundamental_cycles_to_time(cycles, n_particles),
        ),
    ).set_xlabel("fundamental cycles")
    return recurrence


def create_hamiltonian_figure(result: FPUTResult) -> Figure:
    """Separate energy-scale curves from the cached relative numerical error."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 6), layout="constrained")
    energy_axis, error_axis = figure.subplots(2, 1, sharex=True)
    interaction = result.total_energy - np.sum(result.modal_energy, axis=1)
    for values, label in (
        (result.kinetic_energy, "kinetic"),
        (result.harmonic_potential_energy, "harmonic potential"),
        (result.nonlinear_potential_energy, "nonlinear potential"),
        (result.total_energy, "total H"),
        (interaction, "H - ΣE_k"),
    ):
        energy_axis.plot(result.t, values, label=label)
    scale = max(abs(float(result.total_energy[0])), np.finfo(float).eps)
    error_axis.plot(
        result.t, (result.total_energy - result.total_energy[0]) / scale, label="relative H error"
    )
    energy_axis.set(title="Hamiltonian components and interaction identity", ylabel="energy")
    error_axis.set(xlabel="t", ylabel="relative H error")
    for axis in (energy_axis, error_axis):
        axis.grid(True, alpha=0.25)
        axis.legend()
    return figure


def create_modal_energy_figure(result: FPUTResult, modes: list[int], scale_name: str) -> Figure:
    """Plot cached selected modes above the complete modal-energy heatmap."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 6), layout="constrained")
    curve_axis, heatmap_axis = figure.subplots(2, 1, height_ratios=(1, 1.2))
    for mode in modes:
        curve_axis.plot(result.t, result.modal_energy[:, mode - 1], label=f"E_{mode}")
    curve_axis.set(title="Selected modal energies", ylabel="E_k")
    curve_axis.grid(True, alpha=0.25)
    curve_axis.legend()
    values = result.modal_energy
    label = "E_k"
    if scale_name == "log10 normalized":
        reference = max(float(np.sum(values[0])), np.finfo(float).eps)
        values = np.log10(np.maximum(values / reference, np.finfo(float).eps))
        label = "log10(E_k / initial total linear modal energy)"
    image = heatmap_axis.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        extent=(result.t[0], result.t[-1], 0.5, values.shape[1] + 0.5),
    )
    heatmap_axis.set(title="All-mode heatmap", xlabel="t", ylabel="mode")
    figure.colorbar(image, ax=heatmap_axis, label=label)
    return figure


class FPUTResultDialog:
    """Visualization-rich notebook that never reruns a completed solver."""

    def __init__(
        self, parent: tk.Tk | tk.Toplevel, *, result: FPUTResult | FPUTSweepResult
    ) -> None:
        self.parent, self.result = parent, result
        self.win = tk.Toplevel(parent)
        self.win.title("FPUT Results")
        self._canvases: list[object] = []
        self._animation_canvas: object | None = None
        self._animation_plot_frame: ttk.Frame | None = None
        self._representation_var = tk.StringVar(value="strain")
        self._space_representation_var = tk.StringVar(value="strain")
        self._surface_representation_var = tk.StringVar(value="strain")
        self._modal_scale_var = tk.StringVar(value="Linear")
        self._phase_kind_var = tk.StringVar(value="Normal mode")
        self._phase_index_var = tk.StringVar(value="1")
        self._mode_selection_var = tk.StringVar(value="1, 2, 3")
        self._build_ui()
        self._shell.finish(AdvancedResultSize(1200, 800))
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        """Destroy all embedded figures before closing the result window."""
        for canvas in self._canvases:
            close_embedded_figure(canvas)
        self.win.destroy()

    def _add_plot(self, notebook: ttk.Notebook, title: str, figure: Figure) -> None:
        """Add a canvas-bearing tab and keep a handle for deterministic cleanup."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        self._canvases.append(embed_plot_in_tk(figure, frame))

    def _build_ui(self) -> None:
        summary_text = (
            format_fput_sweep_summary(self.result)
            if isinstance(self.result, FPUTSweepResult)
            else format_fput_summary(self.result)
        )
        self._shell = AdvancedResultShell(
            self.win,
            title="FPUT Results",
            summary=summary_text,
            close_command=self._on_close,
        )
        notebook = self._shell.notebook
        if isinstance(self.result, FPUTSweepResult):
            self._build_sweep(notebook)
            return
        result = self.result
        self._add_plot(notebook, "Overview / Recurrence", create_recurrence_figure(result))
        self._build_animation_tab(notebook)
        self._build_representation_tab(
            notebook, "Space-Time", self._space_representation_var, self._space_time_figure
        )
        self._build_representation_tab(
            notebook, "3D Surface", self._surface_representation_var, self._surface_figure
        )
        self._build_modal_tab(notebook)
        self._build_phase_tab(notebook)
        self._add_plot(notebook, "Hamiltonian", create_hamiltonian_figure(result))
        self._add_plot(
            notebook,
            "Thermalization",
            _line_figure(
                result.t,
                [
                    (result.spectral_entropy, "normalized spectral entropy"),
                    (result.participation_number, "participation number"),
                ],
                "Modal spreading diagnostics",
                "t",
                "diagnostic",
            ),
        )

    def _representation_values(
        self, representation: str
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        """Return cached values and physical coordinates for a selected representation."""
        assert isinstance(self.result, FPUTResult)
        if representation == "displacement":
            values = np.pad(self.result.displacement, ((0, 0), (1, 1)))
            return (
                values,
                np.arange(values.shape[1], dtype=float),
                "particle coordinate",
                "displacement",
            )
        values = bond_strain(self.result.displacement)
        return values, np.arange(values.shape[1], dtype=float) + 0.5, "bond coordinate", "strain"

    def _build_representation_tab(
        self,
        notebook: ttk.Notebook,
        title: str,
        variable: tk.StringVar,
        builder: Callable[[str], Figure],
    ) -> None:
        """Build a cached displacement/strain display with no solver callback."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        controls = make_view_controls(tab)
        group = controls.add_group(requested_width=220)
        ttk.Label(group, text="Display:").pack(side=tk.LEFT)
        combo = ttk.Combobox(
            group,
            textvariable=variable,
            values=("displacement", "strain"),
            state="readonly",
            width=16,
        )
        combo.pack(side=tk.LEFT, padx=6)
        plot_frame = ttk.Frame(tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        canvas: list[object | None] = [None]

        def update(_event: object | None = None) -> None:
            reset_embedded_animation(plot_frame, canvas[0])
            figure = cast("Figure", builder(variable.get()))
            canvas[0] = embed_plot_in_tk(figure, plot_frame)
            self._canvases.append(canvas[0])

        combo.bind("<<ComboboxSelected>>", update)
        update()

    def _space_time_figure(self, representation: str) -> Figure:
        """Build a space-time map from cached displacement or bond strain."""
        assert isinstance(self.result, FPUTResult)
        values, coordinate, xlabel, label = self._representation_values(representation)
        return self._heatmap(
            self.result.t, values, f"{label.title()} space-time map", xlabel, label, coordinate
        )

    def _surface_figure(self, representation: str) -> Figure:
        """Build a coordinate-time surface from cached displacement or bond strain."""
        assert isinstance(self.result, FPUTResult)
        values, coordinate, xlabel, label = self._representation_values(representation)
        return self._surface(
            self.result.t, values, f"{label.title()} surface", coordinate, xlabel, label
        )

    def _build_animation_tab(self, notebook: ttk.Notebook) -> None:
        """Embed selectable cached displacement/strain animation with matching MP4 export."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Lattice Animation")
        controls = make_view_controls(tab)
        group = controls.add_group(requested_width=220)
        ttk.Label(group, text="Display:").pack(side=tk.LEFT)
        combo = ttk.Combobox(
            group,
            textvariable=self._representation_var,
            values=("displacement", "strain"),
            state="readonly",
            width=16,
        )
        combo.pack(side=tk.LEFT, padx=6)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._update_animation())
        self._animation_plot_frame = ttk.Frame(tab)
        self._animation_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_animation()

    def _update_animation(self) -> None:
        """Replace the animation using the existing result, never the numerical solver."""
        if not isinstance(self.result, FPUTResult) or self._animation_plot_frame is None:
            return
        reset_embedded_animation(self._animation_plot_frame, self._animation_canvas)
        representation = self._representation_var.get()
        if representation not in {"displacement", "strain"}:
            raise ValueError("Unknown FPUT animation representation.")
        payload = prepare_fput_animation_payload(
            self.result, cast(Literal["displacement", "strain"], representation)
        )
        self._animation_canvas = embed_animation_plot_in_tk(
            create_fput_animation_figure(payload),
            self._animation_plot_frame,
            on_export_mp4=lambda duration: self._export_animation(payload, duration),
        )
        self._canvases.append(self._animation_canvas)

    def _export_animation(self, payload: FPUTAnimationPayload, duration_seconds: float) -> None:
        """Export a fresh payload figure so the selected display matches the video exactly."""
        filename = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".mp4",
            initialfile=f"fput_{payload.representation}.mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if not filename:
            return
        try:
            export_fput_animation_to_mp4(payload, Path(filename), duration_seconds=duration_seconds)
            messagebox.showinfo(
                "Animation export saved", f"Animation was saved to:\n{filename}", parent=self.win
            )
        except RuntimeError as exc:
            messagebox.showerror("Animation export was not saved", str(exc), parent=self.win)

    def _build_modal_tab(self, notebook: ttk.Notebook) -> None:
        """Provide selected cached modal curves and a selectable all-mode heatmap scale."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Modal Energies")
        controls = make_view_controls(tab)
        mode_group = controls.add_group(requested_width=280)
        ttk.Label(mode_group, text="Modes (1..N):").pack(side=tk.LEFT)
        mode_entry = ttk.Entry(mode_group, textvariable=self._mode_selection_var, width=14)
        mode_entry.pack(side=tk.LEFT, padx=5)
        update_button = ttk.Button(mode_group, text="Update")
        update_button.pack(side=tk.LEFT, padx=(0, 8))
        scale_group = controls.add_group(requested_width=190)
        ttk.Label(scale_group, text="Heatmap scale:").pack(side=tk.LEFT)
        scale = ttk.Combobox(
            scale_group,
            textvariable=self._modal_scale_var,
            values=("Linear", "log10 normalized"),
            state="readonly",
            width=18,
        )
        scale.pack(side=tk.LEFT, padx=5)
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True)
        canvas: list[object | None] = [None]

        def update(_event: object | None = None) -> None:
            reset_embedded_animation(frame, canvas[0])
            assert isinstance(self.result, FPUTResult)
            try:
                modes = [int(value.strip()) for value in self._mode_selection_var.get().split(",")]
            except ValueError:
                modes = [1, 2, 3]
            modes = [mode for mode in modes if 1 <= mode <= self.result.modal_energy.shape[1]] or [
                1
            ]
            figure = create_modal_energy_figure(self.result, modes, self._modal_scale_var.get())
            canvas[0] = embed_plot_in_tk(figure, frame)
            self._canvases.append(canvas[0])

        update_button.configure(command=update)
        mode_entry.bind("<Return>", update)
        scale.bind("<<ComboboxSelected>>", update)
        update()

    def _build_phase_tab(self, notebook: ttk.Notebook) -> None:
        """Provide particle or normal-mode phase portraits from cached coordinates."""
        assert isinstance(self.result, FPUTResult)
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Phase Space")
        controls = make_view_controls(tab)
        kind_group = controls.add_group(requested_width=150)
        kind = ttk.Combobox(
            kind_group,
            textvariable=self._phase_kind_var,
            values=("Particle", "Normal mode"),
            state="readonly",
            width=14,
        )
        kind.pack(side=tk.LEFT)
        index_group = controls.add_group(requested_width=170)
        ttk.Label(index_group, text="Index (1..N):").pack(side=tk.LEFT, padx=(0, 4))
        index = ttk.Spinbox(
            index_group,
            from_=1,
            to=self.result.displacement.shape[1],
            textvariable=self._phase_index_var,
            width=6,
        )
        index.pack(side=tk.LEFT)
        update_button = ttk.Button(index_group, text="Update")
        update_button.pack(side=tk.LEFT, padx=(6, 0))
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True)
        canvas: list[object | None] = [None]

        def update(_event: object | None = None) -> None:
            reset_embedded_animation(frame, canvas[0])
            assert isinstance(self.result, FPUTResult)
            try:
                selected = (
                    min(max(int(self._phase_index_var.get()), 1), self.result.displacement.shape[1])
                    - 1
                )
            except ValueError:
                selected = 0
            if self._phase_kind_var.get() == "Particle":
                figure = _line_figure(
                    self.result.displacement[:, selected],
                    [(self.result.velocity[:, selected], f"particle {selected + 1}")],
                    "Particle phase space",
                    f"x_{selected + 1}",
                    f"v_{selected + 1}",
                )
            else:
                figure = _line_figure(
                    self.result.modal_q[:, selected],
                    [(self.result.modal_p[:, selected], f"mode {selected + 1}")],
                    "Normal-mode phase space",
                    f"Q_{selected + 1}",
                    f"P_{selected + 1}",
                )
            canvas[0] = embed_plot_in_tk(figure, frame)
            self._canvases.append(canvas[0])

        kind.bind("<<ComboboxSelected>>", update)
        update_button.configure(command=update)
        index.bind("<Return>", update)
        update()

    def _heatmap(
        self,
        t: np.ndarray,
        values: np.ndarray,
        title: str,
        xlabel: str,
        label: str,
        coordinate: np.ndarray | None = None,
    ) -> Figure:
        """Create a cached two-dimensional result view."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 5), tight_layout=True)
        axis = figure.add_subplot(111)
        extent = (
            (coordinate[0], coordinate[-1], t[0], t[-1])
            if coordinate is not None
            else (0.5, values.shape[1] + 0.5, t[0], t[-1])
        )
        image = axis.imshow(values, origin="lower", aspect="auto", extent=extent)
        axis.set(title=title, xlabel=xlabel, ylabel="t")
        figure.colorbar(image, ax=axis, label=label)
        return figure

    def _surface(
        self,
        t: np.ndarray,
        values: np.ndarray,
        title: str,
        coordinate: np.ndarray,
        xlabel: str,
        zlabel: str,
    ) -> Figure:
        """Create a cached coordinate-time-amplitude surface."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 5), tight_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        coordinate_grid, time = np.meshgrid(coordinate, t)
        axis.plot_surface(
            coordinate_grid,
            time,
            values,
            cmap="viridis",
            rcount=min(100, values.shape[0]),
            ccount=values.shape[1],
        )
        axis.set(title=title, xlabel=xlabel, ylabel="t", zlabel=zlabel)
        return figure

    def _build_sweep(self, notebook: ttk.Notebook) -> None:
        """Show detected and explicitly undetected recurrence-scaling measurements."""
        result = self.result
        assert isinstance(result, FPUTSweepResult)
        self._add_plot(
            notebook,
            "Recurrence Time",
            _line_figure(
                result.parameter_values,
                [(result.first_recurrence_times, "first recurrence time")],
                "First recurrence time (NaN = undetected)",
                result.sweep_variable,
                "T_R",
            ),
        )
        detected = result.detected & (result.parameter_values > 0.0)
        log_x = np.log(result.parameter_values[detected])
        log_y = np.log(result.first_recurrence_times[detected])
        figure = _line_figure(
            log_x,
            [(log_y, "detected runs")],
            "Log-log recurrence scaling",
            f"log({result.sweep_variable})",
            "log(T_R)",
        )
        if result.slope is not None and result.intercept is not None:
            axis = figure.axes[0]
            axis.plot(log_x, result.slope * log_x + result.intercept, label="least-squares fit")
            axis.set_title(
                f"Log-log scaling: slope={result.slope:.4g}, "
                f"intercept={result.intercept:.4g}, R²={result.r_squared:.4g}"
            )
            axis.legend()
        self._add_plot(notebook, "Log-Log Scaling", figure)
        self._add_plot(
            notebook,
            "Quality",
            _line_figure(
                result.parameter_values,
                [
                    (result.recurrence_fidelities, "recurrence fidelity"),
                    (result.max_relative_hamiltonian_drift, "max relative Hamiltonian drift"),
                ],
                "Sweep quality (NaN = undetected recurrence)",
                result.sweep_variable,
                "quality",
            ),
        )
