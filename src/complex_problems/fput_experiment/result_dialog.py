"""Cached result notebook for FPUT simulations and recurrence sweeps."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from complex_problems.fput_experiment.model import bond_strain
from complex_problems.fput_experiment.solver import FPUTResult, FPUTSweepResult
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.window_utils import center_window, make_modal

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
        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        center_window(self.win, width=1200, height=800, max_width_ratio=0.96, resizable=True)
        make_modal(self.win, parent)

    def _on_close(self) -> None:
        """Destroy all embedded figures before closing the result window."""
        from matplotlib import pyplot as plt

        for canvas in self._canvases:
            figure = getattr(canvas, "figure", None)
            if figure is not None:
                plt.close(figure)
        self.win.destroy()

    def _add_plot(self, notebook: ttk.Notebook, title: str, figure: Figure) -> None:
        """Add a canvas-bearing tab and keep a handle for deterministic cleanup."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"  {title}  ")
        self._canvases.append(embed_plot_in_tk(figure, frame))

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        if isinstance(self.result, FPUTSweepResult):
            self._build_sweep(notebook)
            return
        result = self.result
        peaks = result.recurrence_peak_indices
        recurrence = _line_figure(
            result.t,
            [(result.recurrence_fidelity, "F(t)")],
            "Modal-energy recurrence fidelity",
            "t",
            "F",
        )
        axis = recurrence.axes[0]
        if peaks.size:
            axis.scatter(
                result.t[peaks],
                result.recurrence_fidelity[peaks],
                color="tab:red",
                label="accepted peaks",
            )
            axis.legend()
        self._add_plot(notebook, "Overview / Recurrence", recurrence)
        self._build_animation_tab(notebook)
        strain = bond_strain(result.displacement)
        self._add_plot(
            notebook,
            "Space-Time",
            self._heatmap(result.t, strain, "Strain space-time map", "bond coordinate", "strain"),
        )
        self._add_plot(notebook, "3D Surface", self._surface(result.t, strain, "Strain surface"))
        self._add_plot(
            notebook,
            "Modal Energies",
            self._heatmap(result.t, result.modal_energy, "Linear modal energy", "mode", "E_k"),
        )
        self._add_plot(
            notebook,
            "Phase Space",
            _line_figure(
                result.modal_q[:, 0],
                [(result.modal_p[:, 0], "mode 1")],
                "Normal-mode phase space",
                "Q₁",
                "P₁",
            ),
        )
        self._add_plot(
            notebook,
            "Hamiltonian",
            _line_figure(
                result.t,
                [
                    (result.kinetic_energy, "kinetic"),
                    (result.harmonic_potential_energy, "harmonic potential"),
                    (result.nonlinear_potential_energy, "nonlinear potential"),
                    (result.total_energy, "exact H"),
                ],
                "Hamiltonian components",
                "t",
                "energy",
            ),
        )
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

    def _build_animation_tab(self, notebook: ttk.Notebook) -> None:
        """Embed selectable cached displacement/strain animation with matching MP4 export."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="  Lattice Animation  ")
        controls = ttk.Frame(tab)
        controls.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(controls, text="Display:").pack(side=tk.LEFT)
        combo = ttk.Combobox(
            controls,
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
        from complex_problems.common.result_dialog_ui import reset_embedded_animation

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

    def _heatmap(
        self, t: np.ndarray, values: np.ndarray, title: str, xlabel: str, label: str
    ) -> Figure:
        """Create a cached two-dimensional result view."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 5), tight_layout=True)
        axis = figure.add_subplot(111)
        image = axis.imshow(
            values, origin="lower", aspect="auto", extent=(0.5, values.shape[1] + 0.5, t[0], t[-1])
        )
        axis.set(title=title, xlabel=xlabel, ylabel="t")
        figure.colorbar(image, ax=axis, label=label)
        return figure

    def _surface(self, t: np.ndarray, values: np.ndarray, title: str) -> Figure:
        """Create a cached coordinate-time-amplitude surface."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 5), tight_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        coordinate, time = np.meshgrid(np.arange(values.shape[1]), t)
        axis.plot_surface(
            coordinate,
            time,
            values,
            cmap="viridis",
            rcount=min(100, values.shape[0]),
            ccount=values.shape[1],
        )
        axis.set(title=title, xlabel="bond", ylabel="t", zlabel="strain")
        return figure

    def _build_sweep(self, notebook: ttk.Notebook) -> None:
        """Show detected and explicitly undetected recurrence-scaling measurements."""
        result = self.result
        assert isinstance(result, FPUTSweepResult)
        series = [
            (result.first_recurrence_times, "first recurrence time"),
            (result.recurrence_fidelities, "recurrence fidelity"),
        ]
        self._add_plot(
            notebook,
            "Recurrence Scaling",
            _line_figure(
                result.parameter_values,
                series,
                "Recurrence scaling (NaN = undetected)",
                result.sweep_variable,
                "value",
            ),
        )
