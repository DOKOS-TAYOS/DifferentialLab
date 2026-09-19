"""Result dialog for nonlinear wave simulations (NLSE/KdV)."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, cast

import numpy as np

from complex_problems.common.result_dialog_ui import (
    close_embedded_figures,
    reset_embedded_animation,
)
from complex_problems.nonlinear_waves.solver import NonlinearWavesResult
from config import generate_output_basename, get_env_from_schema, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting import (
    create_contour_plot,
    create_solution_plot,
    export_animated_figure_to_mp4,
)
from plotting.animation_metadata import attach_animation_metadata
from utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)


@dataclass(frozen=True)
class _SpectrumAnimationViewPayload:
    """Prepared spectrum data shared by the embedded view and MP4 export."""

    t: np.ndarray
    k: np.ndarray
    frames: np.ndarray
    title: str


@dataclass(frozen=True)
class _KdvReferenceAnimationPayload:
    """Inputs for the selectable KdV numerical/reference animation."""

    x: np.ndarray
    t: np.ndarray
    numerical: np.ndarray
    amplitudes: tuple[float, ...]
    centers: tuple[float, ...]
    inverse_widths: tuple[float, ...]
    speeds: tuple[float, ...]
    x_min: float
    x_max: float
    selected: tuple[str, ...]


def _periodic_reference_profile(
    x: np.ndarray,
    *,
    center: float,
    amplitude: float,
    inverse_width: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Return one isolated soliton using minimum-image periodic distance."""
    length = x_max - x_min
    delta = ((x - center + length / 2.0) % length) - length / 2.0
    return amplitude / np.cosh(inverse_width * delta) ** 2


def _reference_profiles_at_time(
    payload: _KdvReferenceAnimationPayload, time: float
) -> tuple[np.ndarray, ...]:
    """Construct isolated reference profiles for one animation frame."""
    elapsed = time - float(payload.t[0])
    return tuple(
        _periodic_reference_profile(
            payload.x,
            center=center + speed * elapsed,
            amplitude=amplitude,
            inverse_width=inverse_width,
            x_min=payload.x_min,
            x_max=payload.x_max,
        )
        for amplitude, center, inverse_width, speed in zip(
            payload.amplitudes,
            payload.centers,
            payload.inverse_widths,
            payload.speeds,
            strict=True,
        )
    )


def _create_kdv_reference_animation_payload(
    result: NonlinearWavesResult, selected: tuple[str, ...]
) -> _KdvReferenceAnimationPayload:
    """Prepare a KdV reference animation without materializing reference histories."""
    metadata = result.metadata
    return _KdvReferenceAnimationPayload(
        x=result.x,
        t=result.t,
        numerical=np.real(result.field),
        amplitudes=tuple(float(value) for value in metadata["soliton_amplitudes"]),
        centers=tuple(float(value) for value in metadata["soliton_centers"]),
        inverse_widths=tuple(float(value) for value in metadata["soliton_inverse_widths"]),
        speeds=tuple(float(value) for value in metadata["soliton_speeds"]),
        x_min=float(metadata["x_min"]),
        x_max=float(metadata["x_max"]),
        selected=selected or ("Numerical u",),
    )


def _create_kdv_reference_animation_figure(
    payload: _KdvReferenceAnimationPayload,
) -> Figure:
    """Build the selectable KdV profile/reference animation."""
    import matplotlib.pyplot as plt

    selected = set(payload.selected)
    reference_labels = [
        f"Reference soliton {index + 1}" for index in range(len(payload.amplitudes))
    ]
    needs_references = bool(selected.intersection(reference_labels)) or (
        "Interaction residual" in selected
    )
    global_max = 0.0
    if "Numerical u" in selected:
        global_max = max(global_max, float(np.max(np.abs(payload.numerical))))
    for index, amplitude in enumerate(payload.amplitudes):
        if f"Reference soliton {index + 1}" in selected:
            global_max = max(global_max, abs(amplitude))
    if "Interaction residual" in selected:
        for time, numerical in zip(payload.t, payload.numerical, strict=True):
            reference_sum = np.sum(_reference_profiles_at_time(payload, float(time)), axis=0)
            global_max = max(global_max, float(np.max(np.abs(numerical - reference_sum))))
    y_lim = 1.1 * (global_max if global_max > 0.0 else 1.0)

    fig, ax = plt.subplots()
    lines: dict[str, object] = {}
    initial_profiles = (
        _reference_profiles_at_time(payload, float(payload.t[0])) if needs_references else ()
    )
    initial_sum = np.sum(initial_profiles, axis=0) if initial_profiles else np.zeros_like(payload.x)
    if "Numerical u" in selected:
        (line,) = ax.plot(payload.x, payload.numerical[0], linestyle="-", label="Numerical u")
        lines["Numerical u"] = line
    colors = plt.get_cmap("tab20")
    for index, label in enumerate(reference_labels):
        if label not in selected:
            continue
        (line,) = ax.plot(
            payload.x,
            initial_profiles[index],
            linestyle="--",
            color=colors(index % 20),
            label=label,
        )
        lines[label] = line
    if "Interaction residual" in selected:
        (line,) = ax.plot(
            payload.x,
            payload.numerical[0] - initial_sum,
            linestyle=":",
            label="Interaction residual",
        )
        lines["Interaction residual"] = line

    ax.set_xlim(float(payload.x[0]), float(payload.x[-1]))
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"KdV profile animation (t={payload.t[0]:.3g})")
    if len(lines) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    def _update(index: int) -> None:
        """Update existing line artists for one selected frame."""
        frame = max(0, min(index, len(payload.t) - 1))
        profiles = (
            _reference_profiles_at_time(payload, float(payload.t[frame]))
            if needs_references
            else ()
        )
        reference_sum = np.sum(profiles, axis=0) if profiles else np.zeros_like(payload.x)
        numerical_line = lines.get("Numerical u")
        if numerical_line is not None:
            numerical_line.set_ydata(payload.numerical[frame])  # type: ignore[attr-defined]
        for soliton_index, label in enumerate(reference_labels):
            line = lines.get(label)
            if line is not None:
                line.set_ydata(profiles[soliton_index])  # type: ignore[attr-defined]
        residual_line = lines.get("Interaction residual")
        if residual_line is not None:
            residual_line.set_ydata(payload.numerical[frame] - reference_sum)  # type: ignore[attr-defined]
        ax.set_title(f"KdV profile animation (t={payload.t[frame]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(fig, update=_update, n_points=len(payload.t))


def _create_spectrum_animation_figure(payload: _SpectrumAnimationViewPayload) -> Figure:
    """Build the spectrum figure used by Tk and by MP4 export."""
    import matplotlib.pyplot as plt

    y_max = float(np.max(payload.frames))
    y_lim = 1.1 * (y_max if y_max > 0.0 else 1.0)
    fig, ax = plt.subplots()
    (line,) = ax.plot(payload.k, payload.frames[0], linewidth=2.0)
    ax.set_xlim(float(payload.k[0]), float(payload.k[-1]))
    ax.set_ylim(0.0, y_lim)
    ax.set_xlabel("k")
    ax.set_ylabel("Power")
    ax.set_title(f"{payload.title} (t={payload.t[0]:.3g})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    def _update(index: int) -> None:
        """Draw one bounded spectrum frame."""
        idx = max(0, min(index, len(payload.t) - 1))
        line.set_ydata(payload.frames[idx])
        ax.set_title(f"{payload.title} (t={payload.t[idx]:.3g})")
        fig.canvas.draw_idle()

    return attach_animation_metadata(fig, update=_update, n_points=len(payload.t))


def _create_line_animation_figure(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> Figure:
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

    return attach_animation_metadata(fig, update=_update, n_points=len(t))


class NonlinearWavesResultDialog:
    """Result window for nonlinear waves."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, *, result: NonlinearWavesResult) -> None:
        self.parent = parent
        self._result = result
        self.win = tk.Toplevel(parent)
        self.win.title("Nonlinear Waves Results")
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
        close_embedded_figures(
            self,
            ("_anim_canvas", "_st_canvas", "_phase_canvas", "_spec_canvas", "_inv_canvas"),
        )
        self.win.destroy()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        top = ttk.Frame(self.win, padding=pad)
        top.pack(fill=tk.BOTH, expand=True)

        drift_text = ", ".join(f"{k}: {v:+.3e}" for k, v in self._result.magnitudes.items())
        ttk.Label(top, text=drift_text, style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))
        if self._result.model_type == "kdv" and "soliton_count" in self._result.metadata:
            metadata = self._result.metadata
            summary = (
                f"Solitons: N={metadata['soliton_count']} | "
                f"A={metadata['soliton_amplitudes']} | "
                f"x0={metadata['soliton_centers']} | "
                f"v={metadata['soliton_speeds']}"
            )
            ttk.Label(top, text=summary, style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))

        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_anim = ttk.Frame(nb)
        nb.add(tab_anim, text="  Profile Animation  ")
        self._build_anim_tab(tab_anim)

        tab_st = ttk.Frame(nb)
        nb.add(tab_st, text="  Space-Time Map  ")
        self._build_spacetime_tab(tab_st)

        if self._result.phase is not None:
            tab_phase = ttk.Frame(nb)
            nb.add(tab_phase, text="  Phase  ")
            self._build_phase_tab(tab_phase)

        tab_spec = ttk.Frame(nb)
        nb.add(tab_spec, text="  Spectrum  ")
        self._spectrum_tab = tab_spec
        self._spectrum_tab_initialized = False
        nb.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        tab_inv = ttk.Frame(nb)
        nb.add(tab_inv, text="  Invariants  ")
        self._build_invariants_tab(tab_inv)

        btn_frame = ttk.Frame(self.win, padding=(pad, 0, pad, pad))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", style="Cancel.TButton", command=self._on_close).pack(
            side=tk.RIGHT
        )

    def _on_notebook_tab_changed(self, event: tk.Event[tk.Misc]) -> None:
        """Initialize the deferred Spectrum tab on its first selection."""
        if self._spectrum_tab_initialized:
            return
        notebook = cast(ttk.Notebook, event.widget)
        selected_tab = notebook.nametowidget(notebook.select())
        if selected_tab is self._spectrum_tab:
            self._build_spectrum_tab(self._spectrum_tab)
            self._spectrum_tab_initialized = True

    def _build_anim_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        self._kdv_reference_mode = self._is_kdv_soliton_result()
        if self._kdv_reference_mode:
            count = int(self._result.metadata["soliton_count"])
            self._anim_selection_labels = (
                ["Numerical u"]
                + [f"Reference soliton {index + 1}" for index in range(count)]
                + ["Interaction residual"]
            )
            ttk.Label(ctrl, text="Show:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
            self._anim_selection = tk.Listbox(
                ctrl,
                selectmode=tk.EXTENDED,
                exportselection=False,
                height=min(len(self._anim_selection_labels), 6),
                width=24,
                font=get_font(),
            )
            for label in self._anim_selection_labels:
                self._anim_selection.insert(tk.END, label)
            self._anim_selection.selection_set(0)
            self._anim_selection.pack(side=tk.LEFT, padx=(0, 8))
            ttk.Label(
                ctrl,
                text=(
                    "Dashed curves are freely propagating isolated-soliton references. "
                    "Residual = u - sum of references (diagnostic, not a unique decomposition)."
                ),
                style="Small.TLabel",
                wraplength=700,
            ).pack(side=tk.LEFT, anchor=tk.W)
            self._anim_selection.bind("<<ListboxSelect>>", lambda _e: self._update_anim())
        else:
            options = (
                ["Field"] if self._result.model_type == "kdv" else ["Intensity", "Real", "Imag"]
            )
            self._anim_view_var = tk.StringVar(value=options[0])
            ttk.Label(ctrl, text="Display:", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
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

    def _is_kdv_soliton_result(self) -> bool:
        """Return whether this result carries supported KdV soliton metadata."""
        return self._result.model_type == "kdv" and self._result.metadata.get("profile") in {
            "kdv_soliton",
            "kdv_soliton_train",
        }

    def _selected_animation_labels(self) -> tuple[str, ...]:
        """Read the multi-select control, keeping a valid numerical fallback."""
        selected = tuple(
            self._anim_selection.get(index) for index in self._anim_selection.curselection()
        )
        return selected or ("Numerical u",)

    def _update_anim(self) -> None:
        reset_embedded_animation(self._anim_frame, self._anim_canvas)

        if self._kdv_reference_mode:
            payload = _create_kdv_reference_animation_payload(
                self._result, self._selected_animation_labels()
            )
            fig = _create_kdv_reference_animation_figure(payload)
        elif self._result.model_type == "kdv":
            y = np.real(self._result.field)
            title = "KdV profile"
            ylabel = "u"
        else:
            view = self._anim_view_var.get()
            if view == "Real":
                y = np.real(self._result.field)
                ylabel = "Re(ψ)"
            elif view == "Imag":
                y = np.imag(self._result.field)
                ylabel = "Im(ψ)"
            else:
                y = self._result.magnitude
                ylabel = "|ψ|²"
            title = f"NLSE profile - {view}"

        if not self._kdv_reference_mode:
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
        payload = self._get_spectrum_animation_payload()
        self._spec_canvas = embed_animation_plot_in_tk(
            _create_spectrum_animation_figure(payload),
            parent,
            on_export_mp4=lambda duration: self._on_export_animation_mp4(payload, duration),
        )

    def _get_spectrum_animation_payload(self) -> _SpectrumAnimationViewPayload:
        """Return the prepared spectrum history for display or export."""
        return _SpectrumAnimationViewPayload(
            t=self._result.t,
            k=self._result.k,
            frames=self._result.spectrum_power_history,
            title=f"{self._result.model_type.upper()} spectrum evolution",
        )

    def _on_export_animation_mp4(
        self,
        payload: _SpectrumAnimationViewPayload,
        duration_seconds: float,
    ) -> None:
        """Export the spectrum animation through the shared MP4 path."""
        default_path = get_output_dir() / (
            f"{generate_output_basename(prefix='nonlinear_waves')}.mp4"
        )
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
            export_animated_figure_to_mp4(
                _create_spectrum_animation_figure(payload),
                filepath,
                duration_seconds=duration_seconds,
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
