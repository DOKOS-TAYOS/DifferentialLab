"""Regression coverage for membrane animation view selection and export."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

import complex_problems.membrane_2d.result_dialog as result_dialog
from complex_problems.membrane_2d.model import compute_fft_power_history_2d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class _FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def _make_result() -> SimpleNamespace:
    t = np.array([0.0, 0.5, 1.0])
    displacement = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    velocity = -displacement
    kx, ky, spectrum_history = compute_fft_power_history_2d(displacement)
    return SimpleNamespace(
        t=t,
        displacement=displacement,
        velocity=velocity,
        kx=kx,
        ky=ky,
        spectrum_power=spectrum_history[-1],
        spectrum_power_history=spectrum_history,
    )


def _make_dialog(view: str) -> result_dialog.Membrane2DResultDialog:
    dialog = object.__new__(result_dialog.Membrane2DResultDialog)
    dialog._result = _make_result()
    dialog._anim_field_var = _FakeVar(view)
    dialog._anim_frame = object()
    dialog._anim_canvas = None
    dialog._spectrum_power_history = dialog._result.spectrum_power_history
    dialog.win = object()
    return dialog


def test_animation_payloads_use_the_selected_membrane_data() -> None:
    dialog = _make_dialog("2D Field")

    field = dialog._get_animation_view_payload()
    dialog._anim_field_var.value = "3D Surface"
    surface = dialog._get_animation_view_payload()
    dialog._anim_field_var.value = "Spectrum"
    spectrum = dialog._get_animation_view_payload()

    assert field.kind == "image"
    np.testing.assert_array_equal(field.frames, dialog._result.displacement)
    assert surface.kind == "surface"
    np.testing.assert_array_equal(surface.x, np.arange(5))
    np.testing.assert_array_equal(surface.y, np.arange(4))
    np.testing.assert_array_equal(surface.frames, dialog._result.displacement)
    assert spectrum.kind == "image"
    np.testing.assert_array_equal(spectrum.frames, dialog._result.spectrum_power_history)
    np.testing.assert_array_equal(spectrum.frames[-1], dialog._result.spectrum_power)


def test_animation_switch_replaces_canvas_and_passes_matching_export_payload() -> None:
    dialog = _make_dialog("2D Field")
    export_calls: list[tuple[object, float]] = []

    def embed_figure(_figure: object, _frame: object, **kwargs: object) -> object:
        kwargs["on_export_mp4"](3.0)  # type: ignore[operator]
        return object()

    dialog._on_export_animation_mp4 = lambda payload, duration: export_calls.append(  # type: ignore[method-assign]
        (payload, duration)
    )

    with (
        patch.object(result_dialog, "reset_embedded_animation") as reset,
        patch.object(result_dialog, "_create_animation_figure", return_value=plt.figure()),
        patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed_figure),
    ):
        dialog._update_animation()
        dialog._anim_field_var.value = "Spectrum"
        dialog._update_animation()

    assert reset.call_count == 2
    assert len(export_calls) == 2
    assert export_calls[0][0].title == "Membrane displacement field"
    assert export_calls[1][0].title == "2D FFT power spectrum"
    assert export_calls[1][1] == 3.0
    plt.close("all")


def test_mp4_export_cancel_success_and_ffmpeg_error_are_user_facing() -> None:
    dialog = _make_dialog("3D Surface")
    payload = dialog._get_animation_view_payload()
    export = MagicMock()
    show_info = MagicMock()
    show_error = MagicMock()

    with (
        patch.object(result_dialog.filedialog, "asksaveasfilename", return_value=""),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
    ):
        dialog._on_export_animation_mp4(payload, 2.0)
    export.assert_not_called()

    with (
        patch.object(
            result_dialog.filedialog, "asksaveasfilename", return_value="output/surface.mp4"
        ),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
        patch.object(result_dialog.messagebox, "showinfo", show_info),
        patch.object(result_dialog, "_create_animation_figure", return_value=plt.figure()),
    ):
        dialog._on_export_animation_mp4(payload, 2.0)

    assert export.call_args.kwargs["duration_seconds"] == 2.0
    show_info.assert_called_once()

    export.side_effect = RuntimeError("FFMpeg is not available")
    with (
        patch.object(
            result_dialog.filedialog, "asksaveasfilename", return_value="output/surface.mp4"
        ),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
        patch.object(result_dialog.messagebox, "showerror", show_error),
        patch.object(result_dialog, "_create_animation_figure", return_value=plt.figure()),
    ):
        dialog._on_export_animation_mp4(payload, 2.0)

    assert "ffmpeg" in show_error.call_args.args[1].lower()
    plt.close("all")



def test_result_tabs_prioritize_visual_membrane_views() -> None:
    dialog = object.__new__(result_dialog.Membrane2DResultDialog)
    dialog.win = MagicMock()
    dialog._result = SimpleNamespace(magnitudes={})
    notebook = MagicMock()
    shell = MagicMock(notebook=notebook)

    with (
        patch.object(result_dialog, "AdvancedResultShell", return_value=shell),
        patch.object(result_dialog.ttk, "Frame", side_effect=lambda _parent: MagicMock()),
        patch.object(dialog, "_build_animation_tab"),
        patch.object(dialog, "_build_surface_tab"),
        patch.object(dialog, "_build_spectrum_tab"),
        patch.object(dialog, "_build_space_time_tab"),
        patch.object(dialog, "_build_energy_tab"),
    ):
        dialog._build_ui()

    assert [call.kwargs["text"] for call in notebook.add.call_args_list] == [
        "Animation",
        "Surface 3D",
        "Spectrum",
        "Centerline Map",
        "Energy",
    ]
