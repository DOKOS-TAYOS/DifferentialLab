"""Regression tests for coupled-oscillator animation views and export."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

import complex_problems.coupled_oscillators.result_dialog as result_dialog

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class _FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def _make_result(n: int = 3, boundary: str = "fixed") -> SimpleNamespace:
    x = np.linspace(0.0, 1.0, 5)
    y = np.vstack(
        (
            np.arange(n, dtype=float)[:, np.newaxis] + x,
            np.arange(n, dtype=float)[:, np.newaxis] - x,
        )
    )
    return SimpleNamespace(
        x=x,
        y=y,
        n_oscillators=n,
        masses=np.ones(n),
        M_modes=np.eye(n),
        has_modes=True,
        metadata={"boundary": boundary},
    )


def _make_dialog(
    result: SimpleNamespace,
    view: str,
) -> result_dialog.CoupledOscillatorsResultDialog:
    dialog = object.__new__(result_dialog.CoupledOscillatorsResultDialog)
    dialog._result = result
    dialog._anim_view_var = _FakeVar(view)
    dialog._anim_plot_frame = object()
    dialog._anim_canvas = None
    dialog.win = object()
    return dialog


def test_animation_update_passes_mp4_callback_and_rebuilds_for_view_changes() -> None:
    dialog = _make_dialog(_make_result(), "Modes")
    created_payloads: list[dict[str, object]] = []
    export_calls: list[tuple[object, float]] = []

    def create_figure(*_args: object, **kwargs: object) -> object:
        created_payloads.append(kwargs)
        return plt.figure()

    def embed_figure(_figure: object, _frame: object, **kwargs: object) -> object:
        assert callable(kwargs["on_export_mp4"])
        kwargs["on_export_mp4"](4.0)
        return object()

    dialog._on_export_animation_mp4 = lambda payload, duration: export_calls.append(  # type: ignore[method-assign]
        (payload, duration)
    )

    try:
        with (
            patch.object(result_dialog, "reset_embedded_animation"),
            patch.object(result_dialog, "create_vector_animation_plot", side_effect=create_figure),
            patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed_figure),
        ):
            dialog._update_animation()
            mode_payload = export_calls[-1][0]
            dialog._anim_view_var.value = "Oscillators"
            dialog._update_animation()
            oscillator_payload = export_calls[-1][0]

        assert len(created_payloads) == 2
        assert len(export_calls) == 2
        assert mode_payload.y.shape == (6, 5)
        assert mode_payload.vector_components == 3
        assert mode_payload.component_labels == ["Mode 1", "Mode 2", "Mode 3"]
        assert oscillator_payload.y.shape == (10, 5)
        assert oscillator_payload.vector_components == 5
        assert oscillator_payload.component_labels == ["-1", "0", "1", "2", "3"]
        np.testing.assert_array_equal(oscillator_payload.y[[0, 1, -2, -1]], 0.0)
        assert export_calls[0][1] == 4.0
    finally:
        for figure_number in plt.get_fignums():
            plt.close(figure_number)


def test_mp4_export_cancel_success_and_ffmpeg_error_are_user_facing() -> None:
    dialog = _make_dialog(_make_result(32), "Modes")
    payload = dialog._get_animation_view_payload()
    export = MagicMock()
    show_info = MagicMock()
    show_error = MagicMock()

    with (
        patch.object(result_dialog.filedialog, "asksaveasfilename", return_value=""),
        patch.object(result_dialog, "export_animation_to_mp4", export),
    ):
        dialog._on_export_animation_mp4(payload, 3.5)
    export.assert_not_called()

    with (
        patch.object(
            result_dialog.filedialog,
            "asksaveasfilename",
            return_value="output/modes.mp4",
        ),
        patch.object(result_dialog, "export_animation_to_mp4", export),
        patch.object(result_dialog.messagebox, "showinfo", show_info),
    ):
        dialog._on_export_animation_mp4(payload, 3.5)

    export.assert_called_once()
    assert export.call_args.kwargs["duration_seconds"] == 3.5
    assert export.call_args.kwargs["component_labels"] == [f"Mode {i}" for i in range(1, 33)]
    show_info.assert_called_once()

    export.reset_mock()
    export.side_effect = RuntimeError("FFMpeg is not available")
    with (
        patch.object(
            result_dialog.filedialog,
            "asksaveasfilename",
            return_value="output/modes.mp4",
        ),
        patch.object(result_dialog, "export_animation_to_mp4", export),
        patch.object(result_dialog.messagebox, "showerror", show_error),
    ):
        dialog._on_export_animation_mp4(payload, 3.5)

    show_error.assert_called_once()
    assert "ffmpeg" in show_error.call_args.args[1].lower()
