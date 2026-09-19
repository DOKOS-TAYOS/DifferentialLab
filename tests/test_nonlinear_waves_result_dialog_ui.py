"""Regression coverage for nonlinear-wave spectrum animation views."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import complex_problems.nonlinear_waves.result_dialog as result_dialog
import complex_problems.nonlinear_waves.ui as nonlinear_ui


class _StubVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


def _make_input_dialog(profile: str, **values: str) -> nonlinear_ui.NonlinearWavesDialog:
    defaults = {
        "model": "kdv",
        "x_min": "-20.0",
        "x_max": "20.0",
        "nx": "128",
        "t_min": "0.0",
        "t_max": "1.0",
        "dt": "0.01",
        "amp": "1.0",
        "sigma": "1.0",
        "center": "0.0",
        "train_n": "2",
        "train_amplitudes": "1.2, 0.5",
        "train_centers": "-8.0, -2.0",
        "c": "0.0",
        "alpha": "6.0",
        "beta_disp": "1.0",
    }
    defaults.update(values)
    dialog: nonlinear_ui.NonlinearWavesDialog = object.__new__(nonlinear_ui.NonlinearWavesDialog)
    for name, value in defaults.items():
        setattr(dialog, f"_{name}_var", _StubVar(value))
    dialog._profile_var = _StubVar(profile)
    return dialog


def _make_dialog(model_type: str = "nlse") -> result_dialog.NonlinearWavesResultDialog:
    t = np.array([0.0, 0.5, 1.0])
    k = np.linspace(-3.0, 3.0, 5)
    spectrum_history = np.arange(15, dtype=np.float32).reshape(3, 5)
    result = SimpleNamespace(
        model_type=model_type,
        t=t,
        k=k,
        spectrum_power=spectrum_history[-1],
        spectrum_power_history=spectrum_history,
    )
    dialog = object.__new__(result_dialog.NonlinearWavesResultDialog)
    dialog._result = result
    dialog.win = object()
    return dialog


def test_spectrum_animation_uses_full_history_and_nonnegative_y_range() -> None:
    dialog = _make_dialog("kdv")
    payload = dialog._get_spectrum_animation_payload()
    figure = result_dialog._create_spectrum_animation_figure(payload)
    try:
        assert payload.title == "KDV spectrum evolution"
        np.testing.assert_array_equal(payload.frames, dialog._result.spectrum_power_history)
        assert figure.axes[0].get_ylim()[0] == 0.0
        assert figure.axes[0].get_ylim()[1] > float(np.max(payload.frames))
        figure._animation_update(2)  # type: ignore[attr-defined]
        assert figure.axes[0].get_title().endswith("t=1)")
    finally:
        plt.close(figure)


def test_spectrum_tab_uses_animation_embedding_and_export_callback() -> None:
    dialog = _make_dialog()
    export_calls: list[tuple[object, float]] = []

    def embed_figure(_figure: object, _parent: object, **kwargs: object) -> object:
        kwargs["on_export_mp4"](3.0)  # type: ignore[operator]
        return object()

    dialog._on_export_animation_mp4 = lambda payload, duration: export_calls.append(  # type: ignore[method-assign]
        (payload, duration)
    )
    with (
        patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed_figure),
        patch.object(result_dialog, "_create_spectrum_animation_figure", return_value=plt.figure()),
    ):
        dialog._build_spectrum_tab(object())

    assert len(export_calls) == 1
    assert export_calls[0][0].title == "NLSE spectrum evolution"
    assert export_calls[0][1] == 3.0
    plt.close("all")


def test_spectrum_tab_initializes_only_on_first_selection() -> None:
    dialog = _make_dialog()
    spectrum_tab = object()
    dialog._spectrum_tab = spectrum_tab
    dialog._spectrum_tab_initialized = False
    build_tab = MagicMock()
    dialog._build_spectrum_tab = build_tab  # type: ignore[method-assign]

    class _NotebookStub:
        def __init__(self) -> None:
            self.selected = object()

        def select(self) -> object:
            return self.selected

        def nametowidget(self, _selected: object) -> object:
            return self.selected

    notebook = _NotebookStub()
    dialog._on_notebook_tab_changed(SimpleNamespace(widget=notebook))
    build_tab.assert_not_called()
    assert dialog._spectrum_tab_initialized is False

    notebook.selected = spectrum_tab
    dialog._on_notebook_tab_changed(SimpleNamespace(widget=notebook))
    build_tab.assert_called_once_with(spectrum_tab)
    assert dialog._spectrum_tab_initialized is True

    dialog._on_notebook_tab_changed(SimpleNamespace(widget=notebook))
    build_tab.assert_called_once_with(spectrum_tab)


def test_mp4_export_cancel_success_and_ffmpeg_error_are_user_facing() -> None:
    dialog = _make_dialog()
    payload = dialog._get_spectrum_animation_payload()
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
            result_dialog.filedialog, "asksaveasfilename", return_value="output/spectrum.mp4"
        ),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
        patch.object(result_dialog.messagebox, "showinfo", show_info),
        patch.object(result_dialog, "_create_spectrum_animation_figure", return_value=plt.figure()),
    ):
        dialog._on_export_animation_mp4(payload, 2.0)
    assert export.call_args.kwargs["duration_seconds"] == 2.0
    show_info.assert_called_once()

    export.side_effect = RuntimeError("FFMpeg is not available")
    with (
        patch.object(
            result_dialog.filedialog, "asksaveasfilename", return_value="output/spectrum.mp4"
        ),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
        patch.object(result_dialog.messagebox, "showerror", show_error),
        patch.object(result_dialog, "_create_spectrum_animation_figure", return_value=plt.figure()),
    ):
        dialog._on_export_animation_mp4(payload, 2.0)

    assert "ffmpeg" in show_error.call_args.args[1].lower()
    plt.close("all")


def test_kdv_train_csv_parser_rejects_non_finite_and_keeps_count_check_local() -> None:
    assert nonlinear_ui._parse_csv_floats("1.2, 0.5", name="Amplitudes") == [1.2, 0.5]
    with pytest.raises(ValueError, match="finite"):
        nonlinear_ui._parse_csv_floats("1.2, nan", name="Amplitudes")


def test_collect_inputs_kdv_train_ignores_hidden_scalar_fields() -> None:
    dialog = _make_input_dialog(
        "Separated soliton train",
        amp="not-a-number",
        sigma="not-a-number",
        center="not-a-number",
    )

    params = dialog._collect_inputs()

    assert params["profile"] == "kdv_soliton_train"
    assert params["soliton_amplitudes"] == [1.2, 0.5]
    assert params["soliton_centers"] == [-8.0, -2.0]
    assert params["c"] == 0.0
    assert params["alpha"] == 6.0
    assert params["beta_disp"] == 1.0


def test_collect_inputs_kdv_train_rejects_n_list_mismatch() -> None:
    dialog = _make_input_dialog("Separated soliton train", train_n="3")

    with pytest.raises(ValueError, match="equal N"):
        dialog._collect_inputs()


def test_collect_inputs_kdv_single_soliton_maps_scalar_controls() -> None:
    dialog = _make_input_dialog(
        "Single soliton",
        amp="1.25",
        center="-3.5",
        sigma="not-a-number",
    )

    params = dialog._collect_inputs()

    assert params["profile"] == "kdv_soliton"
    assert params["amplitude"] == 1.25
    assert params["center"] == -3.5
