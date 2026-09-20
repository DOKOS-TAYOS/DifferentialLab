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
import complex_problems.nonlinear_waves.solver as nonlinear_solver
import complex_problems.nonlinear_waves.ui as nonlinear_ui
from complex_problems.nonlinear_waves.model import build_kdv_soliton_train
from complex_problems.nonlinear_waves.solver import solve_nonlinear_waves


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


def test_profile_animation_wires_export_for_nlse_and_plain_kdv() -> None:
    for model_type in ("nlse", "kdv"):
        dialog = _make_dialog(model_type)
        dialog._result.x = np.linspace(-2.0, 2.0, 5)
        dialog._result.field = np.ones((3, 5), dtype=complex)
        dialog._result.magnitude = np.abs(dialog._result.field) ** 2
        dialog._anim_frame = object()
        dialog._anim_canvas = None
        dialog._anim_view_var = _StubVar("Intensity" if model_type == "nlse" else "Field")
        dialog._kdv_reference_mode = False
        captured: dict[str, object] = {}

        def embed_figure(_figure: object, _parent: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        with (
            patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed_figure),
            patch.object(result_dialog, "reset_embedded_animation"),
        ):
            dialog._update_anim()

        assert callable(captured["on_export_mp4"])


def test_kdv_profile_export_reuses_selected_overlays_and_cached_centers() -> None:
    dialog = object.__new__(result_dialog.NonlinearWavesResultDialog)
    x = np.linspace(-10.0, 10.0, 32, endpoint=False)
    dialog._result = _make_kdv_reference_result(numerical=np.zeros((2, len(x))))
    dialog._result.x = x
    dialog._kdv_reference_mode = True
    dialog._anim_frame = object()
    dialog._anim_canvas = None
    dialog._selected_animation_labels = MagicMock(return_value=("Soliton 1",))
    dialog._show_interaction_residual = _StubVar("True")
    dialog._show_interaction_residual.get = lambda: True  # type: ignore[method-assign]
    tracked = SimpleNamespace(centers=np.array([[-8.0, -2.0], [-7.5, -1.5]]))
    dialog._get_tracked_soliton_centers = MagicMock(return_value=tracked)
    captured: dict[str, object] = {}

    def embed_figure(_figure: object, _parent: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    with (
        patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed_figure),
        patch.object(result_dialog, "reset_embedded_animation"),
        patch.object(result_dialog, "track_kdv_soliton_centers") as tracker,
    ):
        dialog._update_anim()

    assert callable(captured["on_export_mp4"])
    dialog._get_tracked_soliton_centers.assert_called_once()
    tracker.assert_not_called()


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


def _make_kdv_reference_result(
    *, numerical: np.ndarray, t: np.ndarray | None = None
) -> SimpleNamespace:
    times = np.array([0.0, 1.0]) if t is None else t
    x = np.linspace(-20.0, 20.0, numerical.shape[1], endpoint=False)
    return SimpleNamespace(
        model_type="kdv",
        x=x,
        t=times,
        field=numerical,
        metadata={
            "profile": "kdv_soliton_train",
            "soliton_count": 2,
            "soliton_amplitudes": [1.2, 0.5],
            "soliton_centers": [-8.0, -2.0],
            "soliton_inverse_widths": [1.0, 0.5],
            "soliton_speeds": [2.4, 1.0],
            "x_min": -20.0,
            "x_max": 20.0,
        },
    )


def test_kdv_tracked_profiles_wrap_from_unwrapped_centers() -> None:
    x = np.linspace(-20.0, 20.0, 2048, endpoint=False)
    payload = result_dialog._KdvReferenceAnimationPayload(
        x=x,
        t=np.array([0.0, 1.0]),
        numerical=np.zeros((2, len(x))),
        amplitudes=(1.0,),
        centers=(19.0,),
        inverse_widths=(1.0,),
        speeds=(2.0,),
        x_min=-20.0,
        x_max=20.0,
        selected=("Soliton 1",),
        show_residual=False,
        tracked_centers=np.array([[19.0], [21.0]]),
    )
    initial = result_dialog._tracked_profiles_at_frame(payload, 0)[0]
    wrapped = result_dialog._tracked_profiles_at_frame(payload, 1)[0]
    assert x[np.argmax(initial)] == pytest.approx(19.0, abs=0.05)
    assert x[np.argmax(wrapped)] == pytest.approx(-19.0, abs=0.05)


def test_kdv_reference_residual_is_zero_then_reproduces_perturbation() -> None:
    x = np.linspace(-20.0, 20.0, 256, endpoint=False)
    base = result_dialog._KdvReferenceAnimationPayload(
        x=x,
        t=np.array([0.0, 1.0]),
        numerical=np.zeros((2, len(x))),
        amplitudes=(1.2, 0.5),
        centers=(-8.0, -2.0),
        inverse_widths=(1.0, 0.5),
        speeds=(2.4, 1.0),
        x_min=-20.0,
        x_max=20.0,
        selected=(),
        show_residual=True,
        tracked_centers=np.array([[-8.0, -2.0], [-5.6, -1.0]]),
    )
    references = np.array(
        [
            np.sum(result_dialog._tracked_profiles_at_frame(base, index), axis=0)
            for index in range(len(base.t))
        ]
    )
    exact = base.__class__(**{**base.__dict__, "numerical": references})
    figure = result_dialog._create_kdv_reference_animation_figure(exact)
    try:
        residual = next(
            line.get_ydata()
            for line in figure.axes[0].lines
            if line.get_label() == "Interaction residual"
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    finally:
        plt.close(figure)
    perturbation = np.full_like(references, 0.125)
    perturbed = base.__class__(**{**base.__dict__, "numerical": references + perturbation})
    figure = result_dialog._create_kdv_reference_animation_figure(perturbed)
    try:
        residual = next(
            line.get_ydata()
            for line in figure.axes[0].lines
            if line.get_label() == "Interaction residual"
        )
        np.testing.assert_allclose(residual, perturbation[0])
        assert figure.axes[0].get_ylim()[1] > 1.2
    finally:
        plt.close(figure)


def test_kdv_reference_y_limit_uses_only_selected_reference_amplitudes() -> None:
    x = np.linspace(-20.0, 20.0, 256, endpoint=False)
    result = _make_kdv_reference_result(numerical=np.zeros((2, len(x))))
    payload = result_dialog._create_kdv_reference_animation_payload(
        result, ("Soliton 2",), tracked_centers=np.array([[-8.0, -2.0], [-5.6, -1.0]])
    )
    figure = result_dialog._create_kdv_reference_animation_figure(payload)
    try:
        lower, upper = figure.axes[0].get_ylim()
        assert upper == pytest.approx(0.55, abs=0.001)
        assert lower == pytest.approx(-0.55, abs=0.001)
    finally:
        plt.close(figure)


def test_kdv_reference_figure_styles_labels_and_animation() -> None:
    x = np.linspace(-20.0, 20.0, 128, endpoint=False)
    result = _make_kdv_reference_result(numerical=np.zeros((2, len(x))))
    payload = result_dialog._create_kdv_reference_animation_payload(
        result,
        ("Soliton 1", "Soliton 2"),
        show_residual=True,
        tracked_centers=np.array([[-8.0, -2.0], [-5.6, -1.0]]),
    )
    figure = result_dialog._create_kdv_reference_animation_figure(payload)
    try:
        lines = figure.axes[0].lines
        assert [line.get_label() for line in lines] == [
            "Numerical u",
            "Soliton 1",
            "Soliton 2",
            "Interaction residual",
        ]
        assert lines[0].get_linestyle() == "-"
        assert lines[1].get_linestyle() == "--"
        assert lines[2].get_linestyle() == "--"
        assert lines[3].get_linestyle() == ":"
        assert lines[1].get_color() != lines[2].get_color()
        assert figure._animation_n_points == 2  # type: ignore[attr-defined]
        figure._animation_update(1)  # type: ignore[attr-defined]
        assert figure.axes[0].get_title().endswith("t=1)")
        assert np.max(lines[1].get_ydata()) > 0.0
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("model_type", "profile", "expected"),
    [
        ("kdv", "kdv_soliton_train", True),
        ("kdv", "kdv_soliton", True),
        ("kdv", "sech", False),
        ("nlse", "kdv_soliton", False),
    ],
)
def test_kdv_reference_mode_only_supports_kdv_soliton_profiles(
    model_type: str, profile: str, expected: bool
) -> None:
    dialog = object.__new__(result_dialog.NonlinearWavesResultDialog)
    dialog._result = SimpleNamespace(model_type=model_type, metadata={"profile": profile})
    assert dialog._is_kdv_soliton_result() is expected


def test_kdv_selector_contract_defaults_to_no_optional_overlays() -> None:
    dialog = object.__new__(result_dialog.NonlinearWavesResultDialog)
    selection = MagicMock()
    selection.curselection.return_value = (0, 1)
    selection.get.side_effect = lambda index: ("Soliton 1", "Soliton 2")[index]
    dialog._anim_selection = selection
    assert dialog._selected_animation_labels() == ("Soliton 1", "Soliton 2")

    selection.curselection.return_value = ()
    assert dialog._selected_animation_labels() == ()


def test_kdv_selector_uses_extended_mode_and_compact_unselected_list() -> None:
    created: dict[str, object] = {}

    class _Listbox:
        def __init__(self, _parent: object, **kwargs: object) -> None:
            created.update(kwargs)

        def insert(self, _index: object, _label: str) -> None:
            pass

        def pack(self, **_kwargs: object) -> None:
            pass

        def bind(self, *_args: object) -> None:
            pass

        def configure(self, **_kwargs: object) -> None:
            pass

        def yview(self, *_args: object) -> None:
            pass

    dialog = object.__new__(result_dialog.NonlinearWavesResultDialog)
    dialog.win = object()
    dialog._result = SimpleNamespace(
        model_type="kdv",
        metadata={"profile": "kdv_soliton_train", "soliton_count": 4},
    )
    dialog._update_anim = MagicMock()
    dialog._tracked_soliton_centers = None
    with (
        patch.object(result_dialog.tk, "Listbox", _Listbox),
        patch.object(result_dialog.tk, "BooleanVar", return_value=MagicMock(get=lambda: False)),
        patch.object(result_dialog.ttk, "Checkbutton", return_value=MagicMock(pack=MagicMock())),
        patch.object(result_dialog.ttk, "Scrollbar", return_value=MagicMock(pack=MagicMock())),
        patch.object(result_dialog, "ToolTip"),
        patch.object(result_dialog, "get_font", return_value=None),
    ):
        dialog._build_anim_tab(MagicMock())

    assert created["selectmode"] == result_dialog.tk.EXTENDED
    assert created["height"] <= 3
    assert dialog._anim_selection_labels == [
        "Soliton 1",
        "Soliton 2",
        "Soliton 3",
        "Soliton 4",
    ]


def test_kdv_selection_passes_multiple_labels_without_solving() -> None:
    dialog = object.__new__(result_dialog.NonlinearWavesResultDialog)
    dialog._result = _make_kdv_reference_result(numerical=np.zeros((2, 64)))
    dialog._kdv_reference_mode = True
    dialog._anim_frame = object()
    dialog._anim_canvas = object()
    dialog._selected_animation_labels = MagicMock(return_value=("Soliton 2",))
    dialog._show_interaction_residual = _StubVar("False")
    dialog._show_interaction_residual.get = lambda: False  # type: ignore[method-assign]
    created_payload = MagicMock()
    fake_figure = object()
    reset = MagicMock()
    embed = MagicMock(return_value="new-canvas")
    previous_canvas = dialog._anim_canvas
    with (
        patch.object(result_dialog, "reset_embedded_animation", reset),
        patch.object(
            result_dialog,
            "_create_kdv_reference_animation_payload",
            return_value=created_payload,
        ) as create_payload,
        patch.object(
            result_dialog,
            "_create_kdv_reference_animation_figure",
            return_value=fake_figure,
        ),
        patch.object(
            dialog,
            "_get_tracked_soliton_centers",
            return_value=SimpleNamespace(centers=np.zeros((2, 2))),
        ) as tracked,
        patch.object(result_dialog, "embed_animation_plot_in_tk", embed),
        patch.object(nonlinear_solver, "solve_nonlinear_waves") as solve,
    ):
        dialog._update_anim()

    reset.assert_called_once_with(dialog._anim_frame, previous_canvas)
    create_payload.assert_called_once()
    assert create_payload.call_args.args == (dialog._result, ("Soliton 2",))
    assert create_payload.call_args.kwargs["show_residual"] is False
    np.testing.assert_array_equal(
        create_payload.call_args.kwargs["tracked_centers"], np.zeros((2, 2))
    )
    tracked.assert_called_once()
    embed.assert_called_once()
    assert embed.call_args.args == (fake_figure, dialog._anim_frame)
    assert callable(embed.call_args.kwargs["on_export_mp4"])
    assert dialog._anim_canvas == "new-canvas"
    solve.assert_not_called()


def test_kdv_tracked_centers_are_cached_without_rerunning_the_solver() -> None:
    dialog = object.__new__(result_dialog.NonlinearWavesResultDialog)
    dialog._result = _make_kdv_reference_result(numerical=np.zeros((2, 64)))
    dialog._tracked_soliton_centers = None
    fitted = SimpleNamespace(centers=np.zeros((2, 2)))
    with patch.object(result_dialog, "track_kdv_soliton_centers", return_value=fitted) as tracker:
        assert dialog._get_tracked_soliton_centers() is fitted
        assert dialog._get_tracked_soliton_centers() is fitted
    tracker.assert_called_once()


def test_kdv_reference_initialization_matches_production_train_definition() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-40.0,
        x_max=40.0,
        nx=512,
        t_max=0.01,
        dt=0.01,
        profile="kdv_soliton_train",
        soliton_amplitudes=[3.0, 0.5],
        soliton_centers=[-15.0, 12.0],
        c=0.0,
        alpha=6.0,
        beta_disp=1.0,
    )
    expected, characteristics = build_kdv_soliton_train(
        result.x,
        amplitudes=result.metadata["soliton_amplitudes"],
        centers=result.metadata["soliton_centers"],
        c=result.metadata["c"],
        alpha=result.metadata["alpha"],
        beta_disp=result.metadata["beta_disp"],
    )
    assert len(characteristics) == result.metadata["soliton_count"]
    payload = result_dialog._create_kdv_reference_animation_payload(
        result,
        (),
        tracked_centers=np.asarray(result.metadata["soliton_centers"], dtype=float)[None, :],
    )
    references = np.sum(result_dialog._tracked_profiles_at_frame(payload, 0), axis=0)
    np.testing.assert_allclose(references, expected, rtol=1e-10, atol=2e-12)
    np.testing.assert_allclose(result.field[0].real, expected, rtol=1e-12, atol=1e-12)
