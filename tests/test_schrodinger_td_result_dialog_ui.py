"""Regression coverage for Schrodinger TD animation result views."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import complex_problems.schrodinger_td.result_dialog as result_dialog


def _make_result(dimension: int = 2) -> SimpleNamespace:
    t = np.array([0.0, 0.5, 1.0])
    x = np.linspace(-2.0, 2.0, 5, endpoint=False)
    density = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5) / 10.0
    spectrum = density + 1.0
    if dimension == 1:
        psi = np.array([[1.0 + 0.5j, 2.0 - 0.25j, 0.0, -1.0j, 0.5 + 0.75j] for _ in t])
        return SimpleNamespace(
            dimension=1,
            t=t,
            x=x,
            psi=psi,
            magnitude=np.abs(psi) ** 2,
            potential=np.ones(5),
            spectrum_power=spectrum[-1, 0],
        )
    return SimpleNamespace(
        dimension=2,
        t=t,
        x=x,
        y=np.linspace(-1.0, 1.0, 4, endpoint=False),
        kx=np.linspace(-3.0, 3.0, 5),
        ky=np.linspace(-2.0, 2.0, 4),
        magnitude=density,
        phase=np.sin(density),
        spectrum_power=spectrum[-1],
        spectrum_power_history=spectrum.astype(np.float32),
    )


def _make_dialog(result: SimpleNamespace) -> result_dialog.SchrodingerTDResultDialog:
    dialog = object.__new__(result_dialog.SchrodingerTDResultDialog)
    dialog._result = result
    dialog.win = object()
    return dialog


def test_2d_animation_payloads_use_all_frames_and_scientific_labels() -> None:
    dialog = _make_dialog(_make_result())

    spectrum = dialog._get_spectrum_animation_payload()
    surface = dialog._get_density_surface_animation_payload()

    assert spectrum.kind == "image"
    np.testing.assert_array_equal(spectrum.frames, dialog._result.spectrum_power_history)
    np.testing.assert_array_equal(spectrum.x, dialog._result.kx)
    np.testing.assert_array_equal(spectrum.y, dialog._result.ky)
    assert (spectrum.xlabel, spectrum.ylabel) == ("kₓ", "kᵧ")
    assert surface.kind == "surface"
    np.testing.assert_array_equal(surface.frames, dialog._result.magnitude)
    assert surface.frames.shape[0] == len(dialog._result.t)
    assert (surface.zlabel, surface.colorbar_label) == ("|ψ|²", "|ψ|²")
    assert surface.symmetric_z_range is False


def test_2d_tabs_use_animation_embedding_and_export_callbacks() -> None:
    dialog = _make_dialog(_make_result())
    export_calls: list[tuple[object, float]] = []

    def embed_figure(_figure: object, _parent: object, **kwargs: object) -> object:
        kwargs["on_export_mp4"](2.5)  # type: ignore[operator]
        return object()

    dialog._on_export_animation_mp4 = lambda payload, duration: export_calls.append(  # type: ignore[method-assign]
        (payload, duration)
    )

    with (
        patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed_figure),
        patch.object(result_dialog, "_create_animation_figure", return_value=plt.figure()),
    ):
        dialog._build_spectrum_tab(object())
        dialog._build_extra_tab(object())

    assert len(export_calls) == 2
    assert export_calls[0][0].title == "2D k-space power"
    assert export_calls[1][0].title == "2D density surface"
    assert all(duration == 2.5 for _payload, duration in export_calls)
    plt.close("all")


def test_1d_potential_remains_static() -> None:
    dialog = _make_dialog(_make_result(dimension=1))
    with (
        patch.object(result_dialog, "create_solution_plot") as create_plot,
        patch.object(result_dialog, "embed_plot_in_tk") as embed_plot,
        patch.object(result_dialog, "embed_animation_plot_in_tk") as embed_animation,
    ):
        dialog._build_extra_tab(object())

    create_plot.assert_called_once()
    embed_plot.assert_called_once()
    embed_animation.assert_not_called()


def test_main_animation_uses_selected_representation_and_export_callback() -> None:
    for dimension, views in ((1, ("Density", "Real", "Imag")), (2, ("Density", "Phase"))):
        dialog = _make_dialog(_make_result(dimension))
        dialog._anim_view_var = SimpleNamespace(get=lambda: views[-1])
        dialog._anim_frame = object()
        dialog._anim_canvas = None
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
        payload = dialog._get_main_animation_payload()
        figure = result_dialog._create_animation_figure(payload)
        try:
            assert figure._animation_n_points == len(dialog._result.t)  # type: ignore[attr-defined]
        finally:
            plt.close(figure)


def test_2d_spectrum_tab_initializes_only_on_first_selection() -> None:
    dialog = _make_dialog(_make_result())
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
    dialog = _make_dialog(_make_result())
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
        patch.object(result_dialog, "_create_animation_figure", return_value=plt.figure()),
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
        patch.object(result_dialog, "_create_animation_figure", return_value=plt.figure()),
    ):
        dialog._on_export_animation_mp4(payload, 2.0)

    assert "ffmpeg" in show_error.call_args.args[1].lower()
    plt.close("all")


def test_result_tabs_are_dimension_specific_and_visual_first() -> None:
    for dimension, expected in (
        (1, ["Animation", "Density Maps", "Spectrum", "Expectations", "Potential"]),
        (2, ["Animation", "Density Surface", "Density Maps", "Spectrum", "Expectations"]),
    ):
        result = _make_result(dimension)
        result.magnitudes = {"norm_drift_rel": 0.0, "max_density": 1.0}
        result.metadata = {}
        dialog = object.__new__(result_dialog.SchrodingerTDResultDialog)
        dialog.win = MagicMock()
        dialog._result = result
        notebook = MagicMock()
        shell = MagicMock(notebook=notebook)

        with (
            patch.object(result_dialog, "AdvancedResultShell", return_value=shell),
            patch.object(result_dialog.ttk, "Frame", side_effect=lambda _parent: MagicMock()),
            patch.object(dialog, "_build_animation_tab"),
            patch.object(dialog, "_build_space_tab"),
            patch.object(dialog, "_build_spectrum_tab"),
            patch.object(dialog, "_build_invariants_tab"),
            patch.object(dialog, "_build_extra_tab"),
        ):
            dialog._build_ui()

        assert [call.kwargs["text"] for call in notebook.add.call_args_list] == expected
