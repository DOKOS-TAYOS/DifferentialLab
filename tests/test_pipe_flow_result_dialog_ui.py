"""Regression coverage for Pipe Flow result figures and dialog wiring."""

from __future__ import annotations

from unittest.mock import DEFAULT, MagicMock, patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import complex_problems.pipe_flow.result_dialog as result_dialog
from complex_problems.pipe_flow.solver import PipeFlowResult, solve_pipe_flow


def _default_result() -> PipeFlowResult:
    return solve_pipe_flow()


def _transient_result() -> PipeFlowResult:
    return solve_pipe_flow(
        model_type="transient",
        nx=48,
        t_max=0.01,
        dt=2e-4,
        sample_every=5,
        p_amp=1.5e3,
        p_freq_hz=1.0,
        wave_speed=150.0,
    )


def test_geometry_figure_separates_diameter_and_area() -> None:
    result = _default_result()
    figure = result_dialog._create_geometry_figure(result)
    try:
        assert len(figure.axes) == 2
        assert figure.axes[0].get_shared_x_axes().joined(figure.axes[0], figure.axes[1])
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), result.x)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_xdata(), result.x)
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), result.diameter)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), result.area)
        assert figure.axes[0].get_ylabel() == "Diameter [m]"
        assert figure.axes[1].get_ylabel() == "Area [m²]"
        assert figure.axes[1].get_xlabel() == "x [m]"
        assert figure._suptitle.get_text() == "Pipe geometry"  # type: ignore[union-attr]
    finally:
        plt.close(figure)


def test_steady_pressure_figure_displays_kpa_on_spatial_axis() -> None:
    result = _default_result()
    figure = result_dialog._create_steady_pressure_figure(result)
    try:
        axis = figure.axes[0]
        assert axis.get_xlabel() == "x [m]"
        assert axis.get_ylabel() == "Pressure [kPa]"
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), result.x)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), result.pressure[0] / 1000.0)
        np.testing.assert_allclose(axis.lines[0].get_ydata()[[0, -1]], [200.0, 190.0])
    finally:
        plt.close(figure)


def test_steady_velocity_figure_uses_three_independent_scales() -> None:
    result = _default_result()
    figure = result_dialog._create_steady_velocity_figure(result)
    try:
        assert len(figure.axes) == 3
        axes = figure.axes
        assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])
        assert not axes[1].get_shared_y_axes().joined(axes[1], axes[2])
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), result.velocity[0])
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), result.reynolds[0])
        np.testing.assert_array_equal(axes[2].lines[0].get_ydata(), result.friction[0])
        assert axes[0].get_ylabel() == "Velocity [m/s]"
        assert axes[1].get_ylabel() == "Re"
        assert axes[2].get_ylabel() == "f_D"
        assert axes[2].get_xlabel() == "x [m]"
    finally:
        plt.close(figure)


def test_steady_flow_diagnostics_use_spatial_q_and_residual() -> None:
    result = _default_result()
    figure = result_dialog._create_steady_flow_diagnostics_figure(result)
    try:
        q_profile = result.velocity[0] * result.area
        q_mean = result.flow_rate_mean[0]
        axes = figure.axes
        assert len(axes) == 2
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), result.x)
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), q_profile)
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(), q_mean)
        np.testing.assert_allclose(axes[1].lines[0].get_xdata(), result.x)
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), q_profile - q_mean)
        np.testing.assert_allclose(axes[1].lines[1].get_ydata(), 0.0)
        assert axes[0].get_ylabel() == "Q [m³/s]"
        assert axes[1].get_ylabel() == "Q - Q_mean [m³/s]"
        assert axes[1].get_xlabel() == "x [m]"
        assert axes[0].get_ylim()[0] < axes[0].get_ylim()[1]
        assert axes[1].get_ylim()[0] < axes[1].get_ylim()[1]
    finally:
        plt.close(figure)


def test_steady_flow_diagnostics_render_zero_and_reverse_flow() -> None:
    for p_in, p_out in ((2.0e5, 2.0e5), (1.9e5, 2.0e5)):
        result = solve_pipe_flow(p_in=p_in, p_out=p_out, nx=48)
        figure = result_dialog._create_steady_flow_diagnostics_figure(result)
        try:
            assert all(np.isfinite(axis.get_ylim()).all() for axis in figure.axes)
            assert all(axis.get_ylim()[0] < axis.get_ylim()[1] for axis in figure.axes)
            np.testing.assert_allclose(
                figure.axes[1].lines[0].get_ydata(),
                result.velocity[0] * result.area - result.flow_rate_mean[0],
            )
        finally:
            plt.close(figure)


def test_transient_flow_diagnostics_separate_pressure_and_flow_scales() -> None:
    result = _transient_result()
    figure = result_dialog._create_transient_flow_diagnostics_figure(result)
    try:
        axes = figure.axes
        assert len(axes) == 2
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), result.t)
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), result.pressure[:, 0] / 1000.0)
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(), result.pressure[:, -1] / 1000.0)
        np.testing.assert_array_equal(axes[1].lines[0].get_xdata(), result.t)
        np.testing.assert_array_equal(axes[1].lines[1].get_xdata(), result.t)
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), result.flow_rate_mean)
        np.testing.assert_array_equal(axes[1].lines[1].get_ydata(), result.flow_rate_std)
        assert axes[0].get_ylabel() == "Pressure [kPa]"
        assert axes[1].get_ylabel() == "Q [m³/s]"
        assert axes[1].get_xlabel() == "t"
    finally:
        plt.close(figure)


def test_result_summary_is_model_specific_and_uses_readable_units() -> None:
    steady_summary = result_dialog._format_result_summary(_default_result())
    transient_summary = result_dialog._format_result_summary(_transient_result())

    assert "Q:" in steady_summary
    assert "mean u:" in steady_summary
    assert "Re max:" in steady_summary
    assert "Δp:" in steady_summary
    assert "m³/s" in steady_summary and "kPa" in steady_summary
    assert "flow_rate_m3s" not in steady_summary
    assert "max |p|:" in transient_summary
    assert "CFL:" in transient_summary
    assert "mean Q std:" in transient_summary
    assert "max_pressure_pa" not in transient_summary


def test_result_dialog_wires_new_steady_factories() -> None:
    dialog = object.__new__(result_dialog.PipeFlowResultDialog)
    dialog._result = _default_result()
    dialog._geometry_canvas = None
    dialog._pressure_canvas = None
    dialog._velocity_canvas = None
    dialog._quality_canvas = None

    factories = (
        "_create_geometry_figure",
        "_create_steady_pressure_figure",
        "_create_steady_velocity_figure",
        "_create_steady_flow_diagnostics_figure",
    )
    with (
        patch.object(result_dialog, "embed_plot_in_tk", return_value=object()),
        patch.multiple(
            result_dialog,
            **{name: DEFAULT for name in factories},
        ) as mocked,
    ):
        for name in factories:
            mocked[name].return_value = plt.figure()
        dialog._build_geometry_tab(object())
        dialog._build_pressure_tab(object())
        dialog._build_velocity_tab(object())
        dialog._build_quality_tab(object())

    for name in factories:
        assert mocked[name].call_count == 1
        plt.close(mocked[name].return_value)


def test_steady_result_dialog_does_not_add_animation_tab() -> None:
    dialog = object.__new__(result_dialog.PipeFlowResultDialog)
    dialog.win = MagicMock()
    dialog._result = _default_result()
    frames = [MagicMock() for _ in range(4)]
    notebook = MagicMock()
    shell = MagicMock(notebook=notebook)

    with (
        patch.object(result_dialog, "AdvancedResultShell", return_value=shell),
        patch.object(result_dialog.ttk, "Frame", side_effect=frames),
        patch.object(dialog, "_build_anim_tab") as build_animation,
        patch.object(dialog, "_build_geometry_tab"),
        patch.object(dialog, "_build_pressure_tab"),
        patch.object(dialog, "_build_velocity_tab"),
        patch.object(dialog, "_build_quality_tab"),
    ):
        dialog._build_ui()

    build_animation.assert_not_called()
    assert notebook.add.call_count == 4


def test_result_dialog_keeps_transient_maps_and_diagnostics() -> None:
    dialog = object.__new__(result_dialog.PipeFlowResultDialog)
    dialog._result = _transient_result()
    dialog._pressure_canvas = None
    dialog._velocity_canvas = None
    dialog._quality_canvas = None
    pressure_figure = plt.figure()
    velocity_figure = plt.figure()
    diagnostics_figure = plt.figure()

    with (
        patch.object(
            result_dialog,
            "create_contour_plot",
            side_effect=[pressure_figure, velocity_figure],
        ) as contour,
        patch.object(
            result_dialog,
            "_create_transient_flow_diagnostics_figure",
            return_value=diagnostics_figure,
        ) as diagnostics,
        patch.object(result_dialog, "embed_plot_in_tk", return_value=object()),
    ):
        dialog._build_pressure_tab(object())
        dialog._build_velocity_tab(object())
        dialog._build_quality_tab(object())

    assert contour.call_count == 2
    assert diagnostics.call_count == 1
    for figure in (pressure_figure, velocity_figure, diagnostics_figure):
        plt.close(figure)


def test_animation_factory_remains_available_for_transient_fields() -> None:
    result = _transient_result()
    for field, title, ylabel in (
        (result.pressure, "Pressure profile", "p"),
        (result.velocity, "Velocity profile", "u"),
        (result.reynolds, "Reynolds profile", "Re"),
    ):
        figure = result_dialog._create_line_animation_figure(
            result.x,
            result.t,
            field,
            title=title,
            ylabel=ylabel,
        )
        try:
            assert figure._animation_n_points == len(result.t)  # type: ignore[attr-defined]
        finally:
            plt.close(figure)


def test_transient_animation_wires_export_for_each_selected_field() -> None:
    dialog = object.__new__(result_dialog.PipeFlowResultDialog)
    dialog._result = _transient_result()
    dialog._anim_view_var = MagicMock()
    dialog._anim_frame = object()
    dialog._anim_canvas = None
    captured: dict[str, object] = {}

    def embed_figure(_figure: object, _parent: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    for view in ("pressure", "velocity", "reynolds"):
        dialog._anim_view_var.get.return_value = view
        with (
            patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed_figure),
            patch.object(result_dialog, "reset_embedded_animation"),
        ):
            dialog._update_animation()
        assert callable(captured["on_export_mp4"])
