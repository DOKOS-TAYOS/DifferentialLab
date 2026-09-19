"""Headless tests for the standard PDE axis-sweep result views."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from frontend.ui_dialogs.result_dialog import ResultDialog


class _Var:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class _Child:
    def __init__(self) -> None:
        self.destroyed = 0

    def destroy(self) -> None:
        self.destroyed += 1


class _Frame:
    def __init__(self, children: list[_Child]) -> None:
        self.children = children

    def winfo_children(self) -> list[_Child]:
        return self.children


class _Canvas:
    def __init__(self, figure: object) -> None:
        self.figure = figure
        self.stopped = 0

    def _stop_animation(self) -> None:
        self.stopped += 1


def _dialog(result: SimpleNamespace) -> ResultDialog:
    dialog = ResultDialog.__new__(ResultDialog)
    dialog._result = result
    return dialog


def test_scalar_2d_axis_sweep_builder_maps_both_displayed_coordinates() -> None:
    x = np.array([10.0, 20.0, 30.0])
    y = np.array([1.0, 2.0])
    field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dialog = _dialog(
        SimpleNamespace(
            equation_type="pde",
            x=x,
            y=field,
            y_grid=y,
            metadata={"variables": ["longitude", "height"], "equation_name": "u"},
        )
    )
    dialog._pde_2d_sweep_axis_var = _Var("height")
    with patch("plotting.create_line_animation_plot", return_value="figure") as factory:
        dialog._create_pde_2d_axis_sweep_figure()
        args = factory.call_args.args
        kwargs = factory.call_args.kwargs
        np.testing.assert_allclose(args[0], y)
        np.testing.assert_allclose(args[1], x)
        np.testing.assert_allclose(args[2], field)
        assert kwargs["frame_label"] == "height"
        assert kwargs["xlabel"] == "longitude"

    dialog._pde_2d_sweep_axis_var = _Var("longitude")
    with patch("plotting.create_line_animation_plot", return_value="figure") as factory:
        dialog._create_pde_2d_axis_sweep_figure()
        args = factory.call_args.args
        np.testing.assert_allclose(args[0], x)
        np.testing.assert_allclose(args[1], y)
        np.testing.assert_allclose(args[2], field.T)
        assert factory.call_args.kwargs["frame_label"] == "longitude"
        assert factory.call_args.kwargs["xlabel"] == "height"


def test_vector_2d_axis_sweep_builder_routes_component_and_magnitude() -> None:
    values = np.array(
        [
            [[3.0, 0.0], [0.0, 5.0]],
            [[4.0, 2.0], [1.0, 12.0]],
        ]
    )
    dialog = _dialog(
        SimpleNamespace(
            equation_type="vector_pde",
            vector_components=2,
            x=np.array([0.0, 1.0]),
            y=values,
            y_grid=np.array([0.0, 1.0]),
            metadata={"variables": ["x", "y"]},
        )
    )
    dialog._pde_2d_sweep_axis_var = _Var("y")
    dialog._pde_2d_sweep_field_var = _Var("Magnitude")
    with patch("plotting.create_line_animation_plot") as factory:
        dialog._create_pde_2d_axis_sweep_figure()
        np.testing.assert_allclose(factory.call_args.args[2], np.linalg.norm(values, axis=0))
        assert factory.call_args.kwargs["ylabel"] == "|f|"

    dialog._pde_2d_sweep_field_var = _Var("Component 1")
    with patch("plotting.create_line_animation_plot") as factory:
        dialog._create_pde_2d_axis_sweep_figure()
        np.testing.assert_allclose(factory.call_args.args[2], values[1])
        assert factory.call_args.kwargs["ylabel"] == "f[1]"


def test_scalar_3d_axis_sweep_builder_maps_all_plane_orientations() -> None:
    values = np.arange(24.0).reshape(2, 3, 4)
    dialog = _dialog(
        SimpleNamespace(
            equation_type="pde_3d",
            x=np.array([10.0, 20.0, 30.0, 40.0]),
            y=values,
            y_grid=np.array([1.0, 2.0, 3.0]),
            z_grid=np.array([100.0, 200.0]),
            metadata={"variables": ["lon", "lat", "alt"]},
        )
    )
    expected = {
        "lon": (
            dialog._result.x,
            dialog._result.y_grid,
            dialog._result.z_grid,
            values.transpose(2, 0, 1),
        ),
        "lat": (
            dialog._result.y_grid,
            dialog._result.x,
            dialog._result.z_grid,
            values.transpose(1, 0, 2),
        ),
        "alt": (dialog._result.z_grid, dialog._result.x, dialog._result.y_grid, values),
    }
    for label, (frames_axis, horizontal, vertical, frames) in expected.items():
        dialog._pde_3d_sweep_axis_var = _Var(label)
        with patch("plotting.create_image_animation_plot") as factory:
            dialog._create_pde_3d_axis_sweep_figure()
        args = factory.call_args.args
        np.testing.assert_allclose(args[0], frames_axis)
        np.testing.assert_allclose(args[1], frames)
        np.testing.assert_allclose(factory.call_args.kwargs["x_coordinates"], horizontal)
        np.testing.assert_allclose(factory.call_args.kwargs["y_coordinates"], vertical)
        assert factory.call_args.kwargs["frame_label"] == label


def test_axis_sweep_replacement_owns_only_the_current_canvas() -> None:
    dialog = ResultDialog.__new__(ResultDialog)
    dialog._canvases = []
    dialog._result = SimpleNamespace()
    old_figure = plt.figure()
    new_figure = plt.figure()
    old_canvas = _Canvas(old_figure)
    child = _Child()
    frame = _Frame([child])
    dialog._axis_canvas = old_canvas
    dialog._canvases.append(old_canvas)
    new_canvas = _Canvas(new_figure)

    try:
        with patch(
            "frontend.ui_dialogs.result_dialog.embed_animation_plot_in_tk",
            return_value=new_canvas,
        ) as embed:
            dialog._replace_animation_plot(
                frame,
                lambda: new_figure,
                "_axis_canvas",
                "axis_sweep",
            )

        assert old_canvas.stopped == 1
        assert child.destroyed == 1
        assert dialog._canvases == [new_canvas]
        assert embed.call_args.kwargs["on_export_mp4"] is not None
    finally:
        plt.close(old_figure)
        plt.close(new_figure)


def test_axis_sweep_export_uses_fresh_factory_and_requested_duration() -> None:
    dialog = ResultDialog.__new__(ResultDialog)
    dialog.win = object()
    created: list[object] = []

    def factory() -> object:
        figure = object()
        created.append(figure)
        return figure

    with (
        patch(
            "frontend.ui_dialogs.result_dialog.filedialog.asksaveasfilename",
            return_value="axis.mp4",
        ),
        patch("frontend.ui_dialogs.result_dialog.messagebox.showinfo"),
        patch("plotting.export_animated_figure_to_mp4") as export,
    ):
        dialog._export_axis_sweep_mp4(factory, 7.5)

    assert len(created) == 1
    assert export.call_args.args[0] is created[0]
    assert export.call_args.kwargs["duration_seconds"] == 7.5
