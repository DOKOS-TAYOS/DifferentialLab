"""Parser, pipeline, and display-free UI routing tests for scalar PDE 3D."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from frontend.ui_dialogs.equation_dialog import EquationDialog
from frontend.ui_dialogs.result_dialog import ResultDialog
from pipeline import SolverResult, run_solver_pipeline
from solver.equation_parser import parse_pde_3d_residual_expression


class _FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class _FakeText:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self, _start: str, _end: str) -> str:
        return self.value


class _FakeWindow:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def test_parser_supports_named_and_indexed_3d_notation() -> None:
    """The custom parser exposes every documented 3D derivative name."""
    residual = parse_pde_3d_residual_expression(
        "f[0,0] + fyy + f[2,2] + fxy + fxz + fyz + fx + fy + fz + f + x + y + z",
        ["x[0]", "x[1]", "x[2]"],
    )
    result = residual(
        0.1,
        0.2,
        0.3,
        f=1.0,
        fx=2.0,
        fy=3.0,
        fz=4.0,
        fxx=5.0,
        fxy=6.0,
        fxz=7.0,
        fyy=8.0,
        fyz=9.0,
        fzz=10.0,
    )
    assert result == 55.6


def test_pipeline_dispatches_scalar_pde_3d_and_preserves_three_grids() -> None:
    """PDE 3D is a standard data-only pipeline route."""
    result = run_solver_pipeline(
        expression="-fxx - fyy - fzz",
        function_name=None,
        order=2,
        parameters={},
        equation_name="Homogeneous PDE 3D",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        y0=[],
        n_points=7,
        n_points_y=6,
        n_points_z=5,
        method="fdm",
        selected_stats={"mean"},
        equation_type="pde_3d",
        variables=["x", "y", "z"],
    )

    assert isinstance(result, SolverResult)
    assert result.equation_type == "pde_3d"
    assert result.y.shape == (5, 6, 7)
    assert result.y_grid is not None and result.y_grid.shape == (6,)
    assert result.z_grid is not None and result.z_grid.shape == (5,)
    assert result.metadata["domain"] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert result.metadata["matrix_shape"] == (60, 60)
    assert set(result.statistics) == {"central_xy_slice"}
    np.testing.assert_allclose(result.y, 0.0)


def test_custom_pde_3d_ui_routes_to_standard_parameters_dialog() -> None:
    """The equation chooser sends custom 3D residuals through the standard workflow."""
    dialog = EquationDialog.__new__(EquationDialog)
    dialog.custom_expr = _FakeText("-fxx - fyy - fzz")

    def parse_params() -> dict[str, float]:
        return {"k": 0.0}

    dialog._parse_custom_params = parse_params  # type: ignore[method-assign]
    dialog.win = _FakeWindow()
    dialog.parent = object()

    with patch("frontend.ui_dialogs.parameters_dialog.ParametersDialog") as parameters_dialog:
        dialog._on_next_custom_pde_3d()

    assert dialog.win.destroyed is True
    kwargs = parameters_dialog.call_args.kwargs
    assert kwargs["equation_type"] == "pde_3d"
    assert kwargs["variables"] == ["x", "y", "z"]
    assert kwargs["default_domain"] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert kwargs["expression"] == "-fxx - fyy - fzz"


def test_result_dialog_selects_scalar_pde_3d_orthogonal_slice_without_display() -> None:
    """The result view extracts a scalar XZ slice in public z/y/x ordering."""
    dialog = ResultDialog.__new__(ResultDialog)
    field = np.arange(24, dtype=float).reshape(2, 3, 4)
    dialog._result = SimpleNamespace(
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y_grid=np.array([0.0, 1.0, 2.0]),
        z_grid=np.array([0.0, 1.0]),
        y=field,
        metadata={"variables": ["x", "y", "z"], "equation_name": "Scalar 3D"},
    )
    dialog._pde_3d_slice_plane_var = _FakeVar("XZ")
    dialog._pde_3d_slice_index_var = _FakeVar("1")
    dialog._pde_3d_slice_frame = object()
    captured: dict[str, object] = {}

    def replace_plot(frame: object, figure: object, canvas_attr: str) -> None:
        captured.update(frame=frame, figure=figure, canvas_attr=canvas_attr)

    dialog._replace_plot = replace_plot  # type: ignore[method-assign]
    with patch("plotting.create_contour_plot", return_value="figure") as create_plot:
        dialog._update_pde_3d_slice()

    args = create_plot.call_args.args
    kwargs = create_plot.call_args.kwargs
    np.testing.assert_array_equal(args[0], dialog._result.x)
    np.testing.assert_array_equal(args[1], dialog._result.z_grid)
    np.testing.assert_array_equal(args[2], field[:, 1, :])
    assert kwargs["xlabel"] == "x"
    assert kwargs["ylabel"] == "z"
    assert "y=1" in kwargs["title"]
    assert captured["figure"] == "figure"
