"""Tests for ParametersDialog input collection helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from frontend.ui_dialogs import parameters_dialog as parameters_ui
from utils import DifferentialLabError


class _FakeVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _FakeListbox:
    def __init__(self, selected: tuple[int, ...]) -> None:
        self._selected = selected

    def curselection(self) -> tuple[int, ...]:
        return self._selected


class _FakeParent:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []

    def after(self, delay_ms: int, callback: object) -> None:
        self.after_calls.append((delay_ms, callback))

    def winfo_exists(self) -> bool:
        return True


class _FakeWindow:
    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


def _build_scalar_dialog_stub() -> parameters_ui.ParametersDialog:
    dialog = parameters_ui.ParametersDialog.__new__(parameters_ui.ParametersDialog)
    dialog.expression = "a * f"
    dialog.function_name = None
    dialog.order = 2
    dialog.parameters = {"a": 1.0, "weights[3]": [1.0, 1.0, 1.0]}
    dialog.equation_name = "Test equation"
    dialog.equation_type = "ode"
    dialog.variables = ["x"]
    dialog.vector_expressions = None
    dialog.vector_components = 1
    dialog.pde_operator = "neg_laplacian"
    dialog.component_orders = None
    dialog.is_pde = False
    dialog.is_vector = False
    dialog.parent = _FakeParent()
    dialog.win = _FakeWindow()
    dialog.xmin_var = _FakeVar("0.0")
    dialog.xmax_var = _FakeVar("1.0")
    dialog.npoints_var = _FakeVar("50")
    dialog.method_var = _FakeVar("RK45")
    dialog._eq_param_vars = {
        "a": _FakeVar("2.5"),
        "weights[3]": _FakeVar("1, 2, 3"),
    }
    dialog._x0_vars = [_FakeVar("0.0"), _FakeVar("0.25")]
    dialog._y0_vars = [_FakeVar("1.0"), _FakeVar("0.0")]
    dialog._stats_listbox = _FakeListbox((0,))
    dialog._stat_keys = ["mean", "rms"]
    dialog._bc_vars = []
    dialog._bc_type_vars = []
    dialog._domain_shape_var = None
    dialog._mask_expr_var = None
    dialog._contour_bc_expr_var = None
    dialog._contour_bc_type_var = None
    dialog.ymin_var = None
    dialog.ymax_var = None
    dialog.npoints_y_var = None
    return dialog


def test_collect_solver_inputs_builds_scalar_pipeline_kwargs() -> None:
    dialog = _build_scalar_dialog_stub()

    collected = dialog._collect_solver_inputs()

    assert collected.x_min == 0.0
    assert collected.x_max == 1.0
    assert collected.y0 == [1.0, 0.0]
    assert collected.n_points == 50
    assert collected.method == "RK45"
    assert collected.selected_stats == {"mean"}
    assert collected.x0_list == [0.0, 0.25]
    assert collected.y_min is None
    assert collected.y_max is None
    assert collected.n_points_y is None
    assert collected.parameters["a"] == 2.5
    np.testing.assert_allclose(collected.parameters["weights"], np.array([1.0, 2.0, 3.0]))


def test_collect_solver_inputs_reports_invalid_parameter_with_title() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog._eq_param_vars = {"a": _FakeVar("not-a-number")}

    with pytest.raises(parameters_ui._InputValidationError) as exc_info:
        dialog._collect_solver_inputs()

    assert exc_info.value.title == "Invalid Parameter"
    assert exc_info.value.message == "Parameter 'a' must be a number."


def test_parameters_dialog_formats_solver_errors() -> None:
    user_error = parameters_ui._format_solver_exception(DifferentialLabError("bad input"))
    assert user_error.title == "DifferentialLabError"
    assert user_error.message == "bad input"

    memory_error = parameters_ui._format_solver_exception(MemoryError("too many points"))
    assert memory_error.title == "Memory Error"
    assert "Try reducing the grid size" in memory_error.message

    generic_error = parameters_ui._format_solver_exception(RuntimeError("boom"))
    assert generic_error.title == "Error"
    assert generic_error.message == "boom"


def test_on_solve_delegates_to_shared_background_runner() -> None:
    dialog = _build_scalar_dialog_stub()
    solver_inputs = dialog._collect_solver_inputs()
    dialog._collect_solver_inputs = MagicMock(return_value=solver_inputs)  # type: ignore[method-assign]

    with (
        patch.object(parameters_ui, "run_task_with_loading", create=True) as run_task,
        patch("frontend.ui_dialogs.loading_dialog.LoadingDialog"),
        patch("threading.Thread"),
    ):
        dialog._on_solve()

    run_task.assert_called_once()
    kwargs = run_task.call_args.kwargs
    assert kwargs["parent"] is dialog.parent
    assert kwargs["message"] == "Solving..."
    assert kwargs["format_error"] is parameters_ui._format_solver_exception
    assert kwargs["on_complete"] is not None
    assert callable(kwargs["task"])
    assert callable(kwargs["on_success"])
    assert dialog.win.destroy_calls == 1


def test_on_solve_aborts_when_heavy_request_confirmation_is_declined() -> None:
    dialog = _build_scalar_dialog_stub()
    solver_inputs = dialog._collect_solver_inputs()
    dialog._collect_solver_inputs = MagicMock(return_value=solver_inputs)  # type: ignore[method-assign]
    dialog._confirm_heavy_request = MagicMock(return_value=False)  # type: ignore[attr-defined]

    with patch.object(parameters_ui, "run_task_with_loading", create=True) as run_task:
        dialog._on_solve()

    dialog._confirm_heavy_request.assert_called_once_with(solver_inputs)
    run_task.assert_not_called()
    assert dialog.win.destroy_calls == 0
