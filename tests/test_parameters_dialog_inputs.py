"""Tests for ParametersDialog input collection helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from frontend.ui_dialogs import parameters_dialog as parameters_ui
from utils import DifferentialLabError


class _FakeVar:
    def __init__(self, value: object) -> None:
        self._value = value

    def get(self) -> object:
        return self._value

    def set(self, value: object) -> None:
        self._value = value


class _FakeListbox:
    def __init__(self, selected: tuple[int, ...]) -> None:
        self._selected = selected

    def curselection(self) -> tuple[int, ...]:
        return self._selected

    def selection_clear(self, _first: object, _last: object) -> None:
        self._selected = ()

    def selection_set(self, index: int) -> None:
        self._selected = (*self._selected, index)


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
    dialog.session = None
    dialog.selection = None
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
    dialog.method_desc = MagicMock()
    dialog.event_expression_var = _FakeVar("f[0] - 0.5")
    dialog.event_terminal_var = _FakeVar(True)
    dialog.event_direction_var = _FakeVar("-1")
    dialog._eq_param_vars = {
        "a": _FakeVar("2.5"),
        "weights[3]": _FakeVar("1, 2, 3"),
    }
    dialog._x0_vars = [_FakeVar("0.0"), _FakeVar("0.25")]
    dialog._y0_vars = [_FakeVar("1.0"), _FakeVar("0.0")]
    dialog._stats_listbox = _FakeListbox((0,))
    dialog._stat_keys = ["mean", "rms"]
    dialog._stats_desc_label = MagicMock()
    dialog._bc_vars = []
    dialog._bc_type_vars = []
    dialog._domain_shape_var = None
    dialog._mask_expr_var = None
    dialog._contour_bc_expr_var = None
    dialog._contour_bc_type_var = None
    dialog.ymin_var = None
    dialog.ymax_var = None
    dialog.npoints_y_var = None
    dialog.zmin_var = None
    dialog.zmax_var = None
    dialog.npoints_z_var = None
    return dialog


def _selection(equation_type: str = "ode") -> parameters_ui.EquationSelection:
    return parameters_ui.EquationSelection(
        expression="a * f",
        function_name=None,
        order=2,
        parameters={"a": 1.0},
        equation_name="Test equation",
        default_y0=[1.0, 0.0],
        default_domain=[0.0, 1.0],
        equation_type=equation_type,
    )


@pytest.mark.parametrize(
    ("equation_type", "expected_default"),
    [("vector_pde", 100), ("pde", 1000), ("pde_3d", 25)],
)
def test_pde_grid_defaults_are_practical_and_family_specific(
    equation_type: str,
    expected_default: int,
) -> None:
    """Vector PDE uses its interactive default without changing other PDE defaults."""
    assert parameters_ui._default_pde_grid_points(equation_type) == expected_default


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
    assert collected.event_expression == "f[0] - 0.5"
    assert collected.event_terminal is True
    assert collected.event_direction == -1
    assert collected.parameters["a"] == 2.5
    np.testing.assert_allclose(collected.parameters["weights"], np.array([1.0, 2.0, 3.0]))


def test_scalar_form_snapshot_round_trips_raw_values() -> None:
    dialog = _build_scalar_dialog_stub()

    snapshot = dialog.capture_form_state()
    dialog.xmin_var.set("changed")
    dialog._y0_vars[0].set("changed")
    dialog.method_var.set("changed")
    dialog._eq_param_vars["a"].set("changed")
    dialog.apply_form_state(snapshot)

    assert dialog.xmin_var.get() == "0.0"
    assert dialog.xmax_var.get() == "1.0"
    assert [variable.get() for variable in dialog._y0_vars] == ["1.0", "0.0"]
    assert dialog.method_var.get() == "RK45"
    assert dialog._stats_listbox.curselection() == (0,)
    assert dialog.event_expression_var.get() == "f[0] - 0.5"
    assert dialog.event_terminal_var.get() is True
    assert dialog.event_direction_var.get() == "-1"
    assert dialog._eq_param_vars["a"].get() == "2.5"


def test_pde_form_snapshot_keeps_domains_grid_and_boundary_modes() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "pde"
    dialog.is_pde = True
    dialog.ymin_var = _FakeVar("-2")
    dialog.ymax_var = _FakeVar("3")
    dialog.npoints_var = _FakeVar("41")
    dialog.npoints_y_var = _FakeVar("51")
    dialog._domain_shape_var = _FakeVar("Custom contour")
    dialog._bc_vars = [_FakeVar("x"), _FakeVar("y"), _FakeVar("1"), _FakeVar("2")]
    dialog._bc_type_vars = [
        _FakeVar("Dirichlet"),
        _FakeVar("Neumann"),
        _FakeVar("Dirichlet"),
        _FakeVar("Neumann"),
    ]
    dialog._mask_expr_var = _FakeVar("x**2 + y**2 <= 1")
    dialog._contour_bc_expr_var = _FakeVar("sin(x)")
    dialog._contour_bc_type_var = _FakeVar("Neumann")

    snapshot = dialog.capture_form_state()

    assert (snapshot.y_min, snapshot.y_max) == ("-2", "3")
    assert (snapshot.n_points, snapshot.n_points_y) == ("41", "51")
    assert snapshot.domain_shape == "Custom contour"
    assert snapshot.boundary_expressions == ("x", "y", "1", "2")
    assert snapshot.boundary_types[1] == "Neumann"
    assert snapshot.mask_expression == "x**2 + y**2 <= 1"
    assert snapshot.contour_boundary_expression == "sin(x)"
    assert snapshot.contour_boundary_type == "Neumann"


def test_vector_form_snapshot_keeps_component_initial_state() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "vector_ode"
    dialog.is_vector = True
    dialog.vector_components = 2
    dialog.component_orders = (1, 2)
    dialog._y0_vars = [_FakeVar("1"), _FakeVar("2"), _FakeVar("3")]
    dialog._x0_vars = [_FakeVar("0"), _FakeVar("0.1"), _FakeVar("0.2")]

    snapshot = dialog.capture_form_state()

    assert snapshot.initial_values == ("1", "2", "3")
    assert snapshot.initial_positions == ("0", "0.1", "0.2")


def test_back_saves_snapshot_releases_tk_state_and_reopens_equation() -> None:
    dialog = _build_scalar_dialog_stub()
    session = parameters_ui.SolveSession()
    selection = _selection()
    snapshot = dialog.capture_form_state()
    dialog.session = session
    dialog.selection = selection
    dialog.capture_form_state = MagicMock(return_value=snapshot)  # type: ignore[method-assign]
    dialog._release_tk_state = MagicMock()  # type: ignore[method-assign]

    with patch("frontend.ui_dialogs.equation_dialog.EquationDialog") as equation_dialog:
        dialog._on_back()

    assert session.configuration_for(selection) == snapshot
    assert dialog.win.destroy_calls == 1
    dialog._release_tk_state.assert_called_once_with()
    equation_dialog.assert_called_once_with(dialog.parent, session=session)


def test_collect_solver_inputs_reports_invalid_parameter_with_title() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog._eq_param_vars = {"a": _FakeVar("not-a-number")}

    with pytest.raises(parameters_ui._InputValidationError) as exc_info:
        dialog._collect_solver_inputs()

    assert exc_info.value.title == "Check the parameter value"
    assert exc_info.value.message == "Parameter 'a' must be a number."


def test_parameters_dialog_formats_solver_errors() -> None:
    user_error = parameters_ui._format_solver_exception(DifferentialLabError("bad input"))
    assert user_error.title == "Solver input issue"
    assert user_error.message == "bad input"

    memory_error = parameters_ui._format_solver_exception(MemoryError("too many points"))
    assert memory_error.title == "Not enough memory"
    assert "Try reducing the grid size" in memory_error.message

    generic_error = parameters_ui._format_solver_exception(RuntimeError("boom"))
    assert generic_error.title == "Solver error"
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
    assert kwargs["message"] == "Solving equation..."
    assert kwargs["format_error"] is parameters_ui._format_solver_exception
    assert "on_complete" not in kwargs
    assert callable(kwargs["task"])
    assert callable(kwargs["on_success"])
    assert all(cell.cell_contents is not dialog for cell in kwargs["task"].__closure__ or ())
    assert dialog.win.destroy_calls == 1


def test_on_solve_releases_pde_3d_tk_vars_before_starting_worker() -> None:
    dialog = _build_scalar_dialog_stub()
    solver_inputs = dialog._collect_solver_inputs()
    dialog.equation_type = "pde_3d"
    dialog.variables = ["x", "y", "z"]
    dialog.zmin_var = _FakeVar("0.0")
    dialog.zmax_var = _FakeVar("1.0")
    dialog.npoints_z_var = _FakeVar("9")
    dialog._collect_solver_inputs = MagicMock(return_value=solver_inputs)  # type: ignore[method-assign]
    dialog._confirm_heavy_request = MagicMock(return_value=True)  # type: ignore[method-assign]

    with patch.object(parameters_ui, "run_task_with_loading", create=True):
        dialog._on_solve()

    assert dialog.zmin_var is None
    assert dialog.zmax_var is None
    assert dialog.npoints_z_var is None


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
