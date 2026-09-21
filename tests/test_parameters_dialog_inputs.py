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


class _FakeFrame:
    def __init__(self) -> None:
        self.visible = False

    def pack(self, **_kwargs: object) -> None:
        self.visible = True

    def pack_forget(self) -> None:
        self.visible = False


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
    dialog.event_enabled_var = _FakeVar(True)
    dialog.event_expression_var = _FakeVar("f[0] - 0.5")
    dialog.event_terminal_var = _FakeVar(True)
    dialog.event_direction_var = _FakeVar("-1")
    dialog._event_controls_frame = _FakeFrame()
    dialog._eq_param_vars = {
        "a": _FakeVar("2.5"),
        "weights[3]": _FakeVar("1, 2, 3"),
    }
    dialog._x0_vars = [_FakeVar("0.0"), _FakeVar("0.25")]
    dialog._y0_vars = [_FakeVar("1.0"), _FakeVar("0.0")]
    dialog._stat_keys = ["mean", "rms"]
    dialog._stat_vars = {"mean": _FakeVar(True), "rms": _FakeVar(False)}
    dialog._bc_vars = []
    dialog._bc_type_vars = []
    dialog._bc_expression_vars_by_face = {}
    dialog._bc_type_vars_by_face = {}
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
    assert {key for key, variable in dialog._stat_vars.items() if variable.get()} == {"mean"}
    assert dialog.event_enabled_var.get() is True
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


def test_2d_pde_visible_faces_serialize_in_historical_solver_order() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "pde"
    dialog.is_pde = True
    visual_order, solver_order = parameters_ui._pde_face_orders("pde")
    assert visual_order == ("left", "right", "bottom", "top")
    assert solver_order == ("bottom", "top", "left", "right")

    expressions = {face: _FakeVar(f"expr-{face}") for face in visual_order}
    types = {
        face: _FakeVar("Dirichlet" if index % 2 == 0 else "Neumann")
        for index, face in enumerate(visual_order)
    }
    dialog._bc_vars = [expressions[face] for face in solver_order]
    dialog._bc_type_vars = [types[face] for face in solver_order]

    bc_expressions, bc_types, *_rest = dialog._collect_pde_options()

    assert bc_expressions == [
        "expr-bottom",
        "expr-top",
        "expr-left",
        "expr-right",
    ]
    assert bc_types == ["dirichlet", "neumann", "dirichlet", "neumann"]


def test_3d_pde_visible_faces_serialize_in_historical_solver_order() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "pde_3d"
    dialog.is_pde = True
    visual_order, solver_order = parameters_ui._pde_face_orders("pde_3d")
    assert visual_order == ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    assert solver_order == ("z_min", "z_max", "y_min", "y_max", "x_min", "x_max")

    expressions = {face: _FakeVar(f"expr-{face}") for face in visual_order}
    types = {
        face: _FakeVar("Dirichlet" if index % 2 == 0 else "Neumann")
        for index, face in enumerate(visual_order)
    }
    dialog._bc_vars = [expressions[face] for face in solver_order]
    dialog._bc_type_vars = [types[face] for face in solver_order]

    bc_expressions, bc_types, *_rest = dialog._collect_pde_options()

    assert bc_expressions == [
        "expr-z_min",
        "expr-z_max",
        "expr-y_min",
        "expr-y_max",
        "expr-x_min",
        "expr-x_max",
    ]
    assert bc_types == [
        "dirichlet",
        "neumann",
        "dirichlet",
        "neumann",
        "dirichlet",
        "neumann",
    ]


def test_3d_pde_axes_map_to_solver_bounds_and_grid_sizes() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "pde_3d"
    dialog.is_pde = True
    dialog.xmin_var = _FakeVar("-1")
    dialog.xmax_var = _FakeVar("2")
    dialog.ymin_var = _FakeVar("-3")
    dialog.ymax_var = _FakeVar("4")
    dialog.zmin_var = _FakeVar("-5")
    dialog.zmax_var = _FakeVar("6")
    dialog.npoints_var = _FakeVar("11")
    dialog.npoints_y_var = _FakeVar("12")
    dialog.npoints_z_var = _FakeVar("13")

    collected = dialog._collect_solver_inputs()

    assert (collected.x_min, collected.x_max, collected.n_points) == (-1.0, 2.0, 11)
    assert (collected.y_min, collected.y_max, collected.n_points_y) == (-3.0, 4.0, 12)
    assert (collected.z_min, collected.z_max, collected.n_points_z) == (-5.0, 6.0, 13)


def test_2d_pde_axes_map_to_solver_bounds_and_grid_sizes() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "pde"
    dialog.is_pde = True
    dialog.xmin_var = _FakeVar("-1")
    dialog.xmax_var = _FakeVar("2")
    dialog.ymin_var = _FakeVar("-3")
    dialog.ymax_var = _FakeVar("4")
    dialog.npoints_var = _FakeVar("21")
    dialog.npoints_y_var = _FakeVar("22")

    collected = dialog._collect_solver_inputs()

    assert (collected.x_min, collected.x_max, collected.n_points) == (-1.0, 2.0, 21)
    assert (collected.y_min, collected.y_max, collected.n_points_y) == (-3.0, 4.0, 22)


def test_vector_form_snapshot_keeps_component_initial_state() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "vector_ode"
    dialog.is_vector = True
    dialog.vector_components = 2
    dialog.component_orders = (1, 2)
    dialog._y0_vars = [_FakeVar("1"), _FakeVar("2"), _FakeVar("3")]
    dialog._x0_vars = [_FakeVar("0"), _FakeVar("0.1"), _FakeVar("0.2")]

    snapshot = dialog.capture_form_state()
    for variable in (*dialog._y0_vars, *dialog._x0_vars):
        variable.set("changed")
    dialog.apply_form_state(snapshot)

    assert snapshot.initial_values == ("1", "2", "3")
    assert snapshot.initial_positions == ("0", "0.1", "0.2")
    assert [variable.get() for variable in dialog._y0_vars] == ["1", "2", "3"]
    assert [variable.get() for variable in dialog._x0_vars] == ["0", "0.1", "0.2"]


def test_3d_pde_snapshot_round_trips_all_six_historical_faces() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.equation_type = "pde_3d"
    dialog.is_pde = True
    dialog.ymin_var = _FakeVar("-2")
    dialog.ymax_var = _FakeVar("2")
    dialog.zmin_var = _FakeVar("-3")
    dialog.zmax_var = _FakeVar("3")
    dialog.npoints_var = _FakeVar("11")
    dialog.npoints_y_var = _FakeVar("12")
    dialog.npoints_z_var = _FakeVar("13")
    dialog._bc_vars = [_FakeVar(f"face-{index}") for index in range(6)]
    dialog._bc_type_vars = [
        _FakeVar("Dirichlet" if index % 2 == 0 else "Neumann") for index in range(6)
    ]

    snapshot = dialog.capture_form_state()
    for variable in (*dialog._bc_vars, *dialog._bc_type_vars):
        variable.set("changed")
    dialog.apply_form_state(snapshot)

    assert snapshot.boundary_expressions == tuple(f"face-{index}" for index in range(6))
    assert [variable.get() for variable in dialog._bc_vars] == [
        f"face-{index}" for index in range(6)
    ]
    assert [variable.get() for variable in dialog._bc_type_vars] == [
        "Dirichlet",
        "Neumann",
        "Dirichlet",
        "Neumann",
        "Dirichlet",
        "Neumann",
    ]


def test_metric_catalog_is_fully_grouped_and_defaults_selected() -> None:
    grouped = {key for _group_name, keys in parameters_ui._STATISTIC_GROUPS for key in keys}
    assert grouped == set(parameters_ui.AVAILABLE_STATISTICS)
    assert set(parameters_ui._STATISTIC_LABELS) == set(parameters_ui.AVAILABLE_STATISTICS)

    dialog = _build_scalar_dialog_stub()
    dialog._stat_keys = list(parameters_ui.AVAILABLE_STATISTICS)
    dialog._stat_vars = {key: _FakeVar(True) for key in dialog._stat_keys}
    assert set(dialog.capture_form_state().statistics) == set(parameters_ui.AVAILABLE_STATISTICS)


def test_metric_select_all_and_clear_helpers() -> None:
    dialog = _build_scalar_dialog_stub()

    dialog._clear_statistics()
    assert not any(variable.get() for variable in dialog._stat_vars.values())

    dialog._select_all_statistics()
    assert all(variable.get() for variable in dialog._stat_vars.values())


def test_disabled_event_retains_snapshot_text_but_is_ignored_by_solver() -> None:
    dialog = _build_scalar_dialog_stub()
    dialog.event_enabled_var.set(False)

    snapshot = dialog.capture_form_state()
    collected = dialog._collect_solver_inputs()

    assert snapshot.event_enabled is False
    assert snapshot.event_expression == "f[0] - 0.5"
    assert snapshot.event_terminal is True
    assert snapshot.event_direction == "-1"
    assert collected.event_expression is None
    assert collected.event_terminal is False
    assert collected.event_direction == 0


def test_event_visibility_helper_preserves_retained_values() -> None:
    dialog = _build_scalar_dialog_stub()

    dialog.event_enabled_var.set(False)
    dialog._on_event_enabled_change()
    assert dialog._event_controls_frame.visible is False
    assert dialog.event_expression_var.get() == "f[0] - 0.5"

    dialog.event_enabled_var.set(True)
    dialog._on_event_enabled_change()
    assert dialog._event_controls_frame.visible is True
    assert dialog.event_expression_var.get() == "f[0] - 0.5"


@pytest.mark.parametrize(
    ("equation_type", "expected"),
    [
        ("difference", (760, 620)),
        ("ode", (980, 720)),
        ("vector_ode", (1080, 780)),
        ("pde", (1120, 800)),
        ("pde_3d", (1180, 840)),
    ],
)
def test_preferred_configuration_size_is_family_specific(
    equation_type: str,
    expected: tuple[int, int],
) -> None:
    assert parameters_ui._preferred_configuration_size(equation_type) == expected


def test_responsive_layout_breakpoint() -> None:
    assert parameters_ui._uses_stacked_layout(899)
    assert not parameters_ui._uses_stacked_layout(900)


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
    dialog._bc_expression_vars_by_face = {"z_min": _FakeVar("0")}
    dialog._bc_type_vars_by_face = {"z_min": _FakeVar("Dirichlet")}
    dialog._collect_solver_inputs = MagicMock(return_value=solver_inputs)  # type: ignore[method-assign]
    dialog._confirm_heavy_request = MagicMock(return_value=True)  # type: ignore[method-assign]

    with patch.object(parameters_ui, "run_task_with_loading", create=True):
        dialog._on_solve()

    assert dialog.zmin_var is None
    assert dialog.zmax_var is None
    assert dialog.npoints_z_var is None
    assert dialog._bc_expression_vars_by_face == {}
    assert dialog._bc_type_vars_by_face == {}


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
