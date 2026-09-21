"""Headless regression tests for Custom equation draft capture and restore."""

from __future__ import annotations

import pytest

from frontend.ui_dialogs.equation_dialog import EquationDialog
from frontend.ui_dialogs.solve_session import SolveSession


class _FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class _FakeText:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self, _start: str, _end: str) -> str:
        return f"{self.value}\n"

    def delete(self, _start: str, _end: str) -> None:
        self.value = ""

    def insert(self, _index: str, value: str) -> None:
        self.value = value


class _FakeEntry:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def delete(self, _start: int, _end: str) -> None:
        self.value = ""

    def insert(self, _index: int, value: str) -> None:
        self.value = value


def _dialog(family: str) -> EquationDialog:
    dialog = EquationDialog.__new__(EquationDialog)
    dialog.session = SolveSession(current_family=family)
    dialog.custom_order_var = _FakeVar("2")  # type: ignore[assignment]
    dialog.custom_params = _FakeEntry()  # type: ignore[assignment]
    dialog.custom_expr = _FakeText()  # type: ignore[assignment]
    dialog._restoring_custom = False
    return dialog


@pytest.mark.parametrize(
    ("family", "order", "expression", "parameters"),
    [
        ("ode", "3", "-ω**2*f[0]", "ω, forcing[3]"),
        ("difference", "2", "r*f[0]", "r"),
        ("pde_3d", "2", "-fxx-fyy-fzz", "alpha"),
    ],
)
def test_scalar_family_custom_draft_round_trip(
    family: str, order: str, expression: str, parameters: str
) -> None:
    dialog = _dialog(family)
    dialog.custom_order_var.set(order)
    dialog.custom_expr.value = expression  # type: ignore[attr-defined]
    dialog.custom_params.value = parameters  # type: ignore[attr-defined]

    dialog._capture_custom_state(family)
    dialog.custom_order_var.set("1")
    dialog.custom_expr.value = ""  # type: ignore[attr-defined]
    dialog.custom_params.value = ""  # type: ignore[attr-defined]
    dialog._apply_custom_state(family)

    assert dialog.custom_order_var.get() == order
    assert dialog.custom_expr.value == expression  # type: ignore[attr-defined]
    assert dialog.custom_params.value == parameters  # type: ignore[attr-defined]


def test_2d_pde_custom_draft_round_trip_includes_compatibility_state() -> None:
    dialog = _dialog("pde")
    dialog.custom_expr = _FakeText("sin(pi*x)*sin(pi*y)")  # type: ignore[assignment]
    dialog.custom_params = _FakeEntry("alpha")  # type: ignore[assignment]
    dialog._pde_op_var = _FakeVar("∇²f (Laplacian)")  # type: ignore[assignment]
    dialog._pde_nvars_var = _FakeVar("2")  # type: ignore[assignment]

    dialog._capture_custom_state("pde")
    dialog.custom_expr.value = ""  # type: ignore[attr-defined]
    dialog.custom_params.value = ""  # type: ignore[attr-defined]
    dialog._pde_op_var.set("-∇²f (Poisson)")
    dialog._pde_nvars_var.set("invalid")
    dialog._apply_custom_state("pde")

    draft = dialog.session.family_state("pde").custom_draft
    assert draft["operator"] == "∇²f (Laplacian)"
    assert draft["expression"] == "sin(pi*x)*sin(pi*y)"
    assert draft["parameters"] == "alpha"
    assert draft["variables"] == "2"
    assert dialog._pde_nvars_var.get() == "2"


def test_vector_ode_per_component_custom_draft_round_trip() -> None:
    dialog = _dialog("vector_ode")
    dialog.custom_params = _FakeEntry("ω, k")  # type: ignore[assignment]
    dialog._vec_n_var = _FakeVar("3")  # type: ignore[assignment]
    dialog._vec_mode_var = _FakeVar("per_component")  # type: ignore[assignment]
    dialog._active_vec_mode = "per_component"
    dialog._vec_order_vars = [_FakeVar("2"), _FakeVar("1"), _FakeVar("3")]  # type: ignore[list-item]
    dialog._vec_expr_widgets = [
        _FakeText("f[1,0]"),
        _FakeText("-f[0,0]"),
        _FakeText("ω*f[2,0]"),
    ]  # type: ignore[list-item]

    dialog._capture_custom_state("vector_ode")
    dialog._vec_order_vars = [_FakeVar("1"), _FakeVar("1"), _FakeVar("1")]  # type: ignore[list-item]
    dialog._vec_expr_widgets = [_FakeText(), _FakeText(), _FakeText()]  # type: ignore[list-item]
    dialog._refresh_vec_boxes = lambda: None  # type: ignore[method-assign]
    dialog._apply_custom_state("vector_ode")

    assert dialog._vec_n_var.get() == "3"
    assert dialog.custom_params.get() == "ω, k"
    assert [item.get() for item in dialog._vec_order_vars] == ["2", "1", "3"]
    assert [item.value for item in dialog._vec_expr_widgets] == [  # type: ignore[attr-defined]
        "f[1,0]",
        "-f[0,0]",
        "ω*f[2,0]",
    ]


def test_vector_ode_bulk_custom_draft_round_trip() -> None:
    dialog = _dialog("vector_ode")
    dialog.custom_params = _FakeEntry("omega")  # type: ignore[assignment]
    dialog._vec_n_var = _FakeVar("4")  # type: ignore[assignment]
    dialog._vec_mode_var = _FakeVar("bulk")  # type: ignore[assignment]
    dialog._active_vec_mode = "bulk"
    dialog._vec_order_vars = [_FakeVar("2")]  # type: ignore[list-item]
    dialog._vec_expr_widgets = []
    dialog._vec_bulk_expr = _FakeText("-f[i,0]")  # type: ignore[assignment]

    dialog._capture_custom_state("vector_ode")
    dialog._vec_order_vars = [_FakeVar("1")]  # type: ignore[list-item]
    dialog._vec_bulk_expr = _FakeText()  # type: ignore[assignment]
    dialog._refresh_vec_boxes = lambda: None  # type: ignore[method-assign]
    dialog._apply_custom_state("vector_ode")

    assert dialog._vec_mode_var.get() == "bulk"
    assert dialog._vec_order_vars[0].get() == "2"
    assert dialog._vec_bulk_expr.value == "-f[i,0]"  # type: ignore[attr-defined]


def test_vector_pde_custom_draft_round_trip() -> None:
    dialog = _dialog("vector_pde")
    dialog.custom_params = _FakeEntry("alpha")  # type: ignore[assignment]
    dialog._vec_n_var = _FakeVar("2")  # type: ignore[assignment]
    dialog._vec_expr_widgets = [
        _FakeText("-fxx[0]-fyy[0]"),
        _FakeText("-fxx[1]-fyy[1]"),
    ]  # type: ignore[list-item]

    dialog._capture_custom_state("vector_pde")
    dialog._vec_expr_widgets = [_FakeText(), _FakeText()]  # type: ignore[list-item]
    dialog._refresh_vector_pde_boxes = lambda: None  # type: ignore[method-assign]
    dialog._apply_custom_state("vector_pde")

    assert dialog._vec_n_var.get() == "2"
    assert dialog.custom_params.get() == "alpha"
    assert [item.value for item in dialog._vec_expr_widgets] == [  # type: ignore[attr-defined]
        "-fxx[0]-fyy[0]",
        "-fxx[1]-fyy[1]",
    ]
