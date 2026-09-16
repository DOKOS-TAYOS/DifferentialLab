"""Display-free tests for the standard Vector PDE UI route."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from frontend.ui_dialogs.equation_dialog import EquationDialog
from frontend.ui_dialogs.result_dialog import ResultDialog


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


def test_custom_vector_pde_ui_routes_to_standard_parameters_dialog() -> None:
    dialog = EquationDialog.__new__(EquationDialog)
    dialog._vec_n_var = _FakeVar("2")
    dialog._vec_expr_widgets = [
        _FakeText("fxx[0] + fyy[0] + f[1]"),
        _FakeText("fxx[1] + fyy[1] + f[0]"),
    ]
    dialog._parse_custom_params = lambda: {"k": 0.0}  # type: ignore[method-assign]
    dialog.win = _FakeWindow()
    dialog.parent = object()

    with patch("frontend.ui_dialogs.parameters_dialog.ParametersDialog") as parameters_dialog:
        dialog._on_next_custom_vector_pde()

    assert dialog.win.destroyed is True
    kwargs = parameters_dialog.call_args.kwargs
    assert kwargs["equation_type"] == "vector_pde"
    assert kwargs["vector_components"] == 2
    assert kwargs["vector_expressions"] == [
        "fxx[0] + fyy[0] + f[1]",
        "fxx[1] + fyy[1] + f[0]",
    ]
    assert kwargs["variables"] == ["x", "y"]


def test_vector_pde_result_fields_include_components_and_magnitude_without_display() -> None:
    dialog = ResultDialog.__new__(ResultDialog)
    fields = np.array(
        [
            [[3.0, 0.0], [0.0, 5.0]],
            [[4.0, 2.0], [1.0, 12.0]],
        ]
    )
    dialog._result = SimpleNamespace(
        equation_type="vector_pde",
        y=fields,
        vector_components=2,
    )

    component, component_label = dialog._selected_pde_field("missing")
    np.testing.assert_allclose(component, fields[0])
    assert component_label == "f[0]"

    dialog._field = _FakeVar("Magnitude")
    magnitude, magnitude_label = dialog._selected_pde_field("_field")
    np.testing.assert_allclose(magnitude, [[5.0, 2.0], [1.0, 13.0]])
    assert magnitude_label == "|f|"


def test_three_component_vector_pde_ui_omits_planar_field_views() -> None:
    """A Vector PDE is planar only when it has exactly two components."""
    dialog = ResultDialog.__new__(ResultDialog)
    dialog._result = SimpleNamespace(vector_components=3)

    assert dialog._vector_pde_view_labels() == ["Components", "Magnitude"]


def test_vector_pde_field_view_dispatches_quiver_without_display() -> None:
    """The result dialog selects the plotting view and preserves the chosen origin."""
    dialog = ResultDialog.__new__(ResultDialog)
    dialog._result = SimpleNamespace(
        x=np.array([0.0, 1.0]),
        y=np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
        metadata={"equation_name": "Vector field"},
    )
    dialog._vector_pde_view_var = _FakeVar("Quiver")
    dialog._vector_pde_origin_x_var = _FakeVar("1.5")
    dialog._vector_pde_origin_y_var = _FakeVar("-2")
    dialog._vector_pde_field_frame = object()
    dialog._require_pde_y_grid = lambda: np.array([0.0])  # type: ignore[method-assign]
    captured: dict[str, object] = {}
    dialog._replace_plot = lambda frame, figure, canvas: captured.update(  # type: ignore[method-assign]
        frame=frame, figure=figure, canvas=canvas
    )

    with patch("plotting.create_vector_field_plot", return_value="figure") as create_plot:
        dialog._update_vector_pde_field()

    assert create_plot.call_args.kwargs["view"] == "quiver"
    assert create_plot.call_args.kwargs["origin"] == (1.5, -2.0)
    assert captured["figure"] == "figure"
