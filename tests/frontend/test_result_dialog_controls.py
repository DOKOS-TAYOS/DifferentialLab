"""Deterministic coverage for the ResultDialog scientific control contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from frontend.ui_dialogs.result_dialog import (
    ResultDialog,
    _SeriesSelector,
    extract_pde_line_slice,
    normalized_series_selection,
    pde_3d_fixed_grid,
    pde_fixed_axis_label,
    responsive_control_rows,
)
from solver.notation import FNotation, generate_derivative_labels


class _BoolVar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def get(self) -> bool:
        return self.value

    def set(self, value: bool) -> None:
        self.value = value


class _StringVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def configure(self, *, text: str) -> None:
        self.text = text


class _Combo:
    def __init__(self) -> None:
        self.values: tuple[str, ...] = ()
        self.index = -1

    def configure(self, *, values: list[str]) -> None:
        self.values = tuple(values)

    def current(self, index: int | None = None) -> int:
        if index is not None:
            self.index = index
        return self.index

    def cget(self, option: str) -> tuple[str, ...]:
        assert option == "values"
        return self.values


def _selector_state(selected: set[int], label_count: int) -> _SeriesSelector:
    selector = _SeriesSelector.__new__(_SeriesSelector)
    selector._labels = tuple(f"row {index}" for index in range(label_count))
    selector._variables = [_BoolVar(index in selected) for index in range(label_count)]
    selector._menu_button = None
    selector._on_change = lambda: None
    return selector


def test_series_selection_defaults_to_first_and_preserves_row_indexes() -> None:
    scalar_labels = generate_derivative_labels(FNotation(kind="ode", order=3))
    assert normalized_series_selection((), count=len(scalar_labels)) == (0,)
    assert normalized_series_selection((0, 2), count=len(scalar_labels)) == (0, 2)

    vector_notation = FNotation(
        kind="vector_ode",
        n_components=2,
        order=3,
        component_orders=(1, 3),
    )
    vector_labels = generate_derivative_labels(vector_notation)
    selector = _selector_state({1, 3}, len(vector_labels))
    assert selector.selected_indices() == [1, 3]
    assert [vector_labels[index] for index in selector.selected_indices()] == [
        vector_labels[1],
        vector_labels[3],
    ]


def test_series_selector_enforces_at_least_one_visible_selection() -> None:
    selector = _selector_state(set(), 4)
    selector._toggle(2)
    assert selector.selected_indices() == [2]
    assert selector._variables[2].get() is True


def test_responsive_control_rows_reflow_without_splitting_groups() -> None:
    widths = (150, 180, 130, 160)
    assert responsive_control_rows(700, widths) == ((0, 1, 2, 3),)
    assert responsive_control_rows(360, widths) == ((0, 1), (2, 3))
    assert responsive_control_rows(170, widths) == ((0,), (1,), (2,), (3,))


def test_pde_3d_fixed_grid_maps_planes_to_physical_coordinates() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([10.0, 20.0])
    z = np.array([0.0, 1.0, 2.0, 3.0])
    for plane, expected_axis, expected_grid in (
        ("XY", "z", z),
        ("XZ", "y", y),
        ("YZ", "x", x),
    ):
        axis, grid = pde_3d_fixed_grid(plane, x, y, z)
        assert axis == expected_axis
        assert grid is expected_grid


def test_pde_3d_plane_change_resets_exact_coordinate_and_index_choices() -> None:
    dialog = ResultDialog.__new__(ResultDialog)
    dialog._result = SimpleNamespace(
        x=np.array([-1.0, 0.25, 2.0]),
        y_grid=np.array([0.125, 0.5, 0.875, 1.0]),
        z_grid=np.array([0.0, 0.058823529411764705, 0.2, 0.9, 1.0]),
        metadata={"variables": ["x", "y", "z"]},
    )
    dialog._pde_3d_slice_plane_var = _StringVar("XY")
    dialog._pde_3d_fixed_axis_label = _Label()
    dialog._pde_3d_slice_coordinate_combo = _Combo()
    dialog._pde_3d_slice_index_var = _StringVar("0")
    dialog._pde_3d_slice_index_context_var = _StringVar("")

    dialog._refresh_pde_3d_slice_coordinates(render=False)
    central_index = len(dialog._result.z_grid) // 2
    assert dialog._pde_3d_fixed_axis_label.text == "Fixed z"
    assert dialog._pde_3d_slice_coordinate_combo.index == central_index
    assert float(dialog._pde_3d_slice_coordinate_combo.values[central_index]) == float(
        dialog._result.z_grid[central_index]
    )
    assert dialog._pde_3d_slice_index_var.get() == str(central_index)

    dialog._pde_3d_slice_plane_var.set("XZ")
    dialog._refresh_pde_3d_slice_coordinates(render=False)
    assert dialog._pde_3d_fixed_axis_label.text == "Fixed y"
    np.testing.assert_array_equal(
        [float(value) for value in dialog._pde_3d_slice_coordinate_combo.values],
        dialog._result.y_grid,
    )
    assert dialog._pde_3d_slice_index_var.get() == str(len(dialog._result.y_grid) // 2)


def test_pde_transform_varying_and_fixed_axes_match_extraction() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 20.0])
    field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert pde_fixed_axis_label("longitude", "longitude", "height") == "height"
    x_1d, values, varying, fixed, fixed_index = extract_pde_line_slice(
        "longitude", "longitude", "height", x, y, field, 18.0
    )
    np.testing.assert_array_equal(x_1d, x)
    np.testing.assert_array_equal(values, field[1, :])
    assert (varying, fixed, fixed_index) == ("longitude", "height", 1)

    assert pde_fixed_axis_label("height", "longitude", "height") == "longitude"
    y_1d, values, varying, fixed, fixed_index = extract_pde_line_slice(
        "height", "longitude", "height", x, y, field, 0.9
    )
    np.testing.assert_array_equal(y_1d, y)
    np.testing.assert_array_equal(values, field[:, 1])
    assert (varying, fixed, fixed_index) == ("height", "longitude", 1)
