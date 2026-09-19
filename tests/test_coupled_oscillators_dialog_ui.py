"""Tests for the coupled oscillators configuration dialog."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from complex_problems.coupled_oscillators import ui as coupled_ui


class _FakeVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _FakeListbox:
    def __init__(self, values: tuple[str, ...], selected: tuple[int, ...]) -> None:
        self._values = values
        self._selected = selected

    def curselection(self) -> tuple[int, ...]:
        return self._selected

    def get(self, index: int) -> str:
        return self._values[index]


class _FakeWindow:
    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


class _FakeParent:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, Any]] = []

    def after(self, delay_ms: int, callback: Any) -> None:
        self.after_calls.append((delay_ms, callback))


def _build_dialog_stub() -> coupled_ui.CoupledOscillatorsDialog:
    dialog = coupled_ui.CoupledOscillatorsDialog.__new__(coupled_ui.CoupledOscillatorsDialog)
    dialog.parent = _FakeParent()
    dialog.win = _FakeWindow()
    dialog._n_var = _FakeVar("4")
    dialog._mass_entry_var = _FakeVar("1.0, 1.1, 1.2, 1.3")
    dialog._k_entry_var = _FakeVar("2.0")
    dialog._boundary_var = _FakeVar("Periodic")
    dialog._t_min_var = _FakeVar("0.0")
    dialog._t_max_var = _FakeVar("10.0")
    dialog._n_points_var = _FakeVar("128")
    dialog._method_var = _FakeVar("RK45")
    dialog._ic_space_var = _FakeVar("Oscillators")
    dialog._ic_pos_var = _FakeVar("1.0, 0.5, 0.0, -0.5")
    dialog._ic_vel_var = _FakeVar("0.0, 0.1, 0.2, 0.3")
    dialog._k_2nn_var = _FakeVar("3.5")
    dialog._k_3nn_var = _FakeVar("4.5")
    dialog._k_4nn_var = _FakeVar("5.5")
    dialog._fput_alpha_var = _FakeVar("0.25")
    dialog._nonlinear_coeff_var = _FakeVar("80")
    dialog._nonlinear_quartic_var = _FakeVar("150")
    dialog._nonlinear_quintic_var = _FakeVar("5")
    dialog._external_amp_var = _FakeVar("50")
    dialog._external_freq_var = _FakeVar("1.75")
    dialog._coupling_listbox = _FakeListbox(
        (
            "2nd neighbor",
            "3rd neighbor",
            "4th neighbor",
            "FPUT-\u03b1",
            "Nonlinear (cubic)",
            "Nonlinear (quartic)",
            "Nonlinear (quintic)",
            "External force",
        ),
        (0, 3, 4, 7),
    )
    return dialog


def test_coupled_oscillators_collect_inputs_builds_solver_kwargs() -> None:
    dialog = _build_dialog_stub()

    params = dialog._collect_inputs()

    assert params["n_oscillators"] == 4
    assert params["masses"] == [1.0, 1.1, 1.2, 1.3]
    assert params["k_coupling"] == 2.0
    assert params["boundary"] == "periodic"
    assert set(params["coupling_types"]) == {
        "linear",
        "nonlinear_fput_alpha",
        "nonlinear",
        "external_force",
    }
    assert params["k_2nn"] == 3.5
    assert params["k_3nn"] == 0.0
    assert params["k_4nn"] == 0.0
    assert params["external_frequency"] == 1.75
    assert params["t_min"] == 0.0
    assert params["t_max"] == 10.0
    assert params["n_points"] == 128
    assert params["method"] == "RK45"
    assert params["y0"] == [1.0, 0.5, 0.0, -0.5, 0.0, 0.1, 0.2, 0.3]


def test_coupled_oscillators_collect_inputs_accepts_function_specs() -> None:
    dialog = _build_dialog_stub()
    dialog._mass_entry_var = _FakeVar("1 + 0.5*i")
    dialog._k_entry_var = _FakeVar("2 + i")

    params = dialog._collect_inputs()

    masses = params["masses"]
    k_coupling = params["k_coupling"]
    assert callable(masses)
    assert callable(k_coupling)
    assert masses(2) == 2.0
    assert k_coupling(3) == 5.0


def test_coupled_oscillators_on_solve_uses_shared_dialog_runner() -> None:
    dialog = _build_dialog_stub()
    dialog._collect_inputs = MagicMock(return_value={"n_oscillators": 4})  # type: ignore[method-assign]

    with patch.object(coupled_ui, "run_solver_dialog", create=True) as run_solver_dialog:
        dialog._on_solve()

    run_solver_dialog.assert_called_once()
    kwargs = run_solver_dialog.call_args.kwargs
    assert kwargs["parent"] is dialog.parent
    assert kwargs["window"] is dialog.win
    assert kwargs["collect_inputs"] is dialog._collect_inputs
    assert kwargs["solver"] is coupled_ui.solve_coupled_oscillators
    assert kwargs["message"] == "Solving coupled oscillators..."
    assert kwargs["result_parent"] is dialog.parent
    assert kwargs["result_dialog_factory"] is coupled_ui.CoupledOscillatorsResultDialog
    assert not hasattr(coupled_ui, "threading")
