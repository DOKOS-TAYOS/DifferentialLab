"""Regression tests for the stateful standard solve workflow."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any

from frontend.ui_dialogs.solve_session import (
    EQUATION_FAMILIES,
    EquationSelection,
    ParametersFormState,
    SolveSession,
    choose_filtered_key,
    filter_equation_keys,
    matching_categories,
)


def _catalog() -> dict[str, SimpleNamespace]:
    return {
        "harmonic": SimpleNamespace(
            name="Harmonic Oscillator",
            category="Mechanics",
            description="Undamped periodic motion",
            formula="y'' + omega^2 y = 0",
            expression="-omega**2 * f[0]",
            equation_type="ode",
        ),
        "damped": SimpleNamespace(
            name="Damped Oscillator",
            category="Mechanics",
            description="Motion with viscous damping",
            formula="y'' + gamma y' + omega^2 y = 0",
            expression="-gamma*f[1] - omega**2*f[0]",
            equation_type="ode",
        ),
        "growth": SimpleNamespace(
            name="Logistic Growth",
            category="Population",
            description="Saturating population model",
            formula="y' = r y (1-y/K)",
            expression="r*f*(1-f/K)",
            equation_type="ode",
        ),
        "poisson": SimpleNamespace(
            name="Poisson",
            category="Elliptic",
            description="Potential with a source term",
            formula="-laplacian(u) = s",
            expression="s",
            equation_type="pde",
        ),
    }


def _selection(name: str = "Harmonic Oscillator") -> EquationSelection:
    return EquationSelection(
        expression="-omega**2 * f[0]",
        function_name=None,
        order=2,
        parameters={"omega": 1.0},
        equation_name=name,
        default_y0=[1.0, 0.0],
        default_domain=[0.0, 10.0],
        display_formula="y'' + omega^2 y = 0",
    )


def _snapshot(x_min: str = "0") -> ParametersFormState:
    return ParametersFormState(
        x_min=x_min,
        x_max="10",
        n_points="500",
        initial_values=("1", "0"),
        initial_positions=("0", "0"),
        parameter_values={"omega": "2.5"},
        method="DOP853",
        statistics=("mean", "rms"),
        event_enabled=True,
        event_expression="f[0] - 0.25",
        event_terminal=True,
        event_direction="-1",
    )


def _contains_tk_object(value: Any) -> bool:
    module = type(value).__module__
    if module == "tkinter" or module.startswith("tkinter."):
        return True
    if is_dataclass(value):
        return any(_contains_tk_object(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, dict):
        return any(
            _contains_tk_object(key) or _contains_tk_object(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_tk_object(item) for item in value)
    return False


def test_session_retains_family_modes_built_in_choices_and_custom_drafts() -> None:
    session = SolveSession()
    ode = session.family_state("ode")
    ode.mode = "custom"
    ode.category = "Mechanics"
    ode.equation_key = "damped"
    ode.custom_draft.update(
        expression="-omega**2*f[0]",
        order="2",
        parameters="omega, forcing[3]",
    )
    pde = session.family_state("pde")
    pde.mode = "custom"
    pde.custom_draft.update(
        operator="-nabla^2 f",
        expression="sin(pi*x)*sin(pi*y)",
        parameters="alpha",
    )
    vector = session.family_state("vector_ode")
    vector.custom_draft.update(
        components="3",
        mode="bulk",
        component_orders=["1", "2", "1"],
        component_expressions=["f[1,0]", "-f[0,0]", "0"],
        bulk_expression="-f[i,0]",
    )

    session.current_family = "vector_ode"
    session.current_family = "pde"
    session.current_family = "ode"
    session.select_equation(_selection())
    session.save_configuration(_snapshot())
    session.latest_pipeline_inputs = {"x_min": 0.0, "selected_stats": {"mean"}}

    restored = session.family_state()
    assert restored.mode == "custom"
    assert (restored.category, restored.equation_key) == ("Mechanics", "damped")
    assert restored.custom_draft["expression"] == "-omega**2*f[0]"
    assert session.family_state("pde").custom_draft["operator"] == "-nabla^2 f"
    assert session.family_state("vector_ode").custom_draft["bulk_expression"] == "-f[i,0]"
    assert set(session.family_states) == set(EQUATION_FAMILIES)
    assert not _contains_tk_object(session)


def test_search_matches_name_description_category_and_is_case_insensitive() -> None:
    catalog = _catalog()

    assert filter_equation_keys(catalog, "ode", "HARMONIC") == ["harmonic"]
    assert filter_equation_keys(catalog, "ode", "viscous") == ["damped"]
    assert filter_equation_keys(catalog, "ode", "population") == ["growth"]
    assert matching_categories(catalog, "ode", "motion") == ["Mechanics"]


def test_search_clear_restores_family_catalog_and_no_results_is_safe() -> None:
    catalog = _catalog()
    all_ode_keys = filter_equation_keys(catalog, "ode")

    assert filter_equation_keys(catalog, "ode", "oscillator") == ["harmonic", "damped"]
    assert filter_equation_keys(catalog, "ode", "no such equation") == []
    assert matching_categories(catalog, "ode", "no such equation") == []
    assert filter_equation_keys(catalog, "ode", "") == all_ode_keys
    assert "poisson" not in all_ode_keys


def test_search_result_can_retain_a_still_valid_selection() -> None:
    catalog = _catalog()
    selected = "damped"
    filtered = filter_equation_keys(catalog, "ode", "oscillator", "Mechanics")

    assert choose_filtered_key(filtered, selected) == selected
    assert choose_filtered_key(filtered, "growth") == "harmonic"
    assert choose_filtered_key([], selected) is None


def test_equation_configuration_back_and_continue_state_transition() -> None:
    session = SolveSession()
    selection = _selection()
    edited = _snapshot(x_min="-3.25")

    session.select_equation(selection)
    session.save_configuration(edited)
    session.current_family = "ode"
    restored = session.configuration_for(selection)

    assert session.selection is selection
    assert restored == edited
    assert restored is not None
    assert restored.x_min == "-3.25"
    assert restored.parameter_values == {"omega": "2.5"}


def test_configuration_snapshot_is_not_applied_to_changed_equation() -> None:
    session = SolveSession()
    original = _selection()
    changed = _selection("Different equation")
    session.select_equation(original)
    session.save_configuration(_snapshot())

    assert session.configuration_for(original) == _snapshot()
    assert session.configuration_for(changed) is None
