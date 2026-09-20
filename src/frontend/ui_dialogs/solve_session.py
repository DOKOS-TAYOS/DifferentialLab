"""Pure state for one standard Equation -> Configuration solve workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

EQUATION_FAMILIES = (
    "ode",
    "difference",
    "vector_ode",
    "pde",
    "pde_3d",
    "vector_pde",
)


@dataclass(slots=True)
class EquationFamilyState:
    """Equation-step state retained independently for one equation family."""

    mode: str = "built_in"
    category: str | None = None
    equation_key: str | None = None
    search_query: str = ""
    custom_draft: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EquationSelection:
    """Pure frontend specification used to construct ``ParametersDialog``."""

    expression: str | None
    function_name: str | None
    order: int
    parameters: dict[str, float | list[float]]
    equation_name: str
    default_y0: list[float]
    default_domain: list[float]
    parameters_schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    display_formula: str | None = None
    equation_type: str = "ode"
    variables: list[str] = field(default_factory=lambda: ["x"])
    vector_expressions: list[str] | None = None
    vector_components: int = 1
    pde_operator: str = "neg_laplacian"
    component_orders: tuple[int, ...] | None = None
    default_boundary_conditions: dict[str, dict[str, str]] = field(default_factory=dict)

    def signature(self) -> str:
        """Return a stable identity for compatible configuration snapshots."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def parameters_kwargs(self) -> dict[str, Any]:
        """Return constructor arguments for ``ParametersDialog``."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParametersFormState:
    """Raw, validation-free Configuration form values containing no Tk objects."""

    x_min: str
    x_max: str
    n_points: str
    y_min: str | None = None
    y_max: str | None = None
    n_points_y: str | None = None
    z_min: str | None = None
    z_max: str | None = None
    n_points_z: str | None = None
    initial_values: tuple[str, ...] = ()
    initial_positions: tuple[str, ...] = ()
    parameter_values: dict[str, str] = field(default_factory=dict)
    method: str = ""
    statistics: tuple[str, ...] = ()
    event_expression: str | None = None
    event_terminal: bool = False
    event_direction: str | None = None
    domain_shape: str | None = None
    boundary_expressions: tuple[str, ...] = ()
    boundary_types: tuple[str, ...] = ()
    mask_expression: str | None = None
    contour_boundary_expression: str | None = None
    contour_boundary_type: str | None = None


@dataclass(slots=True)
class SolveSession:
    """State owner for one standard solve workflow."""

    current_family: str = "ode"
    family_states: dict[str, EquationFamilyState] = field(
        default_factory=lambda: {family: EquationFamilyState() for family in EQUATION_FAMILIES}
    )
    selection: EquationSelection | None = None
    configuration_snapshots: dict[str, ParametersFormState] = field(default_factory=dict)
    latest_pipeline_inputs: dict[str, Any] | None = None

    def family_state(self, family: str | None = None) -> EquationFamilyState:
        """Return retained state for *family*, creating it for forward compatibility."""
        key = family or self.current_family
        return self.family_states.setdefault(key, EquationFamilyState())

    def select_equation(self, selection: EquationSelection) -> None:
        """Store the equation that will be configured."""
        self.selection = selection

    def configuration_for(
        self, selection: EquationSelection | None = None
    ) -> ParametersFormState | None:
        """Return a compatible saved Configuration snapshot, if one exists."""
        selected = selection or self.selection
        if selected is None:
            return None
        return self.configuration_snapshots.get(selected.signature())

    def save_configuration(
        self,
        snapshot: ParametersFormState,
        selection: EquationSelection | None = None,
    ) -> None:
        """Associate a raw Configuration snapshot with its exact equation selection."""
        selected = selection or self.selection
        if selected is None:
            raise ValueError("An equation selection is required before saving configuration state.")
        self.configuration_snapshots[selected.signature()] = snapshot


def filter_equation_keys(
    equations: dict[str, Any],
    equation_type: str,
    query: str = "",
    category: str | None = None,
) -> list[str]:
    """Filter catalog keys by family, optional category, and case-insensitive text."""
    normalized_query = query.strip().casefold()
    matches: list[str] = []
    for key, equation in equations.items():
        if getattr(equation, "equation_type", "ode") != equation_type:
            continue
        if category is not None and equation.category != category:
            continue
        haystack = " ".join(
            (
                str(equation.name),
                str(equation.category),
                str(equation.description),
                str(getattr(equation, "formula", "")),
                str(getattr(equation, "expression", "")),
            )
        ).casefold()
        if normalized_query and normalized_query not in haystack:
            continue
        matches.append(key)
    return matches


def matching_categories(
    equations: dict[str, Any], equation_type: str, query: str = ""
) -> list[str]:
    """Return only categories containing catalog matches for the current filter."""
    return sorted(
        {equations[key].category for key in filter_equation_keys(equations, equation_type, query)}
    )


def choose_filtered_key(keys: list[str], preferred_key: str | None) -> str | None:
    """Keep a valid catalog selection, otherwise choose the first available result."""
    if preferred_key in keys:
        return preferred_key
    return keys[0] if keys else None
