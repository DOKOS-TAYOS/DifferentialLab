"""Load and manage predefined ODE equations from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils import get_logger

logger = get_logger(__name__)

_EQUATIONS_PATH = Path(__file__).resolve().parent.parent / "config" / "equations.yaml"
_cache: dict[str, PredefinedEquation] | None = None


EquationType = str  # "ode" | "difference"


@dataclass
class PredefinedEquation:
    """Predefined equation (ODE, difference, PDE, or vector ODE) loaded from YAML.

    formula is always required for display. Either expression or function_name
    must be set for execution. If function_name is set, the equation is resolved by
    importing the function from config.equations; otherwise expression is used.
    For vector ODEs, use vector_expressions or function_name.

    Attributes:
        key: Unique identifier (YAML key).
        name: Human-readable name.
        formula: Compact human-readable equation string (e.g. ``"y'' + ω²y = 0"``).
        description: Multi-line description with formula and context.
        order: Equation order (1, 2, …) for ODE/difference. For vector: order per component.
        parameters: Mapping of param name to ``{default, description}``.
        expression: Python expression for execution (optional if function_name set).
        function_name: Name of function in config.equations to import (optional).
        vector_expressions: For vector ODEs, list of expressions (one per component).
        vector_components: Number of components [f_0, f_1, ...] for vector ODEs.
        default_initial_conditions: Default y0 vector.
        default_domain: Default ``[x_min, x_max]`` for ODE or ``[n_min, n_max]`` for difference.
            For PDE: ``[x_min, x_max, y_min, y_max, ...]`` per variable.
        equation_type: ``"ode"`` (differential), ``"difference"``, ``"pde"``, or ``"vector_ode"``.
        variables: Independent variable names, e.g. ``["x"]`` for 1D, ``["x","y"]`` for 2D.
            If absent or ``["x"]``, treated as 1D ODE.
        partial_derivatives: For PDEs, maps derivative keys (e.g. ``"f_xx"``, ``"f_xy"``)
            to expression strings. Only needed for PDE type.
    """

    key: str
    name: str
    formula: str
    description: str
    order: int
    parameters: dict[str, dict[str, Any]]
    expression: str | None
    function_name: str | None
    default_initial_conditions: list[float]
    default_domain: list[float] = field(default_factory=lambda: [0.0, 10.0])
    vector_expressions: list[str] | None = None
    vector_components: int = 1
    equation_type: EquationType = "ode"
    variables: list[str] = field(default_factory=lambda: ["x"])
    partial_derivatives: dict[str, str] | None = None


def load_predefined_equations() -> dict[str, PredefinedEquation]:
    """Load all predefined equations from the YAML file.

    Results are cached after the first successful load to avoid redundant
    disk I/O on repeated calls.

    Returns:
        Ordered dict mapping equation key to :class:`PredefinedEquation`.

    Raises:
        FileNotFoundError: If the YAML file is missing.
    """
    global _cache
    if _cache is not None:
        return _cache

    if not _EQUATIONS_PATH.exists():
        logger.error("Equations YAML not found: %s", _EQUATIONS_PATH)
        raise FileNotFoundError(f"Equations file not found: {_EQUATIONS_PATH}")

    with open(_EQUATIONS_PATH, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    equations: dict[str, PredefinedEquation] = {}
    for key, data in raw.items():
        formula: str = data.get("formula", "")
        if not formula:
            logger.warning("Equation '%s' has no formula (required for display); skipping", key)
            continue

        expression: str | None = data.get("expression")
        function_name: str | None = data.get("function_name")
        vector_expressions: list[str] | None = data.get("vector_expressions")
        vector_components: int = int(data.get("vector_components", 1))
        has_vector = vector_expressions is not None and len(vector_expressions) > 0
        if not expression and not function_name and not has_vector:
            logger.warning(
                "Equation '%s' has neither expression, function_name, nor vector_expressions; "
                "skipping",
                key,
            )
            continue

        eq = PredefinedEquation(
            key=key,
            name=data.get("name", key),
            formula=formula,
            description=data.get("description", ""),
            order=int(data.get("order", 1)),
            parameters=data.get("parameters", {}),
            expression=expression,
            function_name=function_name,
            vector_expressions=vector_expressions,
            vector_components=vector_components if has_vector else 1,
            default_initial_conditions=list(data.get("default_initial_conditions", [0.0])),
            default_domain=list(data.get("default_domain", [0.0, 10.0])),
            equation_type=str(data.get("equation_type", "ode")),
            variables=list(data.get("variables", ["x"])),
            partial_derivatives=dict(data.get("partial_derivatives", {})) or None,
        )
        equations[key] = eq
        logger.debug("Loaded predefined equation: %s", key)

    logger.info("Loaded %d predefined equations", len(equations))
    _cache = equations
    return equations


def is_multivariate(variables: list[str] | None) -> bool:
    """Return True if the equation has more than one independent variable.

    Args:
        variables: List of variable names (e.g. ["x"] or ["x","y"]).

    Returns:
        True if len(variables) > 1.
    """
    if not variables:
        return False
    return len(variables) > 1


def is_vector_ode(eq: PredefinedEquation) -> bool:
    """Return True if the equation is a vector ODE ([f_0, f_1, ...]).

    Args:
        eq: Predefined equation.

    Returns:
        True if vector_expressions is set or equation_type is vector_ode.
    """
    if getattr(eq, "equation_type", "ode") == "vector_ode":
        return True
    return (
        getattr(eq, "vector_expressions", None) is not None
        and len(getattr(eq, "vector_expressions", [])) > 0
    )

