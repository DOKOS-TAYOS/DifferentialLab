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
    """Representation of a predefined equation (ODE or difference) loaded from YAML.

    formula is always required for display. Either expression or function_name
    must be set for execution. If function_name is set, the equation is resolved by
    importing the function from config.equations; otherwise expression is used.

    Attributes:
        key: Unique identifier (YAML key).
        name: Human-readable name.
        formula: Compact human-readable equation string (e.g. ``"y'' + ω²y = 0"``).
        description: Multi-line description with formula and context.
        order: Equation order (1, 2, …).
        parameters: Mapping of param name to ``{default, description}``.
        expression: Python expression for execution (optional if function_name set).
        function_name: Name of function in config.equations to import (optional).
        default_initial_conditions: Default y0 vector.
        default_domain: Default ``[x_min, x_max]`` for ODE or ``[n_min, n_max]`` for difference.
        equation_type: ``"ode"`` (differential) or ``"difference"``.
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
    equation_type: EquationType = "ode"


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
        if not expression and not function_name:
            logger.warning(
                "Equation '%s' has neither expression nor function_name; skipping", key
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
            default_initial_conditions=list(data.get("default_initial_conditions", [0.0])),
            default_domain=list(data.get("default_domain", [0.0, 10.0])),
            equation_type=str(data.get("equation_type", "ode")),
        )
        equations[key] = eq
        logger.debug("Loaded predefined equation: %s", key)

    logger.info("Loaded %d predefined equations", len(equations))
    _cache = equations
    return equations

