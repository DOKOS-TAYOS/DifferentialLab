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


@dataclass
class PredefinedEquation:
    """Representation of a predefined ODE loaded from YAML.

    Attributes:
        key: Unique identifier (YAML key).
        name: Human-readable name.
        description: Multi-line description/formula.
        order: ODE order (1, 2, …).
        parameters: Mapping of param name to ``{default, description}``.
        expression: Python expression for the highest derivative.
        default_initial_conditions: Default y0 vector.
        default_domain: Default ``[x_min, x_max]``.
    """

    key: str
    name: str
    description: str
    order: int
    parameters: dict[str, dict[str, Any]]
    expression: str
    default_initial_conditions: list[float]
    default_domain: list[float] = field(default_factory=lambda: [0.0, 10.0])


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
        eq = PredefinedEquation(
            key=key,
            name=data.get("name", key),
            description=data.get("description", ""),
            order=int(data.get("order", 1)),
            parameters=data.get("parameters", {}),
            expression=data.get("expression", ""),
            default_initial_conditions=list(data.get("default_initial_conditions", [0.0])),
            default_domain=list(data.get("default_domain", [0.0, 10.0])),
        )
        equations[key] = eq
        logger.debug("Loaded predefined equation: %s", key)

    logger.info("Loaded %d predefined equations", len(equations))
    _cache = equations
    return equations

