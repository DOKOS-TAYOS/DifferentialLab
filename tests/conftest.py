"""Shared pytest fixtures for ode_solver tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_expression_order1() -> str:
    """Valid first-order ODE expression (y' = k*y)."""
    return "k * y[0]"


@pytest.fixture
def sample_expression_order2() -> str:
    """Valid second-order ODE expression (harmonic oscillator)."""
    return "-omega**2 * y[0]"


@pytest.fixture
def sample_y0_order1() -> list[float]:
    """Initial condition for order-1 ODE."""
    return [1.0]


@pytest.fixture
def sample_y0_order2() -> list[float]:
    """Initial conditions for order-2 ODE (y, y')."""
    return [1.0, 0.0]


@pytest.fixture
def sample_parameters() -> dict[str, float]:
    """Sample named parameters."""
    return {"k": 0.5, "omega": 1.0}


@pytest.fixture
def sample_domain() -> tuple[float, float]:
    """Sample integration domain (x_min, x_max)."""
    return (0.0, 10.0)


@pytest.fixture
def sample_t_eval(sample_domain: tuple[float, float]) -> np.ndarray:
    """Uniform grid over sample domain."""
    return np.linspace(sample_domain[0], sample_domain[1], 100)


@pytest.fixture
def equations_yaml_path() -> Path:
    """Path to the predefined equations YAML (in src/config)."""
    return Path(__file__).resolve().parent.parent / "src" / "config" / "equations.yaml"
