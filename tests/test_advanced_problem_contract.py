"""Parameterized contract smoke coverage for registered Advanced Problems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import pytest

from complex_problems.aerodynamics_2d.solver import solve_aerodynamics_2d
from complex_problems.antenna_radiation.solver import solve_antenna_radiation
from complex_problems.coupled_oscillators.solver import solve_coupled_oscillators
from complex_problems.membrane_2d.model import build_initial_displacement
from complex_problems.membrane_2d.solver import solve_membrane_2d
from complex_problems.nonlinear_waves.solver import solve_nonlinear_waves
from complex_problems.pipe_flow.solver import solve_pipe_flow
from complex_problems.problem_registry import (
    _REGISTRATIONS,
    ProblemRegistration,
    ProblemRegistry,
)
from complex_problems.schrodinger_td.solver import solve_schrodinger_td


@dataclass(frozen=True, slots=True)
class _SmokeCase:
    """A low-cost representative solve for one registered plugin."""

    problem_id: str
    solve: Callable[[], object]


def _solve_membrane() -> object:
    """Return a small stable linear membrane solve."""
    return solve_membrane_2d(
        u0=build_initial_displacement(
            nx=8,
            ny=8,
            shape="mode",
            amplitude=0.1,
            sigma=0.2,
            mode_x=1,
            mode_y=1,
            boundary="fixed",
        ),
        t_min=0.0,
        t_max=0.04,
        dt=0.01,
        mass=1.0,
        k_linear=1.0,
        boundary="fixed",
        integrator="verlet",
    )


_SMOKE_CASES: tuple[_SmokeCase, ...] = (
    _SmokeCase(
        "coupled_oscillators",
        lambda: solve_coupled_oscillators(
            n_oscillators=3,
            masses=1.0,
            k_coupling=1.0,
            t_max=0.1,
            n_points=8,
        ),
    ),
    _SmokeCase("membrane_2d", _solve_membrane),
    _SmokeCase(
        "nonlinear_waves",
        lambda: solve_nonlinear_waves(
            model_type="nlse",
            x_min=-5.0,
            x_max=5.0,
            nx=32,
            t_min=0.0,
            t_max=0.02,
            dt=0.01,
            profile="sech",
            amplitude=0.2,
            sigma=1.0,
            center=0.0,
            beta2=1.0,
            gamma=1.0,
            initial_phase_k=0.0,
        ),
    ),
    _SmokeCase(
        "schrodinger_td",
        lambda: solve_schrodinger_td(
            dimension=1,
            x_min=-5.0,
            x_max=5.0,
            nx=32,
            t_min=0.0,
            t_max=0.02,
            dt=0.01,
            hbar=1.0,
            mass=1.0,
            boundary="periodic",
            potential_type="free",
            packet_type="gaussian",
            sigma=0.8,
            x0=0.0,
            k0x=1.0,
        ),
    ),
    _SmokeCase(
        "antenna_radiation",
        lambda: solve_antenna_radiation(n_theta=16, n_phi=20),
    ),
    _SmokeCase(
        "aerodynamics_2d",
        lambda: solve_aerodynamics_2d(
            approximation="stokes",
            nx=24,
            ny=16,
            lx=3.0,
            ly=2.0,
            t_max=0.01,
            dt=0.002,
            sample_every=2,
            rho=1.0,
            nu=0.03,
            u_inf=0.8,
            penalization=0.01,
            obstacle_shape="ellipse",
            obstacle_size_x=0.3,
            obstacle_size_y=0.2,
        ),
    ),
    _SmokeCase(
        "pipe_flow",
        lambda: solve_pipe_flow(
            model_type="steady",
            length=10.0,
            nx=24,
            profile="constant",
            d0=0.05,
            d_in=0.05,
            d_out=0.05,
            rho=1000.0,
            mu=1.0e-3,
            roughness=1.0e-5,
            friction_model="auto",
            p_in=2.0e5,
            p_out=1.9e5,
        ),
    ),
)


def _assert_finite_arrays(value: object) -> None:
    """Assert every numerical array in a structured solver result is finite."""
    if isinstance(value, np.ndarray):
        assert np.all(np.isfinite(value))
    elif is_dataclass(value) and not isinstance(value, type):
        for result_field in fields(value):
            _assert_finite_arrays(getattr(value, result_field.name))
    elif isinstance(value, dict):
        for nested_value in value.values():
            _assert_finite_arrays(nested_value)
    elif isinstance(value, (tuple, list)):
        for nested_value in value:
            _assert_finite_arrays(nested_value)
    elif isinstance(value, (float, np.floating, int, np.integer, complex, np.complexfloating)):
        assert bool(np.isfinite(value))


@pytest.mark.parametrize("registration", _REGISTRATIONS, ids=lambda item: item.module_path)
def test_registered_plugin_contract_is_lazy_and_valid(registration: ProblemRegistration) -> None:
    """Every registration lazily imports a plugin with a valid dialog entry point."""
    registry = ProblemRegistry((registration,))

    descriptors = registry.get_descriptors()

    assert len(descriptors) == 1
    descriptor = next(iter(descriptors.values()))
    assert descriptor.id.strip()
    assert descriptor.name.strip()
    assert descriptor.description.strip()
    assert callable(registry._plugins[descriptor.id].open_dialog)  # type: ignore[index, union-attr]


def test_registered_plugin_descriptors_are_unique() -> None:
    """All seven registered plugin descriptors have distinct identifiers."""
    descriptors = ProblemRegistry(_REGISTRATIONS).get_descriptors()

    assert len(descriptors) == len(_REGISTRATIONS) == 7
    assert set(descriptors) == {case.problem_id for case in _SMOKE_CASES}


@pytest.mark.parametrize("case", _SMOKE_CASES, ids=lambda item: item.problem_id)
def test_registered_plugin_representative_solve_returns_finite_structured_output(
    case: _SmokeCase,
) -> None:
    """Each plugin's low-cost representative solve produces finite structured data."""
    result = case.solve()

    assert is_dataclass(result)
    _assert_finite_arrays(result)
