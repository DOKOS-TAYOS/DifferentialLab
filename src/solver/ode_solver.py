"""Core ODE solving wrappers around SciPy integrators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from config import get_env_from_schema
from utils import SolverFailedError, get_logger

logger = get_logger(__name__)


@dataclass
class ODESolution:
    """Container for ODE solution data.

    Attributes:
        x: Independent variable values.
        y: Solution array — shape ``(n_vars, n_points)``.
        success: Whether the solver converged.
        message: Solver status message.
        method_used: Name of the integration method.
        n_eval: Number of function evaluations (if available).
        raw: The original ``OdeResult`` from SciPy.
    """

    x: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    method_used: str
    n_eval: int = 0
    raw: Any = field(default=None, repr=False)


def solve_ode(
    ode_func: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: list[float],
    method: str | None = None,
    t_eval: np.ndarray | None = None,
    max_step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> ODESolution:
    """Solve an initial-value ODE problem using ``scipy.integrate.solve_ivp``.

    Args:
        ode_func: Right-hand side of the ODE system ``dy/dx = f(x, y)``.
        t_span: Integration interval ``(x_min, x_max)``.
        y0: Initial conditions vector.
        method: Integration method name. Falls back to env default.
        t_eval: Times at which to store the solution. If ``None``, a
            uniform grid is generated from env settings.
        max_step: Maximum allowed step size (0 → ``np.inf``).
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        An :class:`ODESolution` with the results.

    Raises:
        SolverFailedError: If the solver reports failure.
    """
    if method is None:
        method = get_env_from_schema("SOLVER_DEFAULT_METHOD")
    if max_step is None:
        max_step = get_env_from_schema("SOLVER_MAX_STEP")
    if rtol is None:
        rtol = get_env_from_schema("SOLVER_RTOL")
    if atol is None:
        atol = get_env_from_schema("SOLVER_ATOL")

    if t_eval is None:
        n_points: int = get_env_from_schema("SOLVER_NUM_POINTS")
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

    effective_max_step = np.inf if max_step <= 0 else max_step

    from scipy.integrate import solve_ivp

    logger.info(
        "Solving IVP: method=%s, span=%s, y0=%s, rtol=%s, atol=%s",
        method, t_span, y0, rtol, atol,
    )

    sol = solve_ivp(
        fun=ode_func,
        t_span=t_span,
        y0=y0,
        method=method,
        t_eval=t_eval,
        max_step=effective_max_step,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    n_eval = getattr(sol, "nfev", 0)

    result = ODESolution(
        x=sol.t,
        y=sol.y,
        success=sol.success,
        message=sol.message,
        method_used=method,
        n_eval=n_eval,
        raw=sol,
    )

    if not sol.success:
        logger.error("Solver failed: %s", sol.message)
        raise SolverFailedError(f"Solver failed ({method}): {sol.message}")

    logger.info("Solver succeeded: %d points, %d evaluations", len(sol.t), n_eval)
    return result


def solve_multipoint(
    ode_func: Callable[[float, np.ndarray], np.ndarray],
    conditions: list[tuple[int, float, float]],
    order: int,
    x_min: float,
    x_max: float,
    method: str | None = None,
    t_eval: np.ndarray | None = None,
    max_step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> ODESolution:
    """Solve an ODE with initial conditions specified at possibly different x points.

    Uses a shooting method: the full state at ``x_min`` is found via root-finding
    so that all given conditions ``y^(k)(x_i) = a_i`` are satisfied.

    Args:
        ode_func: Right-hand side of the ODE system ``dy/dx = f(x, y)``.
        conditions: List of ``(k, x_i, a_i)`` meaning ``y^(k)(x_i) = a_i``.
        order: ODE order (equals number of conditions).
        x_min: Domain start.
        x_max: Domain end.
        method: Integration method name.
        t_eval: Points at which to store the final solution.
        max_step: Maximum step size.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        An :class:`ODESolution` with the results.

    Raises:
        SolverFailedError: If the solver or shooting method fails.
    """
    if method is None:
        method = get_env_from_schema("SOLVER_DEFAULT_METHOD")
    if max_step is None:
        max_step = get_env_from_schema("SOLVER_MAX_STEP")
    if rtol is None:
        rtol = get_env_from_schema("SOLVER_RTOL")
    if atol is None:
        atol = get_env_from_schema("SOLVER_ATOL")

    effective_max_step = np.inf if max_step <= 0 else max_step

    if t_eval is None:
        n_points: int = get_env_from_schema("SOLVER_NUM_POINTS")
        t_eval = np.linspace(x_min, x_max, n_points)

    all_at_start = all(abs(xi - x_min) < 1e-12 for (_, xi, _) in conditions)
    if all_at_start:
        y0 = [a for (_, _, a) in sorted(conditions, key=lambda c: c[0])]
        return solve_ode(
            ode_func, (x_min, x_max), y0,
            method=method, t_eval=t_eval,
            max_step=max_step, rtol=rtol, atol=atol,
        )

    y0_guess = np.zeros(order)
    for k, _xi, ai in conditions:
        y0_guess[k] = ai

    x_max_needed = max(max(xi for (_, xi, _) in conditions), x_max)
    n_fine = max(2000, len(t_eval) * 2)
    t_eval_fine = np.linspace(x_min, x_max_needed, n_fine)

    from scipy.integrate import solve_ivp

    def _residuals(y0: np.ndarray) -> np.ndarray:
        sol = solve_ivp(
            ode_func, (x_min, x_max_needed), y0.tolist(),
            method=method, t_eval=t_eval_fine,
            max_step=effective_max_step, rtol=rtol, atol=atol,
            dense_output=True,
        )
        if not sol.success:
            return np.full(len(conditions), 1e10)
        return np.array([
            np.interp(xi, sol.t, sol.y[k]) - ai
            for (k, xi, ai) in conditions
        ])

    from scipy.optimize import fsolve

    y0_opt, _, ier, mesg = fsolve(_residuals, y0_guess, full_output=True)

    if ier != 1:
        raise SolverFailedError(f"Shooting method did not converge: {mesg}")

    logger.info("Shooting method converged; y0_opt=%s", y0_opt.tolist())

    return solve_ode(
        ode_func, (x_min, x_max), y0_opt.tolist(),
        method=method, t_eval=t_eval,
        max_step=max_step, rtol=rtol, atol=atol,
    )
