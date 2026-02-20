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
