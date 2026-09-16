"""Core ODE and boundary-value solving wrappers around SciPy integrators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from scipy.integrate import solve_bvp as scipy_solve_bvp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from config import DEFAULT_SOLVER_METHOD, get_env_from_schema
from utils import SolverFailedError, ValidationError, get_logger

logger = get_logger(__name__)

ODEFunction = Callable[[float, np.ndarray], np.ndarray]
ODEEvent = Callable[[float, np.ndarray], float]
ODEJacobian = Callable[[float, np.ndarray], np.ndarray]
BVPBoundaryFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
BVPInitialGuess = np.ndarray | Callable[[np.ndarray], np.ndarray]
MultipointStrategy = Literal["auto", "shooting", "bvp"]


@dataclass(frozen=True, slots=True)
class IVPOptions:
    """Optional :func:`scipy.integrate.solve_ivp` capabilities.

    ``max_step``, tolerances, the method, and output points remain explicit
    arguments of :func:`solve_ode` for backwards compatibility. Less commonly
    used capabilities live here so the public function does not keep growing.
    """

    events: tuple[ODEEvent, ...] = ()
    jac: ODEJacobian | None = None
    vectorized: bool = False
    first_step: float | None = None


@dataclass(frozen=True, slots=True)
class BVPOptions:
    """Controls for the bounded deterministic :func:`solve_bvp` wrapper."""

    initial_mesh_points: int = 25
    tol: float = 1e-3
    max_nodes: int = 1000
    bc_tol: float | None = None
    verbose: Literal[0, 1, 2] = 0


@dataclass
class ODESolution:
    """Container for initial-value and multipoint ODE solution data.

    ``n_eval`` is retained as the backwards-compatible alias used by the
    existing pipeline. The newer fields mirror SciPy's structured diagnostics.
    """

    x: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    method_used: str
    n_eval: int = 0
    nfev: int = 0
    njev: int | None = None
    nlu: int | None = None
    status: int = 0
    t_events: tuple[np.ndarray, ...] = field(default_factory=tuple)
    y_events: tuple[np.ndarray, ...] = field(default_factory=tuple)
    raw: Any = field(default=None, repr=False)


@dataclass
class BVPSolution:
    """Typed result returned by :func:`solve_bvp`."""

    x: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    status: int
    niter: int
    rms_residuals: np.ndarray
    raw: Any = field(default=None, repr=False)


def _resolve_solver_params(
    method: str | None,
    max_step: float | None,
    rtol: float | None,
    atol: float | None,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None,
) -> tuple[str, float, float, float, np.ndarray]:
    """Resolve IVP parameters using environment defaults only for ``None``."""
    resolved_method = DEFAULT_SOLVER_METHOD if method is None else method
    resolved_max_step = get_env_from_schema("SOLVER_MAX_STEP") if max_step is None else max_step
    resolved_rtol = get_env_from_schema("SOLVER_RTOL") if rtol is None else rtol
    resolved_atol = get_env_from_schema("SOLVER_ATOL") if atol is None else atol

    max_step_value = cast(float, resolved_max_step)
    rtol_value = cast(float, resolved_rtol)
    atol_value = cast(float, resolved_atol)
    effective_max_step = np.inf if max_step_value <= 0 else max_step_value

    if t_eval is None:
        n_points: int = get_env_from_schema("SOLVER_NUM_POINTS")
        resolved_t_eval = np.linspace(t_span[0], t_span[1], n_points)
    else:
        resolved_t_eval = np.asarray(t_eval, dtype=float)

    return resolved_method, effective_max_step, rtol_value, atol_value, resolved_t_eval


def _validate_ivp_options(options: IVPOptions) -> None:
    """Validate optional solve_ivp settings before dispatch."""
    if options.first_step is not None and (
        not np.isfinite(options.first_step) or options.first_step <= 0
    ):
        raise ValidationError("first_step must be a finite positive number")
    if any(not callable(event) for event in options.events):
        raise ValidationError("Every IVP event must be callable")
    if options.jac is not None and not callable(options.jac):
        raise ValidationError("The IVP Jacobian must be callable")


def _optional_counter(result: Any, name: str) -> int | None:
    """Read an optional integer counter from a SciPy result."""
    value = getattr(result, name, None)
    return None if value is None else int(value)


def _event_arrays(result: Any, name: str) -> tuple[np.ndarray, ...]:
    """Normalize SciPy event arrays into an immutable tuple."""
    values = getattr(result, name, None)
    if values is None:
        return ()
    return tuple(np.asarray(value, dtype=float) for value in values)


def solve_ode(
    ode_func: ODEFunction,
    t_span: tuple[float, float],
    y0: list[float],
    method: str | None = None,
    t_eval: np.ndarray | None = None,
    max_step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    options: IVPOptions | None = None,
) -> ODESolution:
    """Solve an initial-value ODE problem using ``scipy.integrate.solve_ivp``.

    ``max_step=0`` is the documented infinity sentinel, including when callers
    provide an explicit ``t_eval``. Event functions may define SciPy's
    ``terminal`` and ``direction`` attributes.

    Raises:
        ValidationError: If an optional IVP setting is invalid.
        SolverFailedError: If the solver reports failure.
    """
    method, effective_max_step, rtol, atol, t_eval = _resolve_solver_params(
        method,
        max_step,
        rtol,
        atol,
        t_span,
        t_eval,
    )
    resolved_options = options if options is not None else IVPOptions()
    _validate_ivp_options(resolved_options)

    logger.info(
        "Solving IVP: method=%s, span=%s, y0=%s, rtol=%s, atol=%s",
        method,
        t_span,
        y0,
        rtol,
        atol,
    )

    scipy_options: dict[str, Any] = {
        "events": list(resolved_options.events) if resolved_options.events else None,
        "vectorized": resolved_options.vectorized,
    }
    if resolved_options.jac is not None:
        scipy_options["jac"] = resolved_options.jac
    if resolved_options.first_step is not None:
        scipy_options["first_step"] = resolved_options.first_step

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
        **scipy_options,
    )

    nfev = int(getattr(sol, "nfev", 0))
    result = ODESolution(
        x=np.asarray(sol.t, dtype=float),
        y=np.asarray(sol.y, dtype=float),
        success=bool(sol.success),
        message=str(sol.message),
        method_used=method,
        n_eval=nfev,
        nfev=nfev,
        njev=_optional_counter(sol, "njev"),
        nlu=_optional_counter(sol, "nlu"),
        status=int(getattr(sol, "status", 0)),
        t_events=_event_arrays(sol, "t_events"),
        y_events=_event_arrays(sol, "y_events"),
        raw=sol,
    )

    if not sol.success:
        logger.error("Solver failed: %s", sol.message)
        raise SolverFailedError(f"Solver failed ({method}): {sol.message}")

    logger.info("Solver succeeded: %d points, %d evaluations", len(sol.t), nfev)
    return result


def _validate_bvp_options(options: BVPOptions) -> None:
    """Validate bounded BVP controls before allocating the initial mesh."""
    if not 2 <= options.initial_mesh_points <= 1000:
        raise ValidationError("initial_mesh_points must be between 2 and 1000")
    if not np.isfinite(options.tol) or options.tol <= 0:
        raise ValidationError("BVP tolerance must be a finite positive number")
    if options.max_nodes < options.initial_mesh_points:
        raise ValidationError("max_nodes must be at least initial_mesh_points")
    if options.bc_tol is not None and (not np.isfinite(options.bc_tol) or options.bc_tol <= 0):
        raise ValidationError("bc_tol must be a finite positive number")


def _prepare_bvp_guess(
    initial_guess: BVPInitialGuess,
    mesh: np.ndarray,
) -> np.ndarray:
    """Evaluate and validate a BVP initial guess on ``mesh``."""
    guess = initial_guess(mesh) if callable(initial_guess) else initial_guess
    guess_array = np.asarray(guess, dtype=float)
    if guess_array.ndim == 1:
        guess_array = np.repeat(guess_array[:, np.newaxis], mesh.size, axis=1)
    if guess_array.ndim != 2 or guess_array.shape[1] != mesh.size:
        raise ValidationError(
            "BVP initial_guess must have shape (state_size,) or (state_size, mesh_points)"
        )
    if guess_array.shape[0] < 1 or not np.all(np.isfinite(guess_array)):
        raise ValidationError("BVP initial_guess must contain finite state values")
    return guess_array


def solve_bvp(
    ode_func: ODEFunction,
    t_span: tuple[float, float],
    boundary_func: BVPBoundaryFunction,
    initial_guess: BVPInitialGuess,
    *,
    options: BVPOptions | None = None,
    mesh: np.ndarray | None = None,
) -> BVPSolution:
    """Solve a two-point boundary-value problem with SciPy.

    The public wrapper accepts the same pointwise RHS convention as
    :func:`solve_ode`. When no mesh is supplied, it creates a bounded,
    deterministic uniform mesh controlled by :class:`BVPOptions`.

    Raises:
        ValidationError: If the mesh, guess, or options are invalid.
        SolverFailedError: If SciPy reports BVP non-convergence.
    """
    resolved_options = options if options is not None else BVPOptions()
    _validate_bvp_options(resolved_options)
    x_min, x_max = t_span
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        raise ValidationError("BVP t_span must contain finite increasing endpoints")

    if mesh is None:
        resolved_mesh = np.linspace(x_min, x_max, resolved_options.initial_mesh_points)
    else:
        resolved_mesh = np.asarray(mesh, dtype=float)
        if (
            resolved_mesh.ndim != 1
            or resolved_mesh.size < 2
            or not np.all(np.isfinite(resolved_mesh))
            or np.any(np.diff(resolved_mesh) <= 0)
            or not np.isclose(resolved_mesh[0], x_min)
            or not np.isclose(resolved_mesh[-1], x_max)
        ):
            raise ValidationError("BVP mesh must be finite, increasing, and span t_span")

    guess = _prepare_bvp_guess(initial_guess, resolved_mesh)

    def vectorized_rhs(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        """Adapt the project's pointwise ODE convention to solve_bvp."""
        return np.column_stack(
            [
                np.asarray(ode_func(float(x_value), y_values[:, index]), dtype=float)
                for index, x_value in enumerate(x_values)
            ]
        )

    sol = scipy_solve_bvp(
        vectorized_rhs,
        boundary_func,
        resolved_mesh,
        guess,
        tol=resolved_options.tol,
        max_nodes=resolved_options.max_nodes,
        verbose=resolved_options.verbose,
        bc_tol=resolved_options.bc_tol,
    )
    result = BVPSolution(
        x=np.asarray(sol.x, dtype=float),
        y=np.asarray(sol.y, dtype=float),
        success=bool(sol.success),
        message=str(sol.message),
        status=int(sol.status),
        niter=int(sol.niter),
        rms_residuals=np.asarray(sol.rms_residuals, dtype=float),
        raw=sol,
    )
    if not result.success:
        raise SolverFailedError(
            f"BVP solver did not converge (status {result.status}): {result.message}"
        )
    return result


def _condition_endpoint(
    x_value: float,
    x_min: float,
    x_max: float,
) -> Literal["left", "right"] | None:
    """Classify a condition point as the left endpoint, right endpoint, or interior."""
    tolerance = 1e-10 * max(1.0, abs(x_min), abs(x_max))
    if abs(x_value - x_min) <= tolerance:
        return "left"
    if abs(x_value - x_max) <= tolerance:
        return "right"
    return None


def _validate_multipoint_conditions(
    conditions: list[tuple[int, float, float]],
    order: int,
) -> None:
    """Validate the common shooting/BVP multipoint contract."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValidationError("ODE order must be a positive integer")
    if len(conditions) != order:
        raise ValidationError(f"Expected exactly {order} conditions, got {len(conditions)}")
    for derivative, x_value, target in conditions:
        if isinstance(derivative, bool) or not isinstance(derivative, int):
            raise ValidationError("Condition derivative indexes must be integers")
        if not 0 <= derivative < order:
            raise ValidationError(
                f"Condition derivative index {derivative} is outside [0, {order})"
            )
        if not np.isfinite(x_value) or not np.isfinite(target):
            raise ValidationError("Multipoint conditions must contain finite values")


def _endpoint_condition_sides(
    conditions: list[tuple[int, float, float]],
    x_min: float,
    x_max: float,
) -> list[Literal["left", "right"]] | None:
    """Return endpoint sides when conditions form a valid solve_bvp boundary map."""
    sides: list[Literal["left", "right"]] = []
    used: set[tuple[Literal["left", "right"], int]] = set()
    for derivative, x_value, _target in conditions:
        side = _condition_endpoint(x_value, x_min, x_max)
        if side is None or (side, derivative) in used:
            return None
        used.add((side, derivative))
        sides.append(side)
    return sides


def _build_multipoint_bvp_guess(
    mesh: np.ndarray,
    conditions: list[tuple[int, float, float]],
    sides: list[Literal["left", "right"]],
    order: int,
) -> np.ndarray:
    """Build a deterministic endpoint-informed initial BVP state guess."""
    guess = np.zeros((order, mesh.size), dtype=float)
    endpoint_values: dict[int, dict[str, float]] = {}
    for (derivative, _x_value, target), side in zip(conditions, sides):
        endpoint_values.setdefault(derivative, {})[side] = target

    fraction = (mesh - mesh[0]) / (mesh[-1] - mesh[0])
    for derivative, values in endpoint_values.items():
        left = values.get("left")
        right = values.get("right")
        if left is not None and right is not None:
            guess[derivative] = left + fraction * (right - left)
        elif left is not None:
            guess[derivative].fill(left)
        elif right is not None:
            guess[derivative].fill(right)
    return guess


def _solve_multipoint_bvp(
    ode_func: ODEFunction,
    conditions: list[tuple[int, float, float]],
    sides: list[Literal["left", "right"]],
    order: int,
    x_min: float,
    x_max: float,
    t_eval: np.ndarray,
    options: BVPOptions | None,
) -> ODESolution:
    """Solve endpoint conditions through the dedicated BVP backend."""
    resolved_options = options if options is not None else BVPOptions()
    mesh = np.linspace(x_min, x_max, resolved_options.initial_mesh_points)
    guess = _build_multipoint_bvp_guess(mesh, conditions, sides, order)

    def boundary_residual(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """Return one residual for each endpoint condition."""
        residuals = np.empty(order, dtype=float)
        for index, ((derivative, _x_value, target), side) in enumerate(zip(conditions, sides)):
            state = ya if side == "left" else yb
            residuals[index] = state[derivative] - target
        return residuals

    bvp_result = solve_bvp(
        ode_func,
        (x_min, x_max),
        boundary_residual,
        guess,
        options=resolved_options,
        mesh=mesh,
    )
    evaluated = np.asarray(bvp_result.raw.sol(t_eval), dtype=float)
    return ODESolution(
        x=t_eval,
        y=evaluated,
        success=True,
        message=bvp_result.message,
        method_used="BVP",
        status=bvp_result.status,
        raw=bvp_result.raw,
    )


def _solve_multipoint_shooting(
    ode_func: ODEFunction,
    conditions: list[tuple[int, float, float]],
    order: int,
    x_min: float,
    x_max: float,
    method: str,
    t_eval: np.ndarray,
    effective_max_step: float,
    max_step: float | None,
    rtol: float,
    atol: float,
) -> ODESolution:
    """Run the legacy shooting implementation for true multipoint conditions."""
    y0_guess = np.zeros(order)
    for derivative, _x_value, target in conditions:
        y0_guess[derivative] = target

    x_max_needed = max(max(x_value for _, x_value, _ in conditions), x_max)
    n_fine = max(2000, len(t_eval) * 2)
    t_eval_fine = np.linspace(x_min, x_max_needed, n_fine)

    def residuals(y0: np.ndarray) -> np.ndarray:
        """Integrate one shooting candidate and evaluate all conditions."""
        sol = solve_ivp(
            ode_func,
            (x_min, x_max_needed),
            y0.tolist(),
            method=method,
            t_eval=t_eval_fine,
            max_step=effective_max_step,
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        if not sol.success:
            return np.full(len(conditions), 1e10)
        return np.array(
            [
                np.interp(x_value, sol.t, sol.y[derivative]) - target
                for derivative, x_value, target in conditions
            ]
        )

    y0_opt, _, ier, message = fsolve(residuals, y0_guess, full_output=True)
    if ier != 1:
        raise SolverFailedError(f"Shooting method did not converge: {message}")

    logger.info("Shooting method converged; y0_opt=%s", y0_opt.tolist())
    return solve_ode(
        ode_func,
        (x_min, x_max),
        y0_opt.tolist(),
        method=method,
        t_eval=t_eval,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )


def solve_multipoint(
    ode_func: ODEFunction,
    conditions: list[tuple[int, float, float]],
    order: int,
    x_min: float,
    x_max: float,
    method: str | None = None,
    t_eval: np.ndarray | None = None,
    max_step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    strategy: MultipointStrategy = "auto",
    bvp_options: BVPOptions | None = None,
) -> ODESolution:
    """Solve endpoint or true multipoint ODE conditions.

    ``auto`` uses SciPy's BVP backend for conventional conditions split across
    both domain endpoints, a direct IVP for conditions all at ``x_min``, and
    shooting for conditions containing an interior point. ``bvp`` requires a
    compatible endpoint-only condition set, while ``shooting`` always selects
    the legacy root-finding route (except the exact all-at-start IVP shortcut).
    """
    if strategy not in ("auto", "shooting", "bvp"):
        raise ValidationError("strategy must be 'auto', 'shooting', or 'bvp'")
    _validate_multipoint_conditions(conditions, order)
    method, effective_max_step, rtol, atol, t_eval = _resolve_solver_params(
        method,
        max_step,
        rtol,
        atol,
        (x_min, x_max),
        t_eval,
    )

    all_at_start = all(_condition_endpoint(xi, x_min, x_max) == "left" for _, xi, _ in conditions)
    if all_at_start and strategy != "bvp":
        y0_by_derivative = {derivative: target for derivative, _xi, target in conditions}
        if len(y0_by_derivative) != order:
            raise ValidationError("Initial-value conditions must specify each derivative once")
        y0 = [y0_by_derivative[index] for index in range(order)]
        return solve_ode(
            ode_func,
            (x_min, x_max),
            y0,
            method=method,
            t_eval=t_eval,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )

    sides = _endpoint_condition_sides(conditions, x_min, x_max)
    if strategy == "bvp" and sides is None:
        raise ValidationError(
            "strategy='bvp' requires unique conditions located only at x_min or x_max"
        )

    use_bvp = strategy == "bvp" or (
        strategy == "auto" and sides is not None and "left" in sides and "right" in sides
    )
    if use_bvp:
        assert sides is not None
        return _solve_multipoint_bvp(
            ode_func,
            conditions,
            sides,
            order,
            x_min,
            x_max,
            t_eval,
            bvp_options,
        )

    return _solve_multipoint_shooting(
        ode_func,
        conditions,
        order,
        x_min,
        x_max,
        method,
        t_eval,
        effective_max_step,
        max_step,
        rtol,
        atol,
    )
