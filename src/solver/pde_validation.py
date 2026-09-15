"""Validation helpers for scalar two-dimensional PDE solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from solver.pde_types import PDECoefficients
from utils import SolverFailedError

_AFFINITY_PROBE = (0.37, -0.61, 1.19, -0.83, 0.47, 1.31)
_AFFINITY_TOLERANCE = 1.0e-9


def finite_scalar(value: object, name: str) -> float:
    """Convert a scalar-like value to a finite real float."""
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise SolverFailedError(f"{name} must be a finite real scalar")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise SolverFailedError(f"{name} must be finite")
    return result


def validate_grid_inputs(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> tuple[float, float, float, float, int, int]:
    """Validate and normalize rectangular grid inputs."""
    x_start, x_end, y_start, y_end = tuple(
        finite_scalar(value, name)
        for value, name in (
            (x_min, "x_min"),
            (x_max, "x_max"),
            (y_min, "y_min"),
            (y_max, "y_max"),
        )
    )
    if x_start >= x_end:
        raise SolverFailedError("x_min must be strictly less than x_max")
    if y_start >= y_end:
        raise SolverFailedError("y_min must be strictly less than y_max")
    for value, name in ((nx, "nx"), (ny, "ny")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise SolverFailedError(f"{name} must be an integer")
        if int(value) < 3:
            raise SolverFailedError("Grid must have at least 3 points per dimension")
    return x_start, x_end, y_start, y_end, int(nx), int(ny)


def probe_coefficients(
    residual_func: Callable[..., float],
    xi: float,
    yj: float,
    params: dict[str, float],
) -> PDECoefficients:
    """Extract and verify affine residual coefficients at one coordinate."""
    coordinate = f"({xi:.12g}, {yj:.12g})"

    def evaluate(state: tuple[float, float, float, float, float, float]) -> float:
        return finite_scalar(
            residual_func(xi, yj, *state, **params),
            f"Residual at {coordinate}",
        )

    zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r0 = evaluate(zero)
    unit_states = (
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    responses = [evaluate(state) - r0 for state in unit_states]
    reconstructed = r0 + sum(
        coefficient * value for coefficient, value in zip(responses, _AFFINITY_PROBE)
    )
    actual = evaluate(_AFFINITY_PROBE)
    scale = max(
        1.0,
        abs(actual),
        abs(reconstructed),
        abs(r0)
        + sum(abs(coefficient * value) for coefficient, value in zip(responses, _AFFINITY_PROBE)),
    )
    if abs(actual - reconstructed) > _AFFINITY_TOLERANCE * scale:
        raise SolverFailedError(
            "PDE residual is not affine in (f, fx, fy, fxx, fxy, fyy) "
            f"at {coordinate}: probe mismatch {abs(actual - reconstructed):.3g}"
        )
    g_f, d_fx, e_fy, a_fxx, b_fxy, c_fyy = responses
    return (a_fxx, b_fxy, c_fyy, d_fx, e_fy, g_f, r0)


def validate_coefficients(
    coefficients: object,
    xi: float,
    yj: float,
) -> tuple[PDECoefficients, int]:
    """Validate coefficient shape/finiteness/ellipticity and return orientation."""
    coordinate = f"({xi:.12g}, {yj:.12g})"
    try:
        array = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(
            f"PDE coefficients at {coordinate} must be seven real scalars"
        ) from exc
    if array.shape != (7,):
        raise SolverFailedError(
            f"PDE coefficients at {coordinate} must have shape (7,), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SolverFailedError(f"PDE coefficients at {coordinate} must be finite")

    a_c, bxy, c_c = (float(value) for value in array[:3])
    principal = np.array(((a_c, 0.5 * bxy), (0.5 * bxy, c_c)))
    principal_scale = float(np.max(np.abs(principal)))
    if principal_scale <= np.finfo(float).tiny:
        raise SolverFailedError(f"PDE principal operator is degenerate at {coordinate}")
    eigenvalues = np.linalg.eigvalsh(principal)
    margin = 100.0 * np.finfo(float).eps * principal_scale
    positive = bool(np.all(eigenvalues > margin))
    negative = bool(np.all(eigenvalues < -margin))
    if not (positive or negative):
        raise SolverFailedError(
            "PDE principal operator is not strictly elliptic "
            f"at {coordinate}: eigenvalues={eigenvalues.tolist()}"
        )
    return cast(PDECoefficients, tuple(float(value) for value in array)), 1 if positive else -1


def connected_component_labels(
    mask: np.ndarray,
    *,
    periodic_x: bool,
    periodic_y: bool,
) -> np.ndarray:
    """Label four-neighbor connected components, including periodic wraps."""
    active = np.asarray(mask, dtype=bool)
    ny, nx = active.shape
    labels = np.full(active.shape, -1, dtype=np.int64)
    component = 0
    for start_j, start_i in np.argwhere(active):
        if labels[start_j, start_i] >= 0:
            continue
        labels[start_j, start_i] = component
        pending = [(int(start_i), int(start_j))]
        while pending:
            i, j = pending.pop()
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if periodic_x:
                    ni %= nx
                if periodic_y:
                    nj %= ny
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                if active[nj, ni] and labels[nj, ni] < 0:
                    labels[nj, ni] = component
                    pending.append((ni, nj))
        component += 1
    return labels
