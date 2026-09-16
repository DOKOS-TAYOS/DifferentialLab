"""Validation helpers for scalar two-dimensional PDE solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from solver.pde_types import PDECoefficients, VectorPDECoefficients, VectorPDEResidual
from utils import SolverFailedError

_AFFINITY_PROBE = (0.37, -0.61, 1.19, -0.83, 0.47, 1.31)
_AFFINITY_TOLERANCE = 1.0e-9
_VECTOR_AFFINITY_TOLERANCE = 1.0e-9
_STRONG_ELLIPTICITY_DIRECTIONS = 32
_STRONG_ELLIPTICITY_RELATIVE_TOLERANCE = 1.0e-10


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


def finite_vector(value: object, size: int, name: str) -> np.ndarray:
    """Convert a vector-like value to a finite real array of exact length ``size``."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must be a finite real vector of shape ({size},)") from exc
    if np.iscomplexobj(raw):
        raise SolverFailedError(f"{name} must be a finite real vector of shape ({size},)")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must be a finite real vector of shape ({size},)") from exc
    if result.shape != (size,):
        raise SolverFailedError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise SolverFailedError(f"{name} must be finite")
    return result


def validate_component_count(components: int) -> int:
    """Validate and normalize a vector PDE component count."""
    if isinstance(components, (bool, np.bool_)) or not isinstance(components, (int, np.integer)):
        raise SolverFailedError("components must be an integer")
    if int(components) < 1:
        raise SolverFailedError("components must be at least 1")
    return int(components)


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


def probe_vector_coefficients(
    residual_func: VectorPDEResidual,
    xi: float,
    yj: float,
    components: int,
    params: dict[str, float],
) -> VectorPDECoefficients:
    """Recover all coefficient matrices and verify full-state residual affinity.

    Every state family is perturbed in every component, so off-diagonal
    responses are recovered. Two deterministic dense probes then vary all six
    vector arguments simultaneously and compare the actual residual with the
    complete matrix-affine reconstruction.
    """
    coordinate = f"({xi:.12g}, {yj:.12g})"

    def evaluate(state: np.ndarray) -> np.ndarray:
        vectors = [state[index].copy() for index in range(6)]
        try:
            raw = residual_func(xi, yj, *vectors, **params)
        except SolverFailedError:
            raise
        except Exception as exc:
            raise SolverFailedError(
                f"Vector PDE residual evaluation failed at {coordinate}: {exc}"
            ) from exc
        return finite_vector(raw, components, f"Vector PDE residual at {coordinate}")

    zero_state = np.zeros((6, components), dtype=float)
    constant = evaluate(zero_state)
    responses = np.empty((6, components, components), dtype=float)
    for state_index in range(6):
        for component in range(components):
            basis = zero_state.copy()
            basis[state_index, component] = 1.0
            responses[state_index, :, component] = evaluate(basis) - constant

    row_indices = np.arange(1, 7, dtype=float)[:, np.newaxis]
    component_indices = np.arange(1, components + 1, dtype=float)[np.newaxis, :]
    probes = (
        0.19 * row_indices - 0.31 * component_indices + 0.07 * row_indices * component_indices,
        np.cos(row_indices * component_indices) + 0.13 * row_indices - 0.17 * component_indices,
    )
    for probe_number, probe in enumerate(probes, start=1):
        reconstructed = constant.copy()
        for state_index in range(6):
            reconstructed += responses[state_index] @ probe[state_index]
        actual = evaluate(probe)
        scale = max(
            1.0,
            float(np.linalg.norm(actual, ord=np.inf)),
            float(np.linalg.norm(reconstructed, ord=np.inf)),
            float(np.linalg.norm(constant, ord=np.inf))
            + sum(
                float(np.linalg.norm(responses[index] @ probe[index], ord=np.inf))
                for index in range(6)
            ),
        )
        mismatch = float(np.linalg.norm(actual - reconstructed, ord=np.inf))
        if mismatch > _VECTOR_AFFINITY_TOLERANCE * scale:
            raise SolverFailedError(
                "Vector PDE residual is not affine in the complete coupled state "
                "(f, fx, fy, fxx, fxy, fyy) "
                f"at {coordinate}: dense probe {probe_number} mismatch {mismatch:.3g}"
            )

    return VectorPDECoefficients(
        fxx=responses[3],
        fxy=responses[4],
        fyy=responses[5],
        fx=responses[1],
        fy=responses[2],
        f=responses[0],
        constant=constant,
    )


def _finite_matrix(value: object, components: int, name: str) -> np.ndarray:
    """Validate one real finite square system matrix."""
    expected = (components, components)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must be a finite real matrix of shape {expected}") from exc
    if np.iscomplexobj(raw):
        raise SolverFailedError(f"{name} must be a finite real matrix of shape {expected}")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must be a finite real matrix of shape {expected}") from exc
    if result.shape != expected:
        raise SolverFailedError(f"{name} must have shape {expected}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise SolverFailedError(f"{name} must be finite")
    return result


def validate_vector_coefficients(
    coefficients: object,
    xi: float,
    yj: float,
    components: int,
) -> tuple[VectorPDECoefficients, int]:
    """Validate matrix coefficients and sampled strong ellipticity.

    For 32 equally spaced unoriented unit directions on ``[0, pi)``, this
    checks the eigenvalues of the symmetric part of
    ``Axx*xi_x**2 + Axy*xi_x*xi_y + Ayy*xi_y**2``. Every sampled symbol must
    be strictly positive or strictly negative definite relative to its own
    infinity-norm scale, and all directions must have one orientation. This
    deterministic numerical screen is not a mathematical proof between the
    sampled directions.
    """
    coordinate = f"({xi:.12g}, {yj:.12g})"
    if not isinstance(coefficients, VectorPDECoefficients):
        raise SolverFailedError(
            f"Vector PDE coefficients at {coordinate} must be a VectorPDECoefficients object"
        )
    matrices = {
        name: _finite_matrix(getattr(coefficients, name), components, f"{name} at {coordinate}")
        for name in ("fxx", "fxy", "fyy", "fx", "fy", "f")
    }
    constant = finite_vector(
        coefficients.constant,
        components,
        f"constant at {coordinate}",
    )

    direction_indices = np.arange(_STRONG_ELLIPTICITY_DIRECTIONS, dtype=float)
    angles = np.pi * direction_indices / _STRONG_ELLIPTICITY_DIRECTIONS
    direction_x = np.cos(angles)
    direction_y = np.sin(angles)
    symbols = (
        matrices["fxx"] * direction_x[:, np.newaxis, np.newaxis] ** 2
        + matrices["fxy"] * (direction_x * direction_y)[:, np.newaxis, np.newaxis]
        + matrices["fyy"] * direction_y[:, np.newaxis, np.newaxis] ** 2
    )
    symmetric_symbols = 0.5 * (symbols + np.swapaxes(symbols, -1, -2))
    symbol_scales = np.maximum(
        np.linalg.norm(symmetric_symbols, ord=np.inf, axis=(-2, -1)),
        np.finfo(float).tiny,
    )
    eigenvalues = np.linalg.eigvalsh(symmetric_symbols)
    tolerances = _STRONG_ELLIPTICITY_RELATIVE_TOLERANCE * symbol_scales
    positive = np.all(eigenvalues > tolerances[:, np.newaxis], axis=1)
    negative = np.all(eigenvalues < -tolerances[:, np.newaxis], axis=1)
    definite = positive | negative
    failing_directions = np.flatnonzero(~definite)
    if failing_directions.size:
        direction_index = int(failing_directions[0])
        raise SolverFailedError(
            "Vector PDE principal symbol is not uniformly definite in sampled direction "
            f"{direction_index} at {coordinate}: direction="
            f"({direction_x[direction_index]:.12g}, {direction_y[direction_index]:.12g}), "
            f"eigenvalues={eigenvalues[direction_index].tolist()}, "
            f"tolerance={tolerances[direction_index]:.3g}"
        )

    orientations = np.where(positive, 1, -1)
    if np.any(orientations != orientations[0]):
        raise SolverFailedError(
            "Vector PDE principal-symbol orientation changes across sampled directions "
            f"at {coordinate}"
        )

    normalized = VectorPDECoefficients(
        fxx=matrices["fxx"],
        fxy=matrices["fxy"],
        fyy=matrices["fyy"],
        fx=matrices["fx"],
        fy=matrices["fy"],
        f=matrices["f"],
        constant=constant,
    )
    return normalized, int(orientations[0])


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
