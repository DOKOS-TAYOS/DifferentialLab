"""Mathematical transforms: Fourier, Laplace, Taylor series."""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable

import numpy as np
from scipy import fft
from scipy.integrate import quad

from utils import get_logger

logger = get_logger(__name__)


def _nth_derivative(
    f: Callable[[float], float],
    x0: float,
    n: int,
    dx: float = 1e-6,
) -> float:
    """Compute the n-th derivative of f at x0 using central differences.

    Args:
        f: Scalar function.
        x0: Point of evaluation.
        n: Derivative order.
        dx: Step size.

    Returns:
        Approximate n-th derivative.
    """
    if n == 0:
        return f(x0)
    try:
        from scipy.misc import derivative as scipy_deriv
        return scipy_deriv(f, x0, n=n, dx=dx, order=2 * n + 1)
    except (ImportError, AttributeError):
        pass
    # Fallback: recursive central difference
    def df(x: float) -> float:
        return (_nth_derivative(f, x + dx, n - 1, dx)
                - _nth_derivative(f, x - dx, n - 1, dx)) / (2 * dx)
    return df(x0)


class TransformKind(str, Enum):
    """Available transformation types."""

    ORIGINAL = "Original (f(x))"
    FOURIER = "Fourier (FFT)"
    LAPLACE = "Laplace (real axis)"
    TAYLOR = "Taylor series"
    HILBERT = "Hilbert (discrete)"
    Z_TRANSFORM = "Z-transform (discrete)"


class DisplayMode(str, Enum):
    """How to display the transform result."""

    CURVE = "Curve (f vs x)"
    COEFFICIENTS = "Coefficients (a_i vs i)"


def compute_function_samples(
    func: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    n_points: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a function over a range.

    Args:
        func: Vectorized callable f(x) -> y.
        x_min: Lower bound.
        x_max: Upper bound.
        n_points: Number of sample points.

    Returns:
        Tuple of (x, y) arrays.
    """
    x = np.linspace(x_min, x_max, n_points)
    y = func(x)
    return x, y


def apply_transform(
    func: Callable[[np.ndarray], np.ndarray],
    kind: TransformKind,
    x_min: float,
    x_max: float,
    n_points: int = 1024,
    *,
    taylor_order: int = 5,
    taylor_center: float | None = None,
    laplace_s_min: float = 0.1,
    laplace_s_max: float = 10.0,
    laplace_n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Apply a transformation to a function and return (x, y) for plotting.

    Args:
        func: Vectorized callable f(x) -> y.
        kind: Type of transformation.
        x_min: Lower bound of the original domain.
        x_max: Upper bound of the original domain.
        n_points: Number of sample points for Original/Fourier/Taylor.
        taylor_order: Order of Taylor expansion (for Taylor kind).
        taylor_center: Center point for Taylor (default: midpoint).
        laplace_s_min: Minimum s for Laplace (real axis).
        laplace_s_max: Maximum s for Laplace.
        laplace_n_points: Number of s values for Laplace.

    Returns:
        Tuple of (x_axis, y_values, x_label, y_label).
    """
    if kind == TransformKind.ORIGINAL:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        return x, y, "x", "f(x)"

    if kind == TransformKind.FOURIER:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_vals = fft.fft(y)
        fft_mag = np.abs(fft_vals[: n_points // 2])
        freqs = fft.fftfreq(n_points, dx)[: n_points // 2]
        freqs = np.abs(freqs)
        return freqs, fft_mag, "Frequency ω", "|F(ω)|"

    if kind == TransformKind.LAPLACE:
        s_vals = np.linspace(laplace_s_min, laplace_s_max, laplace_n_points)

        def integrand(t: float, s: float) -> float:
            if t < x_min or t > x_max:
                return 0.0
            try:
                return float(func(np.array([t]))[0] * np.exp(-s * t))
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0

        laplace_vals = np.zeros_like(s_vals)
        for i, s in enumerate(s_vals):
            try:
                result, _ = quad(
                    lambda t, s_val=s: integrand(t, s_val),
                    x_min,
                    x_max,
                    limit=200,
                    epsabs=1e-10,
                    epsrel=1e-8,
                )
                laplace_vals[i] = result
            except Exception:
                laplace_vals[i] = np.nan

        return s_vals, laplace_vals, "s (real)", "L(s)"

    if kind == TransformKind.TAYLOR:
        center = taylor_center if taylor_center is not None else (x_min + x_max) / 2

        def nth_derivative(n: int, x0: float) -> float:
            return _nth_derivative(
                lambda t: float(func(np.array([t]))[0]),
                x0,
                n=n,
                dx=1e-6,
            )

        coeffs = np.zeros(taylor_order + 1)
        for n in range(taylor_order + 1):
            coeffs[n] = nth_derivative(n, center) / math.factorial(n)

        x = np.linspace(x_min, x_max, n_points)
        x_centered = x - center
        y_taylor = np.zeros_like(x)
        for n, c in enumerate(coeffs):
            y_taylor += c * (x_centered**n)

        return x, y_taylor, "x", f"Taylor_{taylor_order}(x)"

    if kind == TransformKind.HILBERT:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        # Discrete Hilbert transform: H[f] = Im(analytic signal)
        fft_vals = fft.fft(y)
        n = len(fft_vals)
        h = np.zeros(n, dtype=complex)
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1
        h[(n + 1) // 2 :] = 0
        analytic_fft = fft_vals * h
        analytic_signal = fft.ifft(analytic_fft)
        hilbert_signal = np.imag(analytic_signal)
        return x, hilbert_signal, "x", "H[f](x)"

    if kind == TransformKind.Z_TRANSFORM:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        # Z-transform evaluated on unit circle = DFT; show magnitude spectrum
        fft_vals = fft.fft(y)
        fft_mag = np.abs(fft_vals[: n_points // 2])
        freqs = fft.fftfreq(n_points, (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0)
        freqs = np.abs(freqs[: n_points // 2])
        return freqs, fft_mag, "k (index)", "|X[k]|"

    raise ValueError(f"Unknown transform kind: {kind}")


def get_transform_coefficients(
    func: Callable[[np.ndarray], np.ndarray],
    kind: TransformKind,
    x_min: float,
    x_max: float,
    n_points: int = 512,
    *,
    taylor_order: int = 5,
    taylor_center: float | None = None,
    laplace_n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return coefficient representation (index i vs a_i) for the transform.

    Args:
        func: Vectorized callable f(x) -> y.
        kind: Transform kind.
        x_min: Lower bound.
        x_max: Upper bound.
        n_points: Sample count.
        taylor_order: Taylor order.
        taylor_center: Taylor center.
        laplace_n_points: Number of s values for Laplace.

    Returns:
        Tuple of (indices_i, coefficients_a_i, x_label, y_label).
    """
    if kind == TransformKind.ORIGINAL:
        x, y = compute_function_samples(func, x_min, x_max, min(n_points, 200))
        indices = np.arange(len(x))
        return indices, y, "i (sample)", "f(x_i)"

    if kind == TransformKind.TAYLOR:
        center = taylor_center if taylor_center is not None else (x_min + x_max) / 2

        def nth_derivative(n: int, x0: float) -> float:
            return _nth_derivative(
                lambda t: float(func(np.array([t]))[0]),
                x0,
                n=n,
                dx=1e-6,
            )

        coeffs = np.zeros(taylor_order + 1)
        for n in range(taylor_order + 1):
            coeffs[n] = nth_derivative(n, center) / math.factorial(n)

        indices = np.arange(taylor_order + 1)
        return indices, coeffs, "i", "a_i"

    if kind == TransformKind.FOURIER:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        fft_vals = fft.fft(y)
        half = n_points // 2
        coeffs = np.abs(fft_vals[:half])
        indices = np.arange(half)
        return indices, coeffs, "k", "|F[k]|"

    if kind == TransformKind.LAPLACE:
        s_vals = np.linspace(0.1, 10.0, laplace_n_points)

        def integrand(t: float, s: float) -> float:
            if t < x_min or t > x_max:
                return 0.0
            try:
                return float(func(np.array([t]))[0] * np.exp(-s * t))
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0

        coeffs = np.zeros(laplace_n_points)
        for i, s in enumerate(s_vals):
            try:
                result, _ = quad(
                    lambda t, s_val=s: integrand(t, s_val),
                    x_min,
                    x_max,
                    limit=200,
                    epsabs=1e-10,
                    epsrel=1e-8,
                )
                coeffs[i] = result
            except Exception:
                coeffs[i] = np.nan

        indices = np.arange(laplace_n_points)
        return indices, coeffs, "i", "L(s_i)"

    if kind == TransformKind.HILBERT:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        fft_vals = fft.fft(y)
        n = len(fft_vals)
        h = np.zeros(n, dtype=complex)
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1
        analytic_fft = fft_vals * h
        coeffs = np.abs(analytic_fft[: n // 2])
        indices = np.arange(len(coeffs))
        return indices, coeffs, "k", "|H[k]|"

    if kind == TransformKind.Z_TRANSFORM:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        indices = np.arange(len(y))
        return indices, y, "n", "x[n]"

    raise ValueError(f"Unknown transform kind: {kind}")
