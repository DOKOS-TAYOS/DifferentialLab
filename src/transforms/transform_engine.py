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
        logger.debug("scipy.misc.derivative unavailable, using central-difference fallback")

    # Fallback: recursive central difference
    def df(x: float) -> float:
        return (_nth_derivative(f, x + dx, n - 1, dx) - _nth_derivative(f, x - dx, n - 1, dx)) / (
            2 * dx
        )

    return df(x0)


def _compute_taylor_coeffs(
    func: Callable[[np.ndarray], np.ndarray],
    center: float,
    order: int,
    x_min: float | None = None,
    x_max: float | None = None,
) -> np.ndarray:
    """Compute Taylor series coefficients around a center point.

    Uses least-squares polynomial fitting when x_min/x_max are provided (more stable
    than numerical differentiation for high orders). Falls back to derivative-based
    computation otherwise.

    Args:
        func: Vectorized callable f(x) -> y.
        center: Center point for Taylor expansion.
        order: Highest order of derivative.
        x_min: Lower bound of domain (for polynomial fitting).
        x_max: Upper bound of domain (for polynomial fitting).

    Returns:
        Array of Taylor coefficients a_0, a_1, ..., a_order.
    """
    if x_min is not None and x_max is not None and x_max > x_min:
        # Polynomial fitting: f(x) ≈ Σ a_k (x - center)^k. More stable than derivatives.
        span = max(x_max - x_min, 1e-10)
        radius = min(1.0, span / 2.0)
        n_samples = max(order * 2 + 1, 50)
        x_sample = np.linspace(center - radius, center + radius, n_samples)
        x_sample = np.clip(x_sample, x_min, x_max)
        y_sample = func(x_sample)
        t = x_sample - center  # Expand in powers of (x - center)
        # Vandermonde: V[i,k] = t[i]^k
        V = np.vander(t, order + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(V, y_sample, rcond=None)
        # Zero out negligible coefficients (numerical noise)
        max_c = float(np.max(np.abs(coeffs)))
        if max_c > 0:
            coeffs = np.where(np.abs(coeffs) < 1e-10 * max_c, 0.0, coeffs)
        return coeffs

    # Fallback: derivative-based (unstable for order >= 4)
    def nth_derivative(n: int, x0: float) -> float:
        dx = max(1e-6, 1e-2 * (0.5) ** max(0, n - 2))  # Larger dx for high n
        return _nth_derivative(
            lambda t: float(func(np.array([t]))[0]),
            x0,
            n=n,
            dx=dx,
        )

    coeffs = np.zeros(order + 1)
    for n in range(order + 1):
        coeffs[n] = nth_derivative(n, center) / math.factorial(n)
    max_c = float(np.max(np.abs(coeffs)))
    if max_c > 0:
        coeffs = np.where(np.abs(coeffs) < 1e-10 * max_c, 0.0, coeffs)
    return coeffs


def _hilbert_filter_kernel(n: int) -> np.ndarray:
    """Build Hilbert transform filter for FFT of length n."""
    h = np.zeros(n, dtype=complex)
    h[0] = 1
    h[1 : (n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    return h


def _trim_indices_by_amplitude(
    magnitudes: np.ndarray,
    threshold_fraction: float,
    use_nanmax: bool = False,
) -> tuple[int, int] | None:
    """Find indices where magnitude is above threshold (fraction of max).

    Args:
        magnitudes: Array of magnitudes.
        threshold_fraction: Fraction of max amplitude to use as threshold.
        use_nanmax: If True, use np.nanmax (for Laplace with possible NaN).

    Returns:
        (i_min, i_max) if any point above threshold, else None.
    """
    if threshold_fraction <= 0 or len(magnitudes) == 0:
        return None
    max_amp = float(np.nanmax(magnitudes) if use_nanmax else np.max(magnitudes))
    if max_amp <= 0 or not np.isfinite(max_amp):
        return None
    threshold = max_amp * threshold_fraction
    above = np.where(magnitudes >= threshold)[0]
    if len(above) == 0:
        return None
    return int(above[0]), int(above[-1])


def _refine_fft_spectrum_in_range(
    y: np.ndarray,
    dx: float,
    f_low: float,
    f_high: float,
    n_target: int,
    magnitude_fn: Callable[[np.ndarray, int], np.ndarray],
    fallback: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine FFT spectrum via zero-padding and extract range [f_low, f_high].

    Args:
        y: Real signal samples.
        dx: Sample spacing.
        f_low: Lower frequency bound.
        f_high: Upper frequency bound.
        n_target: Target number of points for refinement.
        magnitude_fn: Callable(fft_vals, n) -> magnitudes of length n//2.
        fallback: If mask is empty, return this (freqs, mag, bin_indices) instead.

    Returns:
        (freqs, magnitudes, bin_indices) in the requested range.
    """
    n_points = len(y)
    f_span = max(f_high - f_low, 1.0 / (n_points * dx))
    n_refined = int(np.ceil(n_target / (f_span * dx)))
    n_refined = max(n_refined, n_points)
    # Cap at a reasonable size; if the signal is longer, downsample it
    _MAX_FFT = 65536
    if n_refined > _MAX_FFT:
        n_refined = _MAX_FFT
    if n_points > n_refined:
        step = max(1, n_points // n_refined)
        y = y[::step]
        n_points = len(y)

    y_padded = np.zeros(n_refined, dtype=complex)
    y_padded[:n_points] = y
    fft_ref = fft.fft(y_padded)
    mag_ref = magnitude_fn(fft_ref, n_refined)
    freqs_ref = fft.fftfreq(n_refined, dx)[: n_refined // 2]
    freqs_ref = np.abs(freqs_ref)

    mask = (freqs_ref >= f_low) & (freqs_ref <= f_high)
    if np.any(mask):
        bin_indices = np.where(mask)[0]
        return freqs_ref[mask], mag_ref[mask], bin_indices
    if fallback is not None:
        return fallback
    return freqs_ref, mag_ref, np.arange(len(freqs_ref))


def _trim_and_refine_fft_spectrum(
    y: np.ndarray,
    dx: float,
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    threshold_fraction: float,
    n_target: int,
    magnitude_fn: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim spectrum by amplitude threshold and refine via zero-padding.

    Args:
        y: Real signal samples.
        dx: Sample spacing.
        freqs: Frequency axis.
        magnitudes: Magnitude values.
        threshold_fraction: Relative amplitude threshold.
        n_target: Target number of points after refinement.
        magnitude_fn: Callable(fft_vals, n) -> magnitudes for refined FFT.

    Returns:
        (freqs, magnitudes, bin_indices) trimmed and refined.
        bin_indices are the actual FFT bin indices (k) for coefficient display.
    """
    trimmed = _trim_indices_by_amplitude(magnitudes, threshold_fraction)
    if trimmed is None:
        return freqs, magnitudes, np.arange(len(freqs))

    i_min, i_max = trimmed
    f_low, f_high = float(freqs[i_min]), float(freqs[i_max])
    fallback = (
        freqs[i_min : i_max + 1],
        magnitudes[i_min : i_max + 1],
        np.arange(i_min, i_max + 1),
    )
    return _refine_fft_spectrum_in_range(
        y, dx, f_low, f_high, n_target, magnitude_fn, fallback=fallback
    )


def _trim_and_refine_laplace(
    func: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    s_vals: np.ndarray,
    laplace_vals: np.ndarray,
    threshold_fraction: float,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim Laplace spectrum by amplitude and recompute in refined s range.

    Args:
        func: Vectorized callable f(x) -> y.
        x_min: Lower integration bound.
        x_max: Upper integration bound.
        s_vals: Original s values.
        laplace_vals: Laplace transform values.
        threshold_fraction: Relative amplitude threshold.
        n_target: Target number of points.

    Returns:
        (s_vals, laplace_vals, sample_indices) trimmed and refined.
        sample_indices are the original sample indices (i) for L(s_i) display.
    """
    laplace_mag = np.abs(laplace_vals)
    trimmed = _trim_indices_by_amplitude(laplace_mag, threshold_fraction, use_nanmax=True)
    if trimmed is None:
        return s_vals, laplace_vals, np.arange(len(s_vals))

    i_min, i_max = trimmed
    s_low, s_high = float(s_vals[i_min]), float(s_vals[i_max])
    s_refined = np.linspace(s_low, s_high, n_target)
    laplace_refined = _compute_laplace_samples(func, x_min, x_max, s_refined)
    # Indices map refined points to original grid (i_min..i_max)
    sample_indices = np.linspace(i_min, i_max, n_target)
    return s_refined, laplace_refined, sample_indices


def _compute_laplace_samples(
    func: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    s_vals: np.ndarray,
) -> np.ndarray:
    """Compute Laplace transform samples over given s values.

    Args:
        func: Vectorized callable f(x) -> y.
        x_min: Lower bound of integration.
        x_max: Upper bound of integration.
        s_vals: Array of s values at which to evaluate the Laplace transform.

    Returns:
        Array of Laplace transform values at each s.
    """
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
        except Exception as exc:
            logger.debug("Laplace quad failed at s=%g: %s", s, exc)
            laplace_vals[i] = np.nan

    return laplace_vals


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
    COEFFICIENTS = "Coefficients (a\u1d62 vs i)"


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
    fourier_amp_threshold: float = 0.01,
    z_transform_amp_threshold: float = 0.01,
    laplace_amp_threshold: float = 0.01,
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
        fourier_amp_threshold: Relative amplitude threshold (fraction of max) to trim
            Fourier spectrum at low and high frequencies. Frequencies with |F(ω)| below
            this fraction of the maximum are excluded from the displayed range. Default 0.01.
        z_transform_amp_threshold: Same as fourier_amp_threshold but for Z-transform
            magnitude spectrum. Default 0.01.
        laplace_amp_threshold: Same for Laplace: trim s range where |L(s)| is below
            this fraction of max. Default 0.01.

    Returns:
        Tuple of (x_axis, y_values, x_label, y_label).

    Raises:
        ValueError: If *kind* is not a known transform type.
    """
    if kind == TransformKind.ORIGINAL:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        return x, y, "x", "f(x)"

    if kind == TransformKind.FOURIER:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_vals = fft.fft(y)
        fft_mag = np.abs(fft_vals[: n_points // 2])
        freqs = np.abs(fft.fftfreq(n_points, dx)[: n_points // 2])
        freqs, fft_mag, _ = _trim_and_refine_fft_spectrum(
            y, dx, freqs, fft_mag, fourier_amp_threshold, n_points // 2,
            magnitude_fn=lambda fv, n: np.abs(fv[: n // 2]),
        )
        return freqs, fft_mag, "ω/(2π)", "|F(ω)|"

    if kind == TransformKind.LAPLACE:
        s_vals = np.linspace(laplace_s_min, laplace_s_max, laplace_n_points)
        laplace_vals = _compute_laplace_samples(func, x_min, x_max, s_vals)
        s_vals, laplace_vals, _ = _trim_and_refine_laplace(
            func, x_min, x_max, s_vals, laplace_vals,
            laplace_amp_threshold, laplace_n_points,
        )
        return s_vals, laplace_vals, "s (real)", "L(s)"

    if kind == TransformKind.TAYLOR:
        center = taylor_center if taylor_center is not None else (x_min + x_max) / 2
        coeffs = _compute_taylor_coeffs(func, center, taylor_order, x_min, x_max)

        x = np.linspace(x_min, x_max, n_points)
        x_centered = x - center
        y_taylor = np.zeros_like(x)
        for n, c in enumerate(coeffs):
            y_taylor += c * (x_centered**n)

        return x, y_taylor, "x", f"Taylor_{taylor_order}(x)"

    if kind == TransformKind.HILBERT:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        fft_vals = fft.fft(y)
        analytic_fft = fft_vals * _hilbert_filter_kernel(len(fft_vals))
        hilbert_signal = np.imag(fft.ifft(analytic_fft))
        return x, hilbert_signal, "x", "H[f](x)"

    if kind == TransformKind.Z_TRANSFORM:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_vals = fft.fft(y)
        fft_mag = np.abs(fft_vals[: n_points // 2])
        freqs = np.abs(fft.fftfreq(n_points, dx)[: n_points // 2])
        freqs, fft_mag, _ = _trim_and_refine_fft_spectrum(
            y, dx, freqs, fft_mag, z_transform_amp_threshold, n_points // 2,
            magnitude_fn=lambda fv, n: np.abs(fv[: n // 2]),
        )
        return freqs, fft_mag, "ω/(2π)", "|X[k]|"

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
    laplace_s_min: float = 0.1,
    laplace_s_max: float = 10.0,
    laplace_n_points: int = 50,
    fourier_amp_threshold: float = 0.01,
    z_transform_amp_threshold: float = 0.01,
    laplace_amp_threshold: float = 0.01,
    hilbert_amp_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, str, str, dict[str, object]]:
    """Return coefficient representation with physical axis and metadata.

    Uses frequency ω (or s for Laplace) as x-axis when meaningful for interpretation.

    Args:
        func: Vectorized callable f(x) -> y.
        kind: Transform kind.
        x_min: Lower bound.
        x_max: Upper bound.
        n_points: Sample count.
        taylor_order: Taylor order.
        taylor_center: Taylor center.
        laplace_n_points: Number of s values for Laplace.
        fourier_amp_threshold: Amplitude threshold for Fourier coefficient trimming.
        z_transform_amp_threshold: Amplitude threshold for Z-transform coefficient trimming.
        laplace_amp_threshold: Amplitude threshold for Laplace coefficient trimming.
        hilbert_amp_threshold: Amplitude threshold for Hilbert coefficient trimming.

    Returns:
        Tuple of (x_axis, coefficients, x_label, y_label, metadata).
        metadata: dict with domain, n_points, and transform-specific params.

    Raises:
        ValueError: If *kind* is not a known transform type.
    """
    base_meta: dict[str, object] = {
        "kind": kind.value,
        "domain": (x_min, x_max),
        "n_points": n_points,
    }

    if kind == TransformKind.ORIGINAL:
        x, y = compute_function_samples(func, x_min, x_max, min(n_points, 200))
        indices = np.arange(len(x))
        return indices, y, "i (sample)", "f(x_i)", {**base_meta}

    if kind == TransformKind.TAYLOR:
        center = taylor_center if taylor_center is not None else (x_min + x_max) / 2
        coeffs = _compute_taylor_coeffs(func, center, taylor_order, x_min, x_max)
        indices = np.arange(taylor_order + 1)
        meta = {**base_meta, "taylor_order": taylor_order, "taylor_center": center}
        return indices, coeffs, "i", "a\u1d62", meta

    if kind == TransformKind.FOURIER:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_mag = np.abs(fft.fft(y)[: n_points // 2])
        freqs = np.abs(fft.fftfreq(n_points, dx)[: n_points // 2])
        freqs, coeffs, _ = _trim_and_refine_fft_spectrum(
            y, dx, freqs, fft_mag, fourier_amp_threshold, n_points // 2,
            magnitude_fn=lambda fv, n: np.abs(fv[: n // 2]),
        )
        meta = {**base_meta, "amp_threshold": fourier_amp_threshold}
        return freqs, coeffs, "ω/(2π)", "|F(ω)|", meta

    if kind == TransformKind.LAPLACE:
        s_vals = np.linspace(laplace_s_min, laplace_s_max, laplace_n_points)
        laplace_vals = _compute_laplace_samples(func, x_min, x_max, s_vals)
        s_vals, coeffs, _ = _trim_and_refine_laplace(
            func, x_min, x_max, s_vals, laplace_vals,
            laplace_amp_threshold, laplace_n_points,
        )
        meta = {
            **base_meta,
            "s_range": (laplace_s_min, laplace_s_max),
            "laplace_n_points": laplace_n_points,
            "amp_threshold": laplace_amp_threshold,
        }
        return s_vals, coeffs, "s", "L(s)", meta

    if kind == TransformKind.HILBERT:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_vals = fft.fft(y)
        coeffs = np.abs((fft_vals * _hilbert_filter_kernel(len(fft_vals)))[: len(fft_vals) // 2])
        freqs = np.abs(fft.fftfreq(len(fft_vals), dx)[: len(fft_vals) // 2])
        freqs, coeffs, _ = _trim_and_refine_fft_spectrum(
            y, dx, freqs, coeffs, hilbert_amp_threshold, n_points // 2,
            magnitude_fn=lambda fv, n: np.abs((fv * _hilbert_filter_kernel(n))[: n // 2]),
        )
        meta = {**base_meta, "amp_threshold": hilbert_amp_threshold}
        return freqs, coeffs, "ω/(2π)", "|H(ω)|", meta

    if kind == TransformKind.Z_TRANSFORM:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_mag = np.abs(fft.fft(y)[: n_points // 2])
        freqs = np.abs(fft.fftfreq(n_points, dx)[: n_points // 2])
        freqs, coeffs, _ = _trim_and_refine_fft_spectrum(
            y, dx, freqs, fft_mag, z_transform_amp_threshold, n_points // 2,
            magnitude_fn=lambda fv, n: np.abs(fv[: n // 2]),
        )
        meta = {**base_meta, "amp_threshold": z_transform_amp_threshold}
        return freqs, coeffs, "ω/(2π)", "|X(ω)|", meta

    raise ValueError(f"Unknown transform kind: {kind}")
