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
        freqs = fft.fftfreq(n_points, dx)[: n_points // 2]
        freqs = np.abs(freqs)
        n_target = n_points // 2  # Desired number of points in the displayed spectrum
        # Trim spectrum to frequencies with amplitude above threshold (relative to max)
        f_low, f_high = float(freqs[0]), float(freqs[-1])
        max_amp = float(np.max(fft_mag))
        if max_amp > 0 and fourier_amp_threshold > 0:
            threshold = max_amp * fourier_amp_threshold
            above = np.where(fft_mag >= threshold)[0]
            if len(above) > 0:
                i_min, i_max = int(above[0]), int(above[-1])
                f_low, f_high = float(freqs[i_min]), float(freqs[i_max])
                # Zero-pad to get n_target points in [f_low, f_high].
                # FFT bin spacing df = 1/(n_refined*dx); bins in [f_low,f_high] ≈ (f_high-f_low)*n_refined*dx
                f_span = max(f_high - f_low, 1.0 / (n_points * dx))
                n_refined = int(np.ceil(n_target / (f_span * dx)))
                n_refined = min(max(n_refined, n_points), 65536)  # Cap to avoid huge FFTs
                # Zero-pad: keep original y, pad with zeros to length n_refined
                y_padded = np.zeros(n_refined, dtype=complex)
                y_padded[:n_points] = y
                fft_ref = fft.fft(y_padded)
                fft_mag_ref = np.abs(fft_ref[: n_refined // 2])
                freqs_ref = fft.fftfreq(n_refined, dx)[: n_refined // 2]
                freqs_ref = np.abs(freqs_ref)
                mask = (freqs_ref >= f_low) & (freqs_ref <= f_high)
                if np.any(mask):
                    freqs = freqs_ref[mask]
                    fft_mag = fft_mag_ref[mask]
                else:
                    freqs = freqs[i_min : i_max + 1]
                    fft_mag = fft_mag[i_min : i_max + 1]
        return freqs, fft_mag, "ω/(2π)", "|F(ω)|"

    if kind == TransformKind.LAPLACE:
        s_vals = np.linspace(laplace_s_min, laplace_s_max, laplace_n_points)
        laplace_vals = _compute_laplace_samples(func, x_min, x_max, s_vals)
        laplace_mag = np.abs(laplace_vals)
        n_target = laplace_n_points
        s_low, s_high = float(laplace_s_min), float(laplace_s_max)
        max_amp = float(np.nanmax(laplace_mag))
        if max_amp > 0 and laplace_amp_threshold > 0:
            threshold = max_amp * laplace_amp_threshold
            above = np.where(laplace_mag >= threshold)[0]
            if len(above) > 0:
                i_min, i_max = int(above[0]), int(above[-1])
                s_low, s_high = float(s_vals[i_min]), float(s_vals[i_max])
                s_refined = np.linspace(s_low, s_high, n_target)
                laplace_vals = _compute_laplace_samples(func, x_min, x_max, s_refined)
                s_vals = s_refined
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
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        # Z-transform evaluated on unit circle = DFT; show magnitude spectrum
        fft_vals = fft.fft(y)
        fft_mag = np.abs(fft_vals[: n_points // 2])
        freqs = fft.fftfreq(n_points, dx)[: n_points // 2]
        freqs = np.abs(freqs)
        n_target = n_points // 2
        # Trim spectrum to frequencies with amplitude above threshold (relative to max)
        f_low, f_high = float(freqs[0]), float(freqs[-1])
        max_amp = float(np.max(fft_mag))
        if max_amp > 0 and z_transform_amp_threshold > 0:
            threshold = max_amp * z_transform_amp_threshold
            above = np.where(fft_mag >= threshold)[0]
            if len(above) > 0:
                i_min, i_max = int(above[0]), int(above[-1])
                f_low, f_high = float(freqs[i_min]), float(freqs[i_max])
                # Zero-pad to get n_target points in [f_low, f_high]
                f_span = max(f_high - f_low, 1.0 / (n_points * dx))
                n_refined = int(np.ceil(n_target / (f_span * dx)))
                n_refined = min(max(n_refined, n_points), 65536)
                y_padded = np.zeros(n_refined, dtype=complex)
                y_padded[:n_points] = y
                fft_ref = fft.fft(y_padded)
                fft_mag_ref = np.abs(fft_ref[: n_refined // 2])
                freqs_ref = fft.fftfreq(n_refined, dx)[: n_refined // 2]
                freqs_ref = np.abs(freqs_ref)
                mask = (freqs_ref >= f_low) & (freqs_ref <= f_high)
                if np.any(mask):
                    freqs = freqs_ref[mask]
                    fft_mag = fft_mag_ref[mask]
                else:
                    freqs = freqs[i_min : i_max + 1]
                    fft_mag = fft_mag[i_min : i_max + 1]
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
        fourier_amp_threshold: Amplitude threshold for Fourier coefficient trimming.
        z_transform_amp_threshold: Amplitude threshold for Z-transform coefficient trimming.
        laplace_amp_threshold: Amplitude threshold for Laplace coefficient trimming.
        hilbert_amp_threshold: Amplitude threshold for Hilbert coefficient trimming.

    Returns:
        Tuple of (indices_i, coefficients_a_i, x_label, y_label).

    Raises:
        ValueError: If *kind* is not a known transform type.
    """
    if kind == TransformKind.ORIGINAL:
        x, y = compute_function_samples(func, x_min, x_max, min(n_points, 200))
        indices = np.arange(len(x))
        return indices, y, "i (sample)", "f(x_i)"

    if kind == TransformKind.TAYLOR:
        center = taylor_center if taylor_center is not None else (x_min + x_max) / 2
        coeffs = _compute_taylor_coeffs(func, center, taylor_order, x_min, x_max)

        indices = np.arange(taylor_order + 1)
        return indices, coeffs, "i", "a_i"

    if kind == TransformKind.FOURIER:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_vals = fft.fft(y)
        fft_mag = np.abs(fft_vals[: n_points // 2])
        freqs = fft.fftfreq(n_points, dx)[: n_points // 2]
        freqs = np.abs(freqs)
        n_target = n_points // 2
        f_low, f_high = float(freqs[0]), float(freqs[-1])
        max_amp = float(np.max(fft_mag))
        if max_amp > 0 and fourier_amp_threshold > 0:
            threshold = max_amp * fourier_amp_threshold
            above = np.where(fft_mag >= threshold)[0]
            if len(above) > 0:
                i_min, i_max = int(above[0]), int(above[-1])
                f_low, f_high = float(freqs[i_min]), float(freqs[i_max])
                f_span = max(f_high - f_low, 1.0 / (n_points * dx))
                n_refined = int(np.ceil(n_target / (f_span * dx)))
                n_refined = min(max(n_refined, n_points), 65536)
                y_padded = np.zeros(n_refined, dtype=complex)
                y_padded[:n_points] = y
                fft_ref = fft.fft(y_padded)
                fft_mag_ref = np.abs(fft_ref[: n_refined // 2])
                freqs_ref = fft.fftfreq(n_refined, dx)[: n_refined // 2]
                freqs_ref = np.abs(freqs_ref)
                mask = (freqs_ref >= f_low) & (freqs_ref <= f_high)
                if np.any(mask):
                    coeffs = fft_mag_ref[mask]
                else:
                    coeffs = fft_mag[i_min : i_max + 1]
            else:
                coeffs = fft_mag
        else:
            coeffs = fft_mag
        indices = np.arange(len(coeffs))
        return indices, coeffs, "k", "|F[k]|"

    if kind == TransformKind.LAPLACE:
        s_vals = np.linspace(laplace_s_min, laplace_s_max, laplace_n_points)
        coeffs = _compute_laplace_samples(func, x_min, x_max, s_vals)
        laplace_mag = np.abs(coeffs)
        n_target = laplace_n_points
        s_low, s_high = float(laplace_s_min), float(laplace_s_max)
        max_amp = float(np.nanmax(laplace_mag))
        if max_amp > 0 and laplace_amp_threshold > 0:
            threshold = max_amp * laplace_amp_threshold
            above = np.where(laplace_mag >= threshold)[0]
            if len(above) > 0:
                i_min, i_max = int(above[0]), int(above[-1])
                s_low, s_high = float(s_vals[i_min]), float(s_vals[i_max])
                s_refined = np.linspace(s_low, s_high, n_target)
                coeffs = _compute_laplace_samples(func, x_min, x_max, s_refined)
        indices = np.arange(len(coeffs))
        return indices, coeffs, "i", "L(s_i)"

    if kind == TransformKind.HILBERT:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_vals = fft.fft(y)
        n = len(fft_vals)
        h = np.zeros(n, dtype=complex)
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1
        analytic_fft = fft_vals * h
        coeffs = np.abs(analytic_fft[: n // 2])
        freqs = fft.fftfreq(n, dx)[: n // 2]
        freqs = np.abs(freqs)
        n_target = n_points // 2
        f_low, f_high = float(freqs[0]), float(freqs[-1])
        max_amp = float(np.max(coeffs))
        if max_amp > 0 and hilbert_amp_threshold > 0:
            threshold = max_amp * hilbert_amp_threshold
            above = np.where(coeffs >= threshold)[0]
            if len(above) > 0:
                i_min, i_max = int(above[0]), int(above[-1])
                f_low, f_high = float(freqs[i_min]), float(freqs[i_max])
                f_span = max(f_high - f_low, 1.0 / (n_points * dx))
                n_refined = int(np.ceil(n_target / (f_span * dx)))
                n_refined = min(max(n_refined, n_points), 65536)
                y_padded = np.zeros(n_refined, dtype=complex)
                y_padded[:n_points] = y
                fft_ref = fft.fft(y_padded)
                n_r = len(fft_ref)
                h_ref = np.zeros(n_r, dtype=complex)
                h_ref[0] = 1
                h_ref[1 : (n_r + 1) // 2] = 2
                if n_r % 2 == 0:
                    h_ref[n_r // 2] = 1
                analytic_fft_ref = fft_ref * h_ref
                coeffs_ref = np.abs(analytic_fft_ref[: n_r // 2])
                freqs_ref = fft.fftfreq(n_r, dx)[: n_r // 2]
                freqs_ref = np.abs(freqs_ref)
                mask = (freqs_ref >= f_low) & (freqs_ref <= f_high)
                if np.any(mask):
                    coeffs = coeffs_ref[mask]
                else:
                    coeffs = coeffs[i_min : i_max + 1]
        indices = np.arange(len(coeffs))
        return indices, coeffs, "k", "|H[k]|"

    if kind == TransformKind.Z_TRANSFORM:
        x, y = compute_function_samples(func, x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1) if n_points > 1 else 1.0
        fft_vals = fft.fft(y)
        fft_mag = np.abs(fft_vals[: n_points // 2])
        freqs = fft.fftfreq(n_points, dx)[: n_points // 2]
        freqs = np.abs(freqs)
        n_target = n_points // 2
        f_low, f_high = float(freqs[0]), float(freqs[-1])
        max_amp = float(np.max(fft_mag))
        if max_amp > 0 and z_transform_amp_threshold > 0:
            threshold = max_amp * z_transform_amp_threshold
            above = np.where(fft_mag >= threshold)[0]
            if len(above) > 0:
                i_min, i_max = int(above[0]), int(above[-1])
                f_low, f_high = float(freqs[i_min]), float(freqs[i_max])
                f_span = max(f_high - f_low, 1.0 / (n_points * dx))
                n_refined = int(np.ceil(n_target / (f_span * dx)))
                n_refined = min(max(n_refined, n_points), 65536)
                y_padded = np.zeros(n_refined, dtype=complex)
                y_padded[:n_points] = y
                fft_ref = fft.fft(y_padded)
                fft_mag_ref = np.abs(fft_ref[: n_refined // 2])
                freqs_ref = fft.fftfreq(n_refined, dx)[: n_refined // 2]
                freqs_ref = np.abs(freqs_ref)
                mask = (freqs_ref >= f_low) & (freqs_ref <= f_high)
                if np.any(mask):
                    coeffs = fft_mag_ref[mask]
                else:
                    coeffs = fft_mag[i_min : i_max + 1]
            else:
                coeffs = fft_mag
        else:
            coeffs = fft_mag
        indices = np.arange(len(coeffs))
        return indices, coeffs, "k", "|X[k]|"

    raise ValueError(f"Unknown transform kind: {kind}")
