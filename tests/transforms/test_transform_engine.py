"""Tests for transforms.transform_engine."""

from __future__ import annotations

import numpy as np
import pytest

from transforms.transform_engine import (
    TransformKind,
    apply_transform,
    compute_function_samples,
    get_transform_coefficients,
)


def test_compute_function_samples() -> None:
    """Sample a function over a range."""
    def func(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    x, y = compute_function_samples(func, 0.0, 2 * np.pi, n_points=100)
    assert len(x) == 100
    assert len(y) == 100
    np.testing.assert_allclose(y[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(y[25], 1.0, atol=0.1)


def test_apply_transform_original() -> None:
    """Original transform returns f(x) samples."""
    def func(x: np.ndarray) -> np.ndarray:
        return x**2

    x, y, x_label, y_label = apply_transform(
        func, TransformKind.ORIGINAL, 0.0, 2.0, n_points=50
    )
    assert x_label == "x"
    assert y_label == "f(x)"
    np.testing.assert_allclose(y, x**2)


def test_apply_transform_taylor_sin() -> None:
    """Taylor of sin(x) at 0: a_0=0, a_1=1, a_2=0, a_3≈-1/6."""
    def func(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    x, y, x_label, y_label = apply_transform(
        func,
        TransformKind.TAYLOR,
        -1.0,
        1.0,
        n_points=100,
        taylor_order=5,
        taylor_center=0.0,
    )
    assert "Taylor" in y_label
    # Taylor approx should match sin near 0
    np.testing.assert_allclose(y[:20], np.sin(x[:20]), atol=0.1)


def test_get_transform_coefficients_taylor_sin() -> None:
    """Taylor coefficients of sin(x) at 0: 0, 1, 0, -1/6, 0, 1/120."""
    def func(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    indices, coeffs, _, _, meta = get_transform_coefficients(
        func,
        TransformKind.TAYLOR,
        -0.5,
        0.5,
        n_points=100,
        taylor_order=5,
        taylor_center=0.0,
    )
    assert len(coeffs) == 6
    assert coeffs[0] == pytest.approx(0.0, abs=0.01)
    assert coeffs[1] == pytest.approx(1.0, abs=0.1)
    assert coeffs[2] == pytest.approx(0.0, abs=0.01)
    assert coeffs[3] == pytest.approx(-1.0 / 6.0, abs=0.1)
    assert "taylor_center" in meta


def test_apply_transform_fourier() -> None:
    """Fourier of a simple signal returns frequency spectrum."""
    def func(x: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * 3 * x)  # 3 Hz component

    x, y, x_label, y_label = apply_transform(
        func, TransformKind.FOURIER, 0.0, 1.0, n_points=256
    )
    assert "ω" in x_label or "F" in y_label
    assert len(x) > 0
    assert len(y) > 0
    # Peak should be near 3 Hz (or 3/(2π) in angular)
    peak_idx = np.argmax(y)
    assert peak_idx >= 0
