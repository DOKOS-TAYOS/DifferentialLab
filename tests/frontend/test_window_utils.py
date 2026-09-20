"""Unit tests for viewport-based label wrapping helpers."""

from __future__ import annotations

from frontend.window_utils import calculate_wraplength


def test_calculate_wraplength_reserves_horizontal_padding() -> None:
    assert calculate_wraplength(900, 48, 200) == 852


def test_calculate_wraplength_preserves_a_smallest_safe_width() -> None:
    assert calculate_wraplength(220, 48, 200) == 200
