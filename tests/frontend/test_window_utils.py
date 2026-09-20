"""Unit tests for viewport-based label wrapping helpers."""

from __future__ import annotations

from frontend.window_utils import (
    bind_wraplength,
    calculate_screen_aware_minsize,
    calculate_wraplength,
)


def test_calculate_wraplength_reserves_horizontal_padding() -> None:
    assert calculate_wraplength(900, 48, 200) == 852


def test_calculate_wraplength_preserves_a_smallest_safe_width() -> None:
    assert calculate_wraplength(220, 48, 200) == 200


class _FakeFrame:
    def __init__(self) -> None:
        self.bind_calls: list[tuple[str, object, str | None]] = []
        self.after_calls: list[tuple[int, object]] = []

    def bind(self, sequence: str, callback: object, add: str | None = None) -> None:
        self.bind_calls.append((sequence, callback, add))

    def after(self, delay: int, callback: object) -> str:
        self.after_calls.append((delay, callback))
        return "after-1"

    def winfo_width(self) -> int:
        return 900


class _FakeLabel:
    def winfo_exists(self) -> bool:
        return True

    def configure(self, **_kwargs: object) -> None:
        return None


def test_bind_wraplength_preserves_existing_configure_bindings() -> None:
    frame = _FakeFrame()
    bind_wraplength(frame, [_FakeLabel()], debounce_ms=0)  # type: ignore[arg-type]

    assert frame.bind_calls[0][0] == "<Configure>"
    assert frame.bind_calls[0][2] == "+"


def test_screen_aware_minsize_clamps_to_available_screen_fraction() -> None:
    assert calculate_screen_aware_minsize(640, 480, 700, 520) == (576, 432)
    assert calculate_screen_aware_minsize(1920, 1080, 700, 520) == (700, 520)
