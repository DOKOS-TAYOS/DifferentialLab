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
    def __init__(self, width: int = 900) -> None:
        self.bind_calls: list[tuple[str, object, str | None]] = []
        self.after_calls: list[tuple[int, object]] = []
        self.width = width

    def bind(self, sequence: str, callback: object, add: str | None = None) -> None:
        self.bind_calls.append((sequence, callback, add))

    def after(self, delay: int, callback: object) -> str:
        self.after_calls.append((delay, callback))
        return "after-1"

    def winfo_width(self) -> int:
        return self.width


class _FakeLabel:
    def __init__(self) -> None:
        self.wraplength: int | None = None

    def winfo_exists(self) -> bool:
        return True

    def configure(self, **kwargs: object) -> None:
        value = kwargs.get("wraplength")
        self.wraplength = int(value) if value is not None else None


def test_bind_wraplength_preserves_existing_configure_bindings() -> None:
    frame = _FakeFrame()
    bind_wraplength(frame, [_FakeLabel()], debounce_ms=0)  # type: ignore[arg-type]

    assert frame.bind_calls[0][0] == "<Configure>"
    assert frame.bind_calls[0][2] == "+"


def test_screen_aware_minsize_clamps_to_available_screen_fraction() -> None:
    assert calculate_screen_aware_minsize(640, 480, 700, 520) == (576, 432)
    assert calculate_screen_aware_minsize(1920, 1080, 700, 520) == (700, 520)


def test_bind_wraplength_clamps_to_narrow_visible_width() -> None:
    frame = _FakeFrame(width=150)
    label = _FakeLabel()
    bind_wraplength(frame, [label], pad=48, min_wrap=200, debounce_ms=0)  # type: ignore[arg-type]

    callback = frame.bind_calls[0][1]
    assert callable(callback)
    callback()
    assert label.wraplength == 102
