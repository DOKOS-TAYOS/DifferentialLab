"""Tests for frontend.theme color helpers."""

from __future__ import annotations

from frontend.theme import (
    _color_to_rgb,
    _darken_color,
    _lighten_color,
    get_select_colors,
)


class TestColorToRgb:
    def test_empty_returns_none(self) -> None:
        assert _color_to_rgb("") is None
        assert _color_to_rgb("   ") is None

    def test_hex_6_digits(self) -> None:
        assert _color_to_rgb("#ffffff") == (255, 255, 255)
        assert _color_to_rgb("#000000") == (0, 0, 0)
        assert _color_to_rgb("#ff0000") == (255, 0, 0)

    def test_hex_3_digits(self) -> None:
        assert _color_to_rgb("#fff") == (255, 255, 255)
        assert _color_to_rgb("#f00") == (255, 0, 0)

    def test_invalid_hex_returns_none(self) -> None:
        assert _color_to_rgb("#gggggg") is None
        assert _color_to_rgb("#12") is None


class TestLightenColor:
    def test_white_unchanged(self) -> None:
        result = _lighten_color("#ffffff", factor=0.2)
        assert result == "#ffffff"

    def test_black_lightened(self) -> None:
        result = _lighten_color("#000000", factor=0.2)
        r, g, b = int(result[1:3], 16), int(result[3:5], 16), int(result[5:7], 16)
        assert r == g == b
        assert r > 0

    def test_invalid_returns_white(self) -> None:
        assert _lighten_color("notacolor", factor=0.2) == "#ffffff"


class TestDarkenColor:
    def test_black_unchanged(self) -> None:
        result = _darken_color("#000000", factor=0.25)
        assert result == "#000000"

    def test_white_darkened(self) -> None:
        result = _darken_color("#ffffff", factor=0.25)
        r, g, b = int(result[1:3], 16), int(result[3:5], 16), int(result[5:7], 16)
        assert r == g == b
        assert r < 255

    def test_invalid_returns_dark(self) -> None:
        result = _darken_color("notacolor", factor=0.25)
        assert result == "#1e1e1e"


class TestGetSelectColors:
    def test_returns_tuple_of_two_hex(self) -> None:
        bg, fg = get_select_colors("#1F1F1F", "#CCCCCC")
        assert bg.startswith("#") and len(bg) == 7
        assert fg.startswith("#") and len(fg) == 7

    def test_select_bg_darker_than_element(self) -> None:
        element_bg = "#808080"
        select_bg, _ = get_select_colors(element_bg, "#ffffff")
        r = int(select_bg[1:3], 16)
        assert r < 0x80
