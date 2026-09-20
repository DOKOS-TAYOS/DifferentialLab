"""Unit tests for keyboard-accessible collapsible Help sections."""

from __future__ import annotations

from typing import Any

from frontend.ui_dialogs import collapsible_section as sections


class _FakeWidget:
    def __init__(self, *_args: object, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.bindings: dict[str, tuple[Any, str | None]] = {}
        self.configurations: list[dict[str, Any]] = []
        self.pack_calls: list[dict[str, Any]] = []
        self.visible = False
        self.after_calls: list[tuple[int, Any]] = []

    def pack(self, **kwargs: Any) -> None:
        self.pack_calls.append(kwargs)
        self.visible = True

    def pack_forget(self) -> None:
        self.visible = False

    def configure(self, **kwargs: Any) -> None:
        self.configurations.append(kwargs)

    def bind(self, sequence: str, callback: Any, add: str | None = None) -> None:
        self.bindings[sequence] = (callback, add)

    def after(self, delay: int, callback: Any) -> None:
        self.after_calls.append((delay, callback))


class _FakeScroll:
    def __init__(self) -> None:
        self.bind_calls = 0
        self.refresh_calls = 0

    def bind_new_children(self) -> None:
        self.bind_calls += 1

    def refresh_scroll_region(self) -> None:
        self.refresh_calls += 1


def _make_section(monkeypatch: Any, *, expanded: bool = False) -> sections.CollapsibleSection:
    monkeypatch.setattr(sections.ttk, "Frame", _FakeWidget)
    monkeypatch.setattr(sections.ttk, "Button", _FakeWidget)
    return sections.CollapsibleSection(_FakeWidget(), _FakeScroll(), "About", expanded=expanded)


def test_header_is_focusable_and_binds_explicit_keyboard_activation(monkeypatch: Any) -> None:
    section = _make_section(monkeypatch)

    assert section._header.kwargs["takefocus"] is True
    assert section._header.kwargs["style"] == "SectionHeader.TButton"
    assert set(section._header.bindings) == {"<Return>", "<KP_Enter>", "<space>"}
    assert all(add == "+" for _, add in section._header.bindings.values())


def test_mouse_command_and_keyboard_activation_toggle_the_content(monkeypatch: Any) -> None:
    section = _make_section(monkeypatch)
    scroll = section._scroll

    command = section._header.kwargs["command"]
    command()
    assert section._expanded is True
    assert section.content.visible is True
    assert section._header.configurations[-1]["text"] == "▼  About"
    assert scroll.bind_calls == 1

    for sequence in ("<Return>", "<space>", "<KP_Enter>"):
        callback, _add = section._header.bindings[sequence]
        assert callback(None) == "break"

    assert section._expanded is False
    assert section.content.visible is False
    assert section._header.configurations[-1]["text"] == "▶  About"
    assert scroll.bind_calls == 2
    assert len(section._wrapper.after_calls) == 4
