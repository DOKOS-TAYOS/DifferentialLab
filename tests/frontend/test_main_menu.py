"""Tests for the main menu restart behavior."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import frontend.ui_main_menu as ui_main_menu


def test_restart_executes_installed_console_launcher_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed console launchers restart directly with their arguments intact."""
    launcher = r"C:\QA á\outside-run\DifferentialLab QA\Scripts\differential-lab.exe"
    python_executable = r"C:\Python 3.12\python.exe"
    arguments = [launcher, "--profile", "test profile"]
    monkeypatch.setattr(ui_main_menu.sys, "executable", python_executable)
    monkeypatch.setattr(ui_main_menu.sys, "argv", arguments)

    expected_command = arguments
    assert ui_main_menu._build_restart_command() == expected_command

    with patch.object(ui_main_menu.subprocess, "Popen") as popen:
        ui_main_menu._restart_application()

    popen.assert_called_once_with(expected_command, shell=False)
    assert popen.call_args.args[0][0] == launcher
    assert python_executable not in popen.call_args.args[0]


def test_restart_preserves_direct_source_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct source launches continue to restart through main_program.py."""
    source_script = r"C:\DifferentialLab source á\src\main_program.py"
    python_executable = r"C:\Python 3.12\python.exe"
    arguments = [source_script, "--profile", "test profile"]
    monkeypatch.setattr(ui_main_menu.sys, "executable", python_executable)
    monkeypatch.setattr(ui_main_menu.sys, "argv", arguments)

    expected_command = [python_executable, source_script, *arguments[1:]]
    with patch.object(ui_main_menu.subprocess, "Popen") as popen:
        ui_main_menu._restart_application()

    popen.assert_called_once_with(expected_command, shell=False)


def test_config_keeps_window_open_when_restart_fails() -> None:
    """A failed process creation must leave the current application running."""
    menu = ui_main_menu.MainMenu.__new__(ui_main_menu.MainMenu)
    menu.root = Mock()

    with (
        patch("frontend.ui_dialogs.ConfigDialog") as config_dialog,
        patch.object(ui_main_menu, "_restart_application", side_effect=OSError("not found")),
        patch.object(ui_main_menu.messagebox, "showerror") as showerror,
    ):
        dialog = config_dialog.return_value
        dialog.accepted = True
        menu._on_config()

    menu.root.destroy.assert_not_called()
    showerror.assert_called_once()


def test_config_destroys_window_after_restart_starts() -> None:
    """The current application closes only after spawning its replacement."""
    menu = ui_main_menu.MainMenu.__new__(ui_main_menu.MainMenu)
    menu.root = Mock()
    events: list[str] = []
    menu.root.destroy.side_effect = lambda: events.append("destroy")

    with (
        patch("frontend.ui_dialogs.ConfigDialog") as config_dialog,
        patch.object(
            ui_main_menu,
            "_restart_application",
            side_effect=lambda: events.append("restart"),
        ) as restart,
    ):
        dialog = config_dialog.return_value
        dialog.accepted = True
        menu._on_config()

    restart.assert_called_once_with()
    menu.root.destroy.assert_called_once_with()
    assert events == ["restart", "destroy"]
