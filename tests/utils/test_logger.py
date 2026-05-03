"""Tests for utils.logger."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import ModuleType

import pytest

import config
import config.env as env_module


@pytest.fixture(autouse=True)
def reset_logger_state() -> Iterator[None]:
    """Keep the differential_lab logger isolated across tests."""
    root = logging.getLogger("differential_lab")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)
    root.propagate = True

    logger_module = sys.modules.get("utils.logger")
    if logger_module is not None:
        setattr(logger_module, "_CONFIGURED", False)

    env_module._VALIDATED_CACHE.clear()
    yield

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)
    root.propagate = True

    logger_module = sys.modules.get("utils.logger")
    if logger_module is not None:
        setattr(logger_module, "_CONFIGURED", False)

    env_module._VALIDATED_CACHE.clear()


def _load_logger_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> ModuleType:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "bootstrap.log"))
    monkeypatch.setenv("LOG_CONSOLE", "false")
    env_module._VALIDATED_CACHE.clear()
    return importlib.reload(importlib.import_module("utils.logger"))


def test_get_logger_returns_namespaced_logger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_module = _load_logger_module(monkeypatch, tmp_path)

    logger = logger_module.get_logger("solver.ode_solver")

    assert logger.name == "differential_lab.solver.ode_solver"


def test_setup_uses_rotating_file_handler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_module = _load_logger_module(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_MAX_BYTES", "2048")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "5")

    root = logger_module._configure_root_logger(force=True)

    file_handlers = [
        handler for handler in root.handlers if isinstance(handler, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    assert file_handlers[0].maxBytes == 2048
    assert file_handlers[0].backupCount == 5


def test_setup_creates_parent_directories_for_log_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_module = _load_logger_module(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_FILE", str(Path("logs") / "nested" / "app.log"))
    monkeypatch.setattr(config, "get_project_root", lambda: tmp_path)

    logger_module._configure_root_logger(force=True)

    assert (tmp_path / "logs" / "nested" / "app.log").exists()


def test_setup_does_not_duplicate_handlers_on_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_module = _load_logger_module(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_CONSOLE", "true")

    first_root = logger_module._configure_root_logger(force=True)
    first_count = len(first_root.handlers)
    second_root = logger_module._configure_root_logger(force=True)

    assert first_count == 2
    assert len(second_root.handlers) == 2


def test_setup_falls_back_to_console_when_file_handler_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_module = _load_logger_module(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_CONSOLE", "false")

    def raise_permission_error(*args: object, **kwargs: object) -> RotatingFileHandler:
        raise PermissionError("denied")

    monkeypatch.setattr(logger_module, "RotatingFileHandler", raise_permission_error)

    root = logger_module._configure_root_logger(force=True)

    assert len(root.handlers) == 1
    assert type(root.handlers[0]) is logging.StreamHandler


def test_reset_logging_state_removes_handlers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_module = _load_logger_module(monkeypatch, tmp_path)
    logger_module._configure_root_logger(force=True)

    assert logging.getLogger("differential_lab").handlers

    logger_module._reset_logging_state()

    assert logger_module._CONFIGURED is False
    assert logging.getLogger("differential_lab").handlers == []
