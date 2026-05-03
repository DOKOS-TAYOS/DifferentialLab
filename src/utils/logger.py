"""Application logging setup.

Configures a hierarchical logger under the ``differential_lab`` namespace.
Log level, file output, rotation, and console output are controlled via
environment variables.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_LOGGER_NS = "differential_lab"


def _build_formatter() -> logging.Formatter:
    """Build the standard log formatter for the application."""
    return logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _remove_handlers(root: logging.Logger) -> None:
    """Detach and close all handlers from the given logger."""
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def _add_console_handler(
    root: logging.Logger,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler:
    """Attach a standard stderr console handler to the given logger."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return handler


def _resolve_log_path(log_file: str) -> Path:
    """Resolve the configured log-file path.

    Relative paths are resolved from the project root. Absolute paths are kept
    unchanged so tests and advanced users can point logs elsewhere.
    """
    path = Path(log_file).expanduser()
    if path.is_absolute():
        return path

    from config import get_project_root

    return get_project_root() / path


def _reset_logging_state() -> None:
    """Reset the differential_lab logger to an unconfigured state.

    This is primarily intended for tests so they can reconfigure logging in
    isolation without leaking handlers between test cases.
    """
    global _CONFIGURED

    root = logging.getLogger(_LOGGER_NS)
    _remove_handlers(root)
    root.setLevel(logging.NOTSET)
    root.propagate = True
    _CONFIGURED = False


def _configure_root_logger(*, force: bool = False) -> logging.Logger:
    """Configure the root ``differential_lab`` logger from env settings.

    Args:
        force: When ``True``, clear any previous configuration first. This is
            mainly useful in tests.

    Returns:
        The configured root logger.
    """
    global _CONFIGURED

    root = logging.getLogger(_LOGGER_NS)
    if _CONFIGURED and not force:
        return root

    if force:
        _reset_logging_state()
        root = logging.getLogger(_LOGGER_NS)
    else:
        _remove_handlers(root)

    from config import (
        DEFAULT_LOG_BACKUP_COUNT,
        DEFAULT_LOG_FILE,
        DEFAULT_LOG_LEVEL,
        DEFAULT_LOG_MAX_BYTES,
        get_env,
    )

    level_name: str = get_env("LOG_LEVEL", DEFAULT_LOG_LEVEL, str)
    log_file: str = get_env("LOG_FILE", DEFAULT_LOG_FILE, str)
    max_bytes: int = get_env("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
    backup_count: int = get_env("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)
    console_requested: bool = get_env("LOG_CONSOLE", False, bool)

    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = _build_formatter()
    root.setLevel(level)
    root.propagate = False

    file_error: tuple[Path, OSError] | None = None
    if log_file:
        log_path = _resolve_log_path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            file_error = (log_path, exc)

    if console_requested or file_error is not None or not root.handlers:
        _add_console_handler(root, level, formatter)

    if file_error is not None:
        root.warning(
            "File logging disabled for '%s': %s",
            str(file_error[0]),
            file_error[1],
        )

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``differential_lab`` namespace.

    Args:
        name: Module name (typically ``__name__``).

    Returns:
        A ``logging.Logger`` instance.
    """
    _configure_root_logger()
    qualified = name if name.startswith(f"{_LOGGER_NS}.") else f"{_LOGGER_NS}.{name}"
    return logging.getLogger(qualified)
