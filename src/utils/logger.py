"""Application logging setup.

Configures a hierarchical logger under the ``differential_lab`` namespace.
Log level, file output, and console output are controlled via environment
variables (LOG_LEVEL, LOG_FILE, LOG_CONSOLE).
"""

import logging
import sys

_CONFIGURED = False
_LOGGER_NS = "differential_lab"


def _setup_root_logger() -> None:
    """Configure the root ``differential_lab`` logger once from env settings.

    Idempotent: subsequent calls have no effect after the first configuration.
    Uses LOG_LEVEL, LOG_FILE, and LOG_CONSOLE from environment.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    from config import DEFAULT_LOG_FILE, DEFAULT_LOG_LEVEL, get_env, get_project_root

    level_name: str = get_env("LOG_LEVEL", DEFAULT_LOG_LEVEL, str)
    log_file: str = get_env("LOG_FILE", DEFAULT_LOG_FILE, str)
    console: bool = get_env("LOG_CONSOLE", False, bool)

    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger(_LOGGER_NS)
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        project_root = get_project_root()
        fh = logging.FileHandler(project_root / log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``differential_lab`` namespace.

    Args:
        name: Module name (typically ``__name__``).

    Returns:
        A ``logging.Logger`` instance.
    """
    _setup_root_logger()
    qualified = name if name.startswith(f"{_LOGGER_NS}.") else f"{_LOGGER_NS}.{name}"
    return logging.getLogger(qualified)
