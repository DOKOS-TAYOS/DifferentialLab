"""Application logging setup."""

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def _setup_root_logger() -> None:
    """Configure the root ``ode_solver`` logger once from env settings."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    from config.env import get_env, DEFAULT_LOG_LEVEL, DEFAULT_LOG_FILE

    level_name: str = get_env("LOG_LEVEL", DEFAULT_LOG_LEVEL, str)
    log_file: str = get_env("LOG_FILE", DEFAULT_LOG_FILE, str)
    console: bool = get_env("LOG_CONSOLE", False, bool)

    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger("ode_solver")
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        project_root = Path(__file__).resolve().parent.parent.parent
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
    """Return a child logger under the ``ode_solver`` namespace.

    Args:
        name: Module name (typically ``__name__``).

    Returns:
        A ``logging.Logger`` instance.
    """
    _setup_root_logger()
    if name.startswith("ode_solver."):
        return logging.getLogger(name)
    return logging.getLogger(f"ode_solver.{name}")
