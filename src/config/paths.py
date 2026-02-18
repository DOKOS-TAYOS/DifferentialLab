"""File path management for output files."""

from datetime import datetime
from pathlib import Path

from config.env import get_env_from_schema

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_output_dir() -> Path:
    """Return the absolute output directory, creating it if needed.

    Returns:
        Path to the output directory.
    """
    rel: str = get_env_from_schema("FILE_OUTPUT_DIR")
    out = _PROJECT_ROOT / rel
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_project_root() -> Path:
    """Return the project root directory.

    Returns:
        Absolute path to the project root.
    """
    return _PROJECT_ROOT


def generate_output_basename(prefix: str = "solution") -> str:
    """Generate a timestamped base filename.

    Args:
        prefix: Filename prefix.

    Returns:
        String like ``solution_20260218_143022``.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def get_plot_path(basename: str) -> Path:
    """Return the full path for a plot file.

    Args:
        basename: Base filename (without extension).

    Returns:
        Full path with the configured plot format extension.
    """
    fmt: str = get_env_from_schema("FILE_PLOT_FORMAT")
    return get_output_dir() / f"{basename}.{fmt}"


def get_csv_path(basename: str) -> Path:
    """Return the full path for a CSV file.

    Args:
        basename: Base filename (without extension).

    Returns:
        Full path with ``.csv`` extension.
    """
    return get_output_dir() / f"{basename}.csv"


def get_json_path(basename: str) -> Path:
    """Return the full path for a JSON file.

    Args:
        basename: Base filename (without extension).

    Returns:
        Full path with ``.json`` extension.
    """
    return get_output_dir() / f"{basename}.json"


def get_env_path() -> Path:
    """Return the path to the ``.env`` file.

    Returns:
        Absolute path to ``.env`` in the project root.
    """
    return _PROJECT_ROOT / ".env"
