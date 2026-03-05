"""File path management for output files."""

from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_OUTPUT_DIR: str = "output"


def get_project_root() -> Path:
    """Return the absolute project root directory.

    Returns:
        Path to the project root (parent of src/).
    """
    return _PROJECT_ROOT


def get_output_dir() -> Path:
    """Return the absolute output directory, creating it if needed.

    Returns:
        Path to the output directory (always ``output/`` from project root).
    """
    out = _PROJECT_ROOT / _DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def generate_output_basename(prefix: str = "solution") -> str:
    """Generate a timestamped base filename.

    Args:
        prefix: Filename prefix.

    Returns:
        String like ``solution_20260218_143022``.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def get_csv_path(basename: str) -> Path:
    """Return the full path for a CSV file.

    Args:
        basename: Base filename (without extension).

    Returns:
        Full path with ``.csv`` extension.
    """
    return get_output_dir() / f"{basename}.csv"


def get_env_path() -> Path:
    """Return the path to the ``.env`` file.

    Returns:
        Absolute path to ``.env`` in the project root.
    """
    return _PROJECT_ROOT / ".env"
