"""Update checker for DifferentialLab.

Checks weekly if a newer version is available in the repository.
If so, shows a dialog (when enabled via env) and can perform git pull
without overwriting user data (input/, output/, .env, etc.).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import get_env, get_project_root

# Default URL to fetch latest version (pyproject.toml from main repo)
_DEFAULT_VERSION_URL = (
    "https://raw.githubusercontent.com/DOKOS-TAYOS/DifferentialLab/main/pyproject.toml"
)
_LAST_CHECK_FILE = ".last_update_check"
_DAYS_BETWEEN_CHECKS = 7


def _get_last_check_path() -> Path:
    """Return path to the file storing last update check timestamp."""
    return get_project_root() / _LAST_CHECK_FILE


def should_run_check() -> bool:
    """Return True if we should run the update check (once per week).

    Returns:
        True if enough time has passed since last check, or no previous check.
    """
    if not get_env("CHECK_UPDATES", True, bool):
        return False

    if get_env("CHECK_UPDATES_FORCE", False, bool):
        return True

    path = _get_last_check_path()
    if not path.exists():
        return True

    try:
        import time

        mtime = path.stat().st_mtime
        elapsed_days = (time.time() - mtime) / (24 * 3600)
        return elapsed_days >= _DAYS_BETWEEN_CHECKS
    except OSError:
        return True


def record_check_done() -> None:
    """Record that an update check was performed (touch the file)."""
    path = _get_last_check_path()
    try:
        path.touch()
    except OSError:
        pass


def _parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a version string like '1.0.0' or '1.2.3.dev1' into a comparable tuple.

    Args:
        version_str: Version string from pyproject.toml.

    Returns:
        Tuple of integers for comparison (e.g. (0, 2, 0)).
    """
    match = re.match(r"^(\d+(?:\.\d+)*)", str(version_str).strip())
    if not match:
        return (0,)
    parts = [int(x) for x in match.group(1).split(".")]
    return tuple(parts)


def fetch_latest_version(version_url: Optional[str] = None) -> Optional[str]:
    """Fetch the latest version from the remote pyproject.toml.

    Args:
        version_url: URL to pyproject.toml. If None, uses env UPDATE_CHECK_URL
            or default.

    Returns:
        Version string (e.g. '0.2.1') or None if fetch failed.
    """
    url = (
        version_url
        or get_env("UPDATE_CHECK_URL", _DEFAULT_VERSION_URL, str)
        or ""
    ).strip()
    if not url:
        url = _DEFAULT_VERSION_URL

    try:
        req = Request(url, headers={"User-Agent": "DifferentialLab-UpdateChecker/1.0"})
        with urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, OSError, ValueError) as e:
        try:
            from utils import get_logger

            get_logger(__name__).debug(
                "Update check: could not fetch version from %s: %s", url, e
            )
        except ImportError:
            pass
        return None

    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1).strip()
    return None


def is_update_available(current_version: str) -> Optional[str]:
    """Check if a newer version is available.

    Args:
        current_version: Current application version.

    Returns:
        The latest version string if newer, else None.
    """
    latest = fetch_latest_version()
    if not latest:
        return None

    current_tuple = _parse_version(current_version)
    latest_tuple = _parse_version(latest)

    if latest_tuple > current_tuple:
        return latest
    return None


def perform_git_pull() -> Tuple[bool, str]:
    """Perform git pull in the project root.

    Before pulling, stashes any local changes in input/ and output/
    to prevent them from being overwritten. After the pull completes,
    the stashed changes are restored. Files in .env and other .gitignore
    entries are automatically protected by git.

    Returns:
        Tuple of (success, message). Message is user-friendly.
    """
    root = get_project_root()
    git_dir = root / ".git"
    if not git_dir.exists() or not git_dir.is_dir():
        return False, "This directory is not a git repository. Update manually."

    try:
        stash_result = subprocess.run(
            ["git", "stash", "push", "-u", "--", "input/", "output/"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )

        result = subprocess.run(
            ["git", "pull", "--no-edit"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
        )

        if stash_result.returncode == 0 and "No local changes" not in (
            stash_result.stdout or ""
        ):
            subprocess.run(
                ["git", "stash", "pop"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=30,
            )

        if result.returncode == 0:
            return True, (
                "Update completed successfully. Restart the application to use the new version."
            )
        err = (result.stderr or result.stdout or "").strip()
        return False, err or "Update failed. Check your connection and try again."
    except subprocess.TimeoutExpired:
        return False, "Update timed out. Try again later."
    except FileNotFoundError:
        return False, "Git not found. Install Git to enable automatic updates."
    except Exception as e:
        return False, str(e)
