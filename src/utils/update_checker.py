"""Update checker for DifferentialLab.

Checks weekly if a newer version is available in the repository.
If so, shows a dialog (when enabled via env) and can perform git pull
without overwriting user data (input/, output/, .env, etc.).
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import get_env_from_schema, get_project_root
from utils.logger import get_logger

logger = get_logger(__name__)

_LAST_CHECK_FILE = ".last_update_check"
_UPDATE_CHECK_TIMEOUT = 10
_VERSION_RE = re.compile(r"^(\d+(?:\.\d+)*)")
_PYPROJECT_VERSION_RE = re.compile(r'version\s*=\s*["\']([^"\']+)["\']')
_PROTECTED_UPDATE_PATHS = ("input", "output")
_AUTO_STASH_MESSAGE = "DifferentialLab auto-update backup"


def _run_git(root: Path, args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a git command in project root and capture text output."""
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _normalize_git_path(path: str) -> str:
    """Normalize git-reported paths to forward-slash relative paths."""
    normalized = path.strip().strip('"').replace("\\", "/")
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _extract_paths_from_porcelain(status_output: str) -> list[str]:
    """Extract changed paths from `git status --porcelain` output."""
    paths: list[str] = []
    seen: set[str] = set()
    for raw_line in status_output.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        payload = line[3:] if len(line) > 3 else line
        candidates = payload.split(" -> ") if " -> " in payload else [payload]
        for candidate in candidates:
            path = _normalize_git_path(candidate)
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _is_path_protected(path: str) -> bool:
    """Return True when path belongs to protected user-data folders."""
    normalized = _normalize_git_path(path)
    return any(
        normalized == protected or normalized.startswith(f"{protected}/")
        for protected in _PROTECTED_UPDATE_PATHS
    )


def _format_path_list(paths: list[str], limit: int = 8) -> str:
    """Format path list for user-facing messages."""
    clipped = paths[:limit]
    lines = [f"- {path}" for path in clipped]
    if len(paths) > limit:
        lines.append(f"- ... ({len(paths) - limit} more)")
    return "\n".join(lines)


def _git_error_text(result: subprocess.CompletedProcess[str]) -> str:
    """Return best available error text from a completed git command."""
    return (result.stderr or result.stdout or "").strip()


def _get_conflicted_paths(root: Path) -> list[str]:
    """Return conflicted paths after merge/stash operations."""
    result = _run_git(root, ["diff", "--name-only", "--diff-filter=U"], timeout=20)
    if result.returncode != 0:
        return []
    return [
        _normalize_git_path(line)
        for line in (result.stdout or "").splitlines()
        if _normalize_git_path(line)
    ]


def _get_stash_head(root: Path) -> str | None:
    """Return hash at refs/stash, or None when no stash exists."""
    result = _run_git(root, ["rev-parse", "--verify", "-q", "refs/stash"], timeout=20)
    if result.returncode != 0:
        return None
    head = (result.stdout or "").strip()
    return head or None


def _get_last_check_path() -> Path:
    """Return path to the file storing last update check timestamp.

    Returns:
        Path to ``.last_update_check`` in project root.
    """
    root = get_project_root()
    return root / _LAST_CHECK_FILE


def should_run_check() -> bool:
    """Return True if we should run the update check (once per week).

    Returns:
        True if enough time has passed since last check, or no previous check.
    """
    if not get_env_from_schema("CHECK_UPDATES"):
        return False

    if get_env_from_schema("CHECK_UPDATES_FORCE"):
        return True

    path = _get_last_check_path()
    if not path.exists():
        return True

    try:
        mtime = path.stat().st_mtime
        elapsed_days = (time.time() - mtime) / (24 * 3600)
        days_between: int = get_env_from_schema("UPDATE_CHECK_INTERVAL_DAYS")
        return elapsed_days >= days_between
    except OSError as exc:
        logger.debug("Could not stat last-check file, assuming check needed: %s", exc)
        return True


def record_check_done() -> None:
    """Record that an update check was performed (touch the file).

    Updates the modification time of ``.last_update_check`` so the next
    check is deferred by UPDATE_CHECK_INTERVAL_DAYS.
    """
    path = _get_last_check_path()
    try:
        path.touch()
    except OSError as exc:
        logger.debug("Could not touch last-check file: %s", exc)


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '1.0.0' or '1.2.3.dev1' into a comparable tuple.

    Args:
        version_str: Version string from pyproject.toml.

    Returns:
        Tuple of integers for comparison (e.g. (0, 2, 0)).
    """
    match = _VERSION_RE.match(str(version_str).strip())
    if not match:
        return (0,)
    return tuple(int(x) for x in match.group(1).split("."))


def _fetch_latest_version(version_url: str | None = None) -> str | None:
    """Fetch the latest version from the remote pyproject.toml.

    Args:
        version_url: URL to pyproject.toml. If None, uses env UPDATE_CHECK_URL
            or default.

    Returns:
        Version string (e.g. '0.4.1') or None if fetch failed.
    """
    url = version_url or get_env_from_schema("UPDATE_CHECK_URL")

    try:
        req = Request(url, headers={"User-Agent": "DifferentialLab-UpdateChecker/1.0"})
        with urlopen(req, timeout=_UPDATE_CHECK_TIMEOUT) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, OSError, ValueError) as e:
        logger.debug("Update check: could not fetch version from %s: %s", url, e)
        return None

    match = _PYPROJECT_VERSION_RE.search(content)
    if match:
        return match.group(1).strip()
    return None


def is_update_available(current_version: str) -> str | None:
    """Check if a newer version is available.

    Args:
        current_version: Current application version.

    Returns:
        The latest version string if newer, else None.
    """
    latest = _fetch_latest_version()
    if not latest:
        return None

    current_tuple = _parse_version(current_version)
    latest_tuple = _parse_version(latest)

    if latest_tuple > current_tuple:
        return latest
    return None


def perform_git_pull() -> tuple[bool, str]:
    """Perform git pull in the project root.

    Strategy:
    1. Preflight check the working tree. If files outside input/output are dirty,
       abort and ask the user to commit/stash first.
    2. Stash local input/output changes.
    3. Pull updates.
    4. Re-apply stashed changes and validate conflicts explicitly.

    Returns:
        Tuple of (success, message). Message is user-friendly.
    """
    root = get_project_root()
    git_dir = root / ".git"
    if not git_dir.exists() or not git_dir.is_dir():
        return False, "This directory is not a git repository. Update manually."

    try:
        preflight = _run_git(
            root,
            ["status", "--porcelain", "--untracked-files=all"],
            timeout=20,
        )
        if preflight.returncode != 0:
            detail = _git_error_text(preflight)
            return False, detail or "Could not inspect repository status before update."

        dirty_paths = _extract_paths_from_porcelain(preflight.stdout or "")
        blocking_paths = [path for path in dirty_paths if not _is_path_protected(path)]
        if blocking_paths:
            return False, (
                "Update canceled: local changes were detected outside input/ and output/.\n"
                "Commit, stash, or discard these changes and retry.\n\n"
                "Changed paths:\n"
                f"{_format_path_list(blocking_paths)}"
            )

        stash_head_before = _get_stash_head(root)

        stash_result = _run_git(
            root,
            ["stash", "push", "-u", "-m", _AUTO_STASH_MESSAGE, "--", "input/", "output/"],
            timeout=30,
        )
        if stash_result.returncode != 0:
            detail = _git_error_text(stash_result)
            return False, (
                "Could not protect local input/output files before updating."
                f"{f' Git said: {detail}' if detail else ''}"
            )

        stash_head_after = _get_stash_head(root)
        stash_created = stash_head_after is not None and stash_head_after != stash_head_before

        pull_result = _run_git(root, ["pull", "--no-edit"], timeout=60)

        restore_error: str | None = None
        if stash_created:
            pop_result = _run_git(root, ["stash", "pop"], timeout=30)
            if pop_result.returncode != 0:
                conflicted_paths = _get_conflicted_paths(root)
                if conflicted_paths:
                    restore_error = (
                        "Repository update completed, but restoring local input/output files "
                        "caused merge conflicts.\n"
                        "Conflicted paths:\n"
                        f"{_format_path_list(conflicted_paths)}\n"
                        "Resolve conflicts manually and review your local data."
                    )
                else:
                    detail = _git_error_text(pop_result)
                    restore_error = (
                        "Repository update completed, but stashed input/output files could not be "
                        "restored automatically.\n"
                        "Run `git stash list` and restore manually after reviewing the tree."
                        f"{f' Git said: {detail}' if detail else ''}"
                    )

        if pull_result.returncode == 0 and not restore_error:
            return True, (
                "Update completed successfully. Restart the application to use the new version."
            )

        pull_error = _git_error_text(pull_result)
        if pull_result.returncode != 0:
            base_error = pull_error or "Update failed. Check your connection and try again."
            if restore_error:
                return False, f"{base_error}\n\nAlso, {restore_error}"
            return False, base_error

        # Pull succeeded but restoring local files failed/conflicted.
        return False, restore_error or "Update completed with unresolved local restore issues."
    except subprocess.TimeoutExpired:
        logger.warning("Git pull timed out")
        return False, "Update timed out. Try again later."
    except FileNotFoundError:
        logger.warning("Git not found in PATH")
        return False, "Git not found. Install Git to enable automatic updates."
    except Exception as e:
        logger.exception("Git pull failed")
        return False, str(e)
