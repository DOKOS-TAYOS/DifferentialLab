"""Release metadata consistency tests."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from config import APP_VERSION

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    """Read the canonical project version from pyproject metadata."""
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def _citation_version() -> str:
    """Read the static release version required by CITATION.cff."""
    citation = (_PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(.+)$", citation, flags=re.MULTILINE)
    assert match is not None
    return match.group(1).strip().strip('"')


def test_runtime_and_citation_versions_match_project_metadata() -> None:
    """Keep runtime metadata and unavoidable citation metadata synchronized."""
    project_version = _project_version()
    assert APP_VERSION == project_version
    assert _citation_version() == project_version
