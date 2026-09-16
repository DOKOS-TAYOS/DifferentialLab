"""Tests for package version consistency."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from config import APP_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_version_matches_pyproject() -> None:
    """The runtime version must come from the project metadata."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert APP_VERSION == pyproject["project"]["version"]


def test_citation_version_matches_pyproject() -> None:
    """Static citation metadata must remain synchronized with project metadata."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", citation, re.MULTILINE)
    assert match is not None
    assert match.group(1) == pyproject["project"]["version"]
