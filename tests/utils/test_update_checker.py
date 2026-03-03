"""Tests for utils.update_checker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from utils.update_checker import (
    _parse_version,
    record_check_done,
    should_run_check,
)


class TestParseVersion:
    def test_simple_version(self) -> None:
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_single_component(self) -> None:
        assert _parse_version("5") == (5,)

    def test_with_dev_suffix(self) -> None:
        assert _parse_version("0.3.1.dev1") == (0, 3, 1)

    def test_with_whitespace(self) -> None:
        assert _parse_version("  2.0.0  ") == (2, 0, 0)

    def test_invalid_returns_zero(self) -> None:
        assert _parse_version("abc") == (0,)
        assert _parse_version("") == (0,)


class TestShouldRunCheck:
    def test_disabled_returns_false(self) -> None:
        with patch("utils.update_checker.get_env_from_schema") as mock:
            mock.side_effect = lambda k: {
                "CHECK_UPDATES": False,
                "CHECK_UPDATES_FORCE": False,
                "UPDATE_CHECK_INTERVAL_DAYS": 7,
            }.get(k, None)
            assert should_run_check() is False

    def test_force_returns_true(self) -> None:
        with patch("utils.update_checker.get_env_from_schema") as mock:
            def env_effect(k: str) -> object:
                if k == "CHECK_UPDATES":
                    return "true"
                if k == "CHECK_UPDATES_FORCE":
                    return "true"
                if k == "UPDATE_CHECK_INTERVAL_DAYS":
                    return 7
                return None
            mock.side_effect = env_effect
            assert should_run_check() is True

    def test_no_file_returns_true(self) -> None:
        def env_effect(k: str) -> object:
            if k == "CHECK_UPDATES":
                return "true"
            if k == "CHECK_UPDATES_FORCE":
                return "false"
            if k == "UPDATE_CHECK_INTERVAL_DAYS":
                return 7
            return None

        with (
            patch("utils.update_checker.get_env_from_schema") as mock_env,
            patch("utils.update_checker._get_last_check_path") as mock_path,
        ):
            mock_env.side_effect = env_effect
            mock_path.return_value = Path("/nonexistent/.last_update_check")
            assert should_run_check() is True


class TestRecordCheckDone:
    def test_touches_file(self, tmp_path: Path) -> None:
        with patch("utils.update_checker.get_project_root", return_value=tmp_path):
            record_check_done()
            check_file = tmp_path / ".last_update_check"
            assert check_file.exists()
