"""Tests for utils.update_checker."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from utils.update_checker import (
    _parse_version,
    perform_git_pull,
    record_check_done,
    should_run_check,
)


class TestParseVersion:
    def test_simple_version(self) -> None:
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_single_component(self) -> None:
        assert _parse_version("5") == (5,)

    def test_with_dev_suffix(self) -> None:
        assert _parse_version("0.3.2.dev1") == (0, 3, 2)

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
    def test_touches_file(self) -> None:
        check_file = Path("C:/repo/.last_update_check")
        with (
            patch("utils.update_checker._get_last_check_path", return_value=check_file),
            patch("pathlib.Path.touch") as touch_mock,
        ):
            record_check_done()
        touch_mock.assert_called_once_with()


class TestPerformGitPull:
    def test_blocks_when_non_protected_paths_are_dirty(self) -> None:
        root = Path("C:/repo")
        status_result = subprocess.CompletedProcess(
            args=["git", "status"],
            returncode=0,
            stdout=" M src/main_program.py\n?? input/new_case.txt\n",
            stderr="",
        )

        with (
            patch("utils.update_checker.get_project_root", return_value=root),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("utils.update_checker.subprocess.run", return_value=status_result) as run_mock,
        ):
            success, msg = perform_git_pull()

        assert success is False
        assert "outside input/ and output/" in msg
        assert "src/main_program.py" in msg
        assert run_mock.call_count == 1

    def test_reports_conflicts_when_stash_pop_fails(self) -> None:
        root = Path("C:/repo")
        responses = [
            subprocess.CompletedProcess(
                args=["git", "status"],
                returncode=0,
                stdout="?? input/new_case.txt\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "rev-parse", "--verify", "-q", "refs/stash"],
                returncode=1,
                stdout="",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "stash", "push"],
                returncode=0,
                stdout="Saved working directory and index state WIP on main: abc123 test\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "rev-parse", "--verify", "-q", "refs/stash"],
                returncode=0,
                stdout="0123456789abcdef\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "pull"],
                returncode=0,
                stdout="Already up to date.\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "stash", "pop"],
                returncode=1,
                stdout="CONFLICT (content): Merge conflict in input/new_case.txt\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=0,
                stdout="input/new_case.txt\n",
                stderr="",
            ),
        ]

        with (
            patch("utils.update_checker.get_project_root", return_value=root),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("utils.update_checker.subprocess.run", side_effect=responses),
        ):
            success, msg = perform_git_pull()

        assert success is False
        assert "caused merge conflicts" in msg
        assert "input/new_case.txt" in msg
