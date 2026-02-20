"""Tests for config.env."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.env import (
    _ENV_SCHEMA_BY_KEY,
    ENV_SCHEMA,
    _validate_env_value,
    get_current_env_values,
    get_env,
    get_env_from_schema,
    write_env_file,
)


class TestGetEnv:
    def test_missing_key_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        assert get_env("NONEXISTENT_KEY", "default", str) == "default"

    def test_known_key_from_schema_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert get_env_from_schema("LOG_LEVEL") == "DEBUG"

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown env key"):
            get_env_from_schema("UNKNOWN_KEY_XYZ")

    def test_bool_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_BOOL", "true")
        assert get_env("SOME_BOOL", False, bool) is True
        monkeypatch.setenv("SOME_BOOL", "false")
        assert get_env("SOME_BOOL", True, bool) is False


class TestValidateEnvValue:
    def test_none_returns_false_and_default(self) -> None:
        schema = {"default": 100, "cast_type": int}
        valid, value = _validate_env_value("SOLVER_NUM_POINTS", None, schema)
        assert valid is False
        assert value == 100

    def test_log_level_invalid_option(self) -> None:
        schema = _ENV_SCHEMA_BY_KEY["LOG_LEVEL"]
        valid, value = _validate_env_value("LOG_LEVEL", "INVALID", schema)
        assert valid is False
        assert value == schema["default"]

    def test_log_level_valid_uppercase(self) -> None:
        schema = _ENV_SCHEMA_BY_KEY["LOG_LEVEL"]
        valid, value = _validate_env_value("LOG_LEVEL", "info", schema)
        assert valid is True
        assert value == "INFO"


class TestGetCurrentEnvValues:
    def test_returns_dict_of_strings(self) -> None:
        result = get_current_env_values()
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, str)


class TestWriteEnvFile:
    def test_writes_file(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        values = {item["key"]: str(item["default"]) for item in ENV_SCHEMA[:3]}
        write_env_file(env_path, values)
        content = env_path.read_text(encoding="utf-8")
        assert "DifferentialLab" in content or "Configuration" in content
        for key in values:
            assert f"{key}=" in content
