"""Environment variable loading, schema definition, and validation."""

import os
from pathlib import Path
from typing import Any, Type, Union

from config.constants import (
    FONT_FAMILIES,
    FONT_SIZES,
    FONT_STYLES,
    FONT_WEIGHTS,
    LINE_STYLES,
    LOG_LEVELS,
    MARKER_FORMATS,
    PLOT_FORMATS,
    SOLVER_METHODS,
)

_EnvCastType = Type[Union[str, int, float, bool]]

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass


DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FILE: str = "differential_lab.log"

_SIZE_FIELDS: frozenset[str] = frozenset({
    "UI_PADDING",
    "UI_BUTTON_WIDTH",
    "UI_FONT_SIZE",
    "UI_ENTRY_WIDTH",
    "PLOT_FIGSIZE_WIDTH",
    "PLOT_FIGSIZE_HEIGHT",
    "PLOT_MARKER_SIZE",
    "FONT_AXIS_SIZE",
    "FONT_TICK_SIZE",
    "SOLVER_NUM_POINTS",
})

ENV_SCHEMA: list[dict[str, Any]] = [
    # --- ui: general ---
    {"key": "UI_BACKGROUND", "default": "#181818", "cast_type": str},
    {"key": "UI_FOREGROUND", "default": "#CCCCCC", "cast_type": str},
    # --- ui: buttons ---
    {"key": "UI_BUTTON_BG", "default": "#1F1F1F", "cast_type": str},
    {"key": "UI_BUTTON_WIDTH", "default": 14, "cast_type": int},
    {"key": "UI_BUTTON_FG", "default": "lime green", "cast_type": str},
    {"key": "UI_BUTTON_FG_CANCEL", "default": "red2", "cast_type": str},
    {"key": "UI_BUTTON_FG_ACCENT2", "default": "yellow", "cast_type": str},
    # --- ui: text / inputs ---
    {"key": "UI_TEXT_SELECT_BG", "default": "steel blue", "cast_type": str},
    {"key": "UI_FONT_SIZE", "default": 16, "cast_type": int},
    {"key": "UI_FONT_FAMILY", "default": "Bahnschrift", "cast_type": str},
    {"key": "UI_PADDING", "default": 8, "cast_type": int},
    {"key": "UI_ENTRY_WIDTH", "default": 25, "cast_type": int},
    # --- plot: size ---
    {"key": "PLOT_FIGSIZE_WIDTH", "default": 12, "cast_type": int},
    {"key": "PLOT_FIGSIZE_HEIGHT", "default": 6, "cast_type": int},
    {"key": "DPI", "default": 100, "cast_type": int},
    {"key": "PLOT_SHOW_TITLE", "default": True, "cast_type": bool},
    {"key": "PLOT_SHOW_GRID", "default": True, "cast_type": bool},
    # --- plot: line ---
    {"key": "PLOT_LINE_COLOR", "default": "royalblue", "cast_type": str},
    {"key": "PLOT_LINE_WIDTH", "default": 1.5, "cast_type": float},
    {"key": "PLOT_LINE_STYLE", "default": "-", "cast_type": str, "options": LINE_STYLES},
    # --- plot: markers ---
    {"key": "PLOT_MARKER_FORMAT", "default": "o", "cast_type": str, "options": MARKER_FORMATS},
    {"key": "PLOT_MARKER_SIZE", "default": 3, "cast_type": int},
    {"key": "PLOT_MARKER_FACE_COLOR", "default": "crimson", "cast_type": str},
    {"key": "PLOT_MARKER_EDGE_COLOR", "default": "crimson", "cast_type": str},
    # --- font (plots) ---
    {"key": "FONT_FAMILY", "default": "serif", "cast_type": str, "options": FONT_FAMILIES},
    {"key": "FONT_TITLE_SIZE", "default": "xx-large", "cast_type": str, "options": FONT_SIZES},
    {"key": "FONT_TITLE_WEIGHT", "default": "semibold", "cast_type": str, "options": FONT_WEIGHTS},
    {"key": "FONT_AXIS_SIZE", "default": 16, "cast_type": int},
    {"key": "FONT_AXIS_STYLE", "default": "italic", "cast_type": str, "options": FONT_STYLES},
    {"key": "FONT_TICK_SIZE", "default": 12, "cast_type": int},
    # --- solver ---
    {"key": "SOLVER_DEFAULT_METHOD", "default": "RK45", "cast_type": str, "options": SOLVER_METHODS},
    {"key": "SOLVER_MAX_STEP", "default": 0.0, "cast_type": float},
    {"key": "SOLVER_RTOL", "default": 1e-8, "cast_type": float},
    {"key": "SOLVER_ATOL", "default": 1e-10, "cast_type": float},
    {"key": "SOLVER_NUM_POINTS", "default": 1000, "cast_type": int},
    # --- paths ---
    {"key": "FILE_OUTPUT_DIR", "default": "output", "cast_type": str},
    {"key": "FILE_PLOT_FORMAT", "default": "png", "cast_type": str, "options": PLOT_FORMATS},
    # --- logging ---
    {"key": "LOG_LEVEL", "default": DEFAULT_LOG_LEVEL, "cast_type": str, "options": LOG_LEVELS},
    {"key": "LOG_FILE", "default": DEFAULT_LOG_FILE, "cast_type": str},
    {"key": "LOG_CONSOLE", "default": False, "cast_type": bool},
]

_ENV_SCHEMA_BY_KEY: dict[str, dict[str, Any]] = {
    item["key"]: item for item in ENV_SCHEMA
}


def _validate_env_value(
    key: str,
    value: Any,
    schema_item: dict[str, Any],
) -> tuple[bool, Any]:
    """Validate an environment variable value according to its schema.

    Args:
        key: Environment variable name.
        value: The value to validate (already cast).
        schema_item: Schema item from ``ENV_SCHEMA``.

    Returns:
        Tuple of ``(is_valid, corrected_value)``.
    """
    default = schema_item["default"]
    cast_type = schema_item["cast_type"]

    if value is None:
        return False, default

    if key == "LOG_LEVEL" and cast_type is str:
        try:
            upper = str(value).strip().upper()
            if upper not in LOG_LEVELS:
                return False, default
            return True, upper
        except (AttributeError, TypeError, ValueError):
            return False, default

    if "options" in schema_item:
        options = schema_item["options"]
        try:
            if cast_type is str:
                if str(value) not in options:
                    return False, default
            else:
                if value not in options:
                    return False, default
        except (AttributeError, TypeError, ValueError):
            return False, default

    if cast_type is int:
        try:
            int_value = int(value)
        except (TypeError, ValueError, OverflowError):
            return False, default
        if key in _SIZE_FIELDS and int_value <= 0:
            return False, default
        if key == "DPI" and (int_value < 50 or int_value > 1000):
            return False, default

    elif cast_type is float:
        try:
            float_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return False, default
        if key == "PLOT_LINE_WIDTH" and (float_value <= 0 or float_value > 20):
            return False, default

    elif cast_type is str:
        try:
            str_value = str(value).strip()
        except (AttributeError, TypeError):
            return False, default
        optional_fields = {"SOLVER_MAX_STEP"}
        if not str_value and key not in optional_fields:
            return False, default

    return True, value


def get_env(
    key: str,
    default: Any,
    cast_type: _EnvCastType = str,
) -> Union[str, int, float, bool]:
    """Get environment variable with type casting, validation, and fallback.

    Args:
        key: Environment variable name.
        default: Default value if variable not found or invalid.
        cast_type: Type to cast the value to.

    Returns:
        The validated value, or *default* if missing/invalid.
    """
    value = os.getenv(key)
    if value is None:
        return default

    schema_item = _ENV_SCHEMA_BY_KEY.get(key)
    if schema_item is None:
        try:
            if cast_type is bool:
                return value.lower() in ("true", "1", "yes")
            return cast_type(value)
        except (ValueError, TypeError):
            return default

    try:
        if cast_type is bool:
            casted = value.lower() in ("true", "1", "yes")
        else:
            casted = cast_type(value)
    except (ValueError, TypeError):
        return default

    _, corrected = _validate_env_value(key, casted, schema_item)
    return corrected


def get_env_from_schema(key: str) -> Any:
    """Get environment variable using ``ENV_SCHEMA`` defaults.

    Args:
        key: Environment variable name (must exist in ``ENV_SCHEMA``).

    Returns:
        The validated value.

    Raises:
        KeyError: If *key* is not in ``ENV_SCHEMA``.
    """
    item = _ENV_SCHEMA_BY_KEY.get(key)
    if item is None:
        raise KeyError(f"Unknown env key: {key}")
    return get_env(key, item["default"], item["cast_type"])


def validate_all_env_values() -> dict[str, tuple[Any, bool]]:
    """Validate all environment values and report corrections.

    Returns:
        Mapping of key to ``(corrected_value, was_corrected)``.
    """
    results: dict[str, tuple[Any, bool]] = {}
    for item in ENV_SCHEMA:
        key = item["key"]
        default = item["default"]
        cast_type = item["cast_type"]
        current = get_env(key, default, cast_type)
        original_raw = os.getenv(key)
        was_corrected = False
        if original_raw is not None:
            try:
                if cast_type is bool:
                    original_casted = original_raw.lower() in ("true", "1", "yes")
                else:
                    original_casted = cast_type(original_raw)
                is_valid, validated = _validate_env_value(key, original_casted, item)
                was_corrected = not is_valid or validated != current
            except (ValueError, TypeError):
                was_corrected = True
        results[key] = (current, was_corrected)
    return results


def get_current_env_values() -> dict[str, str]:
    """Collect current environment values as strings for all schema keys.

    Returns:
        Mapping of key to its string representation.
    """
    result: dict[str, str] = {}
    for item in ENV_SCHEMA:
        key = item["key"]
        val = get_env(key, item["default"], item["cast_type"])
        if item["cast_type"] is bool:
            result[key] = "true" if val else "false"
        else:
            result[key] = str(val)
    return result


def write_env_file(env_path: Path, values: dict[str, str]) -> None:
    """Write a ``.env`` file with the given key=value pairs.

    Args:
        env_path: Destination path for the ``.env`` file.
        values: Mapping from environment keys to string values.
    """
    lines = [
        "# DifferentialLab Configuration - generated by the application",
        "# Edit this file or use the Configuration dialog from the main menu.",
        "",
    ]
    for item in ENV_SCHEMA:
        key = item["key"]
        if key not in values:
            continue
        value = values[key].strip()
        if " " in value or "#" in value or "\n" in value:
            value = f'"{value}"'
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def initialize_and_validate_config() -> None:
    """Initialize and validate all configuration values at startup.

    Invalid values are silently corrected to defaults with a log warning.
    """
    try:
        from utils.logger import get_logger

        logger = get_logger(__name__)
    except ImportError:
        logger = None  # type: ignore[assignment]

    results = validate_all_env_values()
    corrected = [k for k, (_, was) in results.items() if was]

    if corrected and logger:
        logger.warning(
            "Corrected %d invalid env variable(s) to defaults: %s",
            len(corrected),
            ", ".join(corrected),
        )
