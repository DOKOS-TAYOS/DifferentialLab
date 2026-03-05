"""Configuration module for DifferentialLab."""

from config.constants import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_STATISTICS,
    DEFAULT_SOLVER_METHOD,
    SOLVER_METHOD_DESCRIPTIONS,
    SOLVER_METHODS,
)
from config.env import (
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    ENV_SCHEMA,
    SCHEMA_BY_KEY,
    get_current_env_values,
    get_env,
    get_env_from_schema,
    initialize_and_validate_config,
    write_env_file,
)
from config.paths import (
    generate_output_basename,
    get_csv_path,
    get_env_path,
    get_output_dir,
    get_project_root,
)

__all__ = [
    # constants
    "APP_NAME",
    "APP_VERSION",
    "AVAILABLE_STATISTICS",
    "SOLVER_METHODS",
    "SOLVER_METHOD_DESCRIPTIONS",
    "DEFAULT_SOLVER_METHOD",
    # env
    "DEFAULT_LOG_FILE",
    "DEFAULT_LOG_LEVEL",
    "ENV_SCHEMA",
    "SCHEMA_BY_KEY",
    "get_current_env_values",
    "get_env",
    "get_env_from_schema",
    "initialize_and_validate_config",
    "write_env_file",
    # paths
    "generate_output_basename",
    "get_csv_path",
    "get_env_path",
    "get_output_dir",
    "get_project_root",
]
