"""Configuration module for DifferentialLab."""

from config.constants import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_STATISTICS,
    SOLVER_METHOD_DESCRIPTIONS,
    SOLVER_METHODS,
    get_default_solver_method,
)
from config.env import _ENV_SCHEMA_BY_KEY as SCHEMA_BY_KEY
from config.env import (
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    ENV_SCHEMA,
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
    "get_default_solver_method",
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
