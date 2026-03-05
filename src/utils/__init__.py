"""Utility modules for DifferentialLab."""

from utils.exceptions import (
    DifferentialLabError,
    EquationParseError,
    SolverFailedError,
    ValidationError,
)
from utils.export import (
    export_csv_to_path,
    export_json_to_path,
)
from utils.expression_parser_shared import (
    build_eval_namespace,
    normalize_params,
    normalize_unicode_escapes,
    safe_eval,
    validate_exclusive_args,
    validate_expression_ast,
)
from utils.logger import get_logger
from utils.update_checker import (
    is_update_available,
    perform_git_pull,
    record_check_done,
    should_run_check,
)

__all__ = [
    "DifferentialLabError",
    "EquationParseError",
    "SolverFailedError",
    "ValidationError",
    "build_eval_namespace",
    "export_csv_to_path",
    "export_json_to_path",
    "get_logger",
    "normalize_params",
    "normalize_unicode_escapes",
    "safe_eval",
    "validate_exclusive_args",
    "validate_expression_ast",
    "is_update_available",
    "perform_git_pull",
    "record_check_done",
    "should_run_check",
]
