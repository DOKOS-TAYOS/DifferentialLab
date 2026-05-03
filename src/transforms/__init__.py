"""Transform module: function parsing and mathematical transforms.

Public symbols are loaded lazily so importing :mod:`transforms` does not pull in
SciPy-backed transform machinery until a caller asks for it.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "parse_scalar_function": ("transforms.function_parser", "parse_scalar_function"),
    "DisplayMode": ("transforms.transform_engine", "DisplayMode"),
    "TransformKind": ("transforms.transform_engine", "TransformKind"),
    "apply_transform": ("transforms.transform_engine", "apply_transform"),
    "compute_function_samples": ("transforms.transform_engine", "compute_function_samples"),
    "get_transform_coefficients": ("transforms.transform_engine", "get_transform_coefficients"),
}

__all__ = [
    "parse_scalar_function",
    "DisplayMode",
    "TransformKind",
    "apply_transform",
    "compute_function_samples",
    "get_transform_coefficients",
]

if TYPE_CHECKING:
    from transforms.function_parser import parse_scalar_function
    from transforms.transform_engine import (
        DisplayMode,
        TransformKind,
        apply_transform,
        compute_function_samples,
        get_transform_coefficients,
    )


def __getattr__(name: str) -> Any:
    """Load a public transform symbol on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
