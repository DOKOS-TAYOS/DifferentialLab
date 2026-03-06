"""Small parsing helpers for UI numeric fields."""

from __future__ import annotations


def parse_int(value: str, *, name: str) -> int:
    """Parse an integer value from text."""
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def parse_positive_int(value: str, *, name: str, min_value: int = 1) -> int:
    """Parse a strictly positive integer."""
    out = parse_int(value, name=name)
    if out < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    return out


def parse_float(value: str, *, name: str) -> float:
    """Parse a floating-point value from text."""
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric.") from exc


def parse_positive_float(value: str, *, name: str, min_value: float = 0.0) -> float:
    """Parse a positive floating-point value."""
    out = parse_float(value, name=name)
    if out <= min_value:
        relation = ">" if min_value == 0 else f"> {min_value}"
        raise ValueError(f"{name} must be {relation}.")
    return out

