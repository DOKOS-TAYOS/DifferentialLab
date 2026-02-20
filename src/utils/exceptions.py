"""Custom exception hierarchy for DifferentialLab."""


class DifferentialLabError(Exception):
    """Base exception for all DifferentialLab errors."""


class ValidationError(DifferentialLabError):
    """Raised when user input fails validation."""


class EquationParseError(DifferentialLabError):
    """Raised when an ODE expression cannot be parsed or evaluated."""


class SolverFailedError(DifferentialLabError):
    """Raised when the numerical solver fails to converge."""
