"""Custom exception hierarchy for DifferentialLab."""


class DifferentialLabError(Exception):
    """Base exception for all DifferentialLab errors."""


class ValidationError(DifferentialLabError):
    """Raised when user input fails validation."""


class EquationParseError(DifferentialLabError):
    """Raised when an ODE expression cannot be parsed or evaluated."""


class ConfigurationError(DifferentialLabError):
    """Raised when application configuration is invalid."""


class SolverFailedError(DifferentialLabError):
    """Raised when the numerical solver fails to converge."""
