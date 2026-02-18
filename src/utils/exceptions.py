"""Custom exception hierarchy for ODE Solver."""


class ODESolverError(Exception):
    """Base exception for all ODE Solver errors."""


class ValidationError(ODESolverError):
    """Raised when user input fails validation."""


class EquationParseError(ODESolverError):
    """Raised when an ODE expression cannot be parsed or evaluated."""


class ConfigurationError(ODESolverError):
    """Raised when application configuration is invalid."""


class SolverFailedError(ODESolverError):
    """Raised when the numerical solver fails to converge."""
