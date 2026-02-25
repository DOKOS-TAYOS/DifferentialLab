"""Custom exception hierarchy for DifferentialLab.

All application-specific exceptions inherit from DifferentialLabError,
enabling broad ``except DifferentialLabError`` handling when desired.
"""


class DifferentialLabError(Exception):
    """Base exception for all DifferentialLab errors.

    Catch this to handle any application-defined error.
    """


class ValidationError(DifferentialLabError):
    """Raised when user input fails validation (domain, ICs, parameters, etc.)."""


class EquationParseError(DifferentialLabError):
    """Raised when an ODE/difference/PDE expression cannot be parsed or evaluated."""


class SolverFailedError(DifferentialLabError):
    """Raised when the numerical solver fails to converge or encounters an error."""
