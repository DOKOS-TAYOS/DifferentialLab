"""Application constants for DifferentialLab."""

from typing import Final

APP_NAME: Final[str] = "DifferentialLab"
APP_VERSION: Final[str] = "0.3.2"

SOLVER_METHODS: Final[tuple[str, ...]] = (
    "RK45",
    "RK23",
    "DOP853",
    "Radau",
    "BDF",
    "LSODA",
)


def get_default_solver_method() -> str:
    """Return the default ODE integration method (first in available list)."""
    return SOLVER_METHODS[0]


SOLVER_METHOD_DESCRIPTIONS: Final[dict[str, str]] = {
    "RK45": "Runge-Kutta 4(5) — general-purpose explicit method",
    "RK23": "Runge-Kutta 2(3) — low-order, faster per step",
    "DOP853": "Runge-Kutta 8(5,3) — high-order explicit method",
    "Radau": "Implicit Runge-Kutta (Radau IIA) — stiff problems",
    "BDF": "Backward Differentiation Formula — stiff problems",
    "LSODA": "Adams/BDF auto-switching — stiff/non-stiff detection",
}

LINE_STYLES: Final[tuple[str, ...]] = ("-", "--", "-.", ":")
MARKER_FORMATS: Final[tuple[str, ...]] = ("o", "s", "^", "d", "*")
FONT_FAMILIES: Final[tuple[str, ...]] = (
    "serif",
    "sans-serif",
    "monospace",
    "cursive",
    "fantasy",
)
FONT_SIZES: Final[tuple[str, ...]] = (
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
)
FONT_WEIGHTS: Final[tuple[str, ...]] = (
    "normal",
    "bold",
    "light",
    "semibold",
    "heavy",
)
FONT_STYLES: Final[tuple[str, ...]] = ("normal", "italic", "oblique")
LOG_LEVELS: Final[tuple[str, ...]] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)

AVAILABLE_STATISTICS: Final[dict[str, str]] = {
    "mean": "Mean value of f(x)",
    "rms": "Root mean square of f(x)",
    "std": "Standard deviation of f(x)",
    "median": "Median value (robust to outliers)",
    "max": "Maximum value and its location",
    "min": "Minimum value and its location",
    "integral": "Definite integral (area under curve)",
    "l2_norm": "L2 norm sqrt(∫f² dx), energy-like magnitude",
    "zero_crossings": "Number of zero crossings",
    "period": "Estimated period (for oscillatory solutions)",
    "amplitude": "Estimated amplitude (for oscillatory solutions)",
    "dominant_frequency": "Dominant frequency via FFT (cycles per unit)",
    "exponential_rate": "Exponential fit rate λ in f∝exp(λx) (decay/growth)",
    "half_life": "Half-life t_1/2 = ln(2)/|λ| for exponential decay",
    "time_constant": "Time constant τ = 1/|λ| for decay (e.g. RC circuit)",
    "doubling_time": "Doubling time ln(2)/λ for exponential growth",
    "angular_frequency": "Angular frequency ω = 2πf (rad per unit) from FFT",
    "energy": "Energy estimate (kinetic + potential for 2nd order)",
    "gradient_norm": "Mean gradient magnitude |∇u| (2D PDE only)",
}
