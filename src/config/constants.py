"""Application constants for DifferentialLab."""

from typing import Final

APP_NAME: Final[str] = "DifferentialLab"
APP_VERSION: Final[str] = "0.2.0"

SOLVER_METHODS: Final[tuple[str, ...]] = (
    "RK45",
    "RK23",
    "DOP853",
    "Radau",
    "BDF",
    "LSODA",
)

SOLVER_METHOD_DESCRIPTIONS: Final[dict[str, str]] = {
    "RK45": "Runge-Kutta 4(5) — general-purpose explicit method",
    "RK23": "Runge-Kutta 2(3) — low-order, faster per step",
    "DOP853": "Runge-Kutta 8(5,3) — high-order explicit method",
    "Radau": "Implicit Runge-Kutta (Radau IIA) — stiff problems",
    "BDF": "Backward Differentiation Formula — stiff problems",
    "LSODA": "Adams/BDF auto-switching — stiff/non-stiff detection",
}

PLOT_FORMATS: Final[tuple[str, ...]] = ("png", "jpg", "pdf")
LINE_STYLES: Final[tuple[str, ...]] = ("-", "--", "-.", ":")
MARKER_FORMATS: Final[tuple[str, ...]] = ("o", "s", "^", "d", "*")
FONT_FAMILIES: Final[tuple[str, ...]] = (
    "serif", "sans-serif", "monospace", "cursive", "fantasy",
)
FONT_SIZES: Final[tuple[str, ...]] = (
    "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large",
)
FONT_WEIGHTS: Final[tuple[str, ...]] = (
    "normal", "bold", "light", "semibold", "heavy",
)
FONT_STYLES: Final[tuple[str, ...]] = ("normal", "italic", "oblique")
LOG_LEVELS: Final[tuple[str, ...]] = (
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
)

AVAILABLE_STATISTICS: Final[dict[str, str]] = {
    "mean": "Mean value of y(x)",
    "rms": "Root mean square of y(x)",
    "std": "Standard deviation of y(x)",
    "max": "Maximum value and its location",
    "min": "Minimum value and its location",
    "integral": "Definite integral (area under curve)",
    "zero_crossings": "Number of zero crossings",
    "period": "Estimated period (for oscillatory solutions)",
    "amplitude": "Estimated amplitude (for oscillatory solutions)",
    "energy": "Energy estimate (kinetic + potential for 2nd order)",
}
