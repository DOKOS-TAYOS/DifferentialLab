"""UI dialog modules for DifferentialLab."""

from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.tooltip import ToolTip

__all__ = [
    "setup_arrow_enter_navigation",
    "ToolTip",
    "ConfigDialog",
    "EquationDialog",
    "HelpDialog",
    "TransformDialog",
]


def __getattr__(name: str):
    """Lazy-load heavy dialog modules on first access to speed up startup.

    Args:
        name: Attribute name (e.g. ``"ConfigDialog"``, ``"EquationDialog"``).

    Returns:
        The requested dialog class.

    Raises:
        AttributeError: If *name* is not a known lazy-loaded attribute.
    """
    if name == "ConfigDialog":
        from frontend.ui_dialogs.config_dialog import ConfigDialog

        return ConfigDialog
    if name == "EquationDialog":
        from frontend.ui_dialogs.equation_dialog import EquationDialog

        return EquationDialog
    if name == "HelpDialog":
        from frontend.ui_dialogs.help_dialog import HelpDialog

        return HelpDialog
    if name == "TransformDialog":
        from frontend.ui_dialogs.transform_dialog import TransformDialog

        return TransformDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
