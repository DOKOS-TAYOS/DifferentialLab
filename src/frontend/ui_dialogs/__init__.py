"""UI dialog modules for DifferentialLab."""

from frontend.ui_dialogs.config_dialog import ConfigDialog
from frontend.ui_dialogs.equation_dialog import EquationDialog
from frontend.ui_dialogs.help_dialog import HelpDialog
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.ui_dialogs.transform_dialog import TransformDialog

__all__ = [
    "ConfigDialog",
    "EquationDialog",
    "HelpDialog",
    "TransformDialog",
    "setup_arrow_enter_navigation",
    "ToolTip",
]
