"""UI dialog modules for DifferentialLab."""

from frontend.ui_dialogs.config_dialog import ConfigDialog
from frontend.ui_dialogs.equation_dialog import EquationDialog
from frontend.ui_dialogs.help_dialog import HelpDialog
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.parameters_dialog import ParametersDialog
from frontend.ui_dialogs.result_dialog import ResultDialog
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip

__all__ = [
    "ConfigDialog",
    "EquationDialog",
    "HelpDialog",
    "setup_arrow_enter_navigation",
    "ParametersDialog",
    "ResultDialog",
    "ScrollableFrame",
    "ToolTip",
]
