"""UI dialog modules for DifferentialLab."""

from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.ui_dialogs.config_dialog import ConfigDialog
from frontend.ui_dialogs.equation_dialog import EquationDialog
from frontend.ui_dialogs.help_dialog import HelpDialog
from frontend.ui_dialogs.transform_dialog import TransformDialog

__all__ = [
    "CollapsibleSection",
    "ConfigDialog",
    "EquationDialog",
    "HelpDialog",
    "ScrollableFrame",
    "TransformDialog",
    "setup_arrow_enter_navigation",
    "ToolTip",
]
