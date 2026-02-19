"""Frontend module for DifferentialLab."""

from frontend.plot_embed import embed_plot_in_tk
from frontend.theme import configure_ttk_styles, get_font, get_select_colors
from frontend.ui_main_menu import MainMenu

__all__ = [
    "embed_plot_in_tk",
    "configure_ttk_styles",
    "get_font",
    "get_select_colors",
    "MainMenu",
]
