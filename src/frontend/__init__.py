"""Frontend module for DifferentialLab."""

from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import configure_ttk_styles, get_font, get_select_colors
from frontend.window_utils import center_window, fit_and_center, make_modal
from frontend.ui_main_menu import MainMenu

__all__ = [
    "center_window",
    "configure_ttk_styles",
    "embed_animation_plot_in_tk",
    "embed_plot_in_tk",
    "fit_and_center",
    "get_font",
    "get_select_colors",
    "MainMenu",
    "make_modal",
]
