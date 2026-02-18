"""Main menu window for ODE Solver (Tkinter/ttk)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config.constants import APP_NAME, APP_VERSION
from config.theme import configure_ttk_styles, get_font
from config.env import get_env_from_schema
from frontend.window_utils import center_window
from utils.logger import get_logger

logger = get_logger(__name__)


class MainMenu:
    """Application main menu window.

    Presents three primary actions: Solve, Configuration, Information.

    Args:
        root: The root Tk window.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.resizable(True, True)

        configure_ttk_styles(self.root)
        center_window(self.root, 520, 480)

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.root.configure(bg=bg)

        self._build_ui()
        logger.info("Main menu created")

    def _build_ui(self) -> None:
        """Construct the main menu layout."""
        padding: int = get_env_from_schema("UI_PADDING")
        font_family, font_size = get_font()

        main_frame = ttk.Frame(self.root, padding=padding * 3)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, padding * 2))

        ttk.Label(
            title_frame,
            text=APP_NAME,
            style="Title.TLabel",
        ).pack()

        ttk.Label(
            title_frame,
            text=f"v{APP_VERSION} — Numerical ODE Solver",
            style="Small.TLabel",
        ).pack(pady=(4, 0))

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=padding)

        # Description
        ttk.Label(
            main_frame,
            text=(
                "Solve ordinary differential equations numerically.\n"
                "Choose from predefined equations or write your own."
            ),
            style="Small.TLabel",
            justify=tk.CENTER,
        ).pack(pady=(0, padding * 2))

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(expand=True)

        btn_width: int = get_env_from_schema("UI_BUTTON_WIDTH")

        self.btn_solve = ttk.Button(
            btn_frame,
            text="Solve",
            width=btn_width,
            command=self._on_solve,
        )
        self.btn_solve.pack(pady=padding)

        self.btn_config = ttk.Button(
            btn_frame,
            text="Configuration",
            width=btn_width,
            style="Accent2.TButton",
            command=self._on_config,
        )
        self.btn_config.pack(pady=padding)

        self.btn_info = ttk.Button(
            btn_frame,
            text="Information",
            width=btn_width,
            style="Accent2.TButton",
            command=self._on_info,
        )
        self.btn_info.pack(pady=padding)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=padding, side=tk.BOTTOM,
        )

        self.btn_quit = ttk.Button(
            main_frame,
            text="Quit",
            width=btn_width,
            style="Cancel.TButton",
            command=self.root.destroy,
        )
        self.btn_quit.pack(side=tk.BOTTOM, pady=(padding, 0))

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_solve(self) -> None:
        """Open the equation selection dialog."""
        logger.info("User clicked Solve")
        from frontend.ui_dialogs.equation_dialog import EquationDialog

        EquationDialog(self.root)

    def _on_config(self) -> None:
        """Open the configuration dialog."""
        logger.info("User clicked Configuration")
        from frontend.ui_dialogs.config_dialog import ConfigDialog

        ConfigDialog(self.root)

    def _on_info(self) -> None:
        """Open the information / help dialog."""
        logger.info("User clicked Information")
        from frontend.ui_dialogs.help_dialog import HelpDialog

        HelpDialog(self.root)
