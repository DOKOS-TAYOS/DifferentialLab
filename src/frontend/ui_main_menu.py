"""Main menu window for DifferentialLab (Tkinter/ttk)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import APP_NAME, APP_VERSION, get_env_from_schema
from frontend.theme import configure_ttk_styles
from frontend.ui_dialogs import ToolTip, setup_arrow_enter_navigation
from frontend.window_utils import fit_and_center
from utils import get_logger

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

        configure_ttk_styles(self.root)

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.root.configure(bg=bg)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

        fit_and_center(self.root, min_width=520, min_height=480)
        logger.info("Main menu created")

    def _build_ui(self) -> None:
        """Construct the main menu layout."""
        padding: int = get_env_from_schema("UI_PADDING")

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
            text=f"v{APP_VERSION} — DifferentialLab",
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
        ToolTip(self.btn_solve, "Select or write an ODE and solve it numerically.")

        self.btn_config = ttk.Button(
            btn_frame,
            text="Configuration",
            width=btn_width,
            style="Accent2.TButton",
            command=self._on_config,
        )
        self.btn_config.pack(pady=padding)
        ToolTip(self.btn_config, "Adjust solver settings, theme, and other preferences.")

        self.btn_info = ttk.Button(
            btn_frame,
            text="Information",
            width=btn_width,
            style="Accent2.TButton",
            command=self._on_info,
        )
        self.btn_info.pack(pady=padding)
        ToolTip(self.btn_info, "View help, usage instructions, and app information.")

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=padding, side=tk.BOTTOM,
        )

        self.btn_quit = ttk.Button(
            main_frame,
            text="Quit",
            width=btn_width,
            style="Cancel.TButton",
            command=self._on_close,
        )
        self.btn_quit.pack(side=tk.BOTTOM, pady=(padding, 0))
        ToolTip(self.btn_quit, "Close the application.")

        setup_arrow_enter_navigation([
            [self.btn_solve],
            [self.btn_config],
            [self.btn_info],
            [self.btn_quit],
        ])
        self.btn_solve.focus_set()

    def _on_close(self) -> None:
        """Handle window close: stop mainloop and destroy the root window."""
        logger.info("User closed the main window")
        self.root.quit()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_solve(self) -> None:
        """Open the equation selection dialog."""
        logger.info("User clicked Solve")
        from frontend.ui_dialogs import EquationDialog

        EquationDialog(self.root)

    def _on_config(self) -> None:
        """Open the configuration dialog; restart the app if saved."""
        logger.info("User clicked Configuration")
        from frontend.ui_dialogs import ConfigDialog

        dlg = ConfigDialog(self.root)
        self.root.wait_window(dlg.win)

        if dlg.accepted:
            import os
            import sys

            logger.info("Configuration saved — restarting application")
            self.root.destroy()
            os.execv(sys.executable, [sys.executable] + sys.argv)

    def _on_info(self) -> None:
        """Open the information / help dialog."""
        logger.info("User clicked Information")
        from frontend.ui_dialogs import HelpDialog

        HelpDialog(self.root)
