"""Main menu window for DifferentialLab (Tkinter/ttk)."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, cast

from config import APP_NAME, APP_VERSION, get_env_from_schema
from frontend.theme import configure_ttk_styles
from frontend.ui_dialogs import ToolTip, setup_arrow_enter_navigation
from frontend.window_utils import bind_wraplength, fit_and_center
from utils import get_logger

logger = get_logger(__name__)


def _build_restart_command() -> list[str]:
    """Build the command used to restart the application."""
    if Path(sys.argv[0]).suffix.casefold() == ".py":
        return [sys.executable, sys.argv[0], *sys.argv[1:]]

    return [sys.argv[0], *sys.argv[1:]]


def _restart_application() -> None:
    """Start a fresh DifferentialLab process."""
    subprocess.Popen(_build_restart_command(), shell=False)


class MainMenu:
    """Application main menu window.

    Presents actions for solving equations, transforms, advanced problems,
    help, settings, and exit.

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

        main_frame = ttk.Frame(self.root, padding=padding * 2)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Logo
        base_dir = Path(__file__).resolve().parent.parent.parent
        logo_path = base_dir / "images" / "DifferentialLab_logo.png"
        if logo_path.exists():
            # The source asset already includes the DifferentialLab wordmark.
            logo_img = tk.PhotoImage(file=str(logo_path)).subsample(3, 3)
            logo_label = ttk.Label(main_frame, image=logo_img)
            cast(Any, logo_label).image = logo_img  # Keep reference
            logo_label.pack(pady=(0, padding // 2))

        ttk.Label(
            main_frame,
            text=f"v{APP_VERSION}",
            style="Small.TLabel",
        ).pack(pady=(0, padding))

        # Description
        desc_lbl = ttk.Label(
            main_frame,
            text=(
                "Solve ODEs, vector systems, recurrences, and PDEs.\n"
                "Explore transforms, advanced models, plots, and exports."
            ),
            style="Small.TLabel",
            justify=tk.CENTER,
        )
        desc_lbl.pack(pady=(0, padding))

        bind_wraplength(main_frame, desc_lbl, pad=6 * padding, min_wrap=200)

        # Primary application actions share one full-width visual treatment.
        primary_frame = ttk.Frame(main_frame)
        primary_frame.pack(fill=tk.X, expand=True, pady=(padding, 0))

        btn_width: int = get_env_from_schema("UI_BUTTON_WIDTH")

        self.btn_solve = ttk.Button(
            primary_frame,
            text="Solve Equation",
            width=btn_width,
            style="Primary.TButton",
            command=self._on_solve,
        )
        self.btn_solve.pack(fill=tk.X, pady=(0, padding))
        ToolTip(
            self.btn_solve,
            "Choose a built-in or custom equation, then configure and solve it.",
        )

        self.btn_transforms = ttk.Button(
            primary_frame,
            text="Function Transform",
            width=btn_width,
            style="Primary.TButton",
            command=self._on_transforms,
        )
        self.btn_transforms.pack(fill=tk.X, pady=(0, padding))
        ToolTip(
            self.btn_transforms,
            "Enter f(x), apply a transform, inspect the plot, and export the data.",
        )

        self.btn_complex = ttk.Button(
            primary_frame,
            text="Advanced Problems",
            width=btn_width,
            style="Primary.TButton",
            command=self._on_complex_problems,
        )
        self.btn_complex.pack(fill=tk.X)
        ToolTip(
            self.btn_complex,
            "Open specialized physics and engineering solvers with guided settings.",
        )

        # Secondary actions are grouped separately and use a quieter style.
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(padding * 2, padding))
        secondary_frame = ttk.Frame(main_frame)
        secondary_frame.pack(fill=tk.X)
        for column in range(3):
            secondary_frame.columnconfigure(column, weight=1)

        self.btn_info = ttk.Button(
            secondary_frame,
            text="Help & About",
            style="Secondary.TButton",
            command=self._on_info,
        )
        self.btn_info.grid(row=0, column=0, padx=(0, padding // 2), sticky="ew")
        ToolTip(self.btn_info, "Read usage notes, expression syntax, shortcuts, and app details.")

        self.btn_config = ttk.Button(
            secondary_frame,
            text="Settings",
            style="Secondary.TButton",
            command=self._on_config,
        )
        self.btn_config.grid(row=0, column=1, padx=padding // 2, sticky="ew")
        ToolTip(self.btn_config, "Adjust appearance, solver defaults, logging, and output paths.")

        self.btn_quit = ttk.Button(
            secondary_frame,
            text="Exit",
            style="Secondary.TButton",
            command=self._on_close,
        )
        self.btn_quit.grid(row=0, column=2, padx=(padding // 2, 0), sticky="ew")
        ToolTip(self.btn_quit, "Close DifferentialLab.")

        setup_arrow_enter_navigation(
            [
                [self.btn_solve],
                [self.btn_transforms],
                [self.btn_complex],
            ]
        )
        setup_arrow_enter_navigation([[self.btn_info, self.btn_config, self.btn_quit]])
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

    def _on_transforms(self) -> None:
        """Open the function transforms dialog."""
        logger.info("User clicked Transforms")
        from frontend.ui_dialogs import TransformDialog

        TransformDialog(self.root)

    def _on_complex_problems(self) -> None:
        """Open the complex problems selection dialog."""
        logger.info("User clicked Advanced Problems")
        from complex_problems import ComplexProblemsDialog

        ComplexProblemsDialog(self.root)

    def _on_config(self) -> None:
        """Open the configuration dialog; restart the app if saved."""
        logger.info("User clicked Settings")
        from frontend.ui_dialogs import ConfigDialog

        dlg = ConfigDialog(self.root)
        self.root.wait_window(dlg.win)

        if dlg.accepted:
            logger.info("Settings saved - restarting application")
            try:
                _restart_application()
            except OSError as exc:
                logger.exception("Failed to restart application")
                messagebox.showerror(
                    "Restart failed",
                    f"The settings were saved, but DifferentialLab could not restart:\n{exc}",
                    parent=self.root,
                )
                return

            self.root.destroy()

    def _on_info(self) -> None:
        """Open the information / help dialog."""
        logger.info("User clicked Help & About")
        from frontend.ui_dialogs import HelpDialog

        HelpDialog(self.root)
