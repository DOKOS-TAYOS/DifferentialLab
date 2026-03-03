"""Main menu window for DifferentialLab (Tkinter/ttk)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from config import APP_NAME, APP_VERSION, get_env_from_schema
from frontend.theme import configure_ttk_styles
from frontend.ui_dialogs import ToolTip, setup_arrow_enter_navigation
from frontend.window_utils import fit_and_center
from utils import get_logger

logger = get_logger(__name__)


class MainMenu:
    """Application main menu window.

    Presents five actions: Solve, Function Transform, Information,
    Configuration, Quit.

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

        # Logo
        base_dir = Path(__file__).resolve().parent.parent.parent
        logo_path = base_dir / "images" / "DifferentialLab_logo.png"
        if logo_path.exists():
            logo_img = tk.PhotoImage(file=str(logo_path)).subsample(2, 2)
            logo_label = ttk.Label(main_frame, image=logo_img)
            logo_label.image = logo_img  # Keep reference
            logo_label.pack(pady=(0, padding))

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
            text=f"v{APP_VERSION}",
            style="Small.TLabel",
        ).pack(pady=(4, 0))

        # Description
        desc_lbl = ttk.Label(
            main_frame,
            text=(
                "Solve ODEs, vector ODEs, difference equations, and PDEs.\n"
                "Apply Fourier, Laplace, Taylor, and other transforms."
            ),
            style="Small.TLabel",
            justify=tk.CENTER,
        )
        desc_lbl.pack(pady=(0, padding * 2))

        def _update_desc_wrap(_e: tk.Event | None = None) -> None:  # type: ignore[type-arg]
            w = main_frame.winfo_width()
            if w > 100:
                desc_lbl.configure(wraplength=max(200, w - 6 * padding))

        main_frame.bind("<Configure>", _update_desc_wrap)
        self.root.after(50, lambda: _update_desc_wrap(None))

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(expand=True)

        # Center grid: columns and rows expand to center content
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.rowconfigure(0, weight=1)
        btn_frame.rowconfigure(1, weight=0)
        btn_frame.rowconfigure(2, weight=0)
        btn_frame.rowconfigure(3, weight=0)
        btn_frame.rowconfigure(4, weight=1)

        btn_width: int = get_env_from_schema("UI_BUTTON_WIDTH")

        # Row 1: Solve Differential Equation | Function Transform
        self.btn_solve = ttk.Button(
            btn_frame,
            text="Solve Differential\nEquation",
            width=btn_width,
            command=self._on_solve,
        )
        self.btn_solve.grid(row=1, column=0, padx=padding, pady=padding)
        ToolTip(self.btn_solve, "Select or write an ODE and solve it numerically.")

        self.btn_transforms = ttk.Button(
            btn_frame,
            text="Function\nTransform",
            width=btn_width,
            style="Accent2.TButton",
            command=self._on_transforms,
        )
        self.btn_transforms.grid(row=1, column=1, padx=padding, pady=padding)
        ToolTip(
            self.btn_transforms,
            "Enter a function, apply Fourier/Laplace/Taylor transforms, and export data.",
        )

        # Row 2: Information
        self.btn_info = ttk.Button(
            btn_frame,
            text="Information",
            width=btn_width,
            style="Accent2.TButton",
            command=self._on_info,
        )
        self.btn_info.grid(row=2, column=0, columnspan=2, padx=padding, pady=padding)
        ToolTip(self.btn_info, "View help, usage instructions, and app information.")

        # Row 3: Configuration | Quit (sized to text)
        self.btn_config = ttk.Button(
            btn_frame,
            text="Configuration",
            width=len("Configuration"),
            style="SmallMenu.Accent2.TButton",
            command=self._on_config,
        )
        self.btn_config.grid(row=3, column=0, padx=padding, pady=padding)
        ToolTip(self.btn_config, "Adjust solver settings, theme, and other preferences.")

        self.btn_quit = ttk.Button(
            btn_frame,
            text="Quit",
            width=len("Quit"),
            style="SmallMenu.Cancel.TButton",
            command=self._on_close,
        )
        self.btn_quit.grid(row=3, column=1, padx=padding, pady=padding)
        ToolTip(self.btn_quit, "Close the application.")

        setup_arrow_enter_navigation(
            [
                [self.btn_solve, self.btn_transforms],
                [self.btn_info],
                [self.btn_config, self.btn_quit],
            ]
        )
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
