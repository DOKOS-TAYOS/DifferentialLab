"""Configuration dialog — edit .env variables with a scrollable form."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config.env import ENV_SCHEMA, get_current_env_values, write_env_file
from config.paths import get_env_path
from config.env import get_env_from_schema
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import center_window, make_modal
from utils.logger import get_logger

logger = get_logger(__name__)

_SECTION_HEADERS: dict[str, str] = {
    "UI_BACKGROUND": "UI Theme",
    "PLOT_FIGSIZE_WIDTH": "Plot Style",
    "FONT_FAMILY": "Plot Fonts",
    "SOLVER_DEFAULT_METHOD": "Solver Defaults",
    "FILE_OUTPUT_DIR": "File Paths",
    "LOG_LEVEL": "Logging",
}


class ConfigDialog:
    """Scrollable form to edit all ``.env`` configuration values.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Configuration")
        center_window(self.win, 640, 600)
        make_modal(self.win, parent)

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._vars: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        current = get_current_env_values()

        # Scrollable canvas
        canvas = tk.Canvas(self.win, bg=get_env_from_schema("UI_BACKGROUND"),
                           highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.win, orient=tk.VERTICAL, command=canvas.yview)
        form_frame = ttk.Frame(canvas, padding=pad)

        form_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=form_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(form_frame, text="Configuration", style="Title.TLabel").pack(
            anchor=tk.W, pady=(0, pad),
        )
        ttk.Label(
            form_frame,
            text="Changes are saved to the .env file and take effect on next launch.",
            style="Small.TLabel",
        ).pack(anchor=tk.W, pady=(0, pad * 2))

        for item in ENV_SCHEMA:
            key = item["key"]
            cast_type = item["cast_type"]

            if key in _SECTION_HEADERS:
                ttk.Separator(form_frame, orient=tk.HORIZONTAL).pack(
                    fill=tk.X, pady=pad,
                )
                ttk.Label(form_frame, text=_SECTION_HEADERS[key],
                          style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, pad // 2))

            row = ttk.Frame(form_frame)
            row.pack(fill=tk.X, pady=2)

            ttk.Label(row, text=key, width=28, anchor=tk.W).pack(side=tk.LEFT)

            val = current.get(key, str(item["default"]))

            if cast_type is bool:
                bvar = tk.BooleanVar(value=val.lower() in ("true", "1", "yes"))
                cb = ttk.Checkbutton(row, variable=bvar)
                cb.pack(side=tk.LEFT)
                self._vars[key] = bvar
            elif "options" in item:
                svar = tk.StringVar(value=val)
                combo = ttk.Combobox(row, textvariable=svar,
                                      values=list(item["options"]),
                                      state="readonly", width=20)
                combo.pack(side=tk.LEFT)
                self._vars[key] = svar
            else:
                svar = tk.StringVar(value=val)
                entry = ttk.Entry(row, textvariable=svar, width=24)
                entry.pack(side=tk.LEFT)
                self._vars[key] = svar

        # Buttons
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=pad, pady=pad)

        ttk.Button(btn_frame, text="Save", command=self._on_save).pack(
            side=tk.RIGHT, padx=pad,
        )
        ttk.Button(btn_frame, text="Cancel", style="Cancel.TButton",
                   command=self.win.destroy).pack(side=tk.RIGHT)

    def _on_save(self) -> None:
        """Write the edited values to ``.env``."""
        values: dict[str, str] = {}
        for item in ENV_SCHEMA:
            key = item["key"]
            var = self._vars.get(key)
            if var is None:
                continue
            if isinstance(var, tk.BooleanVar):
                values[key] = "true" if var.get() else "false"
            else:
                values[key] = var.get()

        try:
            write_env_file(get_env_path(), values)
            logger.info("Configuration saved to .env")
            messagebox.showinfo("Saved",
                                "Configuration saved. Restart to apply changes.",
                                parent=self.win)
            self.win.destroy()
        except Exception as exc:
            logger.error("Failed to save .env: %s", exc)
            messagebox.showerror("Error", f"Could not save: {exc}",
                                 parent=self.win)
