"""Configuration dialog — edit .env variables with collapsible sections."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import (
    ENV_SCHEMA,
    get_current_env_values,
    get_env_from_schema,
    get_env_path,
    write_env_file,
)
from frontend.theme import get_font
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import center_window, make_modal
from utils import get_logger

logger = get_logger(__name__)

_SECTION_ORDER: list[tuple[str, str, list[str]]] = [
    ("ui_theme", "UI Theme", [
        "UI_BACKGROUND", "UI_FOREGROUND", "UI_BUTTON_BG", "UI_BUTTON_WIDTH",
        "UI_BUTTON_FG", "UI_BUTTON_FG_CANCEL", "UI_BUTTON_FG_ACCENT2",
        "UI_FONT_SIZE", "UI_FONT_FAMILY",
        "UI_PADDING", "UI_ENTRY_WIDTH",
    ]),
    ("plot_style", "Plot Style", [
        "PLOT_FIGSIZE_WIDTH", "PLOT_FIGSIZE_HEIGHT", "DPI",
        "PLOT_SHOW_TITLE", "PLOT_SHOW_GRID",
        "PLOT_LINE_COLOR", "PLOT_LINE_WIDTH", "PLOT_LINE_STYLE",
    ]),
    ("plot_markers", "Plot Markers", [
        "PLOT_MARKER_FORMAT", "PLOT_MARKER_SIZE",
        "PLOT_MARKER_FACE_COLOR", "PLOT_MARKER_EDGE_COLOR",
    ]),
    ("plot_fonts", "Plot Fonts", [
        "FONT_FAMILY", "FONT_TITLE_SIZE", "FONT_TITLE_WEIGHT",
        "FONT_AXIS_SIZE", "FONT_AXIS_STYLE", "FONT_TICK_SIZE",
    ]),
    ("solver", "Solver Defaults", [
        "SOLVER_DEFAULT_METHOD", "SOLVER_MAX_STEP",
        "SOLVER_RTOL", "SOLVER_ATOL", "SOLVER_NUM_POINTS",
    ]),
    ("files", "File Paths", [
        "FILE_OUTPUT_DIR", "FILE_PLOT_FORMAT",
    ]),
    ("logging", "Logging", [
        "LOG_LEVEL", "LOG_FILE", "LOG_CONSOLE",
    ]),
]

_SCHEMA_BY_KEY: dict[str, dict[str, Any]] = {item["key"]: item for item in ENV_SCHEMA}


class ConfigDialog:
    """Scrollable form to edit all ``.env`` configuration values.

    After calling, inspect ``self.accepted`` to know if the user saved.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.accepted = False
        self.win = tk.Toplevel(parent)
        self.win.title("Configuration")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._vars: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self._desc_labels: list[ttk.Label] = []
        self._build_ui()

        self.win.update_idletasks()
        req_width = self.win.winfo_reqwidth()
        req_height = self.win.winfo_reqheight()
        
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        
        win_w = min(max(req_width + 40, 800), int(screen_w * 0.9))
        win_h = min(max(req_height + 40, 700), int(screen_h * 0.9))
        
        center_window(self.win, win_w, win_h)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        bg: str = get_env_from_schema("UI_BACKGROUND")
        current = get_current_env_values()

        # --- Fixed bottom button bar (packed FIRST so it stays at bottom) ---
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        hint = ttk.Label(
            btn_frame,
            text="The application will restart after saving.",
            style="Small.TLabel",
            anchor=tk.CENTER,
        )
        hint.pack(fill=tk.X, pady=(0, pad // 2))

        btn_inner = ttk.Frame(btn_frame)
        btn_inner.pack()

        btn_save = ttk.Button(btn_inner, text="Save", command=self._on_save)
        btn_save.pack(side=tk.LEFT, padx=pad)

        btn_cancel = ttk.Button(
            btn_inner, text="Cancel", style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_cancel.pack(side=tk.LEFT, padx=pad)

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(
            side=tk.BOTTOM, fill=tk.X,
        )

        setup_arrow_enter_navigation([[btn_save, btn_cancel]])

        # --- Scrollable area ---
        self._scroll = ScrollableFrame(self.win)
        self._scroll.apply_bg(bg)
        self._scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        form = self._scroll.inner
        form.configure(padding=pad)

        ttk.Label(form, text="Configuration", style="Title.TLabel").pack(
            anchor=tk.W, pady=(0, pad),
        )

        # --- Collapsible sections ---
        first_section = True
        for _section_id, section_title, keys in _SECTION_ORDER:
            self._add_section(form, section_title, keys, current, pad,
                              expanded=first_section)
            first_section = False

        self._scroll.bind_new_children()

        def _update_wraplength(_e: tk.Event | None = None) -> None:  # type: ignore[type-arg]
            w = form.winfo_width()
            if w > 100:
                wrap = max(200, w - 60)
                for lbl in self._desc_labels:
                    lbl.configure(wraplength=wrap)

        form.bind("<Configure>", _update_wraplength)

        btn_save.focus_set()

    def _add_section(
        self,
        parent: ttk.Frame,
        title: str,
        keys: list[str],
        current: dict[str, str],
        pad: int,
        *,
        expanded: bool = False,
    ) -> None:
        """Add a complete collapsible section with its fields."""
        section = CollapsibleSection(
            parent, self._scroll, title, expanded=expanded, pad=pad,
        )
        section.content.configure(padding=(16, 4, 4, 4))

        for key in keys:
            item = _SCHEMA_BY_KEY.get(key)
            if item is None:
                continue
            self._add_field(section.content, item, current)

    def _add_field(
        self,
        parent: ttk.Frame,
        item: dict[str, Any],
        current: dict[str, str]
    ) -> None:
        key = item["key"]
        cast_type = item["cast_type"]
        val = current.get(key, str(item["default"]))
        desc_text = item.get("description", "")

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text=key, width=28, anchor=tk.W).pack(side=tk.LEFT)

        if cast_type is bool:
            bvar = tk.BooleanVar(value=val.lower() in ("true", "1", "yes"))
            cb = ttk.Checkbutton(row, variable=bvar)
            cb.pack(side=tk.LEFT)
            self._vars[key] = bvar
        elif "options" in item:
            svar = tk.StringVar(value=val)
            combo = ttk.Combobox(
                row, textvariable=svar,
                values=list(item["options"]),
                state="readonly", width=22, font=get_font()
            )
            combo.pack(side=tk.LEFT)
            self._vars[key] = svar
        else:
            svar = tk.StringVar(value=val)
            entry = ttk.Entry(row, textvariable=svar, width=25, font=get_font())
            entry.pack(side=tk.LEFT)
            self._vars[key] = svar

        if desc_text:
            desc = ttk.Label(parent, text=desc_text, style="ConfigDesc.TLabel",
                             wraplength=600, justify=tk.LEFT)
            desc.pack(anchor=tk.W, padx=(12, 0), pady=(0, 4))
            self._desc_labels.append(desc)

    def _on_save(self) -> None:
        """Write the edited values to ``.env`` and flag accepted."""
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
            self.accepted = True
            self.win.destroy()
        except Exception as exc:
            logger.error("Failed to save .env: %s", exc)
            messagebox.showerror("Error", f"Could not save: {exc}",
                                 parent=self.win)
