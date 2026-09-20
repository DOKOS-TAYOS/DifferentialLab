"""Settings dialog with human-readable preferences backed by .env."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import (
    ENV_SCHEMA,
    SCHEMA_BY_KEY,
    get_current_env_values,
    get_env_from_schema,
    get_env_path,
    write_env_file,
)
from frontend.theme import get_font
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import bind_wraplength, fit_and_center, make_modal
from utils import get_logger

logger = get_logger(__name__)

_SECTION_ORDER: list[tuple[str, str, list[str]]] = [
    (
        "appearance",
        "Appearance",
        [
            "UI_BACKGROUND",
            "UI_FOREGROUND",
            "UI_BUTTON_BG",
            "UI_BUTTON_WIDTH",
            "UI_BUTTON_FG",
            "UI_BUTTON_FG_CANCEL",
            "UI_BUTTON_FG_ACCENT2",
            "UI_FONT_SIZE",
            "UI_FONT_FAMILY",
            "UI_PADDING",
            "UI_TOOLTIP_DELAY_MS",
            "UI_TOOLTIP_WRAPLENGTH",
            "UI_TOOLTIP_PADX",
            "UI_TOOLTIP_PADY",
        ],
    ),
    (
        "plots",
        "Plots",
        [
            "PLOT_FIGSIZE_WIDTH",
            "PLOT_FIGSIZE_HEIGHT",
            "DPI",
            "PLOT_SHOW_TITLE",
            "PLOT_SHOW_GRID",
            "PLOT_LINE_COLOR",
            "PLOT_LINE_WIDTH",
            "PLOT_LINE_STYLE",
            "PLOT_COLOR_SCHEME",
            "FONT_FAMILY",
            "FONT_TITLE_SIZE",
            "FONT_TITLE_WEIGHT",
            "FONT_AXIS_SIZE",
            "FONT_AXIS_STYLE",
            "FONT_TICK_SIZE",
            "PLOT_MARKER_FORMAT",
            "PLOT_MARKER_SIZE",
            "PLOT_MARKER_FACE_COLOR",
            "PLOT_MARKER_EDGE_COLOR",
            "PLOT_PHASE_START_COLOR",
            "PLOT_PHASE_END_COLOR",
            "PLOT_PHASE_MARKER_SIZE",
            "PLOT_SURFACE_CMAP",
            "PLOT_CONTOUR_LEVELS",
            "PLOT_GRID_ALPHA",
            "PLOT_SURFACE_ALPHA",
            "PLOT_COLORBAR_SHRINK",
            "PLOT_ANIMATION_LINE_WIDTH",
            "PLOT_VLINES_LINE_WIDTH",
            "PLOT_VLINES_ALPHA",
            "PLOT_ANIMATION_Y_MARGIN",
            "ANIMATION_MAX_FPS",
        ],
    ),
    (
        "solver",
        "Solver Defaults",
        [
            "SOLVER_MAX_STEP",
            "SOLVER_RTOL",
            "SOLVER_ATOL",
            "SOLVER_NUM_POINTS",
        ],
    ),
    (
        "advanced",
        "Advanced",
        [
            "LOG_LEVEL",
            "LOG_FILE",
            "LOG_MAX_BYTES",
            "LOG_BACKUP_COUNT",
            "LOG_CONSOLE",
            "CHECK_UPDATES",
            "UPDATE_CHECK_INTERVAL_DAYS",
            "CHECK_UPDATES_FORCE",
            "UPDATE_CHECK_URL",
        ],
    ),
]

_FIELD_LABELS: dict[str, str] = {
    "UI_BACKGROUND": "Background color",
    "UI_FOREGROUND": "Text color",
    "UI_BUTTON_BG": "Control background",
    "UI_BUTTON_WIDTH": "Main-menu button width",
    "UI_BUTTON_FG": "Primary accent text",
    "UI_BUTTON_FG_CANCEL": "Destructive / cancel text",
    "UI_BUTTON_FG_ACCENT2": "Secondary accent text",
    "UI_FONT_SIZE": "Interface font size",
    "UI_FONT_FAMILY": "Interface font family",
    "UI_PADDING": "Interface spacing",
    "UI_TOOLTIP_DELAY_MS": "Tooltip delay (ms)",
    "UI_TOOLTIP_WRAPLENGTH": "Tooltip maximum width",
    "UI_TOOLTIP_PADX": "Tooltip horizontal padding",
    "UI_TOOLTIP_PADY": "Tooltip vertical padding",
    "PLOT_FIGSIZE_WIDTH": "Figure width (inches)",
    "PLOT_FIGSIZE_HEIGHT": "Figure height (inches)",
    "DPI": "Plot DPI",
    "PLOT_SHOW_TITLE": "Show plot title",
    "PLOT_SHOW_GRID": "Show plot grid",
    "PLOT_LINE_COLOR": "Main line color",
    "PLOT_LINE_WIDTH": "Main line width",
    "PLOT_LINE_STYLE": "Main line style",
    "PLOT_COLOR_SCHEME": "Additional-series colormap",
    "FONT_FAMILY": "Plot font family",
    "FONT_TITLE_SIZE": "Plot title size",
    "FONT_TITLE_WEIGHT": "Plot title weight",
    "FONT_AXIS_SIZE": "Axis-label size",
    "FONT_AXIS_STYLE": "Axis-label style",
    "FONT_TICK_SIZE": "Tick-label size",
    "PLOT_MARKER_FORMAT": "Marker shape",
    "PLOT_MARKER_SIZE": "Marker size",
    "PLOT_MARKER_FACE_COLOR": "Marker fill color",
    "PLOT_MARKER_EDGE_COLOR": "Marker edge color",
    "PLOT_PHASE_START_COLOR": "Phase start color",
    "PLOT_PHASE_END_COLOR": "Phase end color",
    "PLOT_PHASE_MARKER_SIZE": "Phase marker size",
    "PLOT_SURFACE_CMAP": "Surface / contour colormap",
    "PLOT_CONTOUR_LEVELS": "Contour levels",
    "PLOT_GRID_ALPHA": "Grid opacity",
    "PLOT_SURFACE_ALPHA": "Surface opacity",
    "PLOT_COLORBAR_SHRINK": "Colorbar size",
    "PLOT_ANIMATION_LINE_WIDTH": "Animation line width",
    "PLOT_VLINES_LINE_WIDTH": "Guide-line width",
    "PLOT_VLINES_ALPHA": "Guide-line opacity",
    "PLOT_ANIMATION_Y_MARGIN": "Animation y-axis margin",
    "ANIMATION_MAX_FPS": "Maximum animation FPS",
    "SOLVER_MAX_STEP": "Maximum solver step",
    "SOLVER_RTOL": "Relative tolerance",
    "SOLVER_ATOL": "Absolute tolerance",
    "SOLVER_NUM_POINTS": "Default output points",
    "LOG_LEVEL": "Logging level",
    "LOG_FILE": "Log filename",
    "LOG_MAX_BYTES": "Maximum log size (bytes)",
    "LOG_BACKUP_COUNT": "Log backup files",
    "LOG_CONSOLE": "Also log to console",
    "CHECK_UPDATES": "Check for updates on startup",
    "UPDATE_CHECK_INTERVAL_DAYS": "Update check interval (days)",
    "CHECK_UPDATES_FORCE": "Force check every startup",
    "UPDATE_CHECK_URL": "Update metadata URL",
}



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
        self.win.title("Settings")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._vars: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self._desc_labels: list[ttk.Label] = []
        self._build_ui()

        fit_and_center(self.win, min_width=800, min_height=700)
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
            text="Saving writes these values to .env and restarts the app.",
            style="Small.TLabel",
            anchor=tk.CENTER,
        )
        hint.pack(fill=tk.X, pady=(0, pad // 2))

        btn_inner = ttk.Frame(btn_frame)
        btn_inner.pack()

        btn_save = ttk.Button(btn_inner, text="Save & Restart", command=self._on_save)
        btn_save.pack(side=tk.LEFT, padx=pad)

        btn_cancel = ttk.Button(
            btn_inner,
            text="Cancel",
            style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_cancel.pack(side=tk.LEFT, padx=pad)

        setup_arrow_enter_navigation([[btn_save, btn_cancel]])

        # --- Scrollable area ---
        self._scroll = ScrollableFrame(self.win)
        self._scroll.apply_bg(bg)
        self._scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        form = self._scroll.inner
        form.configure(padding=pad)

        ttk.Label(form, text="Settings", style="Title.TLabel").pack(
            anchor=tk.W,
            pady=(0, pad),
        )

        # --- Collapsible sections ---
        first_section = True
        for _section_id, section_title, keys in _SECTION_ORDER:
            self._add_section(form, section_title, keys, current, pad, expanded=first_section)
            first_section = False

        self._scroll.bind_new_children()

        bind_wraplength(form, self._desc_labels, pad=60, min_wrap=200)

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
            parent,
            self._scroll,
            title,
            expanded=expanded,
            pad=pad,
        )
        section.content.configure(padding=(16, 4, 4, 4))

        for key in keys:
            item = SCHEMA_BY_KEY.get(key)
            if item is None:
                continue
            self._add_field(section.content, item, current)

    def _add_field(self, parent: ttk.Frame, item: dict[str, Any], current: dict[str, str]) -> None:
        key = item["key"]
        cast_type = item["cast_type"]
        val = current.get(key, str(item["default"]))
        desc_text = item.get("description", "")

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        label = _FIELD_LABELS.get(key, key)
        field_label = ttk.Label(row, text=label, width=30, anchor=tk.W)
        field_label.pack(side=tk.LEFT)
        ToolTip(field_label, f"Configuration key: {key}")

        if cast_type is bool:
            bvar = tk.BooleanVar(value=val.lower() in ("true", "1", "yes"))
            cb = ttk.Checkbutton(row, variable=bvar)
            cb.pack(side=tk.LEFT)
            self._vars[key] = bvar
        elif "options" in item:
            svar = tk.StringVar(value=val)
            combo = ttk.Combobox(
                row,
                textvariable=svar,
                values=list(item["options"]),
                state="readonly",
                width=22,
                font=get_font(),
            )
            combo.pack(side=tk.LEFT)
            self._vars[key] = svar
        else:
            svar = tk.StringVar(value=val)
            entry = ttk.Entry(row, textvariable=svar, width=25, font=get_font())
            entry.pack(side=tk.LEFT)
            self._vars[key] = svar

        if desc_text:
            desc = ttk.Label(parent, text=desc_text, style="ConfigDesc.TLabel", justify=tk.LEFT)
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
            logger.info("Settings saved to .env")
            self.accepted = True
            self.win.destroy()
        except Exception as exc:
            logger.error("Failed to save .env: %s", exc, exc_info=True)
            messagebox.showerror(
                "Settings were not saved",
                f"DifferentialLab could not save the settings:\n{exc}",
                parent=self.win,
            )
