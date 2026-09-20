"""Settings dialog with human-readable preferences backed by .env."""

from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, messagebox, ttk
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
from frontend.window_utils import (
    bind_wraplength,
    calculate_screen_aware_minsize,
    fit_and_center,
    make_modal,
)
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

_SUBGROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "appearance": [
        (
            "Interface",
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
            ],
        ),
        (
            "Tooltips",
            [
                "UI_TOOLTIP_DELAY_MS",
                "UI_TOOLTIP_WRAPLENGTH",
                "UI_TOOLTIP_PADX",
                "UI_TOOLTIP_PADY",
            ],
        ),
    ],
    "plots": [
        (
            "Figure and Lines",
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
            ],
        ),
        (
            "Fonts",
            [
                "FONT_FAMILY",
                "FONT_TITLE_SIZE",
                "FONT_TITLE_WEIGHT",
                "FONT_AXIS_SIZE",
                "FONT_AXIS_STYLE",
                "FONT_TICK_SIZE",
            ],
        ),
        (
            "Markers",
            [
                "PLOT_MARKER_FORMAT",
                "PLOT_MARKER_SIZE",
                "PLOT_MARKER_FACE_COLOR",
                "PLOT_MARKER_EDGE_COLOR",
            ],
        ),
        (
            "Phase Space",
            [
                "PLOT_PHASE_START_COLOR",
                "PLOT_PHASE_END_COLOR",
                "PLOT_PHASE_MARKER_SIZE",
            ],
        ),
        (
            "3D and Contour",
            [
                "PLOT_SURFACE_CMAP",
                "PLOT_CONTOUR_LEVELS",
                "PLOT_GRID_ALPHA",
                "PLOT_SURFACE_ALPHA",
                "PLOT_COLORBAR_SHRINK",
            ],
        ),
        (
            "Animations",
            [
                "PLOT_ANIMATION_LINE_WIDTH",
                "PLOT_VLINES_LINE_WIDTH",
                "PLOT_VLINES_ALPHA",
                "PLOT_ANIMATION_Y_MARGIN",
                "ANIMATION_MAX_FPS",
            ],
        ),
    ],
    "solver": [
        (
            "Integration and output",
            [
                "SOLVER_MAX_STEP",
                "SOLVER_RTOL",
                "SOLVER_ATOL",
                "SOLVER_NUM_POINTS",
            ],
        )
    ],
    "advanced": [
        (
            "Logging",
            [
                "LOG_LEVEL",
                "LOG_FILE",
                "LOG_MAX_BYTES",
                "LOG_BACKUP_COUNT",
                "LOG_CONSOLE",
            ],
        ),
        (
            "Updates",
            [
                "CHECK_UPDATES",
                "UPDATE_CHECK_INTERVAL_DAYS",
                "CHECK_UPDATES_FORCE",
                "UPDATE_CHECK_URL",
            ],
        ),
    ],
}

_COLOR_KEYS: frozenset[str] = frozenset(
    {
        "UI_BACKGROUND",
        "UI_FOREGROUND",
        "UI_BUTTON_BG",
        "UI_BUTTON_FG",
        "UI_BUTTON_FG_CANCEL",
        "UI_BUTTON_FG_ACCENT2",
        "PLOT_LINE_COLOR",
        "PLOT_MARKER_FACE_COLOR",
        "PLOT_MARKER_EDGE_COLOR",
        "PLOT_PHASE_START_COLOR",
        "PLOT_PHASE_END_COLOR",
    }
)

_INVALID_COLOR_PREVIEW = "#777777"
_STACKED_FIELD_BREAKPOINT = 720


def _default_form_values() -> dict[str, str | bool]:
    """Return every schema default in the representation expected by the form."""
    values: dict[str, str | bool] = {}
    for item in ENV_SCHEMA:
        default = item["default"]
        values[item["key"]] = bool(default) if item["cast_type"] is bool else str(default)
    return values


def _set_form_value(variable: object, value: str | bool) -> None:
    """Set a Tk-compatible form variable without coupling tests to a Tk root."""
    setter = getattr(variable, "set", None)
    if callable(setter):
        setter(value)


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
        self._header_labels: list[ttk.Label] = []
        self._color_swatches: dict[str, tk.Label] = {}
        self._field_layouts: list[tuple[ttk.Frame, ttk.Label, ttk.Frame, ttk.Label | None]] = []
        self._layout_job: str | None = None
        self._build_ui()

        fit_and_center(self.win, min_width=820, min_height=720, resizable=True)
        min_width, min_height = calculate_screen_aware_minsize(
            self.win.winfo_screenwidth(),
            self.win.winfo_screenheight(),
            640,
            500,
        )
        self.win.minsize(min_width, min_height)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        bg: str = get_env_from_schema("UI_BACKGROUND")
        current = get_current_env_values()

        # --- Fixed bottom button bar (packed first so it stays visible) ---
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_frame.columnconfigure(1, weight=1)
        btn_restore = ttk.Button(
            btn_frame,
            text="Restore Defaults",
            style="Secondary.TButton",
            command=self._on_restore_defaults,
        )
        btn_restore.grid(row=0, column=0, sticky=tk.W)
        btn_cancel = ttk.Button(
            btn_frame,
            text="Cancel",
            style="Secondary.TButton",
            command=self.win.destroy,
        )
        btn_cancel.grid(row=0, column=2, padx=(pad, pad // 2))
        btn_save = ttk.Button(
            btn_frame,
            text="Save & Restart",
            style="Primary.TButton",
            command=self._on_save,
        )
        btn_save.grid(row=0, column=3, padx=(pad // 2, 0))

        setup_arrow_enter_navigation([[btn_restore, btn_cancel, btn_save]])

        # --- Fixed header ---
        header = ttk.Frame(self.win, padding=(pad * 2, pad * 2, pad * 2, pad))
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="Settings", style="Title.TLabel").pack(anchor=tk.W)
        intro = ttk.Label(
            header,
            text="Configure interface, plots, solver defaults, and advanced behavior.",
            justify=tk.LEFT,
        )
        intro.pack(anchor=tk.W, fill=tk.X, pady=(pad // 2, 0))
        restart_note = ttk.Label(
            header,
            text="Changes are saved to .env and applied after DifferentialLab restarts.",
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        restart_note.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
        self._header_labels.extend((intro, restart_note))

        # --- Scrollable area ---
        self._scroll = ScrollableFrame(self.win)
        self._scroll.apply_bg(bg)
        self._scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        form = self._scroll.inner
        form.configure(padding=pad)

        # --- Collapsible sections ---
        first_section = True
        for section_id, section_title, _keys in _SECTION_ORDER:
            self._add_section(
                form,
                section_id,
                section_title,
                current,
                pad,
                expanded=first_section,
            )
            first_section = False

        self._scroll.bind_new_children()

        bind_wraplength(self._scroll.viewport, self._desc_labels, pad=72, min_wrap=200)
        bind_wraplength(header, self._header_labels, pad=pad * 4, min_wrap=200)
        self._scroll.viewport.bind("<Configure>", self._schedule_field_layout, add="+")
        self._scroll.viewport.after(100, self._apply_field_layout)

        btn_save.focus_set()

    def _add_section(
        self,
        parent: ttk.Frame,
        section_id: str,
        title: str,
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

        for subgroup_index, (subgroup_title, keys) in enumerate(_SUBGROUPS[section_id]):
            self._add_subgroup(
                section.content,
                subgroup_title,
                keys,
                current,
                pad,
                add_top_space=subgroup_index > 0,
            )

    def _add_subgroup(
        self,
        parent: ttk.Frame,
        title: str,
        keys: list[str],
        current: dict[str, str],
        pad: int,
        *,
        add_top_space: bool,
    ) -> None:
        """Add a lightweight heading and its aligned setting rows."""
        group = ttk.Frame(parent)
        group.pack(fill=tk.X, pady=((pad * 2 if add_top_space else pad // 2), pad // 2))

        heading = ttk.Frame(group)
        heading.pack(fill=tk.X, pady=(0, pad // 2))
        ttk.Label(heading, text=title, style="Subtitle.TLabel").pack(side=tk.LEFT)
        ttk.Separator(heading, orient=tk.HORIZONTAL).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(pad, 0),
        )

        for key in keys:
            self._add_field(group, SCHEMA_BY_KEY[key], current, pad)

    def _add_field(
        self,
        parent: ttk.Frame,
        item: dict[str, Any],
        current: dict[str, str],
        pad: int,
    ) -> None:
        """Add one labeled control, optional color actions, and supporting copy."""
        key: str = item["key"]
        cast_type = item["cast_type"]
        val = current.get(key, str(item["default"]))
        desc_text = item.get("description", "")

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(2, pad // 2))
        row.columnconfigure(1, weight=1)

        label = _FIELD_LABELS.get(key, key)
        field_label = ttk.Label(row, text=label, width=30, anchor=tk.W)
        field_label.grid(row=0, column=0, sticky=tk.W, padx=(0, pad))
        ToolTip(field_label, f"Configuration key: {key}")

        controls = ttk.Frame(row)
        controls.grid(row=0, column=1, sticky=tk.EW)
        controls.columnconfigure(0, weight=1)

        if cast_type is bool:
            bvar = tk.BooleanVar(value=val.lower() in ("true", "1", "yes"))
            cb = ttk.Checkbutton(controls, variable=bvar)
            cb.grid(row=0, column=0, sticky=tk.W)
            self._vars[key] = bvar
        elif "options" in item:
            svar = tk.StringVar(value=val)
            combo = ttk.Combobox(
                controls,
                textvariable=svar,
                values=list(item["options"]),
                state="readonly",
                width=22,
                font=get_font(),
            )
            combo.grid(row=0, column=0, sticky=tk.EW)
            self._vars[key] = svar
        else:
            svar = tk.StringVar(value=val)
            entry = ttk.Entry(controls, textvariable=svar, width=25, font=get_font())
            entry.grid(row=0, column=0, sticky=tk.EW)
            self._vars[key] = svar

            if key in _COLOR_KEYS:
                swatch = tk.Label(
                    controls,
                    width=3,
                    relief=tk.SUNKEN,
                    borderwidth=1,
                    takefocus=False,
                    font=get_font(),
                )
                swatch.grid(row=0, column=1, padx=(pad, pad // 2), sticky=tk.NS)
                ToolTip(swatch, "Color preview; ? means the value cannot be previewed.")
                self._color_swatches[key] = swatch
                ttk.Button(
                    controls,
                    text="Choose...",
                    style="Small.TButton",
                    command=lambda color_key=key: self._choose_color(color_key),
                ).grid(row=0, column=2, sticky=tk.E)
                svar.trace_add(
                    "write", lambda *_args, color_key=key: self._update_color_swatch(color_key)
                )
                self._update_color_swatch(key)

        desc: ttk.Label | None = None
        if desc_text:
            desc = ttk.Label(
                row,
                text=desc_text,
                style="ConfigDesc.TLabel",
                justify=tk.LEFT,
            )
            desc.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))
            self._desc_labels.append(desc)
        self._field_layouts.append((row, field_label, controls, desc))

    def _schedule_field_layout(self, _event: tk.Event[tk.Misc]) -> None:
        """Debounce responsive field rearrangement during window resizing."""
        if self._layout_job is not None:
            try:
                self._scroll.viewport.after_cancel(self._layout_job)
            except tk.TclError:
                pass
        self._layout_job = self._scroll.viewport.after(40, self._apply_field_layout)

    def _apply_field_layout(self) -> None:
        """Stack labels above controls when the visible viewport is narrow."""
        self._layout_job = None
        stacked = self._scroll.viewport.winfo_width() < _STACKED_FIELD_BREAKPOINT
        pad: int = get_env_from_schema("UI_PADDING")

        for row, field_label, controls, desc in self._field_layouts:
            if stacked:
                row.columnconfigure(0, weight=1)
                row.columnconfigure(1, weight=0)
                field_label.configure(width=0)
                field_label.grid_configure(
                    row=0,
                    column=0,
                    columnspan=2,
                    sticky=tk.W,
                    padx=0,
                )
                controls.grid_configure(
                    row=1,
                    column=0,
                    columnspan=2,
                    sticky=tk.EW,
                    pady=(2, 0),
                )
                if desc is not None:
                    desc.grid_configure(row=2, column=0, columnspan=2)
            else:
                row.columnconfigure(0, weight=0)
                row.columnconfigure(1, weight=1)
                field_label.configure(width=30)
                field_label.grid_configure(
                    row=0,
                    column=0,
                    columnspan=1,
                    sticky=tk.W,
                    padx=(0, pad),
                )
                controls.grid_configure(
                    row=0,
                    column=1,
                    columnspan=1,
                    sticky=tk.EW,
                    pady=0,
                )
                if desc is not None:
                    desc.grid_configure(row=1, column=0, columnspan=2)

    def _resolve_color(self, value: str) -> str | None:
        """Return a Tk-resolvable color value, or ``None`` for invalid input."""
        candidate = value.strip().strip('"').strip("'")
        if not candidate:
            return None
        try:
            self.win.winfo_rgb(candidate)
        except tk.TclError:
            return None
        return candidate

    def _update_color_swatch(self, key: str) -> None:
        """Synchronize one color preview without rejecting editable text."""
        var = self._vars.get(key)
        swatch = self._color_swatches.get(key)
        if not isinstance(var, tk.StringVar) or swatch is None:
            return
        resolved = self._resolve_color(var.get())
        if resolved is None:
            swatch.configure(background=_INVALID_COLOR_PREVIEW, foreground="#ffffff", text="?")
        else:
            swatch.configure(background=resolved, text="")

    def _update_color_swatches(self) -> None:
        """Refresh every color preview from its current form value."""
        for key in _COLOR_KEYS:
            self._update_color_swatch(key)

    def _choose_color(self, key: str) -> None:
        """Open the platform color chooser and apply the selected hex color."""
        var = self._vars.get(key)
        if not isinstance(var, tk.StringVar):
            return
        initial = self._resolve_color(var.get())
        if initial is None:
            initial = self._resolve_color(str(SCHEMA_BY_KEY[key]["default"]))
        _rgb, selected = colorchooser.askcolor(
            color=initial,
            parent=self.win,
            title=f"Choose {_FIELD_LABELS[key].lower()}",
        )
        if selected:
            var.set(selected)

    def _on_restore_defaults(self) -> None:
        """Replace form values with schema defaults after confirmation only."""
        confirmed = messagebox.askyesno(
            "Restore default settings?",
            "Restore all settings to their defaults?\n\n"
            "This updates the form only. Nothing will be saved until you choose "
            "Save & Restart.",
            parent=self.win,
        )
        if not confirmed:
            return

        for key, value in _default_form_values().items():
            var = self._vars.get(key)
            if var is not None:
                _set_form_value(var, value)
        self._update_color_swatches()

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
