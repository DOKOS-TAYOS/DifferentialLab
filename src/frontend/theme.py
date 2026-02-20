"""TTK theme configuration built from environment variables."""

import tkinter as tk
from tkinter import ttk

from config.env import get_env_from_schema

# -----------------------------------------------------------------------------
# Color conversion and transformation helpers (RegressionLab-style)
# -----------------------------------------------------------------------------


def _color_to_rgb(color: str) -> tuple[int, int, int] | None:
    """Convert color name or hex to RGB tuple (0-255).

    Args:
        color: Color name (e.g. 'steel blue', 'lime green') or hex (#rrggbb).

    Returns:
        Tuple of (r, g, b) in 0-255 range, or None if conversion fails.
    """
    if not isinstance(color, str) or not color.strip():
        return None
    color = color.strip().strip('"').strip("'")
    if not color:
        return None

    # Parse hex directly
    if color.startswith("#"):
        try:
            hex_part = color.lstrip("#").strip()
            if len(hex_part) == 6:
                return (
                    int(hex_part[0:2], 16),
                    int(hex_part[2:4], 16),
                    int(hex_part[4:6], 16),
                )
            if len(hex_part) == 3:
                return (
                    int(hex_part[0] * 2, 16),
                    int(hex_part[1] * 2, 16),
                    int(hex_part[2] * 2, 16),
                )
        except ValueError:
            pass
        return None

    # Use tkinter for named colors — reuse existing root when available
    try:
        existing_root = tk._default_root  # type: ignore[attr-defined]
        if existing_root is not None and existing_root.winfo_exists():
            r, g, b = existing_root.winfo_rgb(color)
            return (r // 256, g // 256, b // 256)
        tmp = tk.Tk()
        tmp.withdraw()
        r, g, b = tmp.winfo_rgb(color)
        tmp.destroy()
        return (r // 256, g // 256, b // 256)
    except (tk.TclError, Exception):
        pass

    # Fallback: matplotlib (handles many color names)
    try:
        from matplotlib.colors import to_rgb

        r, g, b = to_rgb(color)
        return (int(r * 255), int(g * 255), int(b * 255))
    except (ValueError, TypeError, ImportError):
        pass

    return None


def _lighten_color(color: str, factor: float = 0.20) -> str:
    """Return a lighter shade by moving toward white.

    Args:
        color: Source color (name or hex).
        factor: Fraction to move toward white (0.20 = 20% lighter).

    Returns:
        Hex color string (#rrggbb).
    """
    rgb = _color_to_rgb(color)
    if rgb is None:
        return "#ffffff"
    r, g, b = rgb
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _darken_color(color: str, factor: float = 0.25) -> str:
    """Return a darker shade by reducing each channel.

    Args:
        color: Source color (name or hex).
        factor: Fraction to darken (0.25 = 25% darker).

    Returns:
        Hex color string (#rrggbb).
    """
    rgb = _color_to_rgb(color)
    if rgb is None:
        return "#1e1e1e"
    r, g, b = rgb
    mult = 1.0 - factor
    r = int(r * mult)
    g = int(g * mult)
    b = int(b * mult)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_select_colors(
    element_bg: str,
    text_fg: str,
) -> tuple[str, str]:
    """Compute select background and foreground from element colors.

    Selected text: 20% lighter than unselected text.
    Selected background: 25% darker than the element's background.

    Args:
        element_bg: Background color of the element (e.g. UI_BUTTON_BG).
        text_fg: Foreground/text color (e.g. UI_FOREGROUND).

    Returns:
        Tuple of (select_background, select_foreground) as hex strings.
    """
    select_bg = _darken_color(element_bg, 0.25)
    select_fg = _lighten_color(text_fg, 0.20)
    return (select_bg, select_fg)


def get_font() -> tuple[str, int]:
    """Return the configured ``(family, size)`` font tuple.

    Returns:
        Tuple of font family name and size.
    """
    family: str = get_env_from_schema("UI_FONT_FAMILY")
    size: int = get_env_from_schema("UI_FONT_SIZE")
    return (family, size)


def configure_ttk_styles(root: tk.Tk) -> None:
    """Apply the dark theme to all ttk widgets based on env configuration.

    Args:
        root: The root Tk window.
    """
    bg: str = get_env_from_schema("UI_BACKGROUND")
    fg: str = get_env_from_schema("UI_FOREGROUND")
    btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
    btn_fg: str = get_env_from_schema("UI_BUTTON_FG")
    btn_fg_cancel: str = get_env_from_schema("UI_BUTTON_FG_CANCEL")
    btn_fg_accent2: str = get_env_from_schema("UI_BUTTON_FG_ACCENT2")
    select_bg, select_fg = get_select_colors(element_bg=btn_bg, text_fg=fg)
    padding: int = get_env_from_schema("UI_PADDING")
    font_family: str = get_env_from_schema("UI_FONT_FAMILY")
    font_size: int = get_env_from_schema("UI_FONT_SIZE")

    focus_bg = select_bg

    font = (font_family, font_size)
    font_bold = (font_family, font_size, "bold")
    font_small = (font_family, max(10, font_size - 4))
    font_large = (font_family, font_size + 6, "bold")
    font_desc = (font_family, max(9, int(font_size * 0.72)))

    root.configure(bg=bg)

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=bg, foreground=fg, font=font,
                    borderwidth=0, focuscolor=focus_bg)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg, font=font)
    style.configure("TLabelframe", background=bg, foreground=fg, font=font)
    style.configure("TLabelframe.Label", background=bg, foreground=fg, font=font_bold)

    # --- Buttons ---
    style.configure("TButton", background=btn_bg, foreground=btn_fg,
                    font=font, padding=padding, borderwidth=1, relief="raised")
    style.map("TButton",
              background=[("pressed", focus_bg), ("focus", focus_bg),
                          ("active", focus_bg)],
              foreground=[("active", btn_fg), ("focus", btn_fg)])

    style.configure("Cancel.TButton", foreground=btn_fg_cancel)
    style.map("Cancel.TButton",
              background=[("pressed", focus_bg), ("focus", focus_bg),
                          ("active", focus_bg)],
              foreground=[("active", btn_fg_cancel), ("focus", btn_fg_cancel)])

    style.configure("Accent2.TButton", foreground=btn_fg_accent2)
    style.map("Accent2.TButton",
              background=[("pressed", focus_bg), ("focus", focus_bg),
                          ("active", focus_bg)],
              foreground=[("active", btn_fg_accent2), ("focus", btn_fg_accent2)])

    style.configure("Small.TButton", background=btn_bg, foreground=btn_fg,
                    font=font, padding=(4, 2), borderwidth=1, relief="raised")
    style.map("Small.TButton",
              background=[("pressed", focus_bg), ("focus", focus_bg),
                          ("active", focus_bg)],
              foreground=[("active", btn_fg), ("focus", btn_fg)])

    # --- Labels ---
    style.configure("Title.TLabel", font=font_large, foreground=btn_fg)
    style.configure("Subtitle.TLabel", font=font_bold, foreground=fg)
    style.configure("Small.TLabel", font=font_small, foreground=fg)
    style.configure("ConfigDesc.TLabel", font=font_desc, foreground=fg)

    # Collapsible-section header style
    style.configure("SectionHeader.TFrame", background=btn_bg)
    style.configure("SectionHeader.TLabel", background=btn_bg, foreground=btn_fg,
                    font=font_bold)

    # --- Entry (larger font + focus highlight) ---
    style.configure("TEntry", fieldbackground=btn_bg, foreground=fg,
                    insertcolor=fg, selectbackground=select_bg,
                    selectforeground=select_fg, padding=6, font=font)
    style.map("TEntry",
              fieldbackground=[("focus", "#2a2a2a")])

    # --- Spinbox ---
    style.configure("TSpinbox", fieldbackground=btn_bg, foreground=fg,
                    arrowcolor=fg, insertcolor=fg, selectbackground=select_bg,
                    selectforeground=select_fg, padding=6, font=font)
    style.map("TSpinbox",
              fieldbackground=[("focus", "#2a2a2a")])

    # --- Combobox ---
    style.configure("TCombobox", fieldbackground=btn_bg, foreground=fg,
                    selectbackground=select_bg, selectforeground=select_fg,
                    padding=6, font=font)
    style.map("TCombobox",
              fieldbackground=[("readonly", btn_bg), ("focus", "#2a2a2a")],
              foreground=[("readonly", fg)])

    # --- Checkbutton ---
    style.configure("TCheckbutton", background=bg, foreground=fg, font=font,
                    indicatorcolor=btn_bg)
    style.map("TCheckbutton",
              background=[("active", bg), ("focus", "#2a2a2a")],
              indicatorcolor=[("selected", btn_fg)])

    # --- Treeview ---
    style.configure("Treeview", background=btn_bg, foreground=fg,
                    fieldbackground=btn_bg, font=font_small,
                    rowheight=int(font_size * 1.8))
    style.configure("Treeview.Heading", background=bg, foreground=fg, font=font_bold)
    style.map("Treeview",
              background=[("selected", select_bg)],
              foreground=[("selected", select_fg)])

    # --- Scrollbar ---
    style.configure("TScrollbar", background=btn_bg, troughcolor=bg,
                    arrowcolor=fg, borderwidth=0)

    # --- Notebook ---
    style.configure("TNotebook", background=bg, borderwidth=0)
    style.configure("TNotebook.Tab", background=btn_bg, foreground=fg,
                    font=font, padding=[padding, padding // 2])
    style.map("TNotebook.Tab",
              background=[("selected", bg)],
              foreground=[("selected", btn_fg)])

    # --- Separator ---
    style.configure("TSeparator", background=btn_bg)

    # --- PanedWindow ---
    style.configure("TPanedwindow", background=bg)

    root.option_add("*TCombobox*Listbox*Background", btn_bg)
    root.option_add("*TCombobox*Listbox*Foreground", fg)
    root.option_add("*TCombobox*Listbox*selectBackground", select_bg)
    root.option_add("*TCombobox*Listbox*selectForeground", select_fg)
    root.option_add("*TCombobox*Listbox*Font", font)
