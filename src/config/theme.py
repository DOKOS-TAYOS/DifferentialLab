"""TTK theme configuration built from environment variables."""

import tkinter as tk
from tkinter import ttk

from config.env import get_env_from_schema


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
    select_bg: str = get_env_from_schema("UI_TEXT_SELECT_BG")
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
                    padding=6, font=font)
    style.map("TEntry",
              fieldbackground=[("focus", "#2a2a2a")])

    # --- Spinbox ---
    style.configure("TSpinbox", fieldbackground=btn_bg, foreground=fg,
                    arrowcolor=fg, insertcolor=fg, padding=6, font=font)
    style.map("TSpinbox",
              fieldbackground=[("focus", "#2a2a2a")])

    # --- Combobox ---
    style.configure("TCombobox", fieldbackground=btn_bg, foreground=fg,
                    selectbackground=select_bg, padding=6, font=font)
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
              foreground=[("selected", fg)])

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
    root.option_add("*TCombobox*Listbox*selectForeground", fg)
    root.option_add("*TCombobox*Listbox*Font", font)
