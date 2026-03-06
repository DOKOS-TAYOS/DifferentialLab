"""Dialog for selecting which complex problem to solve."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from complex_problems.problem_docs import get_problem_doc
from complex_problems.problem_registry import PROBLEM_REGISTRY, open_problem_dialog
from config import get_env_from_schema
from frontend.theme import get_font, get_select_colors
from frontend.ui_dialogs import ToolTip, setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import bind_wraplength, fit_and_center, make_modal
from utils import get_logger

logger = get_logger(__name__)


class ComplexProblemsDialog:
    """Window for selecting a complex problem to solve.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Complex Problems")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._build_ui()

        fit_and_center(self.win, min_width=1120, min_height=760, resizable=True)
        make_modal(self.win, parent)
        logger.info("Complex problems dialog opened")

    def _build_ui(self) -> None:
        """Construct the dialog layout."""
        pad: int = get_env_from_schema("UI_PADDING")
        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        select_bg, select_fg = get_select_colors(element_bg=btn_bg, text_fg=fg)

        main_frame = ttk.Frame(self.win, padding=pad * 2)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Complex Problems",
            style="Title.TLabel",
        ).pack(pady=(0, pad))

        intro = ttk.Label(
            main_frame,
            text=(
                "Choose one problem on the left. The right panel summarizes"
                " physical context, configurable options, and outputs."
            ),
            style="Small.TLabel",
            justify=tk.CENTER,
        )
        intro.pack(pady=(0, pad * 2))
        bind_wraplength(main_frame, intro, pad=4 * pad, min_wrap=260)

        content = ttk.Frame(main_frame)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=0, minsize=300)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, pad))
        ttk.Label(left, text="Problems", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 4))

        left_list_frame = ttk.Frame(left)
        left_list_frame.pack(fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(left_list_frame, orient=tk.VERTICAL)
        self._problem_listbox = tk.Listbox(
            left_list_frame,
            width=28,
            height=18,
            bg=btn_bg,
            fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=get_font(),
            yscrollcommand=list_scrollbar.set,
            exportselection=False,
        )
        list_scrollbar.config(command=self._problem_listbox.yview)
        self._problem_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._problem_listbox.bind("<<ListboxSelect>>", self._on_problem_select)
        self._problem_listbox.bind("<Double-Button-1>", self._on_open)
        self._problem_listbox.bind("<Return>", self._on_open)

        right = ttk.LabelFrame(content, text="Problem details", padding=pad)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)

        ttk.Label(right, text="Problem:", style="Subtitle.TLabel").grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(0, 8),
            pady=(0, 4),
        )
        self._name_label = ttk.Label(right, text="", style="Subtitle.TLabel", justify=tk.LEFT)
        self._name_label.grid(row=0, column=1, sticky="nw", pady=(0, 4))

        ttk.Label(right, text="Type:", style="Small.TLabel").grid(
            row=1,
            column=0,
            sticky="nw",
            padx=(0, 8),
            pady=(0, 8),
        )
        self._type_label = ttk.Label(right, text="", style="Small.TLabel", justify=tk.LEFT)
        self._type_label.grid(row=1, column=1, sticky="nw", pady=(0, 8))

        details_frame = ttk.Frame(right)
        details_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        self._details_scroll = ScrollableFrame(details_frame)
        self._details_scroll.apply_bg(btn_bg)
        self._details_scroll.grid(row=0, column=0, sticky="nsew")
        self._details_label = ttk.Label(
            self._details_scroll.inner,
            text="",
            style="Small.TLabel",
            justify=tk.LEFT,
            anchor=tk.NW,
        )
        self._details_label.pack(fill=tk.X, anchor=tk.NW)

        bind_wraplength(
            right,
            [self._name_label, self._type_label, self._details_label],
            pad=2 * pad,
            min_wrap=220,
        )

        self._problem_ids: list[str] = []
        for problem_id, descriptor in PROBLEM_REGISTRY.items():
            self._problem_ids.append(problem_id)
            self._problem_listbox.insert(tk.END, descriptor.name)
        ToolTip(self._problem_listbox, "Double-click a problem to open it.")

        if self._problem_ids:
            self._problem_listbox.selection_set(0)
            self._on_problem_select(None)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=(pad * 2, 0))
        self._open_btn = ttk.Button(
            btn_frame,
            text="Open",
            command=self._on_open,
        )
        self._open_btn.pack(side=tk.LEFT, padx=(0, pad))
        btn_close = ttk.Button(
            btn_frame,
            text="Close",
            style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack(side=tk.LEFT)

        setup_arrow_enter_navigation([[self._open_btn, btn_close]])
        self._problem_listbox.focus_set()

    def _get_selected_problem_id(self) -> str | None:
        selected = self._problem_listbox.curselection()
        if not selected:
            return None
        idx = selected[0]
        if idx < 0 or idx >= len(self._problem_ids):
            return None
        return self._problem_ids[idx]

    def _on_problem_select(self, _event: tk.Event | None) -> None:  # type: ignore[type-arg]
        problem_id = self._get_selected_problem_id()
        if problem_id is None:
            self._name_label.config(text="")
            self._type_label.config(text="")
            self._set_details_text("")
            return

        descriptor = PROBLEM_REGISTRY[problem_id]
        doc = get_problem_doc(problem_id)
        self._name_label.config(text=descriptor.name)
        self._type_label.config(text=doc.problem_type)
        details = (
            f"Description:\n{doc.extended_description}\n\n"
            "Configurable options:\n"
            + "\n".join(f"• {line}" for line in doc.config_options_summary)
            + "\n\nOutputs / visualizations:\n"
            + "\n".join(f"• {line}" for line in doc.visualizations_summary)
        )
        self._set_details_text(details)

    def _set_details_text(self, text: str) -> None:
        self._details_label.config(text=text)
        self._details_scroll.refresh_scroll_region()

    def _on_open(self, _event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        problem_id = self._get_selected_problem_id()
        if problem_id is None:
            return
        self.win.destroy()
        open_problem_dialog(problem_id, self.parent)
