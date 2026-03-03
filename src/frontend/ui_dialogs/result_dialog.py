"""Result dialog — left panel (stats, info, export) + right panel (interactive plots)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any

import numpy as np

from config import generate_output_basename, get_env_from_schema, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import center_window, make_modal
from solver.notation import FNotation, generate_derivative_labels, generate_phase_space_options
from utils import export_csv_to_path, export_json_to_path, get_logger

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from pipeline import SolverResult

logger = get_logger(__name__)

_MAGNITUDE_KEYS = {"mean", "rms", "std", "integral"}

_LEFT_MIN_WIDTH = 580


class ResultDialog:
    """Window showing the solution with interactive plot tabs.

    Plots are generated on-demand from the raw solver data.  The user
    selects *what* to visualise (derivatives, phase-space axes, etc.)
    inside the result window rather than before solving.

    Args:
        parent: Parent window.
        result: A data-only ``SolverResult`` from the pipeline.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        result: SolverResult,
    ) -> None:
        self.parent = parent
        self._result = result
        self._notation: FNotation = result.notation or FNotation(
            kind="ode", order=result.vector_order
        )

        self.win = tk.Toplevel(parent)
        self.win.title(f"Results — {result.metadata.get('equation_name', 'ODE')}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        # Canvas references for cleanup
        self._canvases: list[FigureCanvasTkAgg] = []

        self._build_ui()

        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        win_w = int(screen_w * 0.88)
        win_h = min(int(screen_h * 0.85), 900)

        center_window(self.win, win_w, win_h, max_width_ratio=0.92, resizable=True)
        self.win.minsize(_LEFT_MIN_WIDTH + 500, 500)
        make_modal(self.win, parent)
        logger.info("Result dialog displayed")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        result = self._result

        # ── Fixed bottom button bar ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_close = ttk.Button(
            btn_frame,
            text="Close",
            style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack()
        setup_arrow_enter_navigation([[btn_close]])
        btn_close.focus_set()

        # ── Main content area (grid: left info | right plot) ──
        content = ttk.Frame(self.win)
        content.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        content.columnconfigure(0, weight=0, minsize=_LEFT_MIN_WIDTH)
        content.columnconfigure(1, weight=1, minsize=400)
        content.rowconfigure(0, weight=1)

        # ── LEFT: scrollable info panel ──
        left_frame = ttk.Frame(content, width=_LEFT_MIN_WIDTH)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, pad))
        left_frame.grid_propagate(False)

        left_scroll = ScrollableFrame(left_frame)
        left_scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        left_scroll.pack(fill=tk.BOTH, expand=True)
        left_inner = left_scroll.inner
        left_inner.configure(padding=pad)

        self._build_left_panel(left_inner, left_scroll, result.statistics, result.metadata, pad)
        left_scroll.bind_new_children()

        # ── RIGHT: plots ──
        right_frame = ttk.Frame(content)
        right_frame.grid(row=0, column=1, sticky="nsew")

        self._notebook = ttk.Notebook(right_frame)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        self._build_plot_tabs()

    def _build_left_panel(
        self,
        inner: ttk.Frame,
        scroll: ScrollableFrame,
        statistics: dict[str, Any],
        metadata: dict[str, Any],
        pad: int,
    ) -> None:
        """Build magnitudes, statistics, solver info, and export sections."""
        magnitudes = {k: v for k, v in statistics.items() if k in _MAGNITUDE_KEYS}
        other_stats = {k: v for k, v in statistics.items() if k not in _MAGNITUDE_KEYS}

        if magnitudes:
            mag_section = CollapsibleSection(
                inner, scroll, "Magnitudes", expanded=True, pad=pad
            )
            for key, val in magnitudes.items():
                self._render_stat_entry(mag_section.content, key, val, pad)

        if other_stats:
            stat_section = CollapsibleSection(
                inner, scroll, "Statistics", expanded=True, pad=pad
            )
            for key, val in other_stats.items():
                self._render_stat_entry(stat_section.content, key, val, pad)

        # Solver info
        info_section = CollapsibleSection(
            inner, scroll, "Solver Info", expanded=True, pad=pad
        )
        info_items: list[tuple[str, Any]] = [
            ("Method", metadata.get("method", "?")),
            ("Success", "Yes" if metadata.get("solver_success") else "No"),
            ("Evaluations", metadata.get("n_evaluations", "?")),
            ("Points", metadata.get("num_points", "?")),
        ]
        if metadata.get("rtol") is not None:
            info_items.append(("rtol", metadata["rtol"]))
        if metadata.get("atol") is not None:
            info_items.append(("atol", metadata["atol"]))
        if metadata.get("residual_max") is not None:
            info_items.append(("Residual max", f"{metadata['residual_max']:.2e}"))
        if metadata.get("residual_mean") is not None:
            info_items.append(("Residual mean", f"{metadata['residual_mean']:.2e}"))
        if metadata.get("residual_rms") is not None:
            info_items.append(("Residual RMS", f"{metadata['residual_rms']:.2e}"))
        if metadata.get("n_jacobian_evals") is not None:
            info_items.append(("Jacobian evals", metadata["n_jacobian_evals"]))
        for label, value in info_items:
            row = ttk.Frame(info_section.content)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=16, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, text=str(value), style="Small.TLabel").pack(side=tk.LEFT)

        # Export
        export_section = CollapsibleSection(
            inner, scroll, "Export Data", expanded=True, pad=pad
        )
        btn_row = ttk.Frame(export_section.content)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="Save CSV...", command=self._on_save_csv).pack(
            side=tk.LEFT, padx=(0, pad)
        )
        ttk.Button(btn_row, text="Save JSON...", command=self._on_save_json).pack(
            side=tk.LEFT
        )

    # ------------------------------------------------------------------
    # Plot tab construction
    # ------------------------------------------------------------------

    def _build_plot_tabs(self) -> None:
        """Create the right-side tabs based on equation type."""
        r = self._result
        eq_type = r.equation_type

        is_2d_pde = eq_type == "pde" and r.y_grid is not None

        if is_2d_pde:
            self._build_pde_tabs()
        elif eq_type == "vector_ode" or (r.is_vector and r.vector_components > 1):
            self._build_vector_ode_tabs()
        elif eq_type == "difference":
            self._build_ode_scalar_tabs()  # same layout: solution + (no phase for 1st-order)
        else:
            self._build_ode_scalar_tabs()

    # ── ODE scalar / difference ──────────────────────────────────────

    def _build_ode_scalar_tabs(self) -> None:
        """Solution 2D (multi-select derivatives) + Phase Space (axis dropdowns)."""
        r = self._result
        notation = self._notation
        nb = self._notebook

        # --- Tab 1: Solution 2D ---
        sol_tab = ttk.Frame(nb)
        nb.add(sol_tab, text="  Solution f(x)  ")

        ctrl = ttk.Frame(sol_tab)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        self._sol_labels = generate_derivative_labels(notation)
        ttk.Label(ctrl, text="Show:").pack(side=tk.LEFT, padx=(0, 4))

        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        select_bg: str = get_env_from_schema("UI_BUTTON_FG")

        self._sol_listbox = tk.Listbox(
            ctrl, selectmode=tk.EXTENDED, height=min(len(self._sol_labels), 4),
            width=12, bg=btn_bg, fg=fg, selectbackground=select_bg,
            selectforeground="#000000", exportselection=False,
        )
        for lbl in self._sol_labels:
            self._sol_listbox.insert(tk.END, lbl)
        self._sol_listbox.select_set(0)
        self._sol_listbox.pack(side=tk.LEFT, padx=4)
        self._sol_listbox.bind("<<ListboxSelect>>", lambda _e: self._update_solution_plot())

        self._sol_plot_frame = ttk.Frame(sol_tab)
        self._sol_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._sol_canvas: FigureCanvasTkAgg | None = None
        self._update_solution_plot()

        # --- Tab 2: Phase Space ---
        order = r.vector_order
        if order >= 2 or r.equation_type != "difference":
            phase_tab = ttk.Frame(nb)
            nb.add(phase_tab, text="  Phase Space  ")

            phase_ctrl = ttk.Frame(phase_tab)
            phase_ctrl.pack(fill=tk.X, padx=4, pady=4)

            ps_options = generate_phase_space_options(notation)
            ps_labels = [lbl for lbl, _ in ps_options]

            ttk.Label(phase_ctrl, text="X-axis:").pack(side=tk.LEFT, padx=(0, 2))
            # Default phase portrait: f vs f' for order>=2, x vs f for order 1
            if order >= 2 and len(ps_labels) >= 3:
                default_x = ps_labels[1]  # f
                default_y_ax = ps_labels[2]  # f'
            elif len(ps_labels) >= 2:
                default_x = ps_labels[0]  # x
                default_y_ax = ps_labels[1]  # f
            else:
                default_x = ps_labels[0] if ps_labels else "f"
                default_y_ax = ps_labels[0] if ps_labels else "f"
            self._phase_x_var = tk.StringVar(value=default_x)
            phase_x_combo = ttk.Combobox(
                phase_ctrl, textvariable=self._phase_x_var,
                values=ps_labels, state="readonly", width=12,
            )
            phase_x_combo.pack(side=tk.LEFT, padx=(0, 8))

            ttk.Label(phase_ctrl, text="Y-axis:").pack(side=tk.LEFT, padx=(0, 2))
            self._phase_y_var = tk.StringVar(value=default_y_ax)
            phase_y_combo = ttk.Combobox(
                phase_ctrl, textvariable=self._phase_y_var,
                values=ps_labels, state="readonly", width=12,
            )
            phase_y_combo.pack(side=tk.LEFT, padx=(0, 8))

            ttk.Button(
                phase_ctrl, text="Update", command=self._update_phase_plot
            ).pack(side=tk.LEFT, padx=4)

            self._phase_options_map = {lbl: idx for lbl, idx in ps_options}
            self._phase_plot_frame = ttk.Frame(phase_tab)
            self._phase_plot_frame.pack(fill=tk.BOTH, expand=True)
            self._phase_canvas: FigureCanvasTkAgg | None = None
            self._update_phase_plot()

    def _update_solution_plot(self) -> None:
        """Regenerate the solution f(x) plot with currently selected derivatives."""
        from plotting import create_solution_plot

        r = self._result
        selected = list(self._sol_listbox.curselection())
        if not selected:
            selected = [0]

        xlabel = "n" if r.equation_type == "difference" else "x"
        eq_name = r.metadata.get("equation_name", "f(x)")

        fig = create_solution_plot(
            r.x, r.y,
            title=eq_name,
            xlabel=xlabel,
            ylabel="f",
            selected_derivatives=selected,
            labels=self._sol_labels,
        )
        self._replace_plot(self._sol_plot_frame, fig, "_sol_canvas")

    def _update_phase_plot(self) -> None:
        """Regenerate the phase portrait with selected axes."""
        from plotting import create_phase_plot

        r = self._result
        eq_name = r.metadata.get("equation_name", "Phase")

        x_label = self._phase_x_var.get()
        y_label = self._phase_y_var.get()
        x_idx = self._phase_options_map.get(x_label)
        y_idx = self._phase_options_map.get(y_label)

        y_2d = np.atleast_2d(r.y)
        if y_2d.shape[1] != len(r.x):
            y_2d = y_2d.T

        # Build the two data arrays (None index means independent variable x)
        if x_idx is None:
            horiz = r.x
        elif x_idx < y_2d.shape[0]:
            horiz = y_2d[x_idx]
        else:
            horiz = r.x

        if y_idx is None:
            vert = r.x
        elif y_idx < y_2d.shape[0]:
            vert = y_2d[y_idx]
        else:
            vert = y_2d[0]

        # Build a 2-row array for create_phase_plot
        phase_data = np.vstack([horiz, vert])

        fig = create_phase_plot(
            phase_data,
            title=f"{eq_name} — Phase",
            xlabel=x_label,
            ylabel=y_label,
        )
        self._replace_plot(self._phase_plot_frame, fig, "_phase_canvas")

    # ── Vector ODE ───────────────────────────────────────────────────

    def _build_vector_ode_tabs(self) -> None:
        """Solution 2D + Phase Space 2D + Phase Space 3D + Animation + 3D for vector ODEs."""
        r = self._result
        notation = self._notation
        nb = self._notebook

        # --- Tab 1: Solution 2D ---
        sol_tab = ttk.Frame(nb)
        nb.add(sol_tab, text="  Solution f(x)  ")

        ctrl = ttk.Frame(sol_tab)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        self._vec_sol_labels = generate_derivative_labels(notation)
        ttk.Label(ctrl, text="Show:").pack(side=tk.LEFT, padx=(0, 4))

        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        select_bg: str = get_env_from_schema("UI_BUTTON_FG")

        self._vec_sol_listbox = tk.Listbox(
            ctrl, selectmode=tk.EXTENDED,
            height=min(len(self._vec_sol_labels), 6), width=12,
            bg=btn_bg, fg=fg, selectbackground=select_bg,
            selectforeground="#000000", exportselection=False,
        )
        for lbl in self._vec_sol_labels:
            self._vec_sol_listbox.insert(tk.END, lbl)
        self._vec_sol_listbox.select_set(0)
        self._vec_sol_listbox.pack(side=tk.LEFT, padx=4)
        self._vec_sol_listbox.bind(
            "<<ListboxSelect>>", lambda _e: self._update_vec_solution_plot()
        )

        self._vec_sol_plot_frame = ttk.Frame(sol_tab)
        self._vec_sol_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._vec_sol_canvas: FigureCanvasTkAgg | None = None
        self._update_vec_solution_plot()

        # --- Tab 2: Phase Space 2D ---
        phase_tab = ttk.Frame(nb)
        nb.add(phase_tab, text="  Phase Space  ")

        phase_ctrl = ttk.Frame(phase_tab)
        phase_ctrl.pack(fill=tk.X, padx=4, pady=4)

        ps_options = generate_phase_space_options(notation)
        ps_labels = [lbl for lbl, _ in ps_options]

        # Default: f₀ vs f′₀ (component 0 value vs its first derivative)
        # ps_labels[0] = "x", ps_labels[1] = "f₀", ps_labels[2] = "f′₀" (for order >= 2)
        if r.vector_order >= 2 and len(ps_labels) >= 3:
            default_x_phase = ps_labels[1]   # f₀
            default_y_phase = ps_labels[2]   # f′₀
        elif len(ps_labels) >= 2:
            default_x_phase = ps_labels[0]   # x
            default_y_phase = ps_labels[1]   # f₀
        else:
            default_x_phase = ps_labels[0] if ps_labels else "f"
            default_y_phase = ps_labels[0] if ps_labels else "f"

        ttk.Label(phase_ctrl, text="X-axis:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase_x_var = tk.StringVar(value=default_x_phase)
        ttk.Combobox(
            phase_ctrl, textvariable=self._vec_phase_x_var,
            values=ps_labels, state="readonly", width=14,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(phase_ctrl, text="Y-axis:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase_y_var = tk.StringVar(value=default_y_phase)
        ttk.Combobox(
            phase_ctrl, textvariable=self._vec_phase_y_var,
            values=ps_labels, state="readonly", width=14,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            phase_ctrl, text="Update", command=self._update_vec_phase_plot
        ).pack(side=tk.LEFT, padx=4)

        self._vec_phase_options_map = {lbl: idx for lbl, idx in ps_options}
        self._vec_phase_plot_frame = ttk.Frame(phase_tab)
        self._vec_phase_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._vec_phase_canvas: FigureCanvasTkAgg | None = None
        self._update_vec_phase_plot()

        # --- Tab 3: Phase Space 3D ---
        phase3d_tab = ttk.Frame(nb)
        nb.add(phase3d_tab, text="  Phase 3D  ")

        phase3d_ctrl = ttk.Frame(phase3d_tab)
        phase3d_ctrl.pack(fill=tk.X, padx=4, pady=4)

        # Default axes: f₀, f₁, f₂ for 3+ components; x, f₀, f₁ otherwise
        n_comp = r.vector_components
        if n_comp >= 3 and len(ps_labels) >= 4:
            # ps_labels: x, f₀, f′₀, f₁, f′₁, f₂, f′₂, ...
            # Find the first 3 "base" component labels (derivative 0 of each)
            order = r.vector_order
            def_3d_x = ps_labels[1]                    # f₀
            def_3d_y = ps_labels[1 + order]             # f₁
            def_3d_z = ps_labels[1 + 2 * order]         # f₂
        elif n_comp >= 2 and len(ps_labels) >= 3:
            order = r.vector_order
            def_3d_x = ps_labels[0]                     # x
            def_3d_y = ps_labels[1]                     # f₀
            def_3d_z = ps_labels[1 + order]              # f₁
        else:
            def_3d_x = ps_labels[0] if ps_labels else "x"
            def_3d_y = ps_labels[1] if len(ps_labels) > 1 else def_3d_x
            def_3d_z = ps_labels[2] if len(ps_labels) > 2 else def_3d_y

        ttk.Label(phase3d_ctrl, text="X:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase3d_x_var = tk.StringVar(value=def_3d_x)
        ttk.Combobox(
            phase3d_ctrl, textvariable=self._vec_phase3d_x_var,
            values=ps_labels, state="readonly", width=12,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(phase3d_ctrl, text="Y:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase3d_y_var = tk.StringVar(value=def_3d_y)
        ttk.Combobox(
            phase3d_ctrl, textvariable=self._vec_phase3d_y_var,
            values=ps_labels, state="readonly", width=12,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(phase3d_ctrl, text="Z:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase3d_z_var = tk.StringVar(value=def_3d_z)
        ttk.Combobox(
            phase3d_ctrl, textvariable=self._vec_phase3d_z_var,
            values=ps_labels, state="readonly", width=12,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            phase3d_ctrl, text="Update", command=self._update_vec_phase_3d
        ).pack(side=tk.LEFT, padx=4)

        self._vec_phase3d_plot_frame = ttk.Frame(phase3d_tab)
        self._vec_phase3d_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._vec_phase3d_canvas: FigureCanvasTkAgg | None = None
        self._update_vec_phase_3d()

        # --- Tab 4: Animation ---
        anim_tab = ttk.Frame(nb)
        nb.add(anim_tab, text="  Animation  ")

        anim_ctrl = ttk.Frame(anim_tab)
        anim_ctrl.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(anim_ctrl, text="Derivative order:").pack(side=tk.LEFT, padx=(0, 4))
        self._anim_order_var = tk.StringVar(value="0")
        orders = [str(k) for k in range(r.vector_order)]
        ttk.Combobox(
            anim_ctrl, textvariable=self._anim_order_var,
            values=orders, state="readonly", width=6,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            anim_ctrl, text="Update", command=self._update_animation
        ).pack(side=tk.LEFT, padx=4)

        self._anim_plot_frame = ttk.Frame(anim_tab)
        self._anim_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._update_animation()

        # --- Tab 5: 3D Surface ---
        tab_3d = ttk.Frame(nb)
        nb.add(tab_3d, text="  3D Surface  ")

        ctrl_3d = ttk.Frame(tab_3d)
        ctrl_3d.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(ctrl_3d, text="Derivative order:").pack(side=tk.LEFT, padx=(0, 4))
        self._3d_order_var = tk.StringVar(value="0")
        ttk.Combobox(
            ctrl_3d, textvariable=self._3d_order_var,
            values=orders, state="readonly", width=6,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            ctrl_3d, text="Update", command=self._update_3d_plot
        ).pack(side=tk.LEFT, padx=4)

        self._3d_plot_frame = ttk.Frame(tab_3d)
        self._3d_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._3d_canvas: FigureCanvasTkAgg | None = None
        self._update_3d_plot()

    def _update_vec_solution_plot(self) -> None:
        """Regenerate vector ODE solution plot."""
        from plotting import create_solution_plot

        r = self._result
        selected = list(self._vec_sol_listbox.curselection())
        if not selected:
            selected = [0]

        eq_name = r.metadata.get("equation_name", "f(x)")
        fig = create_solution_plot(
            r.x, r.y,
            title=eq_name,
            xlabel="x",
            ylabel="f",
            selected_derivatives=selected,
            labels=self._vec_sol_labels,
        )
        self._replace_plot(self._vec_sol_plot_frame, fig, "_vec_sol_canvas")

    def _update_vec_phase_plot(self) -> None:
        """Regenerate vector ODE phase portrait."""
        from plotting import create_phase_plot

        r = self._result
        eq_name = r.metadata.get("equation_name", "Phase")

        x_label = self._vec_phase_x_var.get()
        y_label = self._vec_phase_y_var.get()
        x_idx = self._vec_phase_options_map.get(x_label)
        y_idx = self._vec_phase_options_map.get(y_label)

        y_2d = np.atleast_2d(r.y)
        if y_2d.shape[1] != len(r.x):
            y_2d = y_2d.T

        if x_idx is None:
            horiz = r.x
        elif x_idx < y_2d.shape[0]:
            horiz = y_2d[x_idx]
        else:
            horiz = r.x

        if y_idx is None:
            vert = r.x
        elif y_idx < y_2d.shape[0]:
            vert = y_2d[y_idx]
        else:
            vert = y_2d[0]

        phase_data = np.vstack([horiz, vert])
        fig = create_phase_plot(
            phase_data,
            title=f"{eq_name} — Phase",
            xlabel=x_label,
            ylabel=y_label,
        )
        self._replace_plot(self._vec_phase_plot_frame, fig, "_vec_phase_canvas")

    def _update_vec_phase_3d(self) -> None:
        """Regenerate vector ODE 3D phase-space trajectory."""
        from plotting import create_phase_3d_plot

        r = self._result
        eq_name = r.metadata.get("equation_name", "Phase 3D")

        x_label = self._vec_phase3d_x_var.get()
        y_label = self._vec_phase3d_y_var.get()
        z_label = self._vec_phase3d_z_var.get()

        y_2d = np.atleast_2d(r.y)
        if y_2d.shape[1] != len(r.x):
            y_2d = y_2d.T

        def _get_data(label: str) -> np.ndarray:
            idx = self._vec_phase_options_map.get(label)
            if idx is None:
                return r.x
            if idx < y_2d.shape[0]:
                return y_2d[idx]
            return r.x

        fig = create_phase_3d_plot(
            _get_data(x_label),
            _get_data(y_label),
            _get_data(z_label),
            title=f"{eq_name} — Phase 3D",
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
        )
        self._replace_plot(self._vec_phase3d_plot_frame, fig, "_vec_phase3d_canvas")

    def _update_animation(self) -> None:
        """Regenerate the animation tab."""
        from plotting import create_vector_animation_plot

        r = self._result
        eq_name = r.metadata.get("equation_name", "ODE")
        deriv_k = int(self._anim_order_var.get())

        fig = create_vector_animation_plot(
            r.x, r.y,
            order=r.vector_order,
            vector_components=r.vector_components,
            title=f"{eq_name} — f_i(x) (k={deriv_k})",
            deriv_offset=deriv_k,
        )
        # Clear existing widgets
        for w in self._anim_plot_frame.winfo_children():
            w.destroy()

        def _export_cb(dur: float) -> None:
            self._on_export_animation_mp4(dur, deriv_k)

        embed_animation_plot_in_tk(fig, self._anim_plot_frame, on_export_mp4=_export_cb)

    def _update_3d_plot(self) -> None:
        """Regenerate the 3D surface tab."""
        from plotting import create_vector_animation_3d

        r = self._result
        eq_name = r.metadata.get("equation_name", "ODE")
        deriv_k = int(self._3d_order_var.get())

        fig = create_vector_animation_3d(
            r.x, r.y,
            order=r.vector_order,
            vector_components=r.vector_components,
            title=f"{eq_name} — 3D (k={deriv_k})",
            deriv_offset=deriv_k,
        )
        self._replace_plot(self._3d_plot_frame, fig, "_3d_canvas")

    # ── PDE ──────────────────────────────────────────────────────────

    def _build_pde_tabs(self) -> None:
        """Solution 3D (surface) + Solution 2D (contour) + Phase Space slice."""
        nb = self._notebook

        # --- Tab 1: 3D Surface ---
        surf_tab = ttk.Frame(nb)
        nb.add(surf_tab, text="  Solution 3D  ")
        self._pde_3d_frame = ttk.Frame(surf_tab)
        self._pde_3d_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_3d_canvas: FigureCanvasTkAgg | None = None
        self._update_pde_3d()

        # --- Tab 2: 2D Contour ---
        contour_tab = ttk.Frame(nb)
        nb.add(contour_tab, text="  Solution 2D  ")
        self._pde_2d_frame = ttk.Frame(contour_tab)
        self._pde_2d_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_2d_canvas: FigureCanvasTkAgg | None = None
        self._update_pde_2d()

    def _update_pde_3d(self) -> None:
        """Render the 3D surface plot for PDE."""
        from plotting import create_surface_plot

        r = self._result
        eq_name = r.metadata.get("equation_name", "f(x,y)")
        fig = create_surface_plot(
            r.x, r.y_grid, r.y,
            title=eq_name,
            xlabel="x", ylabel="y", zlabel="f",
        )
        self._replace_plot(self._pde_3d_frame, fig, "_pde_3d_canvas")

    def _update_pde_2d(self) -> None:
        """Render the 2D contour plot for PDE."""
        from plotting import create_contour_plot

        r = self._result
        eq_name = r.metadata.get("equation_name", "f(x,y)")
        fig = create_contour_plot(
            r.x, r.y_grid, r.y,
            title=eq_name,
            xlabel="x", ylabel="y",
        )
        self._replace_plot(self._pde_2d_frame, fig, "_pde_2d_canvas")

    # ------------------------------------------------------------------
    # Plot replacement helper
    # ------------------------------------------------------------------

    def _replace_plot(
        self,
        frame: ttk.Frame,
        fig: Figure,
        canvas_attr: str,
    ) -> None:
        """Destroy the old canvas in *frame* and embed *fig* in its place."""
        import matplotlib.pyplot as plt

        old_canvas: FigureCanvasTkAgg | None = getattr(self, canvas_attr, None)
        if old_canvas is not None:
            old_fig = old_canvas.figure
            old_canvas.get_tk_widget().destroy()
            plt.close(old_fig)

        for w in frame.winfo_children():
            w.destroy()

        canvas = embed_plot_in_tk(fig, frame)
        setattr(self, canvas_attr, canvas)

    # ------------------------------------------------------------------
    # Stat rendering
    # ------------------------------------------------------------------

    def _render_stat_entry(self, parent: tk.Widget, key: str, val: Any, pad: int) -> None:
        if isinstance(val, dict):
            hdr = ttk.Frame(parent)
            hdr.pack(fill=tk.X, pady=(2, 0))
            ttk.Label(hdr, text=f"{key}:", width=16, anchor=tk.W, style="Small.TLabel").pack(
                side=tk.LEFT
            )
            for sub_key, sub_val in val.items():
                sub_row = ttk.Frame(parent)
                sub_row.pack(fill=tk.X, pady=0)
                ttk.Label(sub_row, text=f"  {sub_key}:", width=22, anchor=tk.W).pack(side=tk.LEFT)
                formatted = f"{sub_val:.6g}" if isinstance(sub_val, float) else str(sub_val)
                ttk.Label(sub_row, text=formatted, style="Small.TLabel").pack(
                    side=tk.LEFT, padx=(2, 0)
                )
        else:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{key}:", width=16, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, text=self._format_stat(val), style="Small.TLabel").pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_export_file(
        self,
        export_fn,
        ext: str,
        filetypes: list[tuple[str, str]],
        prefix_log: str = "",
    ) -> None:
        default_path = get_output_dir() / f"{generate_output_basename()}{ext}"
        filepath = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=ext,
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=filetypes,
        )
        if not filepath:
            return
        path = Path(filepath)
        try:
            export_fn(path)
            messagebox.showinfo(
                "Export Complete",
                f"{prefix_log} saved to:\n{path}",
                parent=self.win,
            )
        except Exception as exc:
            logger.error(f"{prefix_log} export failed: %s", exc, exc_info=True)
            messagebox.showerror("Export Failed", str(exc), parent=self.win)

    def _on_save_csv(self) -> None:
        r = self._result

        def export_fn(path: str) -> None:
            export_csv_to_path(r.x, r.y, path, y_grid=r.y_grid)

        self._save_export_file(
            export_fn, ".csv",
            [("CSV files", "*.csv"), ("All files", "*.*")],
            "CSV",
        )

    def _on_save_json(self) -> None:
        r = self._result

        def export_fn(path: str) -> None:
            export_json_to_path(r.statistics, r.metadata, path)

        self._save_export_file(
            export_fn, ".json",
            [("JSON files", "*.json"), ("All files", "*.*")],
            "JSON",
        )

    def _on_export_animation_mp4(self, duration_seconds: float, deriv_k: int = 0) -> None:
        r = self._result
        default_path = get_output_dir() / f"{generate_output_basename(prefix='animation')}.mp4"
        filepath_str = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".mp4",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if not filepath_str:
            return
        filepath = Path(filepath_str)
        try:
            from plotting import export_animation_to_mp4

            export_animation_to_mp4(
                r.x, r.y,
                r.vector_order,
                r.vector_components,
                filepath,
                title=f"{r.metadata.get('equation_name', 'ODE')} — f_i(x)",
                duration_seconds=duration_seconds,
            )
            messagebox.showinfo(
                "Export Complete",
                f"Animation saved to:\n{filepath}",
                parent=self.win,
            )
        except RuntimeError as exc:
            logger.warning("MP4 export failed (ffmpeg): %s", exc)
            messagebox.showerror(
                "Export Failed",
                str(exc) + "\n\nInstall ffmpeg and ensure it is in your PATH.",
                parent=self.win,
            )
        except Exception as exc:
            logger.error("MP4 export failed: %s", exc, exc_info=True)
            messagebox.showerror("Export Failed", str(exc), parent=self.win)

    @staticmethod
    def _format_stat(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, dict):
            parts = [
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in value.items()
            ]
            return ", ".join(parts)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
