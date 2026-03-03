"""Result dialog — left panel (stats, info, export) + right panel (interactive plots)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any

import numpy as np

from config import generate_output_basename, get_env_from_schema, get_output_dir
from frontend.plot_embed import embed_animation_plot_in_tk, embed_plot_in_tk
from frontend.theme import get_contrast_foreground, get_font
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

_MAGNITUDE_KEYS = {"mean", "rms", "std", "integral", "l2_norm", "half_life", "time_constant", "doubling_time", "angular_frequency"}

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
        win_w = int(screen_w * 0.94)
        win_h = min(int(screen_h * 0.85), 900)

        center_window(self.win, win_w, win_h, max_width_ratio=0.96, resizable=True)
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
    # Transform controls helper
    # ------------------------------------------------------------------

    def _build_transform_controls(
        self,
        parent: ttk.Frame,
        callback: Any,
        prefix: str,
        *,
        label_style: str | None = None,
    ) -> None:
        """Add a transform dropdown to a tab's control bar.

        The ``StringVar`` is stored as ``self._transform_{prefix}_var``.
        """
        from transforms import TransformKind

        sep = ttk.Separator(parent, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        label_kw = {"style": label_style} if label_style else {}
        ttk.Label(parent, text="Transform:", **label_kw).pack(side=tk.LEFT, padx=(0, 4))
        var = tk.StringVar(value=TransformKind.ORIGINAL.value)
        setattr(self, f"_transform_{prefix}_var", var)

        combo = ttk.Combobox(
            parent,
            textvariable=var,
            values=[k.value for k in TransformKind],
            state="readonly",
            width=20,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT, padx=(0, 4))
        combo.bind("<<ComboboxSelected>>", lambda _e: callback())

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
        select_fg: str = get_contrast_foreground(select_bg)

        self._sol_listbox = tk.Listbox(
            ctrl, selectmode=tk.EXTENDED, height=min(len(self._sol_labels), 4),
            width=12, bg=btn_bg, fg=fg, selectbackground=select_bg,
            selectforeground=select_fg, exportselection=False,
        )
        for lbl in self._sol_labels:
            self._sol_listbox.insert(tk.END, lbl)
        self._sol_listbox.select_set(0)
        self._sol_listbox.pack(side=tk.LEFT, padx=4)
        self._sol_listbox.bind("<<ListboxSelect>>", lambda _e: self._update_solution_plot())

        self._build_transform_controls(ctrl, self._update_solution_plot, "sol")

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
                values=ps_labels, state="readonly", width=6,
                font=get_font(),
            )
            phase_x_combo.pack(side=tk.LEFT, padx=(0, 8))
            phase_x_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_phase_plot())

            ttk.Label(phase_ctrl, text="Y-axis:").pack(side=tk.LEFT, padx=(0, 2))
            self._phase_y_var = tk.StringVar(value=default_y_ax)
            phase_y_combo = ttk.Combobox(
                phase_ctrl, textvariable=self._phase_y_var,
                values=ps_labels, state="readonly", width=6,
                font=get_font(),
            )
            phase_y_combo.pack(side=tk.LEFT, padx=(0, 8))
            phase_y_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_phase_plot())

            self._build_transform_controls(phase_ctrl, self._update_phase_plot, "phase")

            self._phase_options_map = {lbl: idx for lbl, idx in ps_options}
            self._phase_plot_frame = ttk.Frame(phase_tab)
            self._phase_plot_frame.pack(fill=tk.BOTH, expand=True)
            self._phase_canvas: FigureCanvasTkAgg | None = None
            self._update_phase_plot()

    def _apply_transform_multi(
        self,
        x: np.ndarray,
        y_2d: np.ndarray,
        selected: list[int],
        labels: list[str],
        kind: Any,
    ) -> tuple[np.ndarray, np.ndarray, list[str], str, str] | None:
        """Apply a transform to each selected row and align to a common x-axis.

        Returns ``(tx, ty_2d, trans_labels, txlabel, tylabel)`` or ``None``
        if nothing could be computed.
        """
        from scipy.interpolate import interp1d

        from transforms import apply_transform

        x_min_t, x_max_t = float(x[0]), float(x[-1])
        raw: list[tuple[np.ndarray, np.ndarray, str]] = []
        txlabel = tylabel = ""

        for idx in selected:
            if idx >= y_2d.shape[0]:
                continue
            func = interp1d(x, y_2d[idx], kind="cubic", fill_value="extrapolate")
            tx, ty, txlabel, tylabel = apply_transform(
                lambda arr, f=func: f(arr), kind, x_min_t, x_max_t,
            )
            lbl = labels[idx] if idx < len(labels) else f"f[{idx}]"
            raw.append((tx, ty, lbl))

        if not raw:
            return None

        # Use the longest x-axis as the common grid and interpolate the rest
        ref_tx = max(raw, key=lambda r: len(r[0]))[0]
        aligned_rows: list[np.ndarray] = []
        trans_labels: list[str] = []
        for tx_i, ty_i, lbl in raw:
            if len(tx_i) == len(ref_tx) and np.allclose(tx_i, ref_tx):
                aligned_rows.append(ty_i)
            else:
                f_interp = interp1d(tx_i, ty_i, kind="linear",
                                    bounds_error=False, fill_value=0.0)
                aligned_rows.append(f_interp(ref_tx))
            trans_labels.append(lbl)

        return ref_tx, np.vstack(aligned_rows), trans_labels, txlabel, tylabel

    def _get_transform_kind(self, prefix: str) -> Any:
        """Return the current TransformKind for the given control prefix."""
        from transforms import TransformKind

        var = getattr(self, f"_transform_{prefix}_var", None)
        if var is None:
            return TransformKind.ORIGINAL
        try:
            return TransformKind(var.get())
        except ValueError:
            return TransformKind.ORIGINAL

    def _update_solution_plot(self) -> None:
        """Regenerate the solution f(x) plot with currently selected derivatives."""
        from plotting import create_solution_plot
        from transforms import TransformKind

        r = self._result
        selected = list(self._sol_listbox.curselection())
        if not selected:
            selected = [0]

        xlabel = "n" if r.equation_type == "difference" else "x"
        eq_name = r.metadata.get("equation_name", "f(x)")

        kind = self._get_transform_kind("sol")

        if kind == TransformKind.ORIGINAL:
            fig = create_solution_plot(
                r.x, r.y,
                title=eq_name,
                xlabel=xlabel,
                ylabel="f",
                selected_derivatives=selected,
                labels=self._sol_labels,
            )
        else:
            y_2d = np.atleast_2d(r.y)
            if y_2d.shape[1] != len(r.x):
                y_2d = y_2d.T

            result = self._apply_transform_multi(
                r.x, y_2d, selected, self._sol_labels, kind,
            )
            if result is None:
                return
            tx, ty_2d, trans_labels, txlabel, tylabel = result

            fig = create_solution_plot(
                tx, ty_2d,
                title=f"{eq_name} \u2014 {kind.value}",
                xlabel=txlabel,
                ylabel=tylabel,
                selected_derivatives=list(range(ty_2d.shape[0])),
                labels=trans_labels,
            )

        self._replace_plot(self._sol_plot_frame, fig, "_sol_canvas")

    def _transform_phase_axes(
        self,
        x: np.ndarray,
        y_2d: np.ndarray,
        axis_specs: list[tuple[int | None, str]],
        kind: Any,
    ) -> list[tuple[np.ndarray, str]] | None:
        """Transform multiple axis data series for phase-space plots.

        Each element in *axis_specs* is ``(flat_index_or_None, label)``.
        Returns a list of ``(data_array, display_label)`` per axis, or ``None``
        if nothing could be computed.
        """
        # Collect all unique non-None indices that need transforming
        unique_indices: list[int] = []
        for idx, _ in axis_specs:
            if idx is not None and idx not in unique_indices:
                unique_indices.append(idx)

        if not unique_indices:
            # All axes selected the independent variable — nothing to transform
            return None

        labels_for_transform = [
            (f"y[{i}]" if i >= y_2d.shape[0] else f"row{i}")
            for i in unique_indices
        ]
        result = self._apply_transform_multi(
            x, y_2d, unique_indices, labels_for_transform, kind,
        )
        if result is None:
            return None
        tx, ty_2d, _tlabels, txlabel, _tylabel = result

        # Build a lookup from original index to transformed row index
        idx_to_row = {orig: row_i for row_i, orig in enumerate(unique_indices)}

        output: list[tuple[np.ndarray, str]] = []
        for idx, label in axis_specs:
            if idx is None:
                # Independent variable becomes the transform domain axis
                output.append((tx, txlabel))
            elif idx in idx_to_row:
                output.append((ty_2d[idx_to_row[idx]], label))
            else:
                output.append((tx, txlabel))
        return output

    def _update_phase_plot(self) -> None:
        """Regenerate the phase portrait with selected axes."""
        from plotting import create_phase_plot
        from transforms import TransformKind

        r = self._result
        eq_name = r.metadata.get("equation_name", "Phase")

        x_label = self._phase_x_var.get()
        y_label = self._phase_y_var.get()
        x_idx = self._phase_options_map.get(x_label)
        y_idx = self._phase_options_map.get(y_label)

        y_2d = np.atleast_2d(r.y)
        if y_2d.shape[1] != len(r.x):
            y_2d = y_2d.T

        kind = self._get_transform_kind("phase")

        if kind == TransformKind.ORIGINAL:
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

            disp_xlabel = x_label
            disp_ylabel = y_label
            title = f"{eq_name} \u2014 Phase"
        else:
            result = self._transform_phase_axes(
                r.x, y_2d,
                [(x_idx, x_label), (y_idx, y_label)],
                kind,
            )
            if result is None:
                return
            horiz, disp_xlabel = result[0]
            vert, disp_ylabel = result[1]
            title = f"{eq_name} \u2014 Phase \u2014 {kind.value}"

        phase_data = np.vstack([horiz, vert])

        fig = create_phase_plot(
            phase_data,
            title=title,
            xlabel=disp_xlabel,
            ylabel=disp_ylabel,
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
        select_fg: str = get_contrast_foreground(select_bg)

        self._vec_sol_listbox = tk.Listbox(
            ctrl, selectmode=tk.EXTENDED,
            height=min(len(self._vec_sol_labels), 6), width=12,
            bg=btn_bg, fg=fg, selectbackground=select_bg,
            selectforeground=select_fg, exportselection=False,
        )
        for lbl in self._vec_sol_labels:
            self._vec_sol_listbox.insert(tk.END, lbl)
        self._vec_sol_listbox.select_set(0)
        self._vec_sol_listbox.pack(side=tk.LEFT, padx=4)
        self._vec_sol_listbox.bind(
            "<<ListboxSelect>>", lambda _e: self._update_vec_solution_plot()
        )

        self._build_transform_controls(ctrl, self._update_vec_solution_plot, "vec_sol")

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
        vec_phase_x_combo = ttk.Combobox(
            phase_ctrl, textvariable=self._vec_phase_x_var,
            values=ps_labels, state="readonly", width=6,
            font=get_font(),
        )
        vec_phase_x_combo.pack(side=tk.LEFT, padx=(0, 8))
        vec_phase_x_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_plot())

        ttk.Label(phase_ctrl, text="Y-axis:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase_y_var = tk.StringVar(value=default_y_phase)
        vec_phase_y_combo = ttk.Combobox(
            phase_ctrl, textvariable=self._vec_phase_y_var,
            values=ps_labels, state="readonly", width=6,
            font=get_font(),
        )
        vec_phase_y_combo.pack(side=tk.LEFT, padx=(0, 8))
        vec_phase_y_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_plot())

        self._build_transform_controls(phase_ctrl, self._update_vec_phase_plot, "vec_phase")

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
        phase3d_x_combo = ttk.Combobox(
            phase3d_ctrl, textvariable=self._vec_phase3d_x_var,
            values=ps_labels, state="readonly", width=6,
            font=get_font(),
        )
        phase3d_x_combo.pack(side=tk.LEFT, padx=(0, 6))
        phase3d_x_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_3d())

        ttk.Label(phase3d_ctrl, text="Y:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase3d_y_var = tk.StringVar(value=def_3d_y)
        phase3d_y_combo = ttk.Combobox(
            phase3d_ctrl, textvariable=self._vec_phase3d_y_var,
            values=ps_labels, state="readonly", width=6,
            font=get_font(),
        )
        phase3d_y_combo.pack(side=tk.LEFT, padx=(0, 6))
        phase3d_y_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_3d())

        ttk.Label(phase3d_ctrl, text="Z:").pack(side=tk.LEFT, padx=(0, 2))
        self._vec_phase3d_z_var = tk.StringVar(value=def_3d_z)
        phase3d_z_combo = ttk.Combobox(
            phase3d_ctrl, textvariable=self._vec_phase3d_z_var,
            values=ps_labels, state="readonly", width=6,
            font=get_font(),
        )
        phase3d_z_combo.pack(side=tk.LEFT, padx=(0, 6))
        phase3d_z_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_3d())

        self._build_transform_controls(phase3d_ctrl, self._update_vec_phase_3d, "vec_phase3d")

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
        anim_order_combo = ttk.Combobox(
            anim_ctrl, textvariable=self._anim_order_var,
            values=orders, state="readonly", width=4,
            font=get_font(),
        )
        anim_order_combo.pack(side=tk.LEFT, padx=(0, 8))
        anim_order_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_animation())

        self._build_transform_controls(anim_ctrl, self._update_animation, "anim")

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
        order_3d_combo = ttk.Combobox(
            ctrl_3d, textvariable=self._3d_order_var,
            values=orders, state="readonly", width=4,
            font=get_font(),
        )
        order_3d_combo.pack(side=tk.LEFT, padx=(0, 8))
        order_3d_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_3d_plot())

        self._build_transform_controls(ctrl_3d, self._update_3d_plot, "vec_3d")

        self._3d_plot_frame = ttk.Frame(tab_3d)
        self._3d_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._3d_canvas: FigureCanvasTkAgg | None = None
        self._update_3d_plot()

    def _update_vec_solution_plot(self) -> None:
        """Regenerate vector ODE solution plot."""
        from plotting import create_solution_plot
        from transforms import TransformKind

        r = self._result
        selected = list(self._vec_sol_listbox.curselection())
        if not selected:
            selected = [0]

        eq_name = r.metadata.get("equation_name", "f(x)")
        kind = self._get_transform_kind("vec_sol")

        if kind == TransformKind.ORIGINAL:
            fig = create_solution_plot(
                r.x, r.y,
                title=eq_name,
                xlabel="x",
                ylabel="f",
                selected_derivatives=selected,
                labels=self._vec_sol_labels,
            )
        else:
            y_2d = np.atleast_2d(r.y)
            if y_2d.shape[1] != len(r.x):
                y_2d = y_2d.T

            result = self._apply_transform_multi(
                r.x, y_2d, selected, self._vec_sol_labels, kind,
            )
            if result is None:
                return
            tx, ty_2d, trans_labels, txlabel, tylabel = result

            fig = create_solution_plot(
                tx, ty_2d,
                title=f"{eq_name} \u2014 {kind.value}",
                xlabel=txlabel,
                ylabel=tylabel,
                selected_derivatives=list(range(ty_2d.shape[0])),
                labels=trans_labels,
            )

        self._replace_plot(self._vec_sol_plot_frame, fig, "_vec_sol_canvas")

    def _update_vec_phase_plot(self) -> None:
        """Regenerate vector ODE phase portrait."""
        from plotting import create_phase_plot
        from transforms import TransformKind

        r = self._result
        eq_name = r.metadata.get("equation_name", "Phase")

        x_label = self._vec_phase_x_var.get()
        y_label = self._vec_phase_y_var.get()
        x_idx = self._vec_phase_options_map.get(x_label)
        y_idx = self._vec_phase_options_map.get(y_label)

        y_2d = np.atleast_2d(r.y)
        if y_2d.shape[1] != len(r.x):
            y_2d = y_2d.T

        kind = self._get_transform_kind("vec_phase")

        if kind == TransformKind.ORIGINAL:
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

            disp_xlabel = x_label
            disp_ylabel = y_label
            title = f"{eq_name} \u2014 Phase"
        else:
            result = self._transform_phase_axes(
                r.x, y_2d,
                [(x_idx, x_label), (y_idx, y_label)],
                kind,
            )
            if result is None:
                return
            horiz, disp_xlabel = result[0]
            vert, disp_ylabel = result[1]
            title = f"{eq_name} \u2014 Phase \u2014 {kind.value}"

        phase_data = np.vstack([horiz, vert])
        fig = create_phase_plot(
            phase_data,
            title=title,
            xlabel=disp_xlabel,
            ylabel=disp_ylabel,
        )
        self._replace_plot(self._vec_phase_plot_frame, fig, "_vec_phase_canvas")

    def _update_vec_phase_3d(self) -> None:
        """Regenerate vector ODE 3D phase-space trajectory."""
        from plotting import create_phase_3d_plot
        from transforms import TransformKind

        r = self._result
        eq_name = r.metadata.get("equation_name", "Phase 3D")

        x_label = self._vec_phase3d_x_var.get()
        y_label = self._vec_phase3d_y_var.get()
        z_label = self._vec_phase3d_z_var.get()

        y_2d = np.atleast_2d(r.y)
        if y_2d.shape[1] != len(r.x):
            y_2d = y_2d.T

        kind = self._get_transform_kind("vec_phase3d")

        if kind == TransformKind.ORIGINAL:
            def _get_data(label: str) -> np.ndarray:
                idx = self._vec_phase_options_map.get(label)
                if idx is None:
                    return r.x
                if idx < y_2d.shape[0]:
                    return y_2d[idx]
                return r.x

            data_x = _get_data(x_label)
            data_y = _get_data(y_label)
            data_z = _get_data(z_label)
            disp_xlabel = x_label
            disp_ylabel = y_label
            disp_zlabel = z_label
            title = f"{eq_name} \u2014 Phase 3D"
        else:
            x_idx = self._vec_phase_options_map.get(x_label)
            y_idx = self._vec_phase_options_map.get(y_label)
            z_idx = self._vec_phase_options_map.get(z_label)

            result = self._transform_phase_axes(
                r.x, y_2d,
                [(x_idx, x_label), (y_idx, y_label), (z_idx, z_label)],
                kind,
            )
            if result is None:
                return
            data_x, disp_xlabel = result[0]
            data_y, disp_ylabel = result[1]
            data_z, disp_zlabel = result[2]
            title = f"{eq_name} \u2014 Phase 3D \u2014 {kind.value}"

        fig = create_phase_3d_plot(
            data_x, data_y, data_z,
            title=title,
            xlabel=disp_xlabel,
            ylabel=disp_ylabel,
            zlabel=disp_zlabel,
        )
        self._replace_plot(self._vec_phase3d_plot_frame, fig, "_vec_phase3d_canvas")

    def _transform_vector_components(
        self,
        x: np.ndarray,
        y: np.ndarray,
        order: int,
        vector_components: int,
        deriv_offset: int,
        kind: Any,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Transform each vector component independently over x.

        Returns ``(tx, transformed_y, txlabel)`` with the same shape convention
        as the original data, or ``None`` if transform fails.
        """
        y_2d = np.atleast_2d(y)
        if y_2d.shape[1] != len(x):
            y_2d = y_2d.T

        # Extract the relevant rows for each component at the given derivative offset
        indices = [i * order + deriv_offset for i in range(vector_components)]
        labels = [f"f_{i}" for i in range(vector_components)]

        result = self._apply_transform_multi(x, y_2d, indices, labels, kind)
        if result is None:
            return None
        tx, ty_2d, _labels, txlabel, _tylabel = result

        # Rebuild a full-size y array with the transformed components in the right slots
        n_state = vector_components * order
        new_y = np.zeros((n_state, len(tx)))
        for comp_i, orig_idx in enumerate(indices):
            if comp_i < ty_2d.shape[0]:
                new_y[orig_idx] = ty_2d[comp_i]

        return tx, new_y, txlabel

    def _update_animation(self) -> None:
        """Regenerate the animation tab."""
        from plotting import create_vector_animation_plot
        from transforms import TransformKind

        r = self._result
        eq_name = r.metadata.get("equation_name", "ODE")
        deriv_k = int(self._anim_order_var.get())
        kind = self._get_transform_kind("anim")

        if kind == TransformKind.ORIGINAL:
            fig = create_vector_animation_plot(
                r.x, r.y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 f_i(x) (k={deriv_k})",
                deriv_offset=deriv_k,
            )
        else:
            result = self._transform_vector_components(
                r.x, r.y, r.vector_order, r.vector_components, deriv_k, kind,
            )
            if result is None:
                return
            tx, new_y, txlabel = result
            fig = create_vector_animation_plot(
                tx, new_y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 {kind.value} (k={deriv_k})",
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
        from transforms import TransformKind

        r = self._result
        eq_name = r.metadata.get("equation_name", "ODE")
        deriv_k = int(self._3d_order_var.get())
        kind = self._get_transform_kind("vec_3d")

        if kind == TransformKind.ORIGINAL:
            fig = create_vector_animation_3d(
                r.x, r.y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 3D (k={deriv_k})",
                deriv_offset=deriv_k,
            )
        else:
            result = self._transform_vector_components(
                r.x, r.y, r.vector_order, r.vector_components, deriv_k, kind,
            )
            if result is None:
                return
            tx, new_y, txlabel = result
            fig = create_vector_animation_3d(
                tx, new_y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 {kind.value} 3D (k={deriv_k})",
                deriv_offset=deriv_k,
            )

        self._replace_plot(self._3d_plot_frame, fig, "_3d_canvas")

    # ── PDE ──────────────────────────────────────────────────────────

    def _build_pde_tabs(self) -> None:
        """Solution 3D (surface) + Solution 2D (contour) + Phase Space slice."""
        nb = self._notebook
        xlabel, ylabel = self._pde_axis_labels()

        # --- Tab 1: 3D Surface ---
        surf_tab = ttk.Frame(nb)
        nb.add(surf_tab, text="  Solution 3D  ")

        surf_ctrl = ttk.Frame(surf_tab)
        surf_ctrl.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(surf_ctrl, text="Transform along:").pack(side=tk.LEFT, padx=(0, 4))
        self._pde_3d_axis_var = tk.StringVar(value=xlabel)
        ttk.Combobox(
            surf_ctrl, textvariable=self._pde_3d_axis_var,
            values=[xlabel, ylabel], state="readonly", width=4,
            font=get_font(),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._build_transform_controls(surf_ctrl, self._update_pde_3d, "pde_3d")

        self._pde_3d_frame = ttk.Frame(surf_tab)
        self._pde_3d_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_3d_canvas: FigureCanvasTkAgg | None = None
        self._update_pde_3d()

        # --- Tab 2: 2D Contour ---
        contour_tab = ttk.Frame(nb)
        nb.add(contour_tab, text="  Solution 2D  ")

        contour_ctrl = ttk.Frame(contour_tab)
        contour_ctrl.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(contour_ctrl, text="Transform along:").pack(side=tk.LEFT, padx=(0, 4))
        self._pde_2d_axis_var = tk.StringVar(value=xlabel)
        ttk.Combobox(
            contour_ctrl, textvariable=self._pde_2d_axis_var,
            values=[xlabel, ylabel], state="readonly", width=4,
            font=get_font(),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._build_transform_controls(contour_ctrl, self._update_pde_2d, "pde_2d")

        self._pde_2d_frame = ttk.Frame(contour_tab)
        self._pde_2d_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_2d_canvas: FigureCanvasTkAgg | None = None
        self._update_pde_2d()

        # --- Tab 3: Transform (1D slice) ---
        trans_tab = ttk.Frame(nb)
        nb.add(trans_tab, text="  Transform  ")

        trans_ctrl = ttk.Frame(trans_tab)
        trans_ctrl.pack(fill=tk.X, padx=4, pady=4)

        xlabel, ylabel = self._pde_axis_labels()

        ttk.Label(
            trans_ctrl, text="Slice along:", style="Small.TLabel"
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._pde_slice_var = tk.StringVar(value=xlabel)
        ttk.Combobox(
            trans_ctrl, textvariable=self._pde_slice_var,
            values=[xlabel, ylabel], state="readonly", width=2,
            font=get_font(),
        ).pack(side=tk.LEFT, padx=(0, 2))

        r = self._result
        y_mid = (
            float((r.y_grid[0] + r.y_grid[-1]) / 2)
            if r.y_grid is not None and len(r.y_grid) > 0
            else 0.5
        )

        ttk.Label(
            trans_ctrl, text="at fixed value:", style="Small.TLabel"
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._pde_slice_val_var = tk.StringVar(value=str(round(y_mid, 4)))
        ttk.Entry(
            trans_ctrl, textvariable=self._pde_slice_val_var, width=4,
            font=get_font(),
        ).pack(side=tk.LEFT, padx=(0, 2))

        self._build_transform_controls(
            trans_ctrl, self._update_pde_transform, "pde",
            label_style="Small.TLabel",
        )

        ttk.Button(
            trans_ctrl, text="Update", command=self._update_pde_transform,
        ).pack(side=tk.LEFT, padx=4)

        self._pde_trans_frame = ttk.Frame(trans_tab)
        self._pde_trans_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_trans_canvas: FigureCanvasTkAgg | None = None
        self._update_pde_transform()

    def _pde_axis_labels(self) -> tuple[str, str]:
        """Return (xlabel, ylabel) from metadata variable names."""
        variables = self._result.metadata.get("variables", ["x[0]", "x[1]"])
        xlabel = variables[0] if len(variables) > 0 else "x[0]"
        ylabel = variables[1] if len(variables) > 1 else "x[1]"
        return xlabel, ylabel

    def _transform_pde_along_axis(
        self,
        x: np.ndarray,
        y_grid: np.ndarray,
        z: np.ndarray,
        axis_var: str,
        kind: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str] | None:
        """Transform PDE solution along one axis.

        Returns ``(new_x, new_y, new_z, new_xlabel, new_ylabel)`` or
        ``None`` if the transform cannot be applied.

        Because ``apply_transform`` trims spectra by amplitude and may
        return different-length arrays for each slice, we interpolate
        all results onto the domain grid from the first slice.
        """
        from scipy.interpolate import interp1d

        from transforms import apply_transform

        xlabel, ylabel = self._pde_axis_labels()

        if axis_var == xlabel:
            # Transform along x (columns): for each row, transform f vs x
            raw: list[tuple[np.ndarray, np.ndarray]] = []
            txlabel = ""
            for i in range(z.shape[0]):
                func = interp1d(x, z[i, :], kind="cubic", fill_value="extrapolate")
                tx, ty, txlabel, _tylabel = apply_transform(
                    lambda arr, f=func: f(arr), kind,
                    float(x[0]), float(x[-1]),
                )
                raw.append((tx, ty))
            if not raw:
                return None
            # Use first slice's domain as the common grid
            tx_common = raw[0][0]
            new_rows: list[np.ndarray] = []
            for tx_i, ty_i in raw:
                if len(tx_i) == len(tx_common) and np.allclose(tx_i, tx_common):
                    new_rows.append(ty_i)
                else:
                    resamp = interp1d(tx_i, ty_i, kind="linear",
                                      fill_value=0.0, bounds_error=False)
                    new_rows.append(resamp(tx_common))
            new_z = np.array(new_rows)
            return tx_common, y_grid, new_z, txlabel, ylabel
        else:
            # Transform along y_grid (rows): for each column, transform f vs y
            raw_c: list[tuple[np.ndarray, np.ndarray]] = []
            tylabel = ""
            for j in range(z.shape[1]):
                func = interp1d(y_grid, z[:, j], kind="cubic", fill_value="extrapolate")
                ty, tz, tylabel, _tzlabel = apply_transform(
                    lambda arr, f=func: f(arr), kind,
                    float(y_grid[0]), float(y_grid[-1]),
                )
                raw_c.append((ty, tz))
            if not raw_c:
                return None
            ty_common = raw_c[0][0]
            new_cols: list[np.ndarray] = []
            for ty_j, tz_j in raw_c:
                if len(ty_j) == len(ty_common) and np.allclose(ty_j, ty_common):
                    new_cols.append(tz_j)
                else:
                    resamp = interp1d(ty_j, tz_j, kind="linear",
                                      fill_value=0.0, bounds_error=False)
                    new_cols.append(resamp(ty_common))
            new_z = np.column_stack(new_cols)
            return x, ty_common, new_z, xlabel, tylabel

    def _update_pde_3d(self) -> None:
        """Render the 3D surface plot for PDE."""
        from plotting import create_surface_plot
        from transforms import TransformKind

        r = self._result
        xlabel, ylabel = self._pde_axis_labels()
        eq_name = r.metadata.get("equation_name", f"f({xlabel},{ylabel})")

        kind = self._get_transform_kind("pde_3d")
        if kind != TransformKind.ORIGINAL:
            axis_var = self._pde_3d_axis_var.get()
            result = self._transform_pde_along_axis(
                r.x, r.y_grid, r.y, axis_var, kind,
            )
            if result is not None:
                px, py, pz, pxl, pyl = result
                fig = create_surface_plot(
                    px, py, pz,
                    title=f"{eq_name} — {kind.value}",
                    xlabel=pxl, ylabel=pyl, zlabel="|F|",
                )
                self._replace_plot(self._pde_3d_frame, fig, "_pde_3d_canvas")
                return

        fig = create_surface_plot(
            r.x, r.y_grid, r.y,
            title=eq_name,
            xlabel=xlabel, ylabel=ylabel, zlabel="f",
        )
        self._replace_plot(self._pde_3d_frame, fig, "_pde_3d_canvas")

    def _update_pde_2d(self) -> None:
        """Render the 2D contour plot for PDE."""
        from plotting import create_contour_plot
        from transforms import TransformKind

        r = self._result
        xlabel, ylabel = self._pde_axis_labels()
        eq_name = r.metadata.get("equation_name", f"f({xlabel},{ylabel})")

        kind = self._get_transform_kind("pde_2d")
        if kind != TransformKind.ORIGINAL:
            axis_var = self._pde_2d_axis_var.get()
            result = self._transform_pde_along_axis(
                r.x, r.y_grid, r.y, axis_var, kind,
            )
            if result is not None:
                px, py, pz, pxl, pyl = result
                fig = create_contour_plot(
                    px, py, pz,
                    title=f"{eq_name} — {kind.value}",
                    xlabel=pxl, ylabel=pyl,
                )
                self._replace_plot(self._pde_2d_frame, fig, "_pde_2d_canvas")
                return

        fig = create_contour_plot(
            r.x, r.y_grid, r.y,
            title=eq_name,
            xlabel=xlabel, ylabel=ylabel,
        )
        self._replace_plot(self._pde_2d_frame, fig, "_pde_2d_canvas")

    def _update_pde_transform(self) -> None:
        """Render a 1D transform of a slice through the PDE solution."""
        from plotting import create_solution_plot
        from transforms import TransformKind, apply_transform

        r = self._result
        kind = self._get_transform_kind("pde")
        xlabel, ylabel = self._pde_axis_labels()

        slice_var = self._pde_slice_var.get()
        try:
            slice_val = float(self._pde_slice_val_var.get())
        except ValueError:
            slice_val = 0.5

        if slice_var == xlabel:
            # Slice along x[0] at a fixed x[1] value
            y_idx = int(np.argmin(np.abs(r.y_grid - slice_val)))
            data_1d = r.y[y_idx, :]
            x_1d = r.x
            slice_label = f"{ylabel}={slice_val:.3g}"
            axis_label = xlabel
        else:
            # Slice along x[1] at a fixed x[0] value
            x_idx = int(np.argmin(np.abs(r.x - slice_val)))
            data_1d = r.y[:, x_idx]
            x_1d = r.y_grid
            slice_label = f"{xlabel}={slice_val:.3g}"
            axis_label = ylabel

        eq_name = r.metadata.get("equation_name", "PDE")

        if kind == TransformKind.ORIGINAL:
            fig = create_solution_plot(
                x_1d, np.atleast_2d(data_1d),
                title=f"{eq_name} \u2014 slice at {slice_label}",
                xlabel=axis_label, ylabel="f",
                selected_derivatives=[0], labels=["f"],
            )
        else:
            from scipy.interpolate import interp1d

            func = interp1d(x_1d, data_1d, kind="cubic", fill_value="extrapolate")
            x_min_t, x_max_t = float(x_1d[0]), float(x_1d[-1])
            tx, ty, txlabel, tylabel = apply_transform(
                lambda arr: func(arr), kind, x_min_t, x_max_t,
            )
            fig = create_solution_plot(
                tx, np.atleast_2d(ty),
                title=f"{eq_name} \u2014 {kind.value} [slice {slice_label}]",
                xlabel=txlabel, ylabel=tylabel,
                selected_derivatives=[0], labels=[tylabel],
            )

        self._replace_plot(self._pde_trans_frame, fig, "_pde_trans_canvas")

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
