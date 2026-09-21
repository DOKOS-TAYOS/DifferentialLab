"""Result workspace with scientific summaries and interactive plot tabs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable, Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np

from config import (
    AVAILABLE_STATISTICS,
    generate_output_basename,
    get_env_from_schema,
    get_output_dir,
)
from frontend.plot_embed import embed_animation_plot_in_tk, replace_plot_in_tk
from frontend.theme import get_font
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import (
    bind_wraplength,
    calculate_screen_aware_minsize,
    center_window,
    make_modal,
)
from solver.notation import FNotation, generate_derivative_labels, generate_phase_space_options
from utils import export_csv_to_path, export_json_to_path, get_logger

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from frontend.ui_dialogs.solve_session import EquationSelection, SolveSession
    from pipeline import SolverResult

logger = get_logger(__name__)
_EXTRAPOLATE_FILL: Any = "extrapolate"

_METRIC_LABELS = {
    "rms": "RMS",
    "std": "Standard deviation",
    "l2_norm": "L2 norm",
    "dominant_frequency": "Dominant frequency",
    "gradient_norm": "Gradient norm",
    "zero_crossings": "Zero crossings",
    "nnz": "Nonzero count",
}

_DIAGNOSTIC_HELP = {
    "Relative residual": (
        "Residual norm divided by a reference scale. Interpretation depends on the equation, "
        "discretization, and problem setup."
    ),
    "Jacobian evaluations": "Number of Jacobian evaluations reported by the solver.",
    "LU decompositions": "Number of LU matrix decompositions reported by the solver.",
    "Sparse nnz": "Number of nonzero entries in the assembled sparse matrix.",
    "Condition estimate": (
        "Estimated matrix condition number. Its significance depends on scaling and formulation."
    ),
}

_SIDEBAR_INITIAL_WIDTH = 400
_CONTROL_GAP = 10
_SERIES_VISIBLE_LIMIT = 6

SlicePlane = Literal["XY", "XZ", "YZ"]


def responsive_control_rows(
    available_width: int,
    group_widths: Sequence[int],
    *,
    gap: int = _CONTROL_GAP,
) -> tuple[tuple[int, ...], ...]:
    """Lay out ordered control groups without splitting a label from its widget."""
    usable_width = max(1, available_width)
    rows: list[list[int]] = []
    current: list[int] = []
    current_width = 0
    for index, requested_width in enumerate(group_widths):
        width = max(1, requested_width)
        proposed_width = current_width + (gap if current else 0) + width
        if current and proposed_width > usable_width:
            rows.append(current)
            current = [index]
            current_width = width
        else:
            current.append(index)
            current_width = proposed_width
    if current:
        rows.append(current)
    return tuple(tuple(row) for row in rows)


def normalized_series_selection(
    selected: Iterable[int],
    *,
    count: int,
    fallback_index: int = 0,
) -> tuple[int, ...]:
    """Return valid selected row indexes while guaranteeing one visible selection."""
    if count <= 0:
        return ()
    valid = tuple(sorted({index for index in selected if 0 <= index < count}))
    if valid:
        return valid
    return (min(max(fallback_index, 0), count - 1),)


def pde_3d_fixed_grid(
    plane: SlicePlane,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
) -> tuple[str, np.ndarray]:
    """Return the physical grid fixed by an orthogonal slice plane."""
    if plane == "XY":
        return "z", z_grid
    if plane == "XZ":
        return "y", y_grid
    return "x", x_grid


def pde_fixed_axis_label(varying_axis: str, x_label: str, y_label: str) -> str:
    """Return the coordinate held fixed for a varying PDE slice axis."""
    return y_label if varying_axis == x_label else x_label


def extract_pde_line_slice(
    varying_axis: str,
    x_label: str,
    y_label: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field: np.ndarray,
    fixed_value: float,
) -> tuple[np.ndarray, np.ndarray, str, str, int]:
    """Extract a 1D PDE slice using the documented varying/fixed-axis rule."""
    if varying_axis == x_label:
        fixed_index = int(np.argmin(np.abs(y_grid - fixed_value)))
        return x_grid, field[fixed_index, :], x_label, y_label, fixed_index
    fixed_index = int(np.argmin(np.abs(x_grid - fixed_value)))
    return y_grid, field[:, fixed_index], y_label, x_label, fixed_index


def vector_field_uses_origin(view: str) -> bool:
    """Return whether a Vector PDE field view consumes an origin."""
    return view == "Radial/Tangential"


def _format_grid_coordinate(value: float) -> str:
    """Format a coordinate with enough precision to preserve its float identity."""
    return np.format_float_positional(float(value), unique=True, trim="-")


class _ViewControls(ttk.LabelFrame):
    """Result-local responsive container for indivisible label/control groups."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, text="View controls", padding=(10, 6))
        self._groups: list[ttk.Frame] = []
        self._hidden_groups: set[ttk.Frame] = set()
        self._layout_after_id: str | None = None
        self.bind("<Configure>", self._schedule_layout, add="+")
        self.bind("<Destroy>", self._cancel_layout, add="+")

    def add_group(self, label: str | None = None) -> ttk.Frame:
        """Create a group whose label and controls always reflow together."""
        group = ttk.Frame(self)
        if label:
            ttk.Label(group, text=label).pack(side=tk.LEFT, anchor=tk.N, padx=(0, 5), pady=3)
        self._groups.append(group)
        self._layout_groups()
        return group

    def set_group_visible(self, group: ttk.Frame, visible: bool) -> None:
        """Show or hide a group and exclude hidden controls from focus traversal."""
        for child in group.winfo_children():
            if child.winfo_class() in {
                "TButton",
                "TCheckbutton",
                "TCombobox",
                "TEntry",
                "TMenubutton",
            }:
                cast(Any, child).configure(takefocus=visible)
        if visible:
            self._hidden_groups.discard(group)
        else:
            self._hidden_groups.add(group)
        self._layout_groups()

    def _schedule_layout(self, _event: tk.Event[tk.Widget]) -> None:
        if self._layout_after_id is not None:
            try:
                self.after_cancel(self._layout_after_id)
            except tk.TclError:
                return
        self._layout_after_id = self.after_idle(self._layout_groups)

    def _cancel_layout(self, event: tk.Event[tk.Widget]) -> None:
        if event.widget is not self or self._layout_after_id is None:
            return
        try:
            self.after_cancel(self._layout_after_id)
        except tk.TclError:
            pass
        self._layout_after_id = None

    def _layout_groups(self) -> None:
        self._layout_after_id = None
        visible_groups = [group for group in self._groups if group not in self._hidden_groups]
        if not visible_groups:
            return
        for group in self._groups:
            group.place_forget()
        widths = [group.winfo_reqwidth() for group in visible_groups]
        available = max(1, self.winfo_width() - 20)
        rows = responsive_control_rows(available, widths)
        y_position = 4
        for row in rows:
            x_position = 0
            row_height = max(visible_groups[group_index].winfo_reqheight() for group_index in row)
            for group_index in row:
                group = visible_groups[group_index]
                group.place(x=x_position, y=y_position)
                x_position += group.winfo_reqwidth() + _CONTROL_GAP
            y_position += row_height + 6
        self.configure(height=y_position + 28)


class _SeriesSelector(ttk.Frame):
    """Accessible multi-series selector that always keeps one row selected."""

    def __init__(
        self,
        parent: tk.Widget,
        labels: Sequence[str],
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._labels = tuple(labels)
        self._on_change = on_change
        self._variables = [tk.BooleanVar(value=index == 0) for index in range(len(labels))]
        self._menu_button: ttk.Menubutton | None = None
        if len(labels) <= _SERIES_VISIBLE_LIMIT:
            self._build_visible_checkbuttons()
        else:
            self._build_compact_menu()

    def selected_indices(self) -> list[int]:
        """Return selected indexes in exact plotted-row order."""
        return list(
            normalized_series_selection(
                (index for index, variable in enumerate(self._variables) if variable.get()),
                count=len(self._variables),
            )
        )

    def select_all(self) -> None:
        """Select every available series and redraw immediately."""
        for variable in self._variables:
            variable.set(True)
        self._refresh_menu_label()
        self._on_change()

    def _build_visible_checkbuttons(self) -> None:
        for index, (label, variable) in enumerate(zip(self._labels, self._variables)):
            checkbutton = ttk.Checkbutton(
                self,
                text=label,
                variable=variable,
                command=lambda changed=index: self._toggle(changed),
                takefocus=True,
            )
            checkbutton.grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 8), pady=1)
        if len(self._labels) > 1:
            ttk.Button(self, text="Select all", command=self.select_all, takefocus=True).grid(
                row=(len(self._labels) - 1) // 3 + 1,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(3, 0),
            )

    def _build_compact_menu(self) -> None:
        self._menu_button = ttk.Menubutton(self)
        self._menu_button.configure(takefocus=True)
        menu = tk.Menu(self._menu_button, tearoff=False)
        for index, (label, variable) in enumerate(zip(self._labels, self._variables)):
            menu.add_checkbutton(
                label=label,
                variable=variable,
                command=lambda changed=index: self._toggle(changed),
            )
        menu.add_separator()
        menu.add_command(label="Select all", command=self.select_all)
        self._menu_button.configure(menu=menu)
        self._menu_button.pack(side=tk.LEFT)
        self._refresh_menu_label()

    def _toggle(self, changed_index: int) -> None:
        selected = normalized_series_selection(
            (index for index, variable in enumerate(self._variables) if variable.get()),
            count=len(self._variables),
            fallback_index=changed_index,
        )
        for index, variable in enumerate(self._variables):
            variable.set(index in selected)
        self._refresh_menu_label()
        self._on_change()

    def _refresh_menu_label(self) -> None:
        if self._menu_button is None:
            return
        selected = self.selected_indices()
        self._menu_button.configure(text=f"Choose series ({len(selected)} selected)")


def modify_setup_available(session: object | None, selection: object | None) -> bool:
    """Return whether Results can navigate directly back to Configuration."""
    return session is not None and selection is not None


def humanize_metric_label(key: str) -> str:
    """Return a deterministic user-facing label without changing the metric key."""
    return _METRIC_LABELS.get(key, key.replace("_", " ").strip().capitalize())


def computed_metric_items(
    statistics: dict[str, Any],
) -> list[tuple[str, Any, str | None]]:
    """Return humanized metric rows while preserving each structured value."""
    return [
        (humanize_metric_label(key), value, AVAILABLE_STATISTICS.get(key))
        for key, value in statistics.items()
    ]


def _has_display_value(value: Any) -> bool:
    """Return whether *value* contains information worth showing."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.size > 0
    return True


def run_summary_items(metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    """Extract concise, available run-level metadata in display order."""
    items: list[tuple[str, Any]] = []
    success = metadata.get("solver_success")
    if success is True:
        items.append(("Status", "Completed successfully"))
    elif success is False:
        items.append(("Status", "Solver reported failure"))

    fields = (
        ("Method", "method"),
        ("Points", "num_points"),
        ("Function evaluations", "n_evaluations"),
        ("Relative tolerance", "rtol"),
        ("Absolute tolerance", "atol"),
    )
    for label, key in fields:
        value = metadata.get(key)
        if _has_display_value(value):
            items.append((label, value))
    return items


def diagnostic_items(metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    """Extract factual solver diagnostics without applying quality thresholds."""
    fields = (
        ("Solver message", "solver_message"),
        ("Solver status", "solver_status"),
        ("Residual maximum", "residual_max"),
        ("Residual mean", "residual_mean"),
        ("Residual RMS", "residual_rms"),
        ("Discrete residual L2", "discrete_residual_l2"),
        ("Relative residual", "relative_residual_l2"),
        ("Component residuals", "component_residual_l2"),
        ("Component relative residuals", "component_relative_residual_l2"),
        ("Jacobian evaluations", "n_jacobian_evals"),
        ("LU decompositions", "n_lu_decompositions"),
        ("Sparse matrix shape", "matrix_shape"),
        ("Sparse nnz", "nnz"),
        ("Condition estimate", "condition_estimate"),
        ("PDE warnings", "pde_warnings"),
    )
    return [
        (label, metadata[key])
        for label, key in fields
        if key in metadata and _has_display_value(metadata[key])
    ]


def event_summary_items(metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    """Summarize event metadata without exposing event-state arrays."""
    event_times = metadata.get("event_times")
    event_states = metadata.get("event_states")
    if event_times is None and event_states is None:
        return []

    if event_times is None:
        return [("Detected events", 0)]

    groups: list[Any]
    if isinstance(event_times, np.ndarray) and event_times.ndim <= 1:
        groups = [event_times]
    elif isinstance(event_times, (list, tuple)):
        groups = list(event_times)
    else:
        groups = [event_times]

    normalized: dict[str, Any] = {}
    count = 0
    for index, group in enumerate(groups, start=1):
        values = np.asarray(group).reshape(-1).tolist()
        count += len(values)
        if values:
            normalized[f"Event {index}"] = values

    items: list[tuple[str, Any]] = [("Detected events", count)]
    if normalized:
        items.append(("Event times", normalized))
    return items


def _format_display_value(value: Any) -> str:
    """Format a scalar or short sequence for a wrapping display row."""
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, np.ndarray):
        return ", ".join(_format_display_value(item) for item in value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_display_value(item) for item in value)
    return str(value)


class ResultDialog:
    """Window showing the solution with interactive plot tabs.

    Plots are generated on-demand from the raw solver data.  The user
    selects *what* to visualise (derivatives, phase-space axes, etc.)
    inside the result window rather than before solving.

    Args:
        parent: Parent window.
        result: A data-only ``SolverResult`` from the pipeline.
        session: Optional state owner for the standard solve workflow.
        selection: Optional equation selection used to rebuild Configuration.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        result: SolverResult,
        session: SolveSession | None = None,
        selection: EquationSelection | None = None,
    ) -> None:
        self.parent = parent
        self._result = result
        self._session = session
        self._selection = selection
        self._notation: FNotation = result.notation or FNotation(
            kind="ode", order=result.vector_order
        )

        self.win = tk.Toplevel(parent)
        self.win.title(f"Results — {result.metadata.get('equation_name', 'ODE')}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        # Canvas references for cleanup
        self._canvases: list[FigureCanvasTkAgg] = []
        self._closed = False
        self._initial_plot_callbacks: list[Callable[[], None]] = []
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        # Allocate the final window geometry before creating Matplotlib canvases,
        # then materialize every plot frame before the first canvas is embedded.
        self._set_window_geometry()
        self._build_ui()
        self._build_plot_tabs()
        self.win.update_idletasks()
        self._render_initial_plots()
        make_modal(self.win, parent)
        logger.info("Result dialog displayed")

    def _close(self) -> None:
        """Release owned plot resources and close the result window."""
        if self._closed:
            return
        self._closed = True
        self._cleanup_plots()
        try:
            self.win.destroy()
        except tk.TclError:
            pass

    def _modify_setup(self) -> None:
        """Close Results and reopen Configuration from the retained solve session."""
        session = self._session
        selection = self._selection
        if session is None or selection is None:
            return

        parent = self.parent
        parameters_kwargs = selection.parameters_kwargs()
        self._close()

        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        ParametersDialog(
            parent,
            **parameters_kwargs,
            session=session,
            selection=selection,
        )

    @staticmethod
    def _dispose_canvas(canvas: FigureCanvasTkAgg | None) -> None:
        """Stop a canvas animation and close its Matplotlib figure."""
        if canvas is None:
            return

        try:
            stop_animation = getattr(canvas, "_stop_animation", None)
        except Exception:
            stop_animation = None
        if callable(stop_animation):
            try:
                stop_animation()
            except Exception:
                logger.debug("Could not stop a result-dialog animation", exc_info=True)

        try:
            figure = getattr(canvas, "figure", None)
        except Exception:
            return
        if figure is None:
            return

        import matplotlib.pyplot as plt

        try:
            plt.close(figure)
        except Exception:
            logger.debug("Could not close a result-dialog figure", exc_info=True)

    def _cleanup_plots(self) -> None:
        """Close every Matplotlib canvas currently owned by the dialog."""
        canvases = tuple(self._canvases)
        self._canvases.clear()
        for canvas in canvases:
            self._dispose_canvas(canvas)

    def _register_canvas(self, canvas: FigureCanvasTkAgg) -> None:
        """Track a canvas once so its figure is closed with the dialog."""
        if not any(existing is canvas for existing in self._canvases):
            self._canvases.append(canvas)

    def _unregister_canvas(self, canvas: FigureCanvasTkAgg) -> None:
        """Stop tracking a canvas that has already been disposed."""
        self._canvases[:] = [existing for existing in self._canvases if existing is not canvas]

    def _set_window_geometry(self) -> None:
        """Set the dialog geometry before the initial plot canvas is embedded."""
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        win_w = int(screen_w * 0.94)
        win_h = min(int(screen_h * 0.88), 920)

        center_window(self.win, win_w, win_h, max_width_ratio=0.96, resizable=True)
        min_w, min_h = calculate_screen_aware_minsize(
            screen_w,
            screen_h,
            900,
            560,
            max_ratio=0.9,
        )
        self.win.minsize(min_w, min_h)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        result = self._result

        # ── Fixed workflow header ──
        header = ttk.Frame(self.win, padding=(pad, pad, pad, max(4, pad // 2)))
        header.pack(side=tk.TOP, fill=tk.X)
        title_row = ttk.Frame(header)
        title_row.pack(fill=tk.X)
        ttk.Label(title_row, text="Results", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_row, text="Step 3 of 3", style="Small.TLabel").pack(side=tk.RIGHT)
        equation_name = str(result.metadata.get("equation_name") or "Solved equation")
        ttk.Label(header, text=equation_name, style="Subtitle.TLabel").pack(
            fill=tk.X,
            pady=(max(3, pad // 2), 0),
        )
        status_items = run_summary_items(result.metadata)
        status = next((value for label, value in status_items if label == "Status"), None)
        if status is not None:
            ttk.Label(header, text=str(status), style="Small.TLabel").pack(fill=tk.X)

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X)

        # ── Fixed data/action footer ──
        footer = ttk.Frame(self.win, padding=(pad, max(4, pad // 2), pad, pad))
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        navigation_buttons: list[ttk.Button] = []
        if modify_setup_available(self._session, self._selection):
            modify_button = ttk.Button(
                footer,
                text="Modify setup",
                style="Secondary.TButton",
                command=self._modify_setup,
            )
            modify_button.pack(side=tk.LEFT)
            navigation_buttons.append(modify_button)

        button_row = ttk.Frame(footer)
        button_row.pack(side=tk.RIGHT)
        csv_button = ttk.Button(
            button_row,
            text="Export CSV...",
            style="Secondary.TButton",
            command=self._on_save_csv,
        )
        csv_button.pack(side=tk.LEFT, padx=(0, max(4, pad // 2)))
        json_button = ttk.Button(
            button_row,
            text="Export JSON...",
            style="Secondary.TButton",
            command=self._on_save_json,
        )
        json_button.pack(side=tk.LEFT, padx=(0, max(4, pad // 2)))
        close_button = ttk.Button(
            button_row,
            text="Close",
            style="Secondary.TButton",
            command=self._close,
        )
        close_button.pack(side=tk.LEFT)
        navigation_buttons.extend((csv_button, json_button, close_button))
        setup_arrow_enter_navigation([navigation_buttons])
        close_button.focus_set()
        ToolTip(csv_button, "Export the numerical solution data as CSV.")
        ToolTip(json_button, "Export computed metrics and solver metadata as JSON.")

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(side=tk.BOTTOM, fill=tk.X)

        # ── Resizable workspace: summary | plots ──
        content = ttk.Frame(self.win, padding=(pad, pad, pad, max(4, pad // 2)))
        content.pack(fill=tk.BOTH, expand=True)
        self._paned = ttk.Panedwindow(content, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(self._paned, width=_SIDEBAR_INITIAL_WIDTH)
        right_frame = ttk.Frame(self._paned)
        self._paned.add(left_frame, weight=0)
        self._paned.add(right_frame, weight=1)

        left_scroll = ScrollableFrame(left_frame)
        left_scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        left_scroll.pack(fill=tk.BOTH, expand=True)
        left_inner = left_scroll.inner
        left_inner.configure(padding=pad)

        self._build_left_panel(left_inner, left_scroll, result.statistics, result.metadata, pad)
        left_scroll.bind_new_children()

        # ── RIGHT: existing scientific visualization tabs ──
        notebook_style = ttk.Style(self.win)
        notebook_style.configure("Result.TNotebook.Tab", padding=(pad + 2, max(4, pad // 2)))
        self._notebook = ttk.Notebook(right_frame, style="Result.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True)

    def _build_left_panel(
        self,
        inner: ttk.Frame,
        scroll: ScrollableFrame,
        statistics: dict[str, Any],
        metadata: dict[str, Any],
        pad: int,
    ) -> None:
        """Build the run summary, metrics, events, and diagnostics hierarchy."""
        run_section = CollapsibleSection(inner, scroll, "Run summary", expanded=True, pad=pad)
        run_items = run_summary_items(metadata)
        if run_items:
            self._render_display_items(run_section.content, run_items)
        else:
            ttk.Label(
                run_section.content,
                text="No run summary metadata is available.",
                style="Small.TLabel",
                wraplength=300,
            ).pack(fill=tk.X)

        if statistics:
            metrics_section = CollapsibleSection(
                inner,
                scroll,
                "Computed metrics",
                expanded=True,
                pad=pad,
            )
            for label, value, description in computed_metric_items(statistics):
                help_text = {label: description} if description else None
                self._render_display_entry(
                    metrics_section.content,
                    label,
                    value,
                    help_text=help_text,
                )

        events = event_summary_items(metadata)
        if events:
            events_section = CollapsibleSection(inner, scroll, "Events", expanded=True, pad=pad)
            self._render_display_items(events_section.content, events)

        diagnostics = diagnostic_items(metadata)
        if diagnostics:
            diagnostics_section = CollapsibleSection(
                inner,
                scroll,
                "Solver diagnostics",
                expanded=False,
                pad=pad,
            )
            self._render_display_items(
                diagnostics_section.content,
                diagnostics,
                help_text=_DIAGNOSTIC_HELP,
            )

    def _render_display_items(
        self,
        parent: tk.Widget,
        items: list[tuple[str, Any]],
        *,
        help_text: dict[str, str] | None = None,
    ) -> None:
        """Render consistently aligned, wrapping rows for structured information."""
        for label, value in items:
            self._render_display_entry(parent, label, value, help_text=help_text)

    def _render_display_entry(
        self,
        parent: tk.Widget,
        label: str,
        value: Any,
        *,
        indent: int = 0,
        help_text: dict[str, str] | None = None,
    ) -> None:
        """Render one scalar or nested mapping without losing component identity."""
        if isinstance(value, dict):
            heading = ttk.Label(
                parent,
                text=label,
                style="Subtitle.TLabel" if indent == 0 else "Small.TLabel",
            )
            heading.pack(fill=tk.X, padx=(indent * 12, 0), pady=(3, 1))
            if help_text and label in help_text:
                ToolTip(heading, help_text[label])
            for nested_label, nested_value in value.items():
                nested_key = str(nested_label)
                nested_display_label = humanize_metric_label(nested_key)
                nested_help = dict(help_text or {})
                nested_description = AVAILABLE_STATISTICS.get(nested_key)
                if nested_description:
                    nested_help[nested_display_label] = nested_description
                self._render_display_entry(
                    parent,
                    nested_display_label,
                    nested_value,
                    indent=indent + 1,
                    help_text=nested_help or None,
                )
            return

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=(indent * 12, 0), pady=2)
        row.columnconfigure(1, weight=1)
        label_widget = ttk.Label(row, text=label, anchor=tk.NW, width=20)
        label_widget.grid(row=0, column=0, sticky="nw", padx=(0, 8))
        value_widget = ttk.Label(
            row,
            text=_format_display_value(value),
            style="Small.TLabel",
            anchor=tk.NW,
            justify=tk.LEFT,
        )
        value_widget.grid(row=0, column=1, sticky="ew")
        bind_wraplength(row, value_widget, pad=175, min_wrap=120)
        if help_text and label in help_text:
            ToolTip(label_widget, help_text[label])

    # ------------------------------------------------------------------
    # Result-local view controls helpers
    # ------------------------------------------------------------------

    def _create_view_controls(self, parent: ttk.Frame) -> _ViewControls:
        """Create the shared responsive control surface above a plot workspace."""
        controls = _ViewControls(parent)
        controls.pack(fill=tk.X, padx=8, pady=(8, 6))
        return controls

    def _build_transform_controls(
        self,
        parent: ttk.Frame,
        callback: Callable[[], None],
        prefix: str,
    ) -> ttk.Combobox:
        """Add a transform dropdown to an existing labelled control group.

        The ``StringVar`` is stored as ``self._transform_{prefix}_var``.
        """
        from transforms import TransformKind

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
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _e: callback())
        return combo

    # ------------------------------------------------------------------
    # Plot tab construction
    # ------------------------------------------------------------------

    def _build_plot_tabs(self) -> None:
        """Create the right-side tabs based on equation type."""
        r = self._result
        eq_type = r.equation_type

        is_2d_pde = eq_type in ("pde", "vector_pde") and r.y_grid is not None
        is_3d_pde = eq_type == "pde_3d" and r.y_grid is not None and r.z_grid is not None

        if is_3d_pde:
            self._build_pde_3d_slice_tabs()
        elif is_2d_pde:
            self._build_pde_tabs()
        elif eq_type == "vector_ode" or (r.is_vector and r.vector_components > 1):
            self._build_vector_ode_tabs()
        elif eq_type == "difference":
            self._build_ode_scalar_tabs()  # same layout: solution + (no phase for 1st-order)
        else:
            self._build_ode_scalar_tabs()

    def _queue_initial_plot(self, callback: Callable[[], None]) -> None:
        """Defer an initial plot until all tab frames have been laid out."""
        self._initial_plot_callbacks.append(callback)

    def _render_initial_plots(self) -> None:
        """Render each initial plot after Tk has materialized its parent frame."""
        callbacks = self._initial_plot_callbacks
        self._initial_plot_callbacks = []
        for callback in callbacks:
            callback()

    # ── ODE scalar / difference ──────────────────────────────────────

    def _build_ode_scalar_tabs(self) -> None:
        """Solution 2D (multi-select derivatives) + Phase Space (axis dropdowns)."""
        r = self._result
        notation = self._notation
        nb = self._notebook

        # --- Tab 1: Solution 2D ---
        sol_tab = ttk.Frame(nb)
        nb.add(sol_tab, text="  Solution f(x)  ")

        ctrl = self._create_view_controls(sol_tab)

        self._sol_labels = generate_derivative_labels(notation)
        series_group = ctrl.add_group("Series")
        self._sol_series_selector = _SeriesSelector(
            series_group,
            self._sol_labels,
            self._update_solution_plot,
        )
        self._sol_series_selector.pack(side=tk.LEFT)

        transform_group = ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_solution_plot, "sol")

        self._sol_plot_frame = ttk.Frame(sol_tab)
        self._sol_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._sol_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_solution_plot)

        # --- Tab 2: Phase Space ---
        order = r.vector_order
        if order >= 2 or r.equation_type != "difference":
            phase_tab = ttk.Frame(nb)
            nb.add(phase_tab, text="  Phase Space  ")

            phase_ctrl = self._create_view_controls(phase_tab)

            ps_options = generate_phase_space_options(notation)
            ps_labels = [lbl for lbl, _ in ps_options]

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
            x_group = phase_ctrl.add_group("X axis")
            self._phase_x_var = tk.StringVar(value=default_x)
            phase_x_combo = ttk.Combobox(
                x_group,
                textvariable=self._phase_x_var,
                values=ps_labels,
                state="readonly",
                width=6,
                font=get_font(),
            )
            phase_x_combo.pack(side=tk.LEFT)
            phase_x_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_phase_plot())

            y_group = phase_ctrl.add_group("Y axis")
            self._phase_y_var = tk.StringVar(value=default_y_ax)
            phase_y_combo = ttk.Combobox(
                y_group,
                textvariable=self._phase_y_var,
                values=ps_labels,
                state="readonly",
                width=6,
                font=get_font(),
            )
            phase_y_combo.pack(side=tk.LEFT)
            phase_y_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_phase_plot())

            transform_group = phase_ctrl.add_group("Transform")
            self._build_transform_controls(transform_group, self._update_phase_plot, "phase")

            self._phase_options_map = {lbl: idx for lbl, idx in ps_options}
            self._phase_plot_frame = ttk.Frame(phase_tab)
            self._phase_plot_frame.pack(fill=tk.BOTH, expand=True)
            self._phase_canvas: FigureCanvasTkAgg | None = None
            self._queue_initial_plot(self._update_phase_plot)

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
            func = interp1d(x, y_2d[idx], kind="cubic", fill_value=_EXTRAPOLATE_FILL)
            tx, ty, txlabel, tylabel = apply_transform(
                lambda arr, f=func: f(arr),
                kind,
                x_min_t,
                x_max_t,
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
                f_interp = interp1d(tx_i, ty_i, kind="linear", bounds_error=False, fill_value=0.0)
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

    def _require_pde_y_grid(self) -> np.ndarray:
        """Return the PDE y-grid or raise if this dialog is misused."""
        y_grid = self._result.y_grid
        if y_grid is None:
            raise ValueError("2D PDE result is missing its y-grid")
        return y_grid

    def _update_solution_plot(self) -> None:
        """Regenerate the solution f(x) plot with currently selected derivatives."""
        from plotting import create_solution_plot
        from transforms import TransformKind

        r = self._result
        selected = self._sol_series_selector.selected_indices()

        xlabel = "n" if r.equation_type == "difference" else "x"
        eq_name = r.metadata.get("equation_name", "f(x)")

        kind = self._get_transform_kind("sol")

        if kind == TransformKind.ORIGINAL:
            fig = create_solution_plot(
                r.x,
                r.y,
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
                r.x,
                y_2d,
                selected,
                self._sol_labels,
                kind,
            )
            if result is None:
                return
            tx, ty_2d, trans_labels, txlabel, tylabel = result

            fig = create_solution_plot(
                tx,
                ty_2d,
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
            (f"y[{i}]" if i >= y_2d.shape[0] else f"row{i}") for i in unique_indices
        ]
        result = self._apply_transform_multi(
            x,
            y_2d,
            unique_indices,
            labels_for_transform,
            kind,
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
                r.x,
                y_2d,
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

        ctrl = self._create_view_controls(sol_tab)

        self._vec_sol_labels = generate_derivative_labels(notation)
        series_group = ctrl.add_group("Series")
        self._vec_sol_series_selector = _SeriesSelector(
            series_group,
            self._vec_sol_labels,
            self._update_vec_solution_plot,
        )
        self._vec_sol_series_selector.pack(side=tk.LEFT)

        transform_group = ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_vec_solution_plot, "vec_sol")

        self._vec_sol_plot_frame = ttk.Frame(sol_tab)
        self._vec_sol_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._vec_sol_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_vec_solution_plot)

        # --- Tab 2: Phase Space 2D ---
        phase_tab = ttk.Frame(nb)
        nb.add(phase_tab, text="  Phase Space  ")

        phase_ctrl = self._create_view_controls(phase_tab)

        ps_options = generate_phase_space_options(notation)
        ps_labels = [lbl for lbl, _ in ps_options]

        # Default: f₀ vs f′₀ (component 0 value vs its first derivative)
        # ps_labels[0] = "x", ps_labels[1] = "f₀", ps_labels[2] = "f′₀" (for order >= 2)
        if r.vector_order >= 2 and len(ps_labels) >= 3:
            default_x_phase = ps_labels[1]  # f₀
            default_y_phase = ps_labels[2]  # f′₀
        elif len(ps_labels) >= 2:
            default_x_phase = ps_labels[0]  # x
            default_y_phase = ps_labels[1]  # f₀
        else:
            default_x_phase = ps_labels[0] if ps_labels else "f"
            default_y_phase = ps_labels[0] if ps_labels else "f"

        x_group = phase_ctrl.add_group("X axis")
        self._vec_phase_x_var = tk.StringVar(value=default_x_phase)
        vec_phase_x_combo = ttk.Combobox(
            x_group,
            textvariable=self._vec_phase_x_var,
            values=ps_labels,
            state="readonly",
            width=6,
            font=get_font(),
        )
        vec_phase_x_combo.pack(side=tk.LEFT)
        vec_phase_x_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_plot())

        y_group = phase_ctrl.add_group("Y axis")
        self._vec_phase_y_var = tk.StringVar(value=default_y_phase)
        vec_phase_y_combo = ttk.Combobox(
            y_group,
            textvariable=self._vec_phase_y_var,
            values=ps_labels,
            state="readonly",
            width=6,
            font=get_font(),
        )
        vec_phase_y_combo.pack(side=tk.LEFT)
        vec_phase_y_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_plot())

        transform_group = phase_ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_vec_phase_plot, "vec_phase")

        self._vec_phase_options_map = {lbl: idx for lbl, idx in ps_options}
        self._vec_phase_plot_frame = ttk.Frame(phase_tab)
        self._vec_phase_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._vec_phase_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_vec_phase_plot)

        # --- Tab 3: Phase Space 3D ---
        phase3d_tab = ttk.Frame(nb)
        nb.add(phase3d_tab, text="  Phase 3D  ")

        phase3d_ctrl = self._create_view_controls(phase3d_tab)

        # Default axes: f₀, f₁, f₂ for 3+ components; x, f₀, f₁ otherwise
        n_comp = r.vector_components
        if n_comp >= 3 and len(ps_labels) >= 4:
            # ps_labels: x, f₀, f′₀, f₁, f′₁, f₂, f′₂, ...
            # Find the first 3 "base" component labels (derivative 0 of each)
            order = r.vector_order
            def_3d_x = ps_labels[1]  # f₀
            def_3d_y = ps_labels[1 + order]  # f₁
            def_3d_z = ps_labels[1 + 2 * order]  # f₂
        elif n_comp >= 2 and len(ps_labels) >= 3:
            order = r.vector_order
            def_3d_x = ps_labels[0]  # x
            def_3d_y = ps_labels[1]  # f₀
            def_3d_z = ps_labels[1 + order]  # f₁
        else:
            def_3d_x = ps_labels[0] if ps_labels else "x"
            def_3d_y = ps_labels[1] if len(ps_labels) > 1 else def_3d_x
            def_3d_z = ps_labels[2] if len(ps_labels) > 2 else def_3d_y

        x_group = phase3d_ctrl.add_group("X axis")
        self._vec_phase3d_x_var = tk.StringVar(value=def_3d_x)
        phase3d_x_combo = ttk.Combobox(
            x_group,
            textvariable=self._vec_phase3d_x_var,
            values=ps_labels,
            state="readonly",
            width=6,
            font=get_font(),
        )
        phase3d_x_combo.pack(side=tk.LEFT)
        phase3d_x_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_3d())

        y_group = phase3d_ctrl.add_group("Y axis")
        self._vec_phase3d_y_var = tk.StringVar(value=def_3d_y)
        phase3d_y_combo = ttk.Combobox(
            y_group,
            textvariable=self._vec_phase3d_y_var,
            values=ps_labels,
            state="readonly",
            width=6,
            font=get_font(),
        )
        phase3d_y_combo.pack(side=tk.LEFT)
        phase3d_y_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_3d())

        z_group = phase3d_ctrl.add_group("Z axis")
        self._vec_phase3d_z_var = tk.StringVar(value=def_3d_z)
        phase3d_z_combo = ttk.Combobox(
            z_group,
            textvariable=self._vec_phase3d_z_var,
            values=ps_labels,
            state="readonly",
            width=6,
            font=get_font(),
        )
        phase3d_z_combo.pack(side=tk.LEFT)
        phase3d_z_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_vec_phase_3d())

        transform_group = phase3d_ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_vec_phase_3d, "vec_phase3d")

        self._vec_phase3d_plot_frame = ttk.Frame(phase3d_tab)
        self._vec_phase3d_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._vec_phase3d_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_vec_phase_3d)

        # --- Tab 4: Animation ---
        anim_tab = ttk.Frame(nb)
        nb.add(anim_tab, text="  Animation  ")

        anim_ctrl = self._create_view_controls(anim_tab)
        order_group = anim_ctrl.add_group("Derivative order")
        self._anim_order_var = tk.StringVar(value="0")
        orders = [str(k) for k in range(r.vector_order)]
        anim_order_combo = ttk.Combobox(
            order_group,
            textvariable=self._anim_order_var,
            values=orders,
            state="readonly",
            width=4,
            font=get_font(),
        )
        anim_order_combo.pack(side=tk.LEFT)
        anim_order_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_animation())

        transform_group = anim_ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_animation, "anim")

        self._anim_plot_frame = ttk.Frame(anim_tab)
        self._anim_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._anim_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_animation)

        # --- Tab 5: 3D Surface ---
        tab_3d = ttk.Frame(nb)
        nb.add(tab_3d, text="  3D Surface  ")

        ctrl_3d = self._create_view_controls(tab_3d)
        order_group = ctrl_3d.add_group("Derivative order")
        self._3d_order_var = tk.StringVar(value="0")
        order_3d_combo = ttk.Combobox(
            order_group,
            textvariable=self._3d_order_var,
            values=orders,
            state="readonly",
            width=4,
            font=get_font(),
        )
        order_3d_combo.pack(side=tk.LEFT)
        order_3d_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_3d_plot())

        transform_group = ctrl_3d.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_3d_plot, "vec_3d")

        self._3d_plot_frame = ttk.Frame(tab_3d)
        self._3d_plot_frame.pack(fill=tk.BOTH, expand=True)
        self._3d_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_3d_plot)

    def _update_vec_solution_plot(self) -> None:
        """Regenerate vector ODE solution plot."""
        from plotting import create_solution_plot
        from transforms import TransformKind

        r = self._result
        selected = self._vec_sol_series_selector.selected_indices()

        eq_name = r.metadata.get("equation_name", "f(x)")
        kind = self._get_transform_kind("vec_sol")

        if kind == TransformKind.ORIGINAL:
            fig = create_solution_plot(
                r.x,
                r.y,
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
                r.x,
                y_2d,
                selected,
                self._vec_sol_labels,
                kind,
            )
            if result is None:
                return
            tx, ty_2d, trans_labels, txlabel, tylabel = result

            fig = create_solution_plot(
                tx,
                ty_2d,
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
                r.x,
                y_2d,
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
                r.x,
                y_2d,
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
            data_x,
            data_y,
            data_z,
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
                r.x,
                r.y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 f_i(x) (k={deriv_k})",
                deriv_offset=deriv_k,
            )
        else:
            result = self._transform_vector_components(
                r.x,
                r.y,
                r.vector_order,
                r.vector_components,
                deriv_k,
                kind,
            )
            if result is None:
                return
            tx, new_y, txlabel = result
            fig = create_vector_animation_plot(
                tx,
                new_y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 {kind.value} (k={deriv_k})",
                deriv_offset=deriv_k,
            )

        old_canvas = getattr(self, "_anim_canvas", None)
        if old_canvas is not None:
            self._dispose_canvas(old_canvas)
            self._unregister_canvas(old_canvas)

        # Clear existing widgets
        for w in self._anim_plot_frame.winfo_children():
            w.destroy()

        def _export_cb(dur: float) -> None:
            self._on_export_animation_mp4(dur, deriv_k)

        self._anim_canvas = embed_animation_plot_in_tk(
            fig,
            self._anim_plot_frame,
            on_export_mp4=_export_cb,
        )
        self._register_canvas(self._anim_canvas)

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
                r.x,
                r.y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 3D (k={deriv_k})",
                deriv_offset=deriv_k,
            )
        else:
            result = self._transform_vector_components(
                r.x,
                r.y,
                r.vector_order,
                r.vector_components,
                deriv_k,
                kind,
            )
            if result is None:
                return
            tx, new_y, txlabel = result
            fig = create_vector_animation_3d(
                tx,
                new_y,
                order=r.vector_order,
                vector_components=r.vector_components,
                title=f"{eq_name} \u2014 {kind.value} 3D (k={deriv_k})",
                deriv_offset=deriv_k,
            )

        self._replace_plot(self._3d_plot_frame, fig, "_3d_canvas")

    # ── PDE ──────────────────────────────────────────────────────────

    def _build_pde_3d_slice_tabs(self) -> None:
        """Build component-free orthogonal slice access for scalar PDE 3D."""
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Orthogonal Slice  ")
        controls = self._create_view_controls(tab)
        plane_group = controls.add_group("Plane")
        self._pde_3d_slice_plane_var = tk.StringVar(value="XY")
        plane_combo = ttk.Combobox(
            plane_group,
            textvariable=self._pde_3d_slice_plane_var,
            values=["XY", "XZ", "YZ"],
            state="readonly",
            width=4,
            font=get_font(),
        )
        plane_combo.pack(side=tk.LEFT)
        coordinate_group = controls.add_group()
        self._pde_3d_fixed_axis_label = ttk.Label(coordinate_group, text="Fixed z")
        self._pde_3d_fixed_axis_label.pack(side=tk.LEFT, padx=(0, 5))
        self._pde_3d_slice_coordinate_var = tk.StringVar(value="")
        self._pde_3d_slice_coordinate_combo = ttk.Combobox(
            coordinate_group,
            textvariable=self._pde_3d_slice_coordinate_var,
            state="readonly",
            width=14,
            font=get_font(),
        )
        self._pde_3d_slice_coordinate_combo.pack(side=tk.LEFT)
        self._pde_3d_slice_index_var = tk.StringVar(value="0")
        self._pde_3d_slice_index_context_var = tk.StringVar(value="")
        index_group = controls.add_group()
        ttk.Label(index_group, textvariable=self._pde_3d_slice_index_context_var).pack(side=tk.LEFT)

        plane_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_pde_3d_slice_coordinates(),
        )
        self._pde_3d_slice_coordinate_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._select_pde_3d_slice_coordinate()
        )
        self._pde_3d_slice_frame = ttk.Frame(tab)
        self._pde_3d_slice_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_3d_slice_canvas: FigureCanvasTkAgg | None = None
        self._refresh_pde_3d_slice_coordinates(render=False)
        self._queue_initial_plot(self._update_pde_3d_slice)

        sweep_tab = ttk.Frame(self._notebook)
        self._notebook.add(sweep_tab, text="  Axis Sweep  ")
        sweep_controls = self._create_view_controls(sweep_tab)
        sweep_group = sweep_controls.add_group("Sweep along")
        labels = self._pde_coordinate_labels(3)
        self._pde_3d_sweep_axis_var = tk.StringVar(value=labels[2])
        sweep_selector = ttk.Combobox(
            sweep_group,
            textvariable=self._pde_3d_sweep_axis_var,
            values=labels,
            state="readonly",
            width=max(5, max(map(len, labels))),
            font=get_font(),
        )
        sweep_selector.pack(side=tk.LEFT)
        sweep_selector.bind("<<ComboboxSelected>>", lambda _event: self._update_pde_3d_sweep())
        self._pde_3d_sweep_frame = ttk.Frame(sweep_tab)
        self._pde_3d_sweep_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_3d_sweep_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_pde_3d_sweep)

    def _refresh_pde_3d_slice_coordinates(self, *, render: bool = True) -> None:
        """Refresh physical position choices for the selected orthogonal plane."""
        result = self._result
        if result.y_grid is None or result.z_grid is None:
            raise ValueError("PDE 3D result is missing its y/z grids")
        selected_plane = self._pde_3d_slice_plane_var.get()
        plane = cast(SlicePlane, selected_plane) if selected_plane in {"XY", "XZ", "YZ"} else "XY"
        fixed_axis, grid = pde_3d_fixed_grid(plane, result.x, result.y_grid, result.z_grid)
        labels = self._pde_coordinate_labels(3)
        fixed_label = labels["xyz".index(fixed_axis)]
        self._pde_3d_fixed_axis_label.configure(text=f"Fixed {fixed_label}")
        display_values = [_format_grid_coordinate(value) for value in grid]
        self._pde_3d_slice_coordinate_combo.configure(values=display_values)
        index = len(grid) // 2
        self._pde_3d_slice_coordinate_combo.current(index)
        self._pde_3d_slice_index_var.set(str(index))
        self._pde_3d_slice_index_context_var.set(f"index {index} / {len(grid)}")
        if render:
            self._update_pde_3d_slice()

    def _select_pde_3d_slice_coordinate(self) -> None:
        """Map the displayed coordinate choice back to its exact array index."""
        index = self._pde_3d_slice_coordinate_combo.current()
        if index < 0:
            return
        size = len(self._pde_3d_slice_coordinate_combo.cget("values"))
        self._pde_3d_slice_index_var.set(str(index))
        self._pde_3d_slice_index_context_var.set(f"index {index} / {size}")
        self._update_pde_3d_slice()

    def _update_pde_3d_slice(self) -> None:
        """Render the selected XY, XZ, or YZ scalar slice."""
        from plotting import create_contour_plot
        from plotting.coordinates import extract_scalar_3d_slice

        result = self._result
        if result.y_grid is None or result.z_grid is None:
            raise ValueError("PDE 3D result is missing its y/z grids")
        variables = result.metadata.get("variables", ["x", "y", "z"])
        labels = [
            variables[index] if len(variables) > index else "xyz"[index] for index in range(3)
        ]
        selected_plane = self._pde_3d_slice_plane_var.get()
        plane: SlicePlane = (
            cast(SlicePlane, selected_plane) if selected_plane in {"XY", "XZ", "YZ"} else "XY"
        )
        try:
            requested_index = int(self._pde_3d_slice_index_var.get())
        except ValueError:
            requested_index = 0
        slice_data = extract_scalar_3d_slice(
            result.x,
            result.y_grid,
            result.z_grid,
            result.y,
            plane,
            requested_index,
        )
        label_indexes = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[plane]
        axis_1_label, axis_2_label, fixed_axis_label = (labels[index] for index in label_indexes)
        fixed_label = f"{fixed_axis_label}={slice_data.fixed_coordinate:.4g}"
        equation_name = result.metadata.get("equation_name", "PDE 3D")
        figure = create_contour_plot(
            slice_data.axis_1,
            slice_data.axis_2,
            slice_data.values,
            title=f"{equation_name} — {plane} slice at {fixed_label}",
            xlabel=axis_1_label,
            ylabel=axis_2_label,
        )
        self._replace_plot(self._pde_3d_slice_frame, figure, "_pde_3d_slice_canvas")

    def _build_pde_tabs(self) -> None:
        """Solution 3D (surface) + Solution 2D (contour) + Phase Space slice."""
        nb = self._notebook
        xlabel, ylabel = self._pde_axis_labels()

        # --- Tab 1: 3D Surface ---
        surf_tab = ttk.Frame(nb)
        nb.add(surf_tab, text="  Solution 3D  ")

        surf_ctrl = self._create_view_controls(surf_tab)
        axis_group = surf_ctrl.add_group("Transform along")
        self._pde_3d_axis_var = tk.StringVar(value=xlabel)
        axis_combo = ttk.Combobox(
            axis_group,
            textvariable=self._pde_3d_axis_var,
            values=[xlabel, ylabel],
            state="readonly",
            width=4,
            font=get_font(),
        )
        axis_combo.pack(side=tk.LEFT)
        axis_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_pde_3d())

        transform_group = surf_ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_pde_3d, "pde_3d")
        if self._result.equation_type == "vector_pde":
            field_group = surf_ctrl.add_group("Field")
            self._add_vector_pde_field_selector(
                field_group, "_pde_3d_field_var", self._update_pde_3d
            )

        self._pde_3d_frame = ttk.Frame(surf_tab)
        self._pde_3d_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_3d_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_pde_3d)

        # --- Tab 2: 2D Contour ---
        contour_tab = ttk.Frame(nb)
        nb.add(contour_tab, text="  Solution 2D  ")

        contour_ctrl = self._create_view_controls(contour_tab)
        axis_group = contour_ctrl.add_group("Transform along")
        self._pde_2d_axis_var = tk.StringVar(value=xlabel)
        axis_combo = ttk.Combobox(
            axis_group,
            textvariable=self._pde_2d_axis_var,
            values=[xlabel, ylabel],
            state="readonly",
            width=4,
            font=get_font(),
        )
        axis_combo.pack(side=tk.LEFT)
        axis_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_pde_2d())

        transform_group = contour_ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_pde_2d, "pde_2d")
        if self._result.equation_type == "vector_pde":
            field_group = contour_ctrl.add_group("Field")
            self._add_vector_pde_field_selector(
                field_group,
                "_pde_2d_field_var",
                self._update_pde_2d,
            )

        self._pde_2d_frame = ttk.Frame(contour_tab)
        self._pde_2d_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_2d_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_pde_2d)

        if self._result.equation_type == "pde":
            polar_tab = ttk.Frame(nb)
            nb.add(polar_tab, text="  Polar View  ")
            polar_ctrl = self._create_view_controls(polar_tab)
            origin_x_group = polar_ctrl.add_group("Origin x")
            self._pde_polar_origin_x_var = tk.StringVar(value="0")
            ttk.Entry(origin_x_group, textvariable=self._pde_polar_origin_x_var, width=9).pack(
                side=tk.LEFT
            )
            origin_y_group = polar_ctrl.add_group("Origin y")
            self._pde_polar_origin_y_var = tk.StringVar(value="0")
            ttk.Entry(origin_y_group, textvariable=self._pde_polar_origin_y_var, width=9).pack(
                side=tk.LEFT
            )
            action_group = polar_ctrl.add_group()
            ttk.Button(action_group, text="Update", command=self._update_pde_polar).pack(
                side=tk.LEFT
            )
            self._pde_polar_frame = ttk.Frame(polar_tab)
            self._pde_polar_frame.pack(fill=tk.BOTH, expand=True)
            self._pde_polar_canvas: FigureCanvasTkAgg | None = None
            self._queue_initial_plot(self._update_pde_polar)

        if self._result.equation_type == "vector_pde":
            self._build_vector_pde_field_tab()

        sweep_tab = ttk.Frame(nb)
        nb.add(sweep_tab, text="  Axis Sweep  ")
        sweep_ctrl = self._create_view_controls(sweep_tab)
        sweep_group = sweep_ctrl.add_group("Sweep along")
        sweep_labels = self._pde_coordinate_labels(2)
        self._pde_2d_sweep_axis_var = tk.StringVar(value=sweep_labels[1])
        sweep_selector = ttk.Combobox(
            sweep_group,
            textvariable=self._pde_2d_sweep_axis_var,
            values=sweep_labels,
            state="readonly",
            width=max(5, max(map(len, sweep_labels))),
            font=get_font(),
        )
        sweep_selector.pack(side=tk.LEFT)
        sweep_selector.bind("<<ComboboxSelected>>", lambda _event: self._update_pde_2d_sweep())
        if self._result.equation_type == "vector_pde":
            field_group = sweep_ctrl.add_group("Field")
            self._add_vector_pde_field_selector(
                field_group,
                "_pde_2d_sweep_field_var",
                self._update_pde_2d_sweep,
            )
        self._pde_2d_sweep_frame = ttk.Frame(sweep_tab)
        self._pde_2d_sweep_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_2d_sweep_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_pde_2d_sweep)

        # --- Tab 3: Transform (1D slice) ---
        trans_tab = ttk.Frame(nb)
        nb.add(trans_tab, text="  Transform  ")

        trans_ctrl = self._create_view_controls(trans_tab)

        xlabel, ylabel = self._pde_axis_labels()

        varying_group = trans_ctrl.add_group("Vary axis")
        self._pde_slice_var = tk.StringVar(value=xlabel)
        varying_combo = ttk.Combobox(
            varying_group,
            textvariable=self._pde_slice_var,
            values=[xlabel, ylabel],
            state="readonly",
            width=2,
            font=get_font(),
        )
        varying_combo.pack(side=tk.LEFT)

        y_grid = self._require_pde_y_grid()
        y_mid = float((y_grid[0] + y_grid[-1]) / 2) if len(y_grid) > 0 else 0.5

        fixed_group = trans_ctrl.add_group()
        self._pde_fixed_axis_label = ttk.Label(fixed_group, text=f"Fixed {ylabel}")
        self._pde_fixed_axis_label.pack(side=tk.LEFT, padx=(0, 5))
        self._pde_slice_val_var = tk.StringVar(value=str(round(y_mid, 4)))
        ttk.Entry(
            fixed_group,
            textvariable=self._pde_slice_val_var,
            width=9,
            font=get_font(),
        ).pack(side=tk.LEFT)

        transform_group = trans_ctrl.add_group("Transform")
        self._build_transform_controls(transform_group, self._update_pde_transform, "pde")
        if self._result.equation_type == "vector_pde":
            field_group = trans_ctrl.add_group("Field")
            self._add_vector_pde_field_selector(
                field_group,
                "_pde_slice_field_var",
                self._update_pde_transform,
            )

        action_group = trans_ctrl.add_group()
        ttk.Button(
            action_group,
            text="Update",
            command=self._update_pde_transform,
        ).pack(side=tk.LEFT)
        varying_combo.bind("<<ComboboxSelected>>", lambda _event: self._change_pde_slice_axis())

        self._pde_trans_frame = ttk.Frame(trans_tab)
        self._pde_trans_frame.pack(fill=tk.BOTH, expand=True)
        self._pde_trans_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_pde_transform)

    def _add_vector_pde_field_selector(
        self,
        parent: ttk.Frame,
        variable_name: str,
        callback: Callable[[], None],
    ) -> None:
        """Add a component/magnitude selector only for Vector PDE results."""
        if self._result.equation_type != "vector_pde":
            return
        labels = [f"Component {index}" for index in range(self._result.vector_components)]
        labels.append("Magnitude")
        variable = tk.StringVar(value=labels[0])
        setattr(self, variable_name, variable)
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=labels,
            state="readonly",
            width=13,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _event: callback())

    def _selected_pde_field(self, variable_name: str) -> tuple[np.ndarray, str]:
        """Return the selected scalar field and its display label."""
        result = self._result
        if result.equation_type != "vector_pde":
            return np.asarray(result.y), "f"
        variable = getattr(self, variable_name, None)
        label = variable.get() if variable is not None else "Component 0"
        if label == "Magnitude":
            return np.linalg.norm(result.y, axis=0), "|f|"
        try:
            component = int(label.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            component = 0
        component = max(0, min(result.vector_components - 1, component))
        return np.asarray(result.y[component]), f"f[{component}]"

    def _pde_axis_labels(self) -> tuple[str, str]:
        """Return (xlabel, ylabel) from metadata variable names."""
        variables = self._result.metadata.get("variables", ["x[0]", "x[1]"])
        xlabel = variables[0] if len(variables) > 0 else "x[0]"
        ylabel = variables[1] if len(variables) > 1 else "x[1]"
        return xlabel, ylabel

    def _pde_coordinate_labels(self, dimension: int) -> list[str]:
        """Return safe displayed coordinate labels for a PDE result."""
        fallback = ["x", "y", "z"][:dimension]
        variables = self._result.metadata.get("variables", [])
        if not isinstance(variables, (list, tuple)):
            variables = []
        return [
            str(variables[index])
            if index < len(variables) and variables[index]
            else fallback[index]
            for index in range(dimension)
        ]

    def _pde_sweep_axis(self, selected_label: str, dimension: int) -> Literal["x", "y", "z"]:
        """Map a displayed coordinate label to the canonical sweep axis."""
        labels = self._pde_coordinate_labels(dimension)
        index = labels.index(selected_label) if selected_label in labels else dimension - 1
        return cast(Literal["x", "y", "z"], "xyz"[index])

    def _create_pde_2d_axis_sweep_figure(self) -> Figure:
        """Build the selected stationary 2D PDE spatial-sweep animation."""
        from plotting import create_line_animation_plot
        from plotting.coordinates import prepare_scalar_axis_sweep_2d

        result = self._result
        y_grid = self._require_pde_y_grid()
        x_label, y_label = self._pde_coordinate_labels(2)
        field, field_label = self._selected_pde_field("_pde_2d_sweep_field_var")
        axis = cast(Literal["x", "y"], self._pde_sweep_axis(self._pde_2d_sweep_axis_var.get(), 2))
        sweep = prepare_scalar_axis_sweep_2d(result.x, y_grid, field, axis)
        if axis == "x":
            xlabel, frame_label = y_label, x_label
        else:
            xlabel, frame_label = x_label, y_label
        equation_name = result.metadata.get("equation_name", "PDE")
        return create_line_animation_plot(
            sweep.sweep_coordinates,
            sweep.axis_1,
            sweep.frames,
            title=f"{equation_name} — Axis sweep — {field_label}",
            xlabel=xlabel,
            ylabel=field_label,
            frame_label=frame_label,
        )

    def _create_pde_3d_axis_sweep_figure(self) -> Figure:
        """Build the selected stationary 3D PDE spatial-sweep animation."""
        from plotting import create_image_animation_plot
        from plotting.coordinates import prepare_scalar_axis_sweep_3d

        result = self._result
        if result.y_grid is None or result.z_grid is None:
            raise ValueError("PDE 3D result is missing its y/z grids")
        labels = self._pde_coordinate_labels(3)
        axis = self._pde_sweep_axis(self._pde_3d_sweep_axis_var.get(), 3)
        sweep = prepare_scalar_axis_sweep_3d(
            result.x,
            result.y_grid,
            result.z_grid,
            np.asarray(result.y),
            axis,
        )
        axis_labels = {
            "x": (labels[1], labels[2]),
            "y": (labels[0], labels[2]),
            "z": (labels[0], labels[1]),
        }
        xlabel, ylabel = axis_labels[axis]
        equation_name = result.metadata.get("equation_name", "PDE 3D")
        if sweep.axis_2 is None:
            raise ValueError("3D axis sweep is missing its vertical coordinate")
        return create_image_animation_plot(
            sweep.sweep_coordinates,
            sweep.frames,
            title=f"{equation_name} — Axis sweep",
            xlabel=xlabel,
            ylabel=ylabel,
            x_coordinates=sweep.axis_1,
            y_coordinates=sweep.axis_2,
            frame_label=labels["xyz".index(axis)],
        )

    def _replace_animation_plot(
        self,
        frame: ttk.Frame,
        figure_factory: Callable[[], Figure],
        canvas_attr: str,
        export_prefix: str,
    ) -> None:
        """Replace an embedded animation and own only its current canvas."""
        old_canvas: FigureCanvasTkAgg | None = getattr(self, canvas_attr, None)
        if old_canvas is not None:
            self._dispose_canvas(old_canvas)
            self._unregister_canvas(old_canvas)
        for child in frame.winfo_children():
            child.destroy()
        figure = figure_factory()
        canvas = embed_animation_plot_in_tk(
            figure,
            frame,
            on_export_mp4=lambda duration: self._export_axis_sweep_mp4(
                figure_factory, duration, export_prefix
            ),
        )
        setattr(self, canvas_attr, canvas)
        self._register_canvas(canvas)

    def _update_pde_2d_sweep(self) -> None:
        """Render the selected stationary 2D PDE axis sweep."""
        self._replace_animation_plot(
            self._pde_2d_sweep_frame,
            self._create_pde_2d_axis_sweep_figure,
            "_pde_2d_sweep_canvas",
            "axis_sweep",
        )

    def _update_pde_3d_sweep(self) -> None:
        """Render the selected stationary 3D PDE axis sweep."""
        self._replace_animation_plot(
            self._pde_3d_sweep_frame,
            self._create_pde_3d_axis_sweep_figure,
            "_pde_3d_sweep_canvas",
            "axis_sweep",
        )

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
                func = interp1d(x, z[i, :], kind="cubic", fill_value=_EXTRAPOLATE_FILL)
                tx, ty, txlabel, _tylabel = apply_transform(
                    lambda arr, f=func: f(arr),
                    kind,
                    float(x[0]),
                    float(x[-1]),
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
                    resamp = interp1d(tx_i, ty_i, kind="linear", fill_value=0.0, bounds_error=False)
                    new_rows.append(resamp(tx_common))
            new_z = np.array(new_rows)
            return tx_common, y_grid, new_z, txlabel, ylabel
        else:
            # Transform along y_grid (rows): for each column, transform f vs y
            raw_c: list[tuple[np.ndarray, np.ndarray]] = []
            tylabel = ""
            for j in range(z.shape[1]):
                func = interp1d(y_grid, z[:, j], kind="cubic", fill_value=_EXTRAPOLATE_FILL)
                ty, tz, tylabel, _tzlabel = apply_transform(
                    lambda arr, f=func: f(arr),
                    kind,
                    float(y_grid[0]),
                    float(y_grid[-1]),
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
                    resamp = interp1d(ty_j, tz_j, kind="linear", fill_value=0.0, bounds_error=False)
                    new_cols.append(resamp(ty_common))
            new_z = np.column_stack(new_cols)
            return x, ty_common, new_z, xlabel, tylabel

    def _update_pde_3d(self) -> None:
        """Render the 3D surface plot for PDE."""
        from plotting import create_surface_plot
        from transforms import TransformKind

        r = self._result
        y_grid = self._require_pde_y_grid()
        xlabel, ylabel = self._pde_axis_labels()
        eq_name = r.metadata.get("equation_name", f"f({xlabel},{ylabel})")
        field, field_label = self._selected_pde_field("_pde_3d_field_var")

        kind = self._get_transform_kind("pde_3d")
        if kind != TransformKind.ORIGINAL:
            axis_var = self._pde_3d_axis_var.get()
            result = self._transform_pde_along_axis(
                r.x,
                y_grid,
                field,
                axis_var,
                kind,
            )
            if result is not None:
                px, py, pz, pxl, pyl = result
                fig = create_surface_plot(
                    px,
                    py,
                    pz,
                    title=f"{eq_name} — {kind.value}",
                    xlabel=pxl,
                    ylabel=pyl,
                    zlabel=field_label,
                )
                self._replace_plot(self._pde_3d_frame, fig, "_pde_3d_canvas")
                return

        fig = create_surface_plot(
            r.x,
            y_grid,
            field,
            title=f"{eq_name} — {field_label}" if r.equation_type == "vector_pde" else eq_name,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=field_label,
        )
        self._replace_plot(self._pde_3d_frame, fig, "_pde_3d_canvas")

    def _update_pde_2d(self) -> None:
        """Render the 2D contour plot for PDE."""
        from plotting import create_contour_plot
        from transforms import TransformKind

        r = self._result
        y_grid = self._require_pde_y_grid()
        xlabel, ylabel = self._pde_axis_labels()
        eq_name = r.metadata.get("equation_name", f"f({xlabel},{ylabel})")
        field, field_label = self._selected_pde_field("_pde_2d_field_var")

        kind = self._get_transform_kind("pde_2d")
        if kind != TransformKind.ORIGINAL:
            axis_var = self._pde_2d_axis_var.get()
            result = self._transform_pde_along_axis(
                r.x,
                y_grid,
                field,
                axis_var,
                kind,
            )
            if result is not None:
                px, py, pz, pxl, pyl = result
                fig = create_contour_plot(
                    px,
                    py,
                    pz,
                    title=f"{eq_name} — {kind.value}",
                    xlabel=pxl,
                    ylabel=pyl,
                )
                self._replace_plot(self._pde_2d_frame, fig, "_pde_2d_canvas")
                return

        fig = create_contour_plot(
            r.x,
            y_grid,
            field,
            title=f"{eq_name} — {field_label}" if r.equation_type == "vector_pde" else eq_name,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        self._replace_plot(self._pde_2d_frame, fig, "_pde_2d_canvas")

    def _update_pde_polar(self) -> None:
        """Render a scalar PDE field resampled for a polar display."""
        from plotting import create_polar_contour_plot

        try:
            origin = (
                float(self._pde_polar_origin_x_var.get()),
                float(self._pde_polar_origin_y_var.get()),
            )
        except ValueError:
            origin = (0.0, 0.0)
        y_grid = self._require_pde_y_grid()
        xlabel, ylabel = self._pde_axis_labels()
        eq_name = self._result.metadata.get("equation_name", f"f({xlabel},{ylabel})")
        figure = create_polar_contour_plot(
            self._result.x,
            y_grid,
            np.asarray(self._result.y),
            title=f"{eq_name} — polar view about ({origin[0]:.4g}, {origin[1]:.4g})",
            origin=origin,
        )
        self._replace_plot(self._pde_polar_frame, figure, "_pde_polar_canvas")

    def _build_vector_pde_field_tab(self) -> None:
        """Build field-specific views without duplicating the plot canvas lifecycle."""
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Vector Field  ")
        controls = self._create_view_controls(tab)
        self._vector_pde_controls = controls
        view_group = controls.add_group("View")
        self._vector_pde_view_var = tk.StringVar(value="Magnitude")
        views = self._vector_pde_view_labels()
        selector = ttk.Combobox(
            view_group,
            textvariable=self._vector_pde_view_var,
            values=views,
            state="readonly",
            width=19,
            font=get_font(),
        )
        selector.pack(side=tk.LEFT)
        origin_x_group = controls.add_group("Origin x")
        self._vector_pde_origin_x_var = tk.StringVar(value="0")
        ttk.Entry(origin_x_group, textvariable=self._vector_pde_origin_x_var, width=9).pack(
            side=tk.LEFT
        )
        origin_y_group = controls.add_group("Origin y")
        self._vector_pde_origin_y_var = tk.StringVar(value="0")
        ttk.Entry(origin_y_group, textvariable=self._vector_pde_origin_y_var, width=9).pack(
            side=tk.LEFT
        )
        action_group = controls.add_group()
        ttk.Button(action_group, text="Update", command=self._update_vector_pde_field).pack(
            side=tk.LEFT,
        )
        self._vector_pde_origin_groups = (origin_x_group, origin_y_group, action_group)
        selector.bind("<<ComboboxSelected>>", lambda _event: self._change_vector_pde_view())
        self._set_vector_pde_origin_visibility(False)
        self._vector_pde_field_frame = ttk.Frame(tab)
        self._vector_pde_field_frame.pack(fill=tk.BOTH, expand=True)
        self._vector_pde_field_canvas: FigureCanvasTkAgg | None = None
        self._queue_initial_plot(self._update_vector_pde_field)

    def _vector_pde_view_labels(self) -> list[str]:
        """Return views compatible with the Vector PDE component count."""
        views = ["Components", "Magnitude"]
        if self._result.vector_components == 2:
            views.extend(["Quiver", "Streamlines", "Radial/Tangential"])
        return views

    def _set_vector_pde_origin_visibility(self, visible: bool) -> None:
        """Apply progressive disclosure to Radial/Tangential origin controls."""
        for group in self._vector_pde_origin_groups:
            self._vector_pde_controls.set_group_visible(group, visible)

    def _change_vector_pde_view(self) -> None:
        """Refresh disclosure and plot immediately after choosing a field view."""
        self._set_vector_pde_origin_visibility(
            vector_field_uses_origin(self._vector_pde_view_var.get())
        )
        self._update_vector_pde_field()

    def _update_vector_pde_field(self) -> None:
        """Render the selected vector PDE field view."""
        from plotting import create_vector_field_plot

        view_map = {
            "Components": "components",
            "Magnitude": "magnitude",
            "Quiver": "quiver",
            "Streamlines": "stream",
            "Radial/Tangential": "radial_tangential",
        }
        selected = self._vector_pde_view_var.get()
        plot_kwargs: dict[str, Any] = {
            "view": view_map.get(selected, "magnitude"),
            "title": f"{self._result.metadata.get('equation_name', 'Vector PDE')} — {selected}",
        }
        if vector_field_uses_origin(selected):
            try:
                plot_kwargs["origin"] = (
                    float(self._vector_pde_origin_x_var.get()),
                    float(self._vector_pde_origin_y_var.get()),
                )
            except ValueError:
                plot_kwargs["origin"] = (0.0, 0.0)
        figure = create_vector_field_plot(
            self._result.x,
            self._require_pde_y_grid(),
            np.asarray(self._result.y),
            **plot_kwargs,
        )
        self._replace_plot(
            self._vector_pde_field_frame,
            figure,
            "_vector_pde_field_canvas",
        )

    def _change_pde_slice_axis(self) -> None:
        """Synchronize the fixed-coordinate label and redraw the current slice."""
        xlabel, ylabel = self._pde_axis_labels()
        fixed_axis = pde_fixed_axis_label(self._pde_slice_var.get(), xlabel, ylabel)
        self._pde_fixed_axis_label.configure(text=f"Fixed {fixed_axis}")
        self._update_pde_transform()

    def _update_pde_transform(self) -> None:
        """Render a 1D transform of a slice through the PDE solution."""
        from plotting import create_solution_plot
        from transforms import TransformKind, apply_transform

        r = self._result
        y_grid = self._require_pde_y_grid()
        kind = self._get_transform_kind("pde")
        xlabel, ylabel = self._pde_axis_labels()
        field, field_label = self._selected_pde_field("_pde_slice_field_var")

        slice_var = self._pde_slice_var.get()
        try:
            slice_val = float(self._pde_slice_val_var.get())
        except ValueError:
            slice_val = 0.5

        x_1d, data_1d, axis_label, fixed_axis_label, _fixed_index = extract_pde_line_slice(
            slice_var,
            xlabel,
            ylabel,
            r.x,
            y_grid,
            field,
            slice_val,
        )
        slice_label = f"{fixed_axis_label}={slice_val:.3g}"

        eq_name = r.metadata.get("equation_name", "PDE")

        if kind == TransformKind.ORIGINAL:
            fig = create_solution_plot(
                x_1d,
                np.atleast_2d(data_1d),
                title=f"{eq_name} \u2014 slice at {slice_label}",
                xlabel=axis_label,
                ylabel=field_label,
                selected_derivatives=[0],
                labels=[field_label],
            )
        else:
            from scipy.interpolate import interp1d

            func = interp1d(x_1d, data_1d, kind="cubic", fill_value=_EXTRAPOLATE_FILL)
            x_min_t, x_max_t = float(x_1d[0]), float(x_1d[-1])
            tx, ty, txlabel, tylabel = apply_transform(
                lambda arr: func(arr),
                kind,
                x_min_t,
                x_max_t,
            )
            fig = create_solution_plot(
                tx,
                np.atleast_2d(ty),
                title=f"{eq_name} \u2014 {kind.value} [slice {slice_label}]",
                xlabel=txlabel,
                ylabel=tylabel,
                selected_derivatives=[0],
                labels=[tylabel],
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
        """Reuse the existing canvas when replacing a matplotlib figure."""
        old_canvas: FigureCanvasTkAgg | None = getattr(self, canvas_attr, None)
        canvas = replace_plot_in_tk(fig, frame, current_canvas=old_canvas)
        if old_canvas is not None and canvas is not old_canvas:
            self._dispose_canvas(old_canvas)
            self._unregister_canvas(old_canvas)
        setattr(self, canvas_attr, canvas)
        self._register_canvas(canvas)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_export_file(
        self,
        export_fn: Callable[[Path], None],
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
                "Export saved",
                f"{prefix_log} was saved to:\n{path}",
                parent=self.win,
            )
        except Exception as exc:
            logger.error(f"{prefix_log} export failed: %s", exc, exc_info=True)
            messagebox.showerror("Export was not saved", str(exc), parent=self.win)

    def _on_save_csv(self) -> None:
        r = self._result

        def export_fn(path: Path) -> None:
            export_csv_to_path(r.x, r.y, path, y_grid=r.y_grid, z_grid=r.z_grid)

        self._save_export_file(
            export_fn,
            ".csv",
            [("CSV files", "*.csv"), ("All files", "*.*")],
            "CSV",
        )

    def _on_save_json(self) -> None:
        r = self._result

        def export_fn(path: Path) -> None:
            export_json_to_path(r.statistics, r.metadata, path)

        self._save_export_file(
            export_fn,
            ".json",
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
                r.x,
                r.y,
                r.vector_order,
                r.vector_components,
                filepath,
                title=f"{r.metadata.get('equation_name', 'ODE')} — f_i(x)",
                duration_seconds=duration_seconds,
            )
            messagebox.showinfo(
                "Animation export saved",
                f"Animation was saved to:\n{filepath}",
                parent=self.win,
            )
        except RuntimeError as exc:
            logger.warning("MP4 export failed (ffmpeg): %s", exc)
            messagebox.showerror(
                "Animation export was not saved",
                str(exc) + "\n\nInstall ffmpeg and ensure it is in your PATH.",
                parent=self.win,
            )
        except Exception as exc:
            logger.error("MP4 export failed: %s", exc, exc_info=True)
            messagebox.showerror("Animation export was not saved", str(exc), parent=self.win)

    def _export_axis_sweep_mp4(
        self,
        figure_factory: Callable[[], Figure],
        duration_seconds: float,
        prefix: str = "axis_sweep",
    ) -> None:
        """Export a fresh PDE sweep figure using the current dialog selections."""
        default_path = get_output_dir() / f"{generate_output_basename(prefix=prefix)}.mp4"
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
            from plotting import export_animated_figure_to_mp4

            export_animated_figure_to_mp4(
                figure_factory(), filepath, duration_seconds=duration_seconds
            )
            messagebox.showinfo(
                "Animation export saved",
                f"Animation was saved to:\n{filepath}",
                parent=self.win,
            )
        except RuntimeError as exc:
            logger.warning("MP4 export failed (ffmpeg): %s", exc)
            messagebox.showerror(
                "Animation export was not saved",
                str(exc) + "\n\nInstall ffmpeg and ensure it is in your PATH.",
                parent=self.win,
            )
        except Exception as exc:
            logger.error("MP4 export failed: %s", exc, exc_info=True)
            messagebox.showerror("Animation export was not saved", str(exc), parent=self.win)

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
