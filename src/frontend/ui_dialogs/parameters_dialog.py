"""Parameters dialog — configure domain, ICs, method, and statistics."""

from __future__ import annotations

import re
import tkinter as tk
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, cast

from config import (
    AVAILABLE_STATISTICS,
    DEFAULT_SOLVER_METHOD,
    SOLVER_METHOD_DESCRIPTIONS,
    SOLVER_METHODS,
    get_env_from_schema,
)
from frontend.performance_guard import (
    assess_parameters_dialog_request,
    assess_pde_3d_request,
    confirm_performance_advisory,
)
from frontend.theme import get_font
from frontend.ui_dialogs.background_task import BackgroundTaskFailure, run_task_with_loading
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.solve_session import (
    EquationSelection,
    ParametersFormState,
    SolveSession,
)
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import (
    bind_wraplength,
    calculate_screen_aware_minsize,
    fit_and_center,
    make_modal,
)
from solver.predefined import EquationType
from utils import DifferentialLabError, get_logger

logger = get_logger(__name__)

_MAX_PDE_GRID = 1000
_DEFAULT_VECTOR_PDE_GRID = 100
_STACKED_LAYOUT_BREAKPOINT = 900
_LAYOUT_DEBOUNCE_MS = 60

_STATISTIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Summary",
        ("mean", "rms", "std", "median", "max", "min", "integral", "l2_norm"),
    ),
    (
        "Oscillation / dynamics",
        (
            "zero_crossings",
            "period",
            "amplitude",
            "dominant_frequency",
            "angular_frequency",
            "energy",
        ),
    ),
    (
        "Growth / decay",
        ("exponential_rate", "half_life", "time_constant", "doubling_time"),
    ),
    ("PDE", ("gradient_norm",)),
)

_STATISTIC_LABELS: dict[str, str] = {
    "mean": "Mean",
    "rms": "RMS",
    "std": "Standard deviation",
    "median": "Median",
    "max": "Maximum",
    "min": "Minimum",
    "integral": "Integral",
    "l2_norm": "L2 norm",
    "zero_crossings": "Zero crossings",
    "period": "Period",
    "amplitude": "Amplitude",
    "dominant_frequency": "Dominant frequency",
    "angular_frequency": "Angular frequency",
    "energy": "Energy",
    "exponential_rate": "Exponential rate",
    "half_life": "Half-life",
    "time_constant": "Time constant",
    "doubling_time": "Doubling time",
    "gradient_norm": "Gradient norm",
}

_PDE_FACE_SOLVER_ORDER: dict[str, tuple[str, ...]] = {
    "2d": ("bottom", "top", "left", "right"),
    "3d": ("z_min", "z_max", "y_min", "y_max", "x_min", "x_max"),
}

_PDE_FACE_VISUAL_ORDER: dict[str, tuple[str, ...]] = {
    "2d": ("left", "right", "bottom", "top"),
    "3d": ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
}

_PDE_FACE_LABELS: dict[str, str] = {
    "left": "Left, x = xmin",
    "right": "Right, x = xmax",
    "bottom": "Bottom, y = ymin",
    "top": "Top, y = ymax",
    "x_min": "x min",
    "x_max": "x max",
    "y_min": "y min",
    "y_max": "y max",
    "z_min": "z min",
    "z_max": "z max",
}

_PDE_FACE_FREE_VARIABLES: dict[str, str] = {
    "left": "y",
    "right": "y",
    "bottom": "x",
    "top": "x",
    "x_min": "y and z",
    "x_max": "y and z",
    "y_min": "x and z",
    "y_max": "x and z",
    "z_min": "x and y",
    "z_max": "x and y",
}


def _preferred_configuration_size(equation_type: str) -> tuple[int, int]:
    """Return a family-appropriate preferred Configuration window size."""
    return {
        "difference": (760, 620),
        "ode": (980, 720),
        "vector_ode": (1080, 780),
        "pde": (1120, 800),
        "vector_pde": (1120, 800),
        "pde_3d": (1180, 840),
    }.get(equation_type, (980, 720))


def _uses_stacked_layout(viewport_width: int) -> bool:
    """Return whether Configuration content should use one logical column."""
    return viewport_width < _STACKED_LAYOUT_BREAKPOINT


def _pde_face_orders(equation_type: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return visual and solver serialization orders for a PDE family."""
    dimension = "3d" if equation_type == "pde_3d" else "2d"
    return _PDE_FACE_VISUAL_ORDER[dimension], _PDE_FACE_SOLVER_ORDER[dimension]


def _default_pde_grid_points(equation_type: str) -> int:
    """Return the interactive default grid size for a PDE equation type."""
    if equation_type == "pde_3d":
        return 25
    if equation_type == "vector_pde":
        return _DEFAULT_VECTOR_PDE_GRID
    return 1000


def _format_solver_exception(exc: BaseException) -> BackgroundTaskFailure:
    """Map solver exceptions to user-facing dialog failures."""
    if isinstance(exc, DifferentialLabError):
        logger.warning("Solver pipeline failed (user-facing): %s", exc)
        return BackgroundTaskFailure("Solver input issue", str(exc))

    if isinstance(exc, (MemoryError, OSError)):
        logger.error("Solver pipeline: memory/system error: %s", exc, exc_info=True)
        return BackgroundTaskFailure(
            "Not enough memory",
            f"The solver ran out of memory: {exc}\n\nTry reducing the grid size (points per axis).",
        )

    logger.exception("Solver pipeline: unexpected error")
    return BackgroundTaskFailure("Solver error", str(exc))


class _InputValidationError(ValueError):
    """User-facing input validation error with a dialog title."""

    def __init__(self, title: str, message: str) -> None:
        super().__init__(message)
        self.title = title
        self.message = message


@dataclass(frozen=True, slots=True)
class _SolverInputs:
    """Validated inputs ready to pass to the solver pipeline."""

    x_min: float
    x_max: float
    y0: list[float]
    n_points: int
    method: str
    selected_stats: set[str]
    parameters: dict[str, Any]
    x0_list: list[float] | None
    y_min: float | None
    y_max: float | None
    n_points_y: int | None
    z_min: float | None
    z_max: float | None
    n_points_z: int | None
    bc_expressions: list[str] | None
    bc_types: list[str] | None
    mask_expression: str | None
    contour_bc_expression: str | None
    contour_bc_type: str | None
    event_expression: str | None
    event_terminal: bool
    event_direction: int


class ParametersDialog:
    """Dialog for configuring solver parameters, ICs, and statistics.

    Args:
        parent: Parent window.
        expression: ODE expression string (optional).
        function_name: Name of function in config.equations (optional).
        order: ODE order.
        parameters: Parameter name-value mapping.
        equation_name: Display name.
        default_y0: Default initial conditions.
        default_domain: Default ``[x_min, x_max]``.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        expression: str | None = None,
        function_name: str | None = None,
        order: int,
        parameters: Mapping[str, float | list[float]],
        equation_name: str,
        default_y0: list[float],
        default_domain: list[float],
        parameters_schema: dict[str, dict[str, Any]] | None = None,
        display_formula: str | None = None,
        equation_type: str = "ode",
        variables: list[str] | None = None,
        vector_expressions: list[str] | None = None,
        vector_components: int = 1,
        pde_operator: str = "neg_laplacian",
        component_orders: tuple[int, ...] | None = None,
        default_boundary_conditions: dict[str, dict[str, str]] | None = None,
        session: SolveSession | None = None,
        selection: EquationSelection | None = None,
    ) -> None:
        self.parent = parent
        self.expression = expression
        self.function_name = function_name
        self.order = order
        self.parameters: dict[str, float | list[float]] = dict(parameters)
        self.equation_name = equation_name
        self.display_formula = (
            display_formula
            if display_formula is not None
            else (
                "; ".join(vector_expressions)
                if vector_expressions
                else (expression or f"<function:{function_name}>")
            )
        )
        self.parameters_schema = parameters_schema or {}
        self.equation_type = equation_type
        self.variables = variables if variables else ["x"]
        self.vector_expressions = vector_expressions
        self.vector_components = vector_components
        self.is_vector = (
            vector_expressions is not None and len(vector_expressions) > 0
        ) or equation_type in ("vector_ode", "vector_pde")
        self.pde_operator = pde_operator
        self.is_pde = equation_type in ("pde", "pde_3d", "vector_pde") or len(self.variables) > 1
        self.component_orders = component_orders
        self.default_boundary_conditions = default_boundary_conditions or {}
        self.session = session
        self.selection = selection

        self.win = tk.Toplevel(parent)
        self.win.title(f"Solve - {equation_name}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._y0_vars: list[tk.StringVar] = []
        self._x0_vars: list[tk.StringVar] = []
        self._eq_param_vars: dict[str, tk.StringVar] = {}
        self._stat_vars: dict[str, tk.BooleanVar] = {}
        self.event_enabled_var: tk.BooleanVar | None = None
        self.event_expression_var: tk.StringVar | None = None
        self.event_terminal_var: tk.BooleanVar | None = None
        self.event_direction_var: tk.StringVar | None = None
        self._layout_job: str | None = None
        self._stacked_layout: bool | None = None

        self._build_ui(default_y0, default_domain)

        if self.session is not None and self.selection is not None:
            snapshot = self.session.configuration_for(self.selection)
            if snapshot is not None:
                self.apply_form_state(snapshot)

        preferred_width, preferred_height = _preferred_configuration_size(self.equation_type)
        fit_and_center(
            self.win,
            min_width=preferred_width,
            min_height=preferred_height,
            resizable=True,
        )
        min_width, min_height = calculate_screen_aware_minsize(
            self.win.winfo_screenwidth(),
            self.win.winfo_screenheight(),
            640,
            520,
        )
        self.win.minsize(min_width, min_height)
        make_modal(self.win, parent)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, default_y0: list[float], default_domain: list[float]) -> None:
        pad: int = get_env_from_schema("UI_PADDING")

        # ── Fixed bottom button bar ──
        btn_solve = self._build_action_buttons(pad)

        header = ttk.Frame(self.win, padding=(pad, pad, pad, 0))
        header.pack(side=tk.TOP, fill=tk.X)
        title_row = ttk.Frame(header)
        title_row.pack(fill=tk.X)
        ttk.Label(title_row, text="Configure equation", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_row, text="Step 2 of 3", style="Small.TLabel").pack(side=tk.RIGHT)
        ttk.Label(header, text=self.equation_name, style="Subtitle.TLabel").pack(
            anchor=tk.W, pady=(pad, 2)
        )
        formula_lbl = ttk.Label(
            header,
            text=self.display_formula,
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        formula_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, pad))
        bind_wraplength(header, formula_lbl, pad=2 * pad, min_wrap=200)

        # ── Scrollable content ──
        self._scroll = ScrollableFrame(self.win)
        self._scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        self._scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scroll_frame = self._scroll.inner
        scroll_frame.configure(padding=pad)

        # Two-column layout: left = domain + ICs, right = solver + statistics
        left_col, right_col = self._build_layout_columns(scroll_frame, pad)

        is_diff = self._build_domain_and_parameter_sections(left_col, pad, default_domain)
        self._build_initial_or_boundary_sections(
            left_col,
            pad,
            default_y0,
            default_domain,
            is_diff,
        )

        # Solver method (ODE only) — right column
        self._build_solver_method_section(right_col, pad)
        self._build_event_section(right_col, pad)

        self._build_statistics_section(right_col, pad)

        bind_wraplength(self.method_frame, self.method_desc, pad=2 * pad, min_wrap=150)
        self._scroll.bind_new_children()
        self._scroll.viewport.bind("<Configure>", self._schedule_content_layout, add="+")
        self._scroll.viewport.after(100, self._apply_content_layout)
        btn_solve.focus_set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_action_buttons(self, pad: int) -> ttk.Button:
        """Build the fixed bottom solve/cancel button row."""
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        self._btn_back = ttk.Button(
            btn_frame,
            text="Back",
            style="Secondary.TButton",
            command=self._on_back,
        )
        self._btn_back.pack(side=tk.LEFT)

        btn_inner = ttk.Frame(btn_frame)
        btn_inner.pack(side=tk.RIGHT)

        self._btn_cancel = ttk.Button(
            btn_inner,
            text="Cancel",
            style="Secondary.TButton",
            command=self.win.destroy,
        )
        self._btn_cancel.pack(side=tk.LEFT, padx=(0, pad))

        btn_solve = ttk.Button(
            btn_inner,
            text="Solve",
            style="Primary.TButton",
            command=self._on_solve,
        )
        self._btn_solve = btn_solve
        btn_solve.pack(side=tk.LEFT)

        setup_arrow_enter_navigation([[self._btn_back, self._btn_cancel, btn_solve]])
        return btn_solve

    def _build_equation_summary(self, parent: ttk.Frame, pad: int) -> ttk.Label:
        """Build the equation title and formula summary."""
        ttk.Label(parent, text=f"Equation: {self.equation_name}", style="Subtitle.TLabel").pack(
            anchor=tk.W, pady=(0, pad)
        )
        formula_label = ttk.Label(
            parent,
            text=self.display_formula,
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        formula_label.pack(anchor=tk.W, pady=(0, pad))
        return formula_label

    def _build_layout_columns(self, parent: ttk.Frame, pad: int) -> tuple[ttk.Frame, ttk.Frame]:
        """Build logical columns that can stack without recreating their controls."""
        self._columns_frame = ttk.Frame(parent)
        self._columns_frame.pack(fill=tk.BOTH, expand=True, pady=(0, pad))

        self._left_col = ttk.Frame(self._columns_frame)
        self._right_col = ttk.Frame(self._columns_frame)
        return self._left_col, self._right_col

    def _schedule_content_layout(self, _event: tk.Event[tk.Misc]) -> None:
        """Debounce the wide/tall Configuration layout switch."""
        if self._layout_job is not None:
            try:
                self._scroll.viewport.after_cancel(self._layout_job)
            except tk.TclError:
                pass
        self._layout_job = self._scroll.viewport.after(
            _LAYOUT_DEBOUNCE_MS,
            self._apply_content_layout,
        )

    def _apply_content_layout(self) -> None:
        """Use two logical columns when wide and a natural vertical stack when narrow."""
        self._layout_job = None
        stacked = _uses_stacked_layout(self._scroll.viewport.winfo_width())
        if stacked == self._stacked_layout:
            return
        self._stacked_layout = stacked
        pad: int = get_env_from_schema("UI_PADDING")

        self._left_col.grid_forget()
        self._right_col.grid_forget()
        for column in range(2):
            self._columns_frame.columnconfigure(column, weight=0, minsize=0, uniform="")
        for row in range(2):
            self._columns_frame.rowconfigure(row, weight=0)

        if stacked:
            self._columns_frame.columnconfigure(0, weight=1)
            self._left_col.grid(row=0, column=0, sticky=tk.EW, padx=0)
            self._right_col.grid(row=1, column=0, sticky=tk.EW, padx=0)
        else:
            self._columns_frame.columnconfigure(0, weight=1, uniform="configuration")
            self._columns_frame.columnconfigure(1, weight=1, uniform="configuration")
            self._left_col.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, pad // 2))
            self._right_col.grid(row=0, column=1, sticky=tk.NSEW, padx=(pad // 2, 0))
        self._scroll.refresh_scroll_region()

    def _build_domain_and_parameter_sections(
        self,
        left_col: ttk.Frame,
        pad: int,
        default_domain: list[float],
    ) -> bool:
        """Build a simple 1D domain form or a structured PDE axis table."""
        is_diff = self.equation_type == "difference"
        domain_frame = ttk.LabelFrame(left_col, text="Domain", padding=pad)
        domain_frame.pack(fill=tk.X, pady=(0, pad))

        xmin_val = int(default_domain[0]) if is_diff else default_domain[0]
        xmax_val = int(default_domain[1]) if is_diff else default_domain[1]
        self.xmin_var = tk.StringVar(value=str(xmin_val))
        self.xmax_var = tk.StringVar(value=str(xmax_val))
        self.ymin_var: tk.StringVar | None = None
        self.ymax_var: tk.StringVar | None = None
        self.npoints_y_var: tk.StringVar | None = None
        self.zmin_var: tk.StringVar | None = None
        self.zmax_var: tk.StringVar | None = None
        self.npoints_z_var: tk.StringVar | None = None

        if self.is_pde:
            self.npoints_var = tk.StringVar(value=str(_default_pde_grid_points(self.equation_type)))
            self.ymin_var = tk.StringVar(value=str(default_domain[2]))
            self.ymax_var = tk.StringVar(value=str(default_domain[3]))
            self.npoints_y_var = tk.StringVar(
                value=str(_default_pde_grid_points(self.equation_type))
            )
            axis_rows: list[tuple[str, tk.StringVar, tk.StringVar, tk.StringVar]] = [
                ("x", self.xmin_var, self.xmax_var, self.npoints_var),
                ("y", self.ymin_var, self.ymax_var, self.npoints_y_var),
            ]
            if self.equation_type == "pde_3d":
                self.zmin_var = tk.StringVar(value=str(default_domain[4]))
                self.zmax_var = tk.StringVar(value=str(default_domain[5]))
                self.npoints_z_var = tk.StringVar(value="25")
                axis_rows.append(("z", self.zmin_var, self.zmax_var, self.npoints_z_var))

            for column, text in enumerate(("Axis", "Min", "Max", "Grid points")):
                ttk.Label(domain_frame, text=text, style="Small.TLabel").grid(
                    row=0,
                    column=column,
                    sticky=tk.W,
                    padx=(0, pad),
                    pady=(0, pad // 2),
                )
            for row_index, (axis, min_var, max_var, points_var) in enumerate(axis_rows, start=1):
                ttk.Label(domain_frame, text=axis, width=6).grid(
                    row=row_index,
                    column=0,
                    sticky=tk.W,
                    pady=2,
                )
                for column, variable in enumerate((min_var, max_var, points_var), start=1):
                    ttk.Entry(
                        domain_frame,
                        textvariable=variable,
                        width=12,
                        font=get_font(),
                    ).grid(
                        row=row_index,
                        column=column,
                        sticky=tk.EW,
                        padx=(0, pad),
                        pady=2,
                    )
            for column in range(1, 4):
                domain_frame.columnconfigure(column, weight=1)
        else:
            labels = ("n min", "n max") if is_diff else ("x min", "x max")
            for row_index, (label, variable) in enumerate(
                zip(labels, (self.xmin_var, self.xmax_var), strict=True)
            ):
                ttk.Label(domain_frame, text=label).grid(
                    row=row_index,
                    column=0,
                    sticky=tk.W,
                    padx=(0, pad),
                    pady=2,
                )
                ttk.Entry(
                    domain_frame,
                    textvariable=variable,
                    width=16,
                    font=get_font(),
                ).grid(row=row_index, column=1, sticky=tk.EW, pady=2)
            domain_frame.columnconfigure(1, weight=1)
            if not is_diff:
                self.npoints_var = tk.StringVar(value=str(get_env_from_schema("SOLVER_NUM_POINTS")))
                ttk.Label(domain_frame, text="Sample points").grid(
                    row=2,
                    column=0,
                    sticky=tk.W,
                    padx=(0, pad),
                    pady=2,
                )
                ttk.Spinbox(
                    domain_frame,
                    textvariable=self.npoints_var,
                    from_=10,
                    to=10_000_000,
                    increment=100,
                    width=14,
                    font=get_font(),
                ).grid(row=2, column=1, sticky=tk.W, pady=2)

        self._build_equation_parameters_section(left_col, pad)
        return is_diff

    def _build_equation_parameters_section(self, parent: ttk.Frame, pad: int) -> None:
        """Build aligned equation-parameter controls while preserving raw parsing."""
        if not self.parameters:
            return
        sub_digits = "₀₁₂₃₄₅₆₇₈₉"

        def _subscript_n(number: int) -> str:
            return "".join(sub_digits[int(digit)] for digit in str(number))

        def _display_name_for(parameter_name: str) -> str:
            info = self.parameters_schema.get(parameter_name, {})
            match = re.match(r"^(.+)\[(\d+)\]$", parameter_name)
            if match:
                return f"{match.group(1)}{_subscript_n(int(match.group(2)))}"
            display = info.get("display")
            return str(display) if display is not None else parameter_name

        frame = ttk.LabelFrame(parent, text="Equation Parameters", padding=pad)
        frame.pack(fill=tk.X, pady=(0, pad))
        frame.columnconfigure(1, weight=1)
        for row_index, (parameter_name, value) in enumerate(self.parameters.items()):
            info = self.parameters_schema.get(parameter_name, {})
            ttk.Label(frame, text=_display_name_for(parameter_name)).grid(
                row=row_index,
                column=0,
                sticky=tk.W,
                padx=(0, pad),
                pady=2,
            )
            default_text = (
                ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
            )
            variable = tk.StringVar(value=default_text)
            entry = ttk.Entry(frame, textvariable=variable, width=22, font=get_font())
            entry.grid(row=row_index, column=1, sticky=tk.EW, pady=2)
            self._eq_param_vars[parameter_name] = variable
            match = re.match(r"^(.+)\[(\d+)\]$", parameter_name)
            expected_count = (
                int(match.group(2)) if match else (len(value) if isinstance(value, list) else 1)
            )
            help_text = str(info.get("description", ""))
            if isinstance(value, list) and not help_text:
                help_text = f"Comma-separated values ({expected_count} components)"
            ToolTip(entry, help_text)

    def _build_initial_or_boundary_sections(
        self,
        left_col: ttk.Frame,
        pad: int,
        default_y0: list[float],
        default_domain: list[float],
        is_diff: bool,
    ) -> None:
        """Build structured initial-condition or PDE boundary tables."""
        self._bc_vars: list[tk.StringVar] = []
        self._bc_type_vars: list[tk.StringVar] = []
        self._bc_expression_vars_by_face: dict[str, tk.StringVar] = {}
        self._bc_type_vars_by_face: dict[str, tk.StringVar] = {}
        self._domain_shape_var: tk.StringVar | None = None
        self._mask_expr_var: tk.StringVar | None = None
        self._contour_bc_expr_var: tk.StringVar | None = None
        self._contour_bc_type_var: tk.StringVar | None = None
        self._rect_bc_frame: ttk.LabelFrame | None = None
        self._contour_bc_frame: ttk.LabelFrame | None = None

        if self.is_pde:
            self._build_pde_boundary_sections(left_col, pad)
            return

        frame = ttk.LabelFrame(left_col, text="Initial Conditions", padding=pad)
        frame.pack(fill=tk.X, pady=(0, pad))
        headings = (
            ("Quantity", "Initial value")
            if is_diff
            else (
                "Quantity",
                "Initial value",
                "At x",
            )
        )
        for column, text in enumerate(headings):
            ttk.Label(frame, text=text, style="Small.TLabel").grid(
                row=0,
                column=column,
                sticky=tk.W,
                padx=(0, pad),
                pady=(0, pad // 2),
            )
        frame.columnconfigure(1, weight=1)
        if not is_diff:
            frame.columnconfigure(2, weight=1)

        if self.component_orders:
            n_initial = sum(self.component_orders)
        elif self.is_vector:
            n_initial = self.order * self.vector_components
        else:
            n_initial = self.order
        default_position = str(int(default_domain[0]) if is_diff else default_domain[0])
        labels = self._ic_labels()
        for index in range(n_initial):
            ttk.Label(frame, text=labels[index]).grid(
                row=index + 1,
                column=0,
                sticky=tk.W,
                padx=(0, pad),
                pady=2,
            )
            value_var = tk.StringVar(
                value=str(default_y0[index] if index < len(default_y0) else 1.0)
            )
            ttk.Entry(frame, textvariable=value_var, width=14, font=get_font()).grid(
                row=index + 1,
                column=1,
                sticky=tk.EW,
                padx=(0, pad),
                pady=2,
            )
            position_var = tk.StringVar(value=default_position)
            if not is_diff:
                ttk.Entry(
                    frame,
                    textvariable=position_var,
                    width=14,
                    font=get_font(),
                ).grid(row=index + 1, column=2, sticky=tk.EW, pady=2)
            self._y0_vars.append(value_var)
            self._x0_vars.append(position_var)

    def _build_pde_boundary_sections(self, parent: ttk.Frame, pad: int) -> None:
        """Build shape controls and face-keyed boundary tables for PDE families."""
        shape_frame = ttk.LabelFrame(parent, text="Domain Shape", padding=pad)
        shape_frame.pack(fill=tk.X, pady=(0, pad))
        shape_frame.columnconfigure(1, weight=1)
        ttk.Label(shape_frame, text="Shape").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, pad),
        )
        self._domain_shape_var = tk.StringVar(value="Rectangle")
        shape_combo = ttk.Combobox(
            shape_frame,
            textvariable=self._domain_shape_var,
            values=(
                ["Rectangle"] if self.equation_type == "pde_3d" else ["Rectangle", "Custom contour"]
            ),
            state="readonly",
            width=18,
            font=get_font(),
        )
        shape_combo.grid(row=0, column=1, sticky=tk.W)
        shape_combo.bind("<<ComboboxSelected>>", self._on_domain_shape_change)

        self._mask_row = ttk.Frame(shape_frame)
        self._mask_row.columnconfigure(1, weight=1)
        ttk.Label(self._mask_row, text="Mask expression").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, pad),
        )
        self._mask_expr_var = tk.StringVar(value="x**2 + y**2 <= 1")
        mask_entry = ttk.Entry(
            self._mask_row,
            textvariable=self._mask_expr_var,
            width=30,
            font=get_font(),
        )
        mask_entry.grid(row=0, column=1, sticky=tk.EW)
        ToolTip(
            mask_entry,
            "Boolean expression defining the domain, for example x**2 + y**2 <= 1.",
        )

        self._rect_bc_frame = ttk.LabelFrame(
            parent,
            text=(
                "Shared Boundary Conditions (all components)"
                if self.equation_type == "vector_pde"
                else "Boundary Conditions"
            ),
            padding=pad,
        )
        self._rect_bc_frame.pack(fill=tk.X, pady=(0, pad))
        for column, text in enumerate(("Boundary", "Type", "Value")):
            ttk.Label(self._rect_bc_frame, text=text, style="Small.TLabel").grid(
                row=0,
                column=column,
                sticky=tk.W,
                padx=(0, pad),
                pady=(0, pad // 2),
            )
        self._rect_bc_frame.columnconfigure(2, weight=1)

        visual_order, solver_order = _pde_face_orders(self.equation_type)
        for row_index, face in enumerate(visual_order, start=1):
            ttk.Label(self._rect_bc_frame, text=_PDE_FACE_LABELS[face]).grid(
                row=row_index,
                column=0,
                sticky=tk.W,
                padx=(0, pad),
                pady=2,
            )
            preset = self.default_boundary_conditions.get(face, {})
            type_var = tk.StringVar(value=preset.get("type", "Dirichlet"))
            ttk.Combobox(
                self._rect_bc_frame,
                textvariable=type_var,
                values=["Dirichlet", "Neumann"],
                state="readonly",
                width=11,
                font=get_font(),
            ).grid(row=row_index, column=1, sticky=tk.W, padx=(0, pad), pady=2)
            expression_var = tk.StringVar(value=preset.get("expression", "0"))
            entry = ttk.Entry(
                self._rect_bc_frame,
                textvariable=expression_var,
                width=18,
                font=get_font(),
            )
            entry.grid(row=row_index, column=2, sticky=tk.EW, pady=2)
            ToolTip(
                entry,
                f"Expression in {_PDE_FACE_FREE_VARIABLES[face]}. Dirichlet sets the value; "
                "Neumann sets the outward-normal derivative.",
            )
            self._bc_type_vars_by_face[face] = type_var
            self._bc_expression_vars_by_face[face] = expression_var

        # These compatibility lists deliberately retain the historical solver order.
        self._bc_vars = [self._bc_expression_vars_by_face[face] for face in solver_order]
        self._bc_type_vars = [self._bc_type_vars_by_face[face] for face in solver_order]

        self._contour_bc_frame = ttk.LabelFrame(
            parent,
            text=(
                "Shared Contour Boundary (all components)"
                if self.equation_type == "vector_pde"
                else "Contour Boundary Conditions"
            ),
            padding=pad,
        )
        self._contour_bc_frame.columnconfigure(1, weight=1)
        ttk.Label(self._contour_bc_frame, text="Boundary type").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, pad),
            pady=2,
        )
        self._contour_bc_type_var = tk.StringVar(value="Dirichlet")
        ttk.Combobox(
            self._contour_bc_frame,
            textvariable=self._contour_bc_type_var,
            values=["Dirichlet", "Neumann"],
            state="readonly",
            width=11,
            font=get_font(),
        ).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(self._contour_bc_frame, text="Boundary value").grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=(0, pad),
            pady=2,
        )
        self._contour_bc_expr_var = tk.StringVar(value="0")
        contour_entry = ttk.Entry(
            self._contour_bc_frame,
            textvariable=self._contour_bc_expr_var,
            width=25,
            font=get_font(),
        )
        contour_entry.grid(row=1, column=1, sticky=tk.EW, pady=2)
        ToolTip(
            contour_entry,
            "Expression in x and y for a boundary value or outward-normal derivative.",
        )

    def _build_solver_method_section(self, parent: ttk.Frame, pad: int) -> None:
        """Build solver-method controls."""
        self.method_frame = ttk.LabelFrame(parent, text="Solver Method", padding=pad)
        self.method_frame.pack(fill=tk.X, pady=(0, pad))

        self.method_var = tk.StringVar(value=DEFAULT_SOLVER_METHOD)
        combo = ttk.Combobox(
            self.method_frame,
            textvariable=self.method_var,
            values=list(SOLVER_METHODS),
            state="readonly",
            width=15,
            font=get_font(),
        )
        combo.pack(anchor=tk.W)
        self.method_desc = ttk.Label(
            self.method_frame,
            text="",
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        self.method_desc.pack(anchor=tk.W, pady=(2, 0))
        combo.bind("<<ComboboxSelected>>", self._on_method_change)
        self._on_method_change(None)
        if self.equation_type == "difference" or self.is_pde:
            self.method_frame.pack_forget()

    def _build_event_section(self, parent: ttk.Frame, pad: int) -> None:
        """Build progressively disclosed event controls for ODE IVP solves."""
        if self.equation_type == "difference" or self.is_pde:
            self._event_frame = None
            self._event_controls_frame = None
            return

        self._event_frame = ttk.LabelFrame(parent, text="Event", padding=pad)
        self._event_frame.pack(fill=tk.X, pady=(0, pad))
        self.event_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._event_frame,
            text="Detect an event",
            variable=self.event_enabled_var,
            command=self._on_event_enabled_change,
        ).pack(anchor=tk.W)

        self._event_controls_frame = ttk.Frame(self._event_frame)
        self._event_controls_frame.columnconfigure(1, weight=1)
        self.event_expression_var = tk.StringVar(value="")
        ttk.Label(self._event_controls_frame, text="Expression").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, pad),
            pady=(pad, 2),
        )
        expression_entry = ttk.Entry(
            self._event_controls_frame,
            textvariable=self.event_expression_var,
            width=24,
            font=get_font(),
        )
        expression_entry.grid(row=0, column=1, sticky=tk.EW, pady=(pad, 2))
        ToolTip(
            expression_entry,
            "A safe expression in x and the ODE state, for example f[0] - 1. "
            "A zero marks the event.",
        )

        self.event_direction_var = tk.StringVar(value="0")
        ttk.Label(self._event_controls_frame, text="Direction").grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=(0, pad),
            pady=2,
        )
        ttk.Combobox(
            self._event_controls_frame,
            textvariable=self.event_direction_var,
            values=("-1", "0", "1"),
            state="readonly",
            width=5,
            font=get_font(),
        ).grid(row=1, column=1, sticky=tk.W, pady=2)

        self.event_terminal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._event_controls_frame,
            text="Stop integration at event",
            variable=self.event_terminal_var,
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        self._on_event_enabled_change()

    def _build_statistics_section(self, parent: ttk.Frame, pad: int) -> ttk.LabelFrame:
        """Build grouped checkbox controls for every computed metric."""
        frame = ttk.LabelFrame(parent, text="Computed metrics", padding=pad)
        frame.pack(fill=tk.X, pady=(0, pad))
        self._stat_keys = list(AVAILABLE_STATISTICS)
        self._stat_vars = {key: tk.BooleanVar(value=True) for key in self._stat_keys}

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(0, pad // 2))
        ttk.Button(
            actions,
            text="Select all",
            style="Small.TButton",
            command=self._select_all_statistics,
        ).pack(side=tk.LEFT)
        ttk.Button(
            actions,
            text="Clear",
            style="Small.TButton",
            command=self._clear_statistics,
        ).pack(side=tk.LEFT, padx=(pad // 2, 0))

        groups = ttk.Frame(frame)
        groups.pack(fill=tk.X)
        groups.columnconfigure(0, weight=1, uniform="metric-group")
        groups.columnconfigure(1, weight=1, uniform="metric-group")
        for group_index, (title, keys) in enumerate(_STATISTIC_GROUPS):
            group = ttk.Frame(groups)
            group.grid(
                row=group_index // 2,
                column=group_index % 2,
                sticky=tk.NW,
                padx=(0, pad if group_index % 2 == 0 else 0),
                pady=(0, pad // 2),
            )
            ttk.Label(group, text=title, style="Small.TLabel").pack(anchor=tk.W)
            for key in keys:
                checkbox = ttk.Checkbutton(
                    group,
                    text=_STATISTIC_LABELS[key],
                    variable=self._stat_vars[key],
                )
                checkbox.pack(anchor=tk.W)
                ToolTip(checkbox, AVAILABLE_STATISTICS[key])
        return frame

    def _ic_labels(self) -> list[str]:
        subscripts = "₀₁₂₃₄₅₆₇₈₉"

        def _sub(i: int) -> str:
            return subscripts[i] if i < len(subscripts) else str(i)

        if self.equation_type == "difference":
            return [f"f{_sub(i)}" for i in range(self.order)]
        if self.is_vector:
            labels: list[str] = []
            orders = self.component_orders or tuple(
                self.order for _ in range(self.vector_components)
            )
            for c, comp_order in enumerate(orders):
                comp_sub = _sub(c)
                for k in range(comp_order):
                    if k == 0:
                        labels.append(f"f{comp_sub}")
                    else:
                        primes = "\u2032" * k
                        labels.append(f"f{primes}{comp_sub}")
            return labels
        return [
            "f" if derivative == 0 else f"f{'′' * derivative}" for derivative in range(self.order)
        ]

    def _on_domain_shape_change(self, _event: Any) -> None:
        """Toggle visibility between rectangular and custom contour BC sections."""
        is_custom = self._domain_shape_var and self._domain_shape_var.get() == "Custom contour"
        if is_custom:
            self._mask_row.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky=tk.EW,
                pady=(4, 0),
            )
            if self._rect_bc_frame:
                self._rect_bc_frame.pack_forget()
            if self._contour_bc_frame:
                self._contour_bc_frame.pack(fill=tk.X, pady=(0, 4))
            # Suggest symmetric domain for custom contours (e.g. circles)
            if self.xmin_var and float(self.xmin_var.get() or 0) >= 0:
                self.xmin_var.set("-1.0")
                self.xmax_var.set("1.0")
            if self.ymin_var and float(self.ymin_var.get() or 0) >= 0:
                self.ymin_var.set("-1.0")
            if self.ymax_var:
                self.ymax_var.set("1.0")
        else:
            self._mask_row.grid_remove()
            if self._contour_bc_frame:
                self._contour_bc_frame.pack_forget()
            if self._rect_bc_frame:
                self._rect_bc_frame.pack(fill=tk.X, pady=(0, 4))

    def _on_method_change(self, _event: Any) -> None:
        method = self.method_var.get()
        desc = SOLVER_METHOD_DESCRIPTIONS.get(method, "")
        self.method_desc.config(text=desc)

    def _on_event_enabled_change(self) -> None:
        """Show enabled event controls and remove disabled controls from tab traversal."""
        if self._event_controls_frame is None or self.event_enabled_var is None:
            return
        if self.event_enabled_var.get():
            self._event_controls_frame.pack(fill=tk.X)
        else:
            self._event_controls_frame.pack_forget()
        if hasattr(self, "_scroll"):
            self._scroll.refresh_scroll_region()

    def _select_all_statistics(self) -> None:
        """Select every available computed metric."""
        for variable in self._stat_vars.values():
            variable.set(True)

    def _clear_statistics(self) -> None:
        """Clear every computed metric selection."""
        for variable in self._stat_vars.values():
            variable.set(False)

    @staticmethod
    def _raw_value(variable: Any) -> str | None:
        """Return a Tk-like variable value as plain text."""
        if variable is None:
            return None
        return str(variable.get())

    def capture_form_state(self) -> ParametersFormState:
        """Capture every current form value without validating or retaining Tk objects."""
        selected_statistics = tuple(
            key for key in self._stat_keys if bool(self._stat_vars[key].get())
        )
        return ParametersFormState(
            x_min=str(self.xmin_var.get()),
            x_max=str(self.xmax_var.get()),
            n_points=self._raw_value(getattr(self, "npoints_var", None)) or "",
            y_min=self._raw_value(self.ymin_var),
            y_max=self._raw_value(self.ymax_var),
            n_points_y=self._raw_value(self.npoints_y_var),
            z_min=self._raw_value(self.zmin_var),
            z_max=self._raw_value(self.zmax_var),
            n_points_z=self._raw_value(self.npoints_z_var),
            initial_values=tuple(str(variable.get()) for variable in self._y0_vars),
            initial_positions=tuple(str(variable.get()) for variable in self._x0_vars),
            parameter_values={
                name: str(variable.get()) for name, variable in self._eq_param_vars.items()
            },
            method=self._raw_value(getattr(self, "method_var", None)) or "",
            statistics=selected_statistics,
            event_enabled=(
                bool(self.event_enabled_var.get()) if self.event_enabled_var is not None else False
            ),
            event_expression=self._raw_value(self.event_expression_var),
            event_terminal=(
                bool(self.event_terminal_var.get())
                if self.event_terminal_var is not None
                else False
            ),
            event_direction=self._raw_value(self.event_direction_var),
            domain_shape=self._raw_value(self._domain_shape_var),
            boundary_expressions=tuple(str(variable.get()) for variable in self._bc_vars),
            boundary_types=tuple(str(variable.get()) for variable in self._bc_type_vars),
            mask_expression=self._raw_value(self._mask_expr_var),
            contour_boundary_expression=self._raw_value(self._contour_bc_expr_var),
            contour_boundary_type=self._raw_value(self._contour_bc_type_var),
        )

    @staticmethod
    def _set_raw_value(variable: Any, value: str | bool | None) -> None:
        """Apply a plain value to a compatible Tk-like variable when both exist."""
        if variable is not None and value is not None:
            variable.set(value)

    def apply_form_state(self, snapshot: ParametersFormState) -> None:
        """Restore a compatible raw Configuration snapshot after normal control creation."""
        self._set_raw_value(self._domain_shape_var, snapshot.domain_shape)
        if self._domain_shape_var is not None:
            self._on_domain_shape_change(None)

        self._set_raw_value(self.xmin_var, snapshot.x_min)
        self._set_raw_value(self.xmax_var, snapshot.x_max)
        self._set_raw_value(getattr(self, "npoints_var", None), snapshot.n_points)
        self._set_raw_value(self.ymin_var, snapshot.y_min)
        self._set_raw_value(self.ymax_var, snapshot.y_max)
        self._set_raw_value(self.npoints_y_var, snapshot.n_points_y)
        self._set_raw_value(self.zmin_var, snapshot.z_min)
        self._set_raw_value(self.zmax_var, snapshot.z_max)
        self._set_raw_value(self.npoints_z_var, snapshot.n_points_z)

        for variable, value in zip(self._y0_vars, snapshot.initial_values, strict=False):
            variable.set(value)
        for variable, value in zip(self._x0_vars, snapshot.initial_positions, strict=False):
            variable.set(value)
        for name, value in snapshot.parameter_values.items():
            self._set_raw_value(self._eq_param_vars.get(name), value)

        self._set_raw_value(getattr(self, "method_var", None), snapshot.method)
        if getattr(self, "method_var", None) is not None:
            self._on_method_change(None)

        selected = set(snapshot.statistics)
        for key, variable in self._stat_vars.items():
            variable.set(key in selected)

        self._set_raw_value(self.event_enabled_var, snapshot.event_enabled)
        self._set_raw_value(self.event_expression_var, snapshot.event_expression)
        self._set_raw_value(self.event_terminal_var, snapshot.event_terminal)
        self._set_raw_value(self.event_direction_var, snapshot.event_direction)
        if self.event_enabled_var is not None:
            self._on_event_enabled_change()
        for variable, value in zip(self._bc_vars, snapshot.boundary_expressions, strict=False):
            variable.set(value)
        for variable, value in zip(self._bc_type_vars, snapshot.boundary_types, strict=False):
            variable.set(value)
        self._set_raw_value(self._mask_expr_var, snapshot.mask_expression)
        self._set_raw_value(self._contour_bc_expr_var, snapshot.contour_boundary_expression)
        self._set_raw_value(self._contour_bc_type_var, snapshot.contour_boundary_type)

    def _release_tk_state(self) -> None:
        """Drop all Tk variable references while execution is still on the Tk thread."""
        vars_to_discard: list[Any] = []
        vars_to_discard.extend(self._y0_vars)
        vars_to_discard.extend(self._x0_vars)
        vars_to_discard.extend(self._eq_param_vars.values())
        vars_to_discard.extend(self._stat_vars.values())
        vars_to_discard.extend(self._bc_vars)
        vars_to_discard.extend(self._bc_type_vars)
        for attr in (
            "xmin_var",
            "xmax_var",
            "ymin_var",
            "ymax_var",
            "zmin_var",
            "zmax_var",
            "npoints_var",
            "npoints_y_var",
            "npoints_z_var",
            "method_var",
            "event_enabled_var",
            "event_expression_var",
            "event_terminal_var",
            "event_direction_var",
            "_domain_shape_var",
            "_mask_expr_var",
            "_contour_bc_expr_var",
            "_contour_bc_type_var",
        ):
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    vars_to_discard.append(value)
                setattr(self, attr, None)
        self._y0_vars.clear()
        self._x0_vars.clear()
        self._eq_param_vars.clear()
        self._stat_vars.clear()
        self._bc_vars.clear()
        self._bc_type_vars.clear()
        self._bc_expression_vars_by_face.clear()
        self._bc_type_vars_by_face.clear()
        vars_to_discard.clear()

    def _on_back(self) -> None:
        """Save raw Configuration state and return to the retained Equation step."""
        if self.session is None or self.selection is None:
            self.win.destroy()
            return
        self.session.save_configuration(self.capture_form_state(), self.selection)
        parent = self.parent
        session = self.session
        self.win.destroy()
        self._release_tk_state()

        from frontend.ui_dialogs.equation_dialog import EquationDialog

        EquationDialog(parent, session=session)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _parse_equation_parameters(self) -> dict[str, Any]:
        """Parse equation parameter widgets into numeric values."""
        if not self._eq_param_vars:
            return self.parameters

        import numpy as _np

        params: dict[str, Any] = {}
        for pname, var in self._eq_param_vars.items():
            raw = var.get().strip()
            param_match = re.match(r"^(.+)\[(\d+)\]$", pname)
            if param_match:
                base_name = param_match.group(1)
                try:
                    values = [float(v.strip()) for v in raw.split(",")]
                except ValueError:
                    raise _InputValidationError(
                        "Check the parameter value",
                        f"Parameter '{pname}' must be comma-separated numbers.",
                    ) from None
                params[base_name] = _np.array(values)
                continue

            try:
                params[pname] = float(raw)
            except ValueError:
                raise _InputValidationError(
                    "Check the parameter value",
                    f"Parameter '{pname}' must be a number.",
                ) from None

        self.parameters = params
        return params

    def _parse_domain(self) -> tuple[float, float]:
        """Parse the primary x/n domain values."""
        try:
            return float(self.xmin_var.get()), float(self.xmax_var.get())
        except ValueError:
            domain_name = (
                "n\u2098\u1d62\u2099 and n\u2098\u2090\u2093"
                if self.equation_type == "difference"
                else "x\u2098\u1d62\u2099 and x\u2098\u2090\u2093"
            )
            raise _InputValidationError(
                "Check the domain",
                f"{domain_name} must be numbers.",
            ) from None

    def _parse_pde_domain_and_grid(self) -> tuple[float, float, int, int]:
        """Parse 2D PDE y-domain and grid sizes."""
        if self.ymin_var is None or self.ymax_var is None:
            raise _InputValidationError(
                "Check the PDE domain",
                "y\u2098\u1d62\u2099 and y\u2098\u2090\u2093 required.",
            )
        try:
            y_min = float(self.ymin_var.get())
            y_max = float(self.ymax_var.get())
        except ValueError:
            raise _InputValidationError(
                "Check the domain",
                "y\u2098\u1d62\u2099 and y\u2098\u2090\u2093 must be numbers.",
            ) from None
        try:
            n_points = int(self.npoints_var.get())
            n_points_y = int(self.npoints_y_var.get()) if self.npoints_y_var else n_points
        except (ValueError, AttributeError):
            raise _InputValidationError(
                "Check the grid size",
                "Grid points must be integers.",
            ) from None
        if n_points > _MAX_PDE_GRID or n_points_y > _MAX_PDE_GRID:
            raise _InputValidationError(
                "Grid size is too large",
                f"PDE grid is limited to {_MAX_PDE_GRID} points per axis to avoid "
                f"excessive memory use. You entered {n_points}\u00d7{n_points_y}.",
            )
        return y_min, y_max, n_points, n_points_y

    def _parse_pde_3d_domain_and_grid(
        self,
    ) -> tuple[float, float, int, int, float, float, int]:
        """Parse y/z bounds and all three grid sizes for a rectangular PDE 3D."""
        y_min, y_max, n_points, n_points_y = self._parse_pde_domain_and_grid()
        if self.zmin_var is None or self.zmax_var is None or self.npoints_z_var is None:
            raise _InputValidationError(
                "Check the PDE 3D domain",
                "zₘᵢₙ, zₘₐₓ, and the z grid size are required.",
            )
        try:
            z_min = float(self.zmin_var.get())
            z_max = float(self.zmax_var.get())
            n_points_z = int(self.npoints_z_var.get())
        except ValueError:
            raise _InputValidationError(
                "Check the PDE 3D domain",
                "z bounds must be numbers and z grid points must be an integer.",
            ) from None
        if n_points_z > _MAX_PDE_GRID:
            raise _InputValidationError(
                "Grid size is too large",
                f"PDE grid is limited to {_MAX_PDE_GRID} points per axis.",
            )
        return y_min, y_max, n_points, n_points_y, z_min, z_max, n_points_z

    def _parse_ic_points(self) -> list[float]:
        """Parse per-initial-condition x positions."""
        subscripts = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
        x0_list: list[float] = []
        for i, x_var in enumerate(self._x0_vars):
            sub = subscripts[i] if i < len(subscripts) else str(i)
            try:
                x0_list.append(float(x_var.get()))
            except ValueError:
                raise _InputValidationError(
                    "Check the initial-condition point",
                    f"x{sub} must be a number.",
                ) from None
        return x0_list

    def _parse_initial_conditions(self) -> list[float]:
        """Parse initial condition values for non-PDE solves."""
        y0_list: list[float] = []
        for i, var in enumerate(self._y0_vars):
            try:
                y0_list.append(float(var.get()))
            except ValueError:
                raise _InputValidationError(
                    "Check the initial conditions",
                    f"Initial condition {i} must be a number.",
                ) from None
        return y0_list

    def _collect_pde_options(
        self,
    ) -> tuple[list[str] | None, list[str] | None, str | None, str | None, str | None]:
        """Collect PDE boundary, mask, and contour options."""
        bc_expressions: list[str] | None = None
        bc_types: list[str] | None = None
        mask_expression: str | None = None
        contour_bc_expression: str | None = None
        contour_bc_type: str | None = None

        if not self.is_pde:
            return (
                bc_expressions,
                bc_types,
                mask_expression,
                contour_bc_expression,
                contour_bc_type,
            )

        is_custom_contour = (
            self._domain_shape_var is not None and self._domain_shape_var.get() == "Custom contour"
        )
        if is_custom_contour:
            mask_expression = self._mask_expr_var.get().strip() if self._mask_expr_var else None
            if not mask_expression:
                raise _InputValidationError(
                    "Add a mask expression",
                    "Custom contour requires a mask expression.",
                )
            contour_bc_type = (
                self._contour_bc_type_var.get().strip().lower()
                if self._contour_bc_type_var
                else "dirichlet"
            )
            contour_bc_expression = (
                self._contour_bc_expr_var.get().strip() or "0" if self._contour_bc_expr_var else "0"
            )
        else:
            if self._bc_vars:
                bc_expressions = [var.get().strip() or "0" for var in self._bc_vars]
            if self._bc_type_vars:
                bc_types = [var.get().strip().lower() for var in self._bc_type_vars]

        return (
            bc_expressions,
            bc_types,
            mask_expression,
            contour_bc_expression,
            contour_bc_type,
        )

    def _collect_solver_inputs(self) -> _SolverInputs:
        """Collect validated user inputs for ``run_solver_pipeline``."""
        parameters = self._parse_equation_parameters()
        x_min, x_max = self._parse_domain()

        if self.equation_type == "pde_3d":
            (
                y_min,
                y_max,
                n_points,
                n_points_y,
                z_min,
                z_max,
                n_points_z,
            ) = self._parse_pde_3d_domain_and_grid()
            y0 = []
            x0_list = None
            method = "fdm"
        elif self.is_pde:
            y_min, y_max, n_points, n_points_y = self._parse_pde_domain_and_grid()
            z_min = None
            z_max = None
            n_points_z = None
            y0: list[float] = []
            x0_list = None
            method = "fdm"
        elif self.equation_type == "difference":
            n_points = int(x_max) - int(x_min) + 1
            y_min = None
            y_max = None
            n_points_y = None
            z_min = None
            z_max = None
            n_points_z = None
            x0_list = None
            method = "iteration"
            y0 = self._parse_initial_conditions()
        else:
            try:
                n_points = int(self.npoints_var.get())
            except ValueError:
                raise _InputValidationError(
                    "Check the grid size",
                    "Number of points must be an integer.",
                ) from None
            y_min = None
            y_max = None
            n_points_y = None
            z_min = None
            z_max = None
            n_points_z = None
            x0_list = self._parse_ic_points()
            method = self.method_var.get()
            y0 = self._parse_initial_conditions()

        selected_stats = {key for key in self._stat_keys if bool(self._stat_vars[key].get())}
        (
            bc_expressions,
            bc_types,
            mask_expression,
            contour_bc_expression,
            contour_bc_type,
        ) = self._collect_pde_options()

        event_expression: str | None = None
        event_terminal = False
        event_direction = 0
        event_enabled = (
            bool(self.event_enabled_var.get()) if self.event_enabled_var is not None else False
        )
        if event_enabled and self.equation_type not in (
            "difference",
            "pde",
            "pde_3d",
            "vector_pde",
        ):
            if self.event_expression_var is not None:
                event_expression = self.event_expression_var.get().strip() or None
            if self.event_terminal_var is not None:
                event_terminal = bool(self.event_terminal_var.get())
            if self.event_direction_var is not None:
                try:
                    event_direction = int(self.event_direction_var.get())
                except ValueError:
                    raise _InputValidationError(
                        "Check the event direction",
                        "Event direction must be -1, 0, or 1.",
                    ) from None
                if event_direction not in (-1, 0, 1):
                    raise _InputValidationError(
                        "Check the event direction",
                        "Event direction must be -1, 0, or 1.",
                    )

        return _SolverInputs(
            x_min=x_min,
            x_max=x_max,
            y0=y0,
            n_points=n_points,
            method=method,
            selected_stats=selected_stats,
            parameters=parameters,
            x0_list=x0_list,
            y_min=y_min,
            y_max=y_max,
            n_points_y=n_points_y,
            z_min=z_min,
            z_max=z_max,
            n_points_z=n_points_z,
            bc_expressions=bc_expressions,
            bc_types=bc_types,
            mask_expression=mask_expression,
            contour_bc_expression=contour_bc_expression,
            contour_bc_type=contour_bc_type,
            event_expression=event_expression,
            event_terminal=event_terminal,
            event_direction=event_direction,
        )

    def _confirm_heavy_request(self, solver_inputs: _SolverInputs) -> bool:
        """Warn before launching unusually dense standard-equation requests."""
        if self.equation_type == "pde_3d":
            advisory = assess_pde_3d_request(
                nx=solver_inputs.n_points,
                ny=solver_inputs.n_points_y or solver_inputs.n_points,
                nz=solver_inputs.n_points_z or solver_inputs.n_points,
            )
            return confirm_performance_advisory(self.win, advisory)
        if self.is_pde:
            equation_type = "pde"
            state_size = self.vector_components if self.equation_type == "vector_pde" else 1
        elif self.is_vector:
            equation_type = "vector_ode"
            state_size = self.order * self.vector_components
        elif self.equation_type == "difference":
            equation_type = "difference"
            state_size = self.order
        else:
            equation_type = self.equation_type
            state_size = self.order

        advisory = assess_parameters_dialog_request(
            equation_type=equation_type,
            n_points=solver_inputs.n_points,
            state_size=state_size,
            n_points_y=solver_inputs.n_points_y,
        )
        return confirm_performance_advisory(self.win, advisory)

    def _on_solve(self) -> None:
        """Parse inputs, run the solver pipeline, and open the result dialog."""
        session = getattr(self, "session", None)
        selection = getattr(self, "selection", None)
        if session is not None and selection is not None:
            session.save_configuration(self.capture_form_state(), selection)
        try:
            solver_inputs = self._collect_solver_inputs()
        except _InputValidationError as exc:
            messagebox.showerror(exc.title, exc.message, parent=self.win)
            return

        if not self._confirm_heavy_request(solver_inputs):
            return

        pipeline_kwargs: dict[str, Any] = {
            "expression": self.expression,
            "function_name": self.function_name,
            "order": self.order,
            "parameters": solver_inputs.parameters,
            "equation_name": self.equation_name,
            "x_min": solver_inputs.x_min,
            "x_max": solver_inputs.x_max,
            "y0": solver_inputs.y0,
            "n_points": solver_inputs.n_points,
            "method": solver_inputs.method,
            "selected_stats": solver_inputs.selected_stats,
            "x0_list": solver_inputs.x0_list,
            "equation_type": cast(EquationType, self.equation_type),
            "variables": self.variables,
            "y_min": solver_inputs.y_min,
            "y_max": solver_inputs.y_max,
            "n_points_y": solver_inputs.n_points_y,
            "z_min": solver_inputs.z_min,
            "z_max": solver_inputs.z_max,
            "n_points_z": solver_inputs.n_points_z,
            "vector_expressions": self.vector_expressions,
            "vector_components": self.vector_components,
            "pde_operator": self.pde_operator,
            "component_orders": self.component_orders,
            "bc_expressions": solver_inputs.bc_expressions,
            "bc_types": solver_inputs.bc_types,
            "mask_expression": solver_inputs.mask_expression,
            "contour_bc_expression": solver_inputs.contour_bc_expression,
            "contour_bc_type": solver_inputs.contour_bc_type,
            "event_expression": solver_inputs.event_expression,
            "event_terminal": solver_inputs.event_terminal,
            "event_direction": solver_inputs.event_direction,
        }
        if session is not None:
            session.latest_pipeline_inputs = deepcopy(pipeline_kwargs)
        parent = self.parent

        self.win.destroy()

        # Release every Tk-backed value before the worker begins.  The task below
        # closes over only pipeline data, so a solver-thread allocation cannot
        # become the last reference to this dialog or one of its Tk variables.
        self._release_tk_state()

        def _run_solver_pipeline() -> Any:
            from pipeline import run_solver_pipeline

            return run_solver_pipeline(**pipeline_kwargs)

        def _on_success(result: Any) -> None:
            if not parent.winfo_exists():
                return

            from frontend.ui_dialogs.result_dialog import ResultDialog

            ResultDialog(parent, result=result)

        run_task_with_loading(
            parent=self.parent,
            message="Solving equation...",
            task=_run_solver_pipeline,
            on_success=_on_success,
            format_error=_format_solver_exception,
        )
