"""Scrollable configuration dialog for Gravitational N-Body Dynamics."""
# ruff: noqa: E501

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import cast

import numpy as np

from complex_problems.common import add_how_to_config_section
from complex_problems.common.dialog_ui import run_solver_dialog
from complex_problems.gravitational_n_body.model import (
    NBodyState,
    figure_eight_state,
    lagrange_equilateral_state,
    pythagorean_state,
    random_bound_cluster_state,
    ring_state,
    validate_n_body_state,
)
from complex_problems.gravitational_n_body.result_dialog import GravitationalNBodyResultDialog
from complex_problems.gravitational_n_body.solver import solve_n_body
from config.constants import SOLVER_METHODS
from frontend.performance_guard import PerformanceAdvisory, confirm_performance_advisory
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal


def parse_n_body_state_text(
    masses_text: str, positions_text: str, velocities_text: str, *, n_bodies: int, dimension: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse exact comma/line state text without padding or truncating user data."""
    try:
        masses = np.asarray(
            [float(value.strip()) for value in masses_text.split(",") if value.strip()], dtype=float
        )
    except ValueError as exc:
        raise ValueError("Masses must be comma-separated finite numbers.") from exc
    if masses.size != n_bodies:
        raise ValueError(f"Masses must contain exactly {n_bodies} values.")

    def parse_rows(text: str, label: str) -> np.ndarray:
        rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(rows) != n_bodies:
            raise ValueError(f"{label} must have exactly {n_bodies} non-empty rows.")
        values: list[list[float]] = []
        for index, row in enumerate(rows, start=1):
            parts = [part.strip() for part in row.split(",")]
            if len(parts) != dimension:
                raise ValueError(
                    f"{label} row {index} must contain exactly {dimension} comma-separated values."
                )
            try:
                values.append([float(part) for part in parts])
            except ValueError as exc:
                raise ValueError(f"{label} row {index} contains an invalid number.") from exc
        return np.asarray(values, dtype=float)

    positions, velocities = (
        parse_rows(positions_text, "Positions"),
        parse_rows(velocities_text, "Velocities"),
    )
    return validate_n_body_state(masses, positions, velocities, dimension=dimension)[:3]


def assess_n_body_request(*, n_bodies: int, n_points: int) -> PerformanceAdvisory | None:
    """Advise for expensive O(N²)-per-RHS interactive N-body requests."""
    pair_work = max(1, n_bodies) ** 2 * max(1, n_points)
    if pair_work < 500_000:
        return None
    severity = "confirm" if pair_work >= 5_000_000 else "warn"
    return PerformanceAdvisory(
        severity=severity,
        title="Potentially expensive N-body request",
        message=(
            f"This stores {n_points:,} output samples for {n_bodies} bodies and evaluates "
            f"an O(N²) force calculation at each solver RHS call. Estimated display work: "
            f"{pair_work:,} pair-sample units. Consider a shorter interval or fewer samples."
        ),
    )


class GravitationalNBodyDialog:
    """Configuration UI for curated three-body and general N-body states."""

    _THREE_PRESETS = (
        "Figure-eight equal-mass orbit",
        "Lagrange equilateral orbit",
        "Pythagorean three-body problem",
        "Custom",
    )
    _GENERAL_PRESETS = ("Custom", "Random bound cluster", "Rotating ring")

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Gravitational N-Body Dynamics")
        self._build_ui()
        fit_and_center(self.win, min_width=980, min_height=740, padding=40, resizable=True)
        self.win.minsize(820, 620)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.win, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        self._scroll = ScrollableFrame(outer)
        self._scroll.pack(fill=tk.BOTH, expand=True)
        inner = self._scroll.inner
        ttk.Label(inner, text="Gravitational N-Body Dynamics", style="Title.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            inner,
            text="Classical Newtonian point-mass gravity in self-consistent user-selected units. Positive epsilon uses Plummer softening.",
            style="Small.TLabel",
            wraplength=820,
        ).pack(anchor=tk.W, pady=(4, 10))
        self._mode_var = tk.StringVar(value="Three-body")
        self._dimension_var = tk.StringVar(value="2D")
        self._preset_var = tk.StringVar(value=self._THREE_PRESETS[0])
        row = ttk.Frame(inner)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Mode:").pack(side=tk.LEFT)
        mode = ttk.Combobox(
            row,
            textvariable=self._mode_var,
            values=("Three-body", "General N-body"),
            state="readonly",
            width=18,
        )
        mode.pack(side=tk.LEFT, padx=5)
        mode.bind("<<ComboboxSelected>>", self._on_mode_change)
        ttk.Label(row, text="Dimension:").pack(side=tk.LEFT, padx=(18, 0))
        dimension = ttk.Combobox(
            row, textvariable=self._dimension_var, values=("2D", "3D"), state="readonly", width=7
        )
        dimension.pack(side=tk.LEFT, padx=5)
        dimension.bind("<<ComboboxSelected>>", lambda _event: self._apply_selected_preset())
        ttk.Label(row, text="Preset:").pack(side=tk.LEFT, padx=(18, 0))
        self._preset_combo = ttk.Combobox(
            row,
            textvariable=self._preset_var,
            values=self._THREE_PRESETS,
            state="readonly",
            width=28,
        )
        self._preset_combo.pack(side=tk.LEFT, padx=5)
        self._preset_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_selected_preset()
        )
        add_how_to_config_section(
            inner, self._scroll, problem_id="gravitational_n_body", pad=8, wraplength=820
        )
        physics = ttk.Frame(inner)
        physics.pack(fill=tk.X, pady=4)
        self._g_var, self._epsilon_var = tk.StringVar(value="1"), tk.StringVar(value="0")
        self._add_entry(
            physics,
            "G",
            self._g_var,
            "Positive gravitational constant in the selected consistent unit system.",
        )
        self._add_entry(
            physics,
            "Softening epsilon",
            self._epsilon_var,
            "epsilon > 0 uses Plummer-softened gravity; epsilon = 0 is exact point gravity.",
        )
        self._n_var, self._seed_var = tk.StringVar(value="8"), tk.StringVar(value="1234")
        general = ttk.Frame(inner)
        general.pack(fill=tk.X, pady=4)
        self._general_frame = general
        self._add_entry(general, "N", self._n_var, "General mode supports 2 through 100 bodies.")
        self._add_entry(
            general,
            "Random seed",
            self._seed_var,
            "Same random parameters and seed produce the same cluster.",
        )
        (
            self._position_scale_var,
            self._mass_min_var,
            self._mass_max_var,
            self._virial_var,
            self._ring_radius_var,
        ) = (
            tk.StringVar(value="1"),
            tk.StringVar(value="0.8"),
            tk.StringVar(value="1.2"),
            tk.StringVar(value="1"),
            tk.StringVar(value="1"),
        )
        generator = ttk.Frame(inner)
        generator.pack(fill=tk.X, pady=4)
        self._generator_frame = generator
        self._add_entry(
            generator, "Position scale", self._position_scale_var, "Random-cluster spatial scale."
        )
        self._add_entry(
            generator, "Mass range", self._mass_min_var, "Lower random mass; upper mass is next."
        )
        self._add_entry(generator, "to", self._mass_max_var, "Upper random mass.")
        self._add_entry(
            generator,
            "Target virial 2K/|U|",
            self._virial_var,
            "Initial kinetic-energy target for random clusters.",
        )
        self._add_entry(
            generator,
            "Ring radius",
            self._ring_radius_var,
            "Radius used by the coherent equal-mass ring.",
        )
        state = ttk.LabelFrame(inner, text="Editable state (comma-separated; one body per row)")
        state.pack(fill=tk.BOTH, expand=False, pady=8)
        self._masses_var = tk.StringVar()
        ttk.Label(state, text="Masses:").grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        ttk.Entry(state, textvariable=self._masses_var, width=82).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5
        )
        self._positions_text, self._velocities_text = (
            tk.Text(state, height=7, width=48),
            tk.Text(state, height=7, width=48),
        )
        ttk.Label(state, text="Positions:").grid(row=1, column=0, sticky="nw", padx=5, pady=5)
        self._positions_text.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(state, text="Velocities:").grid(row=2, column=0, sticky="nw", padx=5, pady=5)
        self._velocities_text.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        state.columnconfigure(1, weight=1)
        integration = ttk.Frame(inner)
        integration.pack(fill=tk.X, pady=4)
        self._t_min_var, self._t_max_var, self._n_points_var, self._method_var = (
            tk.StringVar(value="0"),
            tk.StringVar(value="6.33"),
            tk.StringVar(value="600"),
            tk.StringVar(value="DOP853"),
        )
        self._add_entry(integration, "t start", self._t_min_var, "Start time.")
        self._add_entry(integration, "t end", self._t_max_var, "End time.")
        self._add_entry(
            integration, "Output samples", self._n_points_var, "Number of sampled output frames."
        )
        ttk.Label(integration, text="Solver:").pack(side=tk.LEFT, padx=(8, 3))
        ttk.Combobox(
            integration,
            textvariable=self._method_var,
            values=SOLVER_METHODS,
            state="readonly",
            width=9,
        ).pack(side=tk.LEFT)
        buttons = ttk.Frame(inner)
        buttons.pack(fill=tk.X, pady=12)
        ttk.Button(
            buttons, text="Generate / apply preset", command=self._apply_selected_preset
        ).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Solve", command=self._on_solve).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Close", command=self.win.destroy).pack(side=tk.LEFT)
        self._apply_selected_preset()
        self._scroll.bind_new_children()

    def _add_entry(
        self, parent: ttk.Frame, label: str, variable: tk.StringVar, tooltip: str
    ) -> None:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(8, 3))
        entry = ttk.Entry(parent, textvariable=variable, width=10)
        entry.pack(side=tk.LEFT)
        ToolTip(entry, tooltip)

    def _on_mode_change(self, _event: tk.Event | None = None) -> None:
        general = self._mode_var.get() == "General N-body"
        self._preset_combo.configure(
            values=self._GENERAL_PRESETS if general else self._THREE_PRESETS
        )
        self._preset_var.set((self._GENERAL_PRESETS if general else self._THREE_PRESETS)[0])
        self._general_frame.pack_forget() if not general else self._general_frame.pack(
            fill=tk.X, pady=4
        )
        self._generator_frame.pack_forget() if not general else self._generator_frame.pack(
            fill=tk.X, pady=4
        )
        self._apply_selected_preset()

    def _selected_state(self) -> NBodyState | None:
        dimension = 2 if self._dimension_var.get() == "2D" else 3
        preset = self._preset_var.get()
        g, epsilon = float(self._g_var.get()), float(self._epsilon_var.get())
        if preset == "Figure-eight equal-mass orbit":
            return figure_eight_state()
        if preset == "Lagrange equilateral orbit":
            return lagrange_equilateral_state(gravitational_constant=g)
        if preset == "Pythagorean three-body problem":
            return pythagorean_state()
        if preset == "Random bound cluster":
            return random_bound_cluster_state(
                n_bodies=int(self._n_var.get()),
                dimension=dimension,
                seed=int(self._seed_var.get()),
                position_scale=float(self._position_scale_var.get()),
                mass_min=float(self._mass_min_var.get()),
                mass_max=float(self._mass_max_var.get()),
                target_virial_ratio=float(self._virial_var.get()),
                gravitational_constant=g,
                epsilon=epsilon,
            )
        if preset == "Rotating ring":
            return ring_state(
                n_bodies=int(self._n_var.get()),
                dimension=dimension,
                radius=float(self._ring_radius_var.get()),
                gravitational_constant=g,
                epsilon=epsilon,
            )
        return None

    def _apply_selected_preset(self) -> None:
        try:
            state = self._selected_state()
        except ValueError:
            return
        if state is None:
            return
        self._dimension_var.set(f"{state.dimension}D")
        self._n_var.set(str(state.masses.size))
        self._masses_var.set(", ".join(f"{value:.12g}" for value in state.masses))
        self._replace_text(self._positions_text, state.positions)
        self._replace_text(self._velocities_text, state.velocities)
        self._t_max_var.set(f"{state.recommended_t_max:.12g}")
        self._n_points_var.set(str(state.recommended_n_points))
        self._epsilon_var.set(f"{state.recommended_epsilon:.12g}")
        self._method_var.set("DOP853")

    def _replace_text(self, widget: tk.Text, values: np.ndarray) -> None:
        widget.delete("1.0", tk.END)
        widget.insert(
            "1.0", "\n".join(", ".join(f"{value:.12g}" for value in row) for row in values)
        )

    def _collect_inputs(self) -> dict[str, object]:
        dimension = 2 if self._dimension_var.get() == "2D" else 3
        n_bodies = 3 if self._mode_var.get() == "Three-body" else int(self._n_var.get())
        masses, positions, velocities = parse_n_body_state_text(
            self._masses_var.get(),
            self._positions_text.get("1.0", tk.END),
            self._velocities_text.get("1.0", tk.END),
            n_bodies=n_bodies,
            dimension=dimension,
        )
        return {
            "masses": masses,
            "positions": positions,
            "velocities": velocities,
            "dimension": dimension,
            "gravitational_constant": float(self._g_var.get()),
            "epsilon": float(self._epsilon_var.get()),
            "t_min": float(self._t_min_var.get()),
            "t_max": float(self._t_max_var.get()),
            "n_points": int(self._n_points_var.get()),
            "method": self._method_var.get(),
        }

    def _on_solve(self) -> None:
        def confirm(params: dict[str, object], parent: tk.Toplevel) -> bool:
            masses = cast(np.ndarray, params["masses"])
            n_points = cast(int, params["n_points"])
            return confirm_performance_advisory(
                parent,
                assess_n_body_request(n_bodies=len(masses), n_points=n_points),
            )

        run_solver_dialog(
            parent=self.parent,
            window=self.win,
            collect_inputs=self._collect_inputs,
            solver=solve_n_body,
            message="Solving gravitational N-body dynamics…",
            result_parent=self.parent,
            result_dialog_factory=GravitationalNBodyResultDialog,
            confirm_run=confirm,
        )
