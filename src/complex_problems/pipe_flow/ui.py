"""UI dialog for configuring pipe-flow simulations."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import cast

from complex_problems.common import (
    add_how_to_config_section,
    compile_scalar_expression,
    parse_float,
    parse_positive_float,
    parse_positive_int,
)
from complex_problems.common.dialog_ui import (
    make_labeled_combo,
    make_labeled_entry,
    make_labeled_spinbox,
    run_solver_dialog,
)
from complex_problems.pipe_flow.solver import solve_pipe_flow
from config import get_env_from_schema
from frontend.performance_guard import assess_pipe_flow_request, confirm_performance_advisory
from frontend.theme import get_font
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal

_MODELS = ("steady", "transient")
_PROFILES = ("constant", "converging", "diverging", "sinusoidal", "custom")
_FRICTION = ("auto", "laminar", "blasius", "swamee_jain")


class PipeFlowDialog:
    """Configuration dialog for steady and transient pipe flow."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Pipe Flow")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))
        self._build_ui()
        fit_and_center(self.win, min_width=1080, min_height=820, padding=32, resizable=True)
        self.win.minsize(940, 720)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        root = ttk.Frame(self.win, padding=pad * 2)
        root.pack(fill=tk.BOTH, expand=True)
        scroll = ScrollableFrame(root)
        scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        scroll.pack(fill=tk.BOTH, expand=True)
        body = scroll.inner
        body.configure(padding=pad)

        ttk.Label(body, text="Pipe Flow", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            body,
            text=(
                "Steady Darcy-Weisbach model and transient pressure-wave model.\n"
                "Define geometry profile, friction correlation, and pressure conditions."
            ),
            style="Small.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, pad))

        add_how_to_config_section(
            body,
            scroll,
            problem_id="pipe_flow",
            pad=pad,
            wraplength=820,
        )

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._model_var = tk.StringVar(value="steady")
        self._profile_var = tk.StringVar(value="constant")
        self._friction_var = tk.StringVar(value="auto")
        model_combo = make_labeled_combo(row, "Model", self._model_var, _MODELS, width=10)
        profile_combo = make_labeled_combo(
            row,
            "Profile",
            self._profile_var,
            _PROFILES,
            width=12,
        )
        make_labeled_combo(row, "Friction", self._friction_var, _FRICTION, width=12)
        model_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())
        profile_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._length_var = tk.StringVar(value="20.0")
        self._nx_var = tk.StringVar(value="256")
        make_labeled_entry(row, "Length L", self._length_var, width=10)
        make_labeled_spinbox(row, "Nₓ", self._nx_var, from_=16, to=32768, width=8)

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Geometry", style="Small.TLabel").pack(anchor=tk.W)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._d_in_var = tk.StringVar(value="0.08")
        self._d_out_var = tk.StringVar(value="0.05")
        self._d0_var = tk.StringVar(value="0.06")
        make_labeled_entry(row, "dᵢₙ (m)", self._d_in_var, width=8)
        make_labeled_entry(row, "dₒᵤₜ (m)", self._d_out_var, width=8)
        make_labeled_entry(row, "d₀ (m)", self._d0_var, width=8)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._amp_var = tk.StringVar(value="0.20")
        self._waves_var = tk.StringVar(value="2.0")
        make_labeled_entry(row, "Sin amplitude", self._amp_var, width=10)
        make_labeled_entry(row, "Sin waves", self._waves_var, width=8)

        self._custom_row = ttk.Frame(body)
        self._custom_row.pack(fill=tk.X, pady=pad // 2)
        ttk.Label(self._custom_row, text="D(x) =").pack(side=tk.LEFT, padx=(0, 4))
        self._custom_expr_var = tk.StringVar(value="0.06 + 0.005*sin(2*pi*x/20)")
        self._custom_entry = ttk.Entry(
            self._custom_row,
            textvariable=self._custom_expr_var,
            width=56,
            font=get_font(),
        )
        self._custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(self._custom_entry, "Custom diameter expression in meters; variable x.")

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Fluid", style="Small.TLabel").pack(anchor=tk.W)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._rho_var = tk.StringVar(value="1000")
        self._mu_var = tk.StringVar(value="0.001")
        self._rough_var = tk.StringVar(value="1e-5")
        make_labeled_entry(row, "ρ (kg/m³)", self._rho_var, width=10)
        make_labeled_entry(row, "μ (Pa·s)", self._mu_var, width=10)
        make_labeled_entry(row, "Roughness (m)", self._rough_var, width=10)

        self._steady_frame = ttk.Frame(body)
        self._steady_frame.pack(fill=tk.X, pady=pad)
        ttk.Label(
            self._steady_frame,
            text="Steady pressure BC",
            style="Small.TLabel",
        ).pack(anchor=tk.W)
        row = ttk.Frame(self._steady_frame)
        row.pack(fill=tk.X, pady=pad // 2)
        self._p_in_var = tk.StringVar(value="200000")
        self._p_out_var = tk.StringVar(value="190000")
        make_labeled_entry(row, "pᵢₙ (Pa)", self._p_in_var, width=10)
        make_labeled_entry(row, "pₒᵤₜ (Pa)", self._p_out_var, width=10)

        self._transient_frame = ttk.Frame(body)
        self._transient_frame.pack(fill=tk.X, pady=pad)
        ttk.Label(self._transient_frame, text="Transient settings", style="Small.TLabel").pack(
            anchor=tk.W
        )

        row = ttk.Frame(self._transient_frame)
        row.pack(fill=tk.X, pady=pad // 2)
        self._p_base_var = tk.StringVar(value="200000")
        self._p_amp_var = tk.StringVar(value="2000")
        self._p_freq_var = tk.StringVar(value="2.0")
        self._wave_speed_var = tk.StringVar(value="200")
        make_labeled_entry(row, "p_base (Pa)", self._p_base_var, width=10)
        make_labeled_entry(row, "p_amp (Pa)", self._p_amp_var, width=10)
        make_labeled_entry(row, "p_freq (Hz)", self._p_freq_var, width=9)
        make_labeled_entry(row, "Wave c (m/s)", self._wave_speed_var, width=10)

        row = ttk.Frame(self._transient_frame)
        row.pack(fill=tk.X, pady=pad // 2)
        self._damping_var = tk.StringVar(value="0.2")
        self._t_max_var = tk.StringVar(value="1.0")
        self._dt_var = tk.StringVar(value="0.0005")
        self._sample_every_var = tk.StringVar(value="10")
        make_labeled_entry(row, "Damping", self._damping_var, width=8)
        make_labeled_entry(row, "tₘₐₓ", self._t_max_var, width=8)
        make_labeled_entry(row, "Δt", self._dt_var, width=8)
        make_labeled_entry(row, "sample_every", self._sample_every_var, width=10)

        self._btn_row = ttk.Frame(body)
        self._btn_row.pack(fill=tk.X, pady=(pad * 2, 0))
        ttk.Button(self._btn_row, text="Solve", command=self._on_solve).pack(
            side=tk.LEFT, padx=(0, pad)
        )
        ttk.Button(
            self._btn_row,
            text="Close",
            style="Cancel.TButton",
            command=self.win.destroy,
        ).pack(side=tk.LEFT)

        self._update_visibility()

        scroll.bind_new_children()

    def _update_visibility(self) -> None:
        if self._profile_var.get() == "custom":
            self._custom_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._custom_row.pack_forget()

        if self._model_var.get() == "steady":
            self._steady_frame.pack(fill=tk.X, pady=8, before=self._btn_row)
            self._transient_frame.pack_forget()
        else:
            self._transient_frame.pack(fill=tk.X, pady=8, before=self._btn_row)
            self._steady_frame.pack_forget()

    def _collect_inputs(self) -> dict[str, object]:
        model_type = self._model_var.get()
        profile = self._profile_var.get()

        length = parse_positive_float(self._length_var.get(), name="Length L")
        nx = parse_positive_int(self._nx_var.get(), name="Nₓ", min_value=16)

        d_in = parse_positive_float(self._d_in_var.get(), name="dᵢₙ")
        d_out = parse_positive_float(self._d_out_var.get(), name="dₒᵤₜ")
        d0 = parse_positive_float(self._d0_var.get(), name="d₀")
        profile_amplitude = parse_float(self._amp_var.get(), name="Sin amplitude")
        profile_waves = parse_positive_float(self._waves_var.get(), name="Sin waves")

        custom_fn: Callable[[float], float] | None = None
        if profile == "custom":
            expr_fn = compile_scalar_expression(self._custom_expr_var.get(), variables=("x",))

            def _custom_fn(x: float, _fn: Callable[..., float] = expr_fn) -> float:
                return _fn(x=float(x))

            custom_fn = _custom_fn

        rho = parse_positive_float(self._rho_var.get(), name="ρ")
        mu = parse_positive_float(self._mu_var.get(), name="μ")
        roughness = parse_float(self._rough_var.get(), name="Roughness")
        if roughness < 0:
            raise ValueError("Roughness must be non-negative.")

        params: dict[str, object] = {
            "model_type": model_type,
            "length": length,
            "nx": nx,
            "profile": profile,
            "d_in": d_in,
            "d_out": d_out,
            "d0": d0,
            "profile_amplitude": profile_amplitude,
            "profile_waves": profile_waves,
            "custom_diameter_fn": custom_fn,
            "rho": rho,
            "mu": mu,
            "roughness": roughness,
            "friction_model": self._friction_var.get(),
            "p_out": parse_float(self._p_out_var.get(), name="pₒᵤₜ"),
        }

        if model_type == "steady":
            params["p_in"] = parse_float(self._p_in_var.get(), name="pᵢₙ")
        else:
            params["p_base"] = parse_float(self._p_base_var.get(), name="p_base")
            params["p_amp"] = parse_float(self._p_amp_var.get(), name="p_amp")
            params["p_freq_hz"] = parse_positive_float(self._p_freq_var.get(), name="p_freq")
            params["wave_speed"] = parse_positive_float(self._wave_speed_var.get(), name="Wave c")
            damping = parse_float(self._damping_var.get(), name="Damping")
            if damping < 0:
                raise ValueError("Damping must be non-negative.")
            params["damping"] = damping
            params["t_max"] = parse_positive_float(self._t_max_var.get(), name="tₘₐₓ")
            params["dt"] = parse_positive_float(self._dt_var.get(), name="Δt")
            params["sample_every"] = parse_positive_int(
                self._sample_every_var.get(),
                name="sample_every",
                min_value=1,
            )
        return params

    def _confirm_heavy_request(self, params: dict[str, object], window: tk.Toplevel) -> bool:
        """Warn before launching unusually large pipe-flow runs."""
        nx = cast(int, params["nx"])
        t_max = cast(float | None, params.get("t_max"))
        dt = cast(float | None, params.get("dt"))
        sample_every = cast(int | None, params.get("sample_every"))
        advisory = assess_pipe_flow_request(
            model_type=str(params["model_type"]),
            nx=nx,
            t_max=t_max,
            dt=dt,
            sample_every=sample_every,
        )
        return confirm_performance_advisory(window, advisory)

    def _on_solve(self) -> None:
        from complex_problems.pipe_flow.result_dialog import PipeFlowResultDialog

        run_solver_dialog(
            parent=self.parent,
            window=self.win,
            collect_inputs=self._collect_inputs,
            solver=solve_pipe_flow,
            message="Solving pipe flow...",
            result_parent=self.parent,
            result_dialog_factory=PipeFlowResultDialog,
            confirm_run=self._confirm_heavy_request,
        )
