"""UI dialog for nonlinear wave simulations (NLSE / KdV)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from complex_problems.common import (
    add_how_to_config_section,
    compile_scalar_expression,
    parse_float,
    parse_positive_float,
    parse_positive_int,
    run_solver_with_loading,
)
from complex_problems.nonlinear_waves.solver import solve_nonlinear_waves
from config import get_env_from_schema
from frontend.theme import get_font
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal

_MODELS = ("nlse", "kdv")
_PROFILES = ("sech", "gaussian", "pulse", "custom")


class NonlinearWavesDialog:
    """Configuration dialog for nonlinear wave propagation models."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Nonlinear Waves (NLSE + KdV)")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))
        self._build_ui()
        fit_and_center(self.win, min_width=980, min_height=760, padding=32, resizable=True)
        self.win.minsize(920, 700)
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

        ttk.Label(body, text="Nonlinear Waves", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            body,
            text=(
                "Choose NLSE (complex envelope) or KdV (real nonlinear dispersive wave).\n"
                "Both use periodic pseudo-spectral solvers."
            ),
            style="Small.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, pad))

        add_how_to_config_section(
            body,
            scroll,
            problem_id="nonlinear_waves",
            pad=pad,
            wraplength=780,
        )

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._model_var = tk.StringVar(value="nlse")
        model_combo = self._make_combo(row, "Model", self._model_var, _MODELS, width=10)
        model_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_model_visibility())
        ToolTip(model_combo, "NLSE: split-step Fourier. KdV: pseudo-spectral ETDRK4.")

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._x_min_var = tk.StringVar(value="-20.0")
        self._x_max_var = tk.StringVar(value="20.0")
        self._nx_var = tk.StringVar(value="512")
        self._make_entry(row, "xₘᵢₙ", self._x_min_var, width=10)
        self._make_entry(row, "xₘₐₓ", self._x_max_var, width=10)
        self._make_spinbox(row, "Nₓ", self._nx_var, from_=64, to=32768, width=9)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._t_min_var = tk.StringVar(value="0.0")
        self._t_max_var = tk.StringVar(value="8.0")
        self._dt_var = tk.StringVar(value="0.002")
        self._make_entry(row, "tₘᵢₙ", self._t_min_var, width=10)
        self._make_entry(row, "tₘₐₓ", self._t_max_var, width=10)
        self._make_entry(row, "Δt", self._dt_var, width=10)

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Initial profile", style="Small.TLabel").pack(anchor=tk.W)

        self._profile_row = ttk.Frame(body)
        self._profile_row.pack(fill=tk.X, pady=pad // 2)
        self._profile_var = tk.StringVar(value="sech")
        profile_combo = self._make_combo(
            self._profile_row, "Profile", self._profile_var, _PROFILES, width=12
        )
        profile_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_profile_visibility())

        self._profile_params_row = ttk.Frame(body)
        self._profile_params_row.pack(fill=tk.X, pady=pad // 2)
        self._amp_var = tk.StringVar(value="1.0")
        self._sigma_var = tk.StringVar(value="1.0")
        self._center_var = tk.StringVar(value="0.0")
        self._make_entry(self._profile_params_row, "Amplitude", self._amp_var, width=8)
        self._make_entry(self._profile_params_row, "σ", self._sigma_var, width=8)
        self._make_entry(self._profile_params_row, "Center x₀", self._center_var, width=9)

        self._custom_row = ttk.Frame(body)
        self._custom_row.pack(fill=tk.X, pady=pad // 2)
        ttk.Label(self._custom_row, text="u₀(x) =").pack(side=tk.LEFT, padx=(0, 4))
        self._custom_expr_var = tk.StringVar(value="exp(-x**2)")
        self._custom_entry = ttk.Entry(
            self._custom_row,
            textvariable=self._custom_expr_var,
            width=52,
            font=get_font(),
        )
        self._custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(self._custom_entry, "Custom expression in x.")

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Model parameters", style="Small.TLabel").pack(anchor=tk.W)

        self._nlse_row = ttk.Frame(body)
        self._nlse_row.pack(fill=tk.X, pady=pad // 2)
        self._beta2_var = tk.StringVar(value="1.0")
        self._gamma_var = tk.StringVar(value="1.0")
        self._phase_k_var = tk.StringVar(value="0.0")
        self._make_entry(self._nlse_row, "β₂", self._beta2_var, width=8)
        self._make_entry(self._nlse_row, "γ", self._gamma_var, width=8)
        self._make_entry(self._nlse_row, "Phase k₀", self._phase_k_var, width=9)

        self._kdv_row = ttk.Frame(body)
        self._kdv_row.pack(fill=tk.X, pady=pad // 2)
        self._c_var = tk.StringVar(value="0.0")
        self._alpha_var = tk.StringVar(value="6.0")
        self._beta_disp_var = tk.StringVar(value="1.0")
        self._make_entry(self._kdv_row, "c", self._c_var, width=8)
        self._make_entry(self._kdv_row, "α", self._alpha_var, width=8)
        self._make_entry(self._kdv_row, "β", self._beta_disp_var, width=8)

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
        ).pack(
            side=tk.LEFT
        )

        self._update_profile_visibility()
        self._update_model_visibility()
        scroll.bind_new_children()

    def _make_entry(
        self, parent: ttk.Frame, label: str, var: tk.StringVar, *, width: int = 10
    ) -> ttk.Entry:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
        entry = ttk.Entry(parent, textvariable=var, width=width, font=get_font())
        entry.pack(side=tk.LEFT, padx=(0, 12))
        return entry

    def _make_spinbox(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        *,
        from_: int,
        to: int,
        width: int = 10,
    ) -> ttk.Spinbox:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
        spin = ttk.Spinbox(
            parent,
            textvariable=var,
            from_=from_,
            to=to,
            width=width,
            font=get_font(),
        )
        spin.pack(side=tk.LEFT, padx=(0, 12))
        return spin

    def _make_combo(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        values: tuple[str, ...],
        *,
        width: int = 12,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            parent,
            textvariable=var,
            values=list(values),
            state="readonly",
            width=width,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT, padx=(0, 12))
        return combo

    def _update_profile_visibility(self) -> None:
        if self._profile_var.get() == "custom":
            self._custom_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._custom_row.pack_forget()

    def _update_model_visibility(self) -> None:
        if self._model_var.get() == "nlse":
            self._nlse_row.pack(fill=tk.X, pady=4, before=self._btn_row)
            self._kdv_row.pack_forget()
        else:
            self._kdv_row.pack(fill=tk.X, pady=4, before=self._btn_row)
            self._nlse_row.pack_forget()

    def _collect_inputs(self) -> dict[str, object]:
        model = self._model_var.get()
        x_min = parse_float(self._x_min_var.get(), name="xₘᵢₙ")
        x_max = parse_float(self._x_max_var.get(), name="xₘₐₓ")
        nx = parse_positive_int(self._nx_var.get(), name="Nₓ", min_value=64)

        t_min = parse_float(self._t_min_var.get(), name="tₘᵢₙ")
        t_max = parse_float(self._t_max_var.get(), name="tₘₐₓ")
        dt = parse_positive_float(self._dt_var.get(), name="Δt")

        profile = self._profile_var.get()
        amplitude = parse_float(self._amp_var.get(), name="Amplitude")
        sigma = parse_positive_float(self._sigma_var.get(), name="σ")
        center = parse_float(self._center_var.get(), name="Center x₀")

        custom_fn = None
        if profile == "custom":
            custom_fn = compile_scalar_expression(
                self._custom_expr_var.get(),
                variables=("x",),
            )

        params: dict[str, object] = {
            "model_type": model,
            "x_min": x_min,
            "x_max": x_max,
            "nx": nx,
            "t_min": t_min,
            "t_max": t_max,
            "dt": dt,
            "profile": profile,
            "amplitude": amplitude,
            "sigma": sigma,
            "center": center,
            "custom_profile_fn": custom_fn,
        }

        if model == "nlse":
            params["beta2"] = parse_float(self._beta2_var.get(), name="β₂")
            params["gamma"] = parse_float(self._gamma_var.get(), name="γ")
            params["initial_phase_k"] = parse_float(self._phase_k_var.get(), name="Phase k₀")
        else:
            params["c"] = parse_float(self._c_var.get(), name="c")
            params["alpha"] = parse_float(self._alpha_var.get(), name="α")
            params["beta_disp"] = parse_float(self._beta_disp_var.get(), name="β")

        return params

    def _on_solve(self) -> None:
        try:
            params = self._collect_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.win)
            return

        self.win.destroy()

        def _task():
            return solve_nonlinear_waves(**params)

        def _on_success(result) -> None:
            from complex_problems.nonlinear_waves.result_dialog import NonlinearWavesResultDialog

            NonlinearWavesResultDialog(self.parent, result=result)

        run_solver_with_loading(
            parent=self.parent,
            message="Solving nonlinear wave propagation...",
            task=_task,
            on_success=_on_success,
        )
