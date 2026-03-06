"""Shared documentation metadata for complex-problem UIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProblemDoc:
    """Structured help content for a complex problem."""

    problem_type: str
    extended_description: str
    equation_summary: str | None
    config_options_summary: tuple[str, ...]
    visualizations_summary: tuple[str, ...]


_DOCS: dict[str, ProblemDoc] = {
    "coupled_oscillators": ProblemDoc(
        problem_type="Lattice dynamics (coupled ODE system)",
        extended_description=(
            "One-dimensional chain of coupled oscillators with optional long-range, "
            "nonlinear, and driven interactions. Useful for normal modes, energy transfer, "
            "and FPUT-like dynamics."
        ),
        equation_summary=(
            "mᵢẍᵢ = kᵢ⁺(xᵢ₊₁−xᵢ) − kᵢ⁻(xᵢ−xᵢ₋₁) + Σ₍d₌₂..₄₎ k_d(xᵢ₊d+xᵢ₋d−2xᵢ) + "
            "α[(xᵢ₊₁−xᵢ)²−(xᵢ−xᵢ₋₁)²] + ε₃[(xᵢ₊₁−xᵢ)³−(xᵢ−xᵢ₋₁)³] + "
            "ε₄ sign(Lᵢ)|Lᵢ|⁴ + ε₅Lᵢ⁵ + Fcos(Ωt)"
        ),
        config_options_summary=(
            "Set N oscillators, mass and nearest-neighbor coupling as "
            "constant/list/function of index.",
            "Choose boundary condition: fixed ends or periodic ring.",
            "Select optional terms: 2nd/3rd/4th neighbors, FPUT-α, "
            "cubic/quartic/quintic, external forcing.",
            "Choose initial conditions in oscillator space (xᵢ, vᵢ) or mode space (qᵢ, q̇ᵢ).",
            "Configure integration interval, output resolution, and ODE solver.",
        ),
        visualizations_summary=(
            "Energy evolution (kinetic, potential, total).",
            "Energy per mode/oscillator and modal decomposition.",
            "Time animation of oscillators or mode amplitudes.",
            "Space-time heatmap and 3D surface views.",
        ),
    ),
    "membrane_2d": ProblemDoc(
        problem_type="Discrete 2D lattice membrane",
        extended_description=(
            "Two-dimensional membrane modeled as a grid of coupled oscillators with optional "
            "nonlinear corrections on the discrete Laplacian."
        ),
        equation_summary="m ü = kΔu + α(Δu)² + β(Δu)³ + cₚ sign(Δu)|Δu|ᵖ",
        config_options_summary=(
            "Set grid size (Nₓ, Nᵧ), boundary condition, integrator, and physical coefficients.",
            "Linear term kΔu is always active; optional nonlinear terms can "
            "be enabled independently.",
            "Choose initial displacement profile: gaussian, mode, random, or custom u₀(x,y).",
            "Configure time window and time step.",
        ),
        visualizations_summary=(
            "Animated displacement/velocity fields.",
            "Center-line space-time contour map.",
            "Final 3D surface shape.",
            "Energy evolution and final 2D FFT power spectrum.",
        ),
    ),
    "nonlinear_waves": ProblemDoc(
        problem_type="Periodic pseudo-spectral PDE propagation",
        extended_description=(
            "Simulation of nonlinear wave propagation with periodic boundary conditions for "
            "NLSE (complex envelope) and KdV (real nonlinear dispersive waves)."
        ),
        equation_summary=(
            "NLSE: i∂ψ/∂t = −(β₂/2)∂²ψ/∂x² + γ|ψ|²ψ    |    "
            "KdV: ∂u/∂t + c∂u/∂x + αu∂u/∂x + β∂³u/∂x³ = 0"
        ),
        config_options_summary=(
            "Choose model (NLSE or KdV), spatial domain, Nₓ, and time stepping.",
            "Pick initial profile type (sech, gaussian, pulse, custom u₀(x)).",
            "Tune model-specific coefficients: β₂/γ/phase slope for NLSE, c/α/β for KdV.",
        ),
        visualizations_summary=(
            "Profile animation (intensity/real/imag for NLSE, field for KdV).",
            "Space-time map and NLSE phase map.",
            "Final spectrum in k-space.",
            "Invariant evolution (norm/momentum/Hamiltonian or mass/L²/Hamiltonian).",
        ),
    ),
    "schrodinger_td": ProblemDoc(
        problem_type="Time-dependent quantum wave dynamics",
        extended_description=(
            "Split-operator spectral integration of the time-dependent Schrödinger equation "
            "in 1D or 2D with configurable potentials and wave-packet initial conditions."
        ),
        equation_summary="iħ∂ψ/∂t = −(ħ²/2m)∇²ψ + Vψ",
        config_options_summary=(
            "Select 1D or 2D domain, grid sizes, time step, and boundary handling.",
            "Choose potential family (free/harmonic/well/barrier/double-well/lattice/custom).",
            "Set packet type and packet parameters (gaussian/superposition/custom amplitude).",
            "Configure absorbing-layer parameters when using absorbing boundaries.",
        ),
        visualizations_summary=(
            "1D line animation or 2D density/phase animation.",
            "Space-time density maps.",
            "Momentum-space spectrum (1D or 2D).",
            "Expectation/invariant curves and potential/final surface views.",
        ),
    ),
    "antenna_radiation": ProblemDoc(
        problem_type="Electromagnetic far-field pattern analysis",
        extended_description=(
            "Compute angular radiation patterns, directivity/gain metrics, and RMS electric "
            "field magnitudes for dipole, loop, patch, and linear-array configurations."
        ),
        equation_summary="G(θ,φ) = η·D(θ,φ),   Eᵣₘₛ ∝ √(S·η₀)",
        config_options_summary=(
            "Choose antenna family and geometry parameters.",
            "Set frequency, transmit power, efficiency, and observation distance.",
            "Configure angular sampling (N_θ, N_φ) and array steering/phase when applicable.",
        ),
        visualizations_summary=(
            "2D angular gain map.",
            "Polar theta cut and phi cut.",
            "Normalized 3D radiation pattern.",
            "RMS electric-field angular map and key summary metrics.",
        ),
    ),
    "aerodynamics_2d": ProblemDoc(
        problem_type="Incompressible 2D flow around obstacles",
        extended_description=(
            "Pseudo-spectral/projection simulation of incompressible flow around immersed "
            "bodies with nonlinear Navier-Stokes or Stokes-limit approximation."
        ),
        equation_summary="∂u/∂t + (u·∇)u = −(1/ρ)∇p + ν∇²u + fₚₑₙ,   ∇·u = 0",
        config_options_summary=(
            "Set numerical domain (Nₓ, Nᵧ, Lₓ, Lᵧ), Δt, and sample rate.",
            "Select approximation, obstacle shape, and geometric parameters.",
            "Configure fluid properties (ρ, ν), inflow speed U∞, and penalization strength.",
        ),
        visualizations_summary=(
            "Animation of speed/vorticity/pressure fields.",
            "Final speed contour map.",
            "Time series of drag and lift coefficients.",
            "Streamlines and centerline diagnostic profiles.",
        ),
    ),
    "pipe_flow": ProblemDoc(
        problem_type="1D steady/transient hydraulic modeling",
        extended_description=(
            "Steady Darcy-Weisbach and transient pressure-wave pipe-flow models with "
            "configurable geometry profile, friction correlation, and forcing conditions."
        ),
        equation_summary=(
            "Steady: dp/dx = −f(ρu²)/(2D)    |    Transient: hyperbolic pressure-velocity system"
        ),
        config_options_summary=(
            "Choose model type (steady/transient), diameter profile, and friction model.",
            "Set pipe length, Nₓ, fluid properties, roughness, and geometry parameters.",
            "Configure pressure boundary conditions and transient "
            "forcing/damping/time-step controls.",
            "Optionally define custom diameter law D(x).",
        ),
        visualizations_summary=(
            "Transient field animation (pressure/velocity/Reynolds).",
            "Geometry plots (diameter/area).",
            "Pressure and velocity profile/space-time maps.",
            "Flow-rate and boundary-signal diagnostics.",
        ),
    ),
}


def get_problem_doc(problem_id: str) -> ProblemDoc:
    """Return help content for the given problem id."""
    doc = _DOCS.get(problem_id)
    if doc is None:
        raise KeyError(f"No ProblemDoc registered for '{problem_id}'.")
    return doc


def get_all_problem_docs() -> dict[str, ProblemDoc]:
    """Return all problem-doc entries."""
    return dict(_DOCS)
