# Advanced Problems Guide

`Advanced Problems` is a ten-plugin subsystem where each problem contributes a
specialized UI, solver, and result dialog.

## Gravitational N-Body Dynamics

This plugin solves classical Newtonian point-mass gravity in 2D or 3D using
`scipy.integrate.solve_ivp`. Coordinates, time, mass, and `G` are a self-consistent
user-selected unit system; built-in three-body presets use normalized `G=1` units.
The force is `G m_j (r_j-r_i)/(r_ij²+epsilon²)^(3/2)` and its potential is
`-G m_i m_j/sqrt(r_ij²+epsilon²)`. Thus `epsilon > 0` is a Plummer-softened model,
not exact point gravity. Coincident initial bodies are rejected for `epsilon=0`.

Choose **Three-body** for the Figure-eight equal-mass benchmark (period about 6.33),
the analytic Lagrange equilateral orbit, the close-encounter Pythagorean 3-4-5 state,
or custom values. The Figure-eight benchmark is associated with Alain Chenciner and
Richard Montgomery, *A remarkable periodic solution of the three-body problem in the
case of equal masses*, Annals of Mathematics 152 (2000), 881--901.

Choose **General N-body** for 2--100 bodies, editable copy-friendly mass and state
rows, a deterministic seeded random bound cluster, or a rotating ring built from its
actual discrete radial acceleration. Pair-force work scales quadratically per RHS
evaluation, so the dialog advises on expensive requests. The result notebook includes
stable-bounds Orbit Animation (with inertial/COM frames and MP4 export), static
trajectories, phase space, energy/conservation diagnostics, and separations.

## Fermi-Pasta-Ulam-Tsingou Experiment

This dedicated fixed-end normalized chain supports the cubic alpha potential
`V(delta) = delta^2/2 + alpha delta^3/3` and the quartic beta potential
`V(delta) = delta^2/2 + beta delta^4/4`. Its exact Hamiltonian includes both
boundary bonds. Linear sine-mode energies are diagnostic quantities, not the complete
nonlinear Hamiltonian. Velocity Verlet is the recommended long-time integrator; RK4 is
available to reproduce the historical numerical method. The notebook provides recurrence
fidelity, modal-energy/thermalization diagnostics, strain space-time and surface views,
phase space, and sequential alpha recurrence-scaling studies. Localized strain is shown
as a solitary-wave-like structure rather than asserted to be an exact soliton; use the
separate Nonlinear Waves plugin for the KdV PDE solver.

## Plugin Contract

Each plugin package follows this shape:

```text
src/complex_problems/<plugin_id>/
  __init__.py
  problem.py
  ui.py
  solver.py
  result_dialog.py
  model.py
```

`model.py` is optional in principle, but every currently registered plugin has
one. `problem.py` exposes `PROBLEM`, which provides a `ProblemDescriptor` and an
`open_dialog(parent)` method.

Plugins are registered lazily in `src/complex_problems/problem_registry.py`.

## Available Modules

| ID | Name | Current scope | Typical outputs |
|---|---|---|---|
| `coupled_oscillators` | Coupled Harmonic Oscillators | One-dimensional oscillator chains with configurable masses, coupling constants, boundaries, nonlinear terms, and forcing | energy evolution, modal energy, oscillator/mode animation, heatmaps |
| `membrane_2d` | 2D Nonlinear Membrane | Discrete 2D membrane lattice with optional nonlinear terms and spectral diagnostics | displacement/velocity animation, energy drift, 3D surface, 2D FFT |
| `nonlinear_waves` | Nonlinear Waves (NLSE + KdV) | Periodic pseudo-spectral propagation for NLSE and KdV, including coefficient-derived one-soliton and separated-train KdV initial states | profile animation, space-time maps, phase/spectrum, invariant curves |
| `schrodinger_td` | Schrodinger TD (1D/2D) | Split-operator spectral time-dependent Schrodinger solver with configurable potentials and packet states | density/phase animation, momentum spectrum, expectations, invariant curves |
| `antenna_radiation` | Antenna Radiation | Far-field patterns for dipole, loop, patch-like aperture, and uniform linear array models | gain/directivity maps, polar cuts, 3D pattern, field metrics |
| `aerodynamics_2d` | Aerodynamics 2D | 2D incompressible flow around obstacles using projection/Stokes-style approximations | speed/vorticity/pressure views, drag/lift curves, streamlines |
| `aerodynamics_3d` | Aerodynamics 3D | Incompressible 3D structured-grid flow in a periodic Cartesian domain with immersed/penalized obstacles; lightweight educational/scientific scope, not industrial CFD | velocity-perturbation/velocity/vorticity vector animation, opt-in cached 3D streamline animation, animated XY/XZ/YZ slices, Cd/Cl/Cs and diagnostics |
| `pipe_flow` | Pipe Flow | Steady Darcy-Weisbach and transient 1D pressure-wave pipe-flow models | pressure/velocity profiles, geometry plots, Reynolds/friction metrics, transient maps |
| `gravitational_n_body` | Gravitational N-Body Dynamics | Softened Newtonian multi-body systems and curated orbit studies | orbit animation, COM diagnostics, energy and angular-momentum drift |
| `fput_experiment` | Fermi-Pasta-Ulam-Tsingou Experiment | Fixed-end alpha/beta FPUT chains | recurrence, modal energy, strain, Hamiltonian drift, scaling fits |

The Aerodynamics 3D v1 configuration requires the complete obstacle to remain inside the
Cartesian periodic box; configurations whose geometry crosses an x, y, or z boundary are
rejected. Results retain only `u`, `v`, `w`, and pressure volumetric histories. Derived slice
histories and streamline polylines are prepared lazily in the Results window and are not added
to the solver result object.

## General Usage Pattern

1. Open `Advanced Problems` from the main menu.
2. Pick a module from the left list.
3. Review the right-side details panel.
4. Open the selected module.
5. Expand `How to configure` if you need parameter guidance.
6. Configure model-specific parameters.
7. Run the solve.
8. Explore module-specific tabs and diagnostics.

## Shared UI Behavior

Common helpers under `src/complex_problems/common/` provide:

- background solver execution with loading dialogs
- safe expression handling for user formulas
- numeric input parsing and validation
- reusable problem documentation panels
- reusable result-dialog cleanup and animation reset helpers

Static explanatory text is shown as labels where possible. Copy-friendly text
areas remain only where users intentionally need to copy or paste expressions.

## Numerical Quality Checks

Most modules expose model-specific diagnostics in results:

- conservation or drift metrics (`energy_drift_rel`, `norm_drift_rel`, etc.)
- stability indicators such as maximum amplitude or divergence
- physical summary metrics such as gain, directivity, drag, lift, Reynolds number,
  or friction factor

Use those diagnostics before trusting conclusions from aggressive settings.

For KdV one-soliton and separated-train results, the profile animation always shows
the numerical solution `u`. Optional dashed overlays are fixed-amplitude,
fixed-width soliton profiles whose centers are tracked from numerical peaks whenever
individual pulses are distinguishable. During strong overlap, identities are ambiguous,
so their displayed trajectories bridge reliable pre- and post-interaction observations;
this makes interaction-induced phase displacement visible without claiming a unique
instantaneous decomposition. The optional diagnostic residual is
`u - Σ(tracked soliton profiles)`; it is not a unique nonlinear decomposition.

## Performance Guidelines

- Start with defaults and validate a low-cost run first.
- Increase grid resolution only after the model behaves as expected.
- Keep time steps conservative for nonlinear or wave-dominated models.
- For 2D models, tune both spatial resolution and output sampling cadence.
- For Aerodynamics 3D, `nx`, `ny`, `nz`, and the saved-frame cadence multiply retained volumetric history; build its streamline cache only when needed.
- Large animations, MP4 export, and high-resolution fields can become memory-heavy.

## Extending with New Plugins

1. Create a package under `src/complex_problems/<plugin_id>/`.
2. Implement `problem.py`, `ui.py`, `solver.py`, and `result_dialog.py`.
3. Add `model.py` when physical/math helper logic is more than trivial.
4. Register the plugin in `src/complex_problems/problem_registry.py`.
5. Add solver and registry tests.
6. Update this guide and `docs/api/complex_problems.rst`.

Detailed implementation guidance is in [Developer Guide](developer-guide.md).
