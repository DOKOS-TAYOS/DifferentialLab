# Complex Problems Guide

`Complex Problems` is a plugin subsystem where each problem contributes a
specialized UI, solver, and result dialog.

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
| `nonlinear_waves` | Nonlinear Waves (NLSE + KdV) | Periodic pseudo-spectral propagation for NLSE and KdV | profile animation, space-time maps, phase/spectrum, invariant curves |
| `schrodinger_td` | Schrodinger TD (1D/2D) | Split-operator spectral time-dependent Schrodinger solver with configurable potentials and packet states | density/phase animation, momentum spectrum, expectations, invariant curves |
| `antenna_radiation` | Antenna Radiation | Far-field patterns for dipole, loop, patch-like aperture, and uniform linear array models | gain/directivity maps, polar cuts, 3D pattern, field metrics |
| `aerodynamics_2d` | Aerodynamics 2D | 2D incompressible flow around obstacles using projection/Stokes-style approximations | speed/vorticity/pressure views, drag/lift curves, streamlines |
| `pipe_flow` | Pipe Flow | Steady Darcy-Weisbach and transient 1D pressure-wave pipe-flow models | pressure/velocity profiles, geometry plots, Reynolds/friction metrics, transient maps |

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

## Performance Guidelines

- Start with defaults and validate a low-cost run first.
- Increase grid resolution only after the model behaves as expected.
- Keep time steps conservative for nonlinear or wave-dominated models.
- For 2D models, tune both spatial resolution and output sampling cadence.
- Large animations and high-resolution fields can become memory-heavy.

## Extending with New Plugins

1. Create a package under `src/complex_problems/<plugin_id>/`.
2. Implement `problem.py`, `ui.py`, `solver.py`, and `result_dialog.py`.
3. Add `model.py` when physical/math helper logic is more than trivial.
4. Register the plugin in `src/complex_problems/problem_registry.py`.
5. Add solver and registry tests.
6. Update this guide and `docs/api/complex_problems.rst`.

Detailed implementation guidance is in [Developer Guide](developer-guide.md).
