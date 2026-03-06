# Complex Problems Guide

`Complex Problems` is a plugin subsystem where each problem contributes:

- `problem.py` descriptor and dialog entry point
- `ui.py` configuration dialog
- `solver.py` specialized numerical model
- `result_dialog.py` custom visualization tabs
- optional `model.py` helpers

## Available modules

| ID | Name | Core model | Typical outputs |
|---|---|---|---|
| `coupled_oscillators` | Coupled Harmonic Oscillators | 1D oscillator chain with linear/nonlinear couplings | mode energies, phase-space, recurrence behavior |
| `membrane_2d` | 2D Nonlinear Membrane | discrete 2D lattice membrane | displacement animation, energy drift, 2D FFT |
| `nonlinear_waves` | Nonlinear Waves (NLSE + KdV) | split-step / pseudo-spectral propagation | space-time map, phase/spectrum, invariants |
| `schrodinger_td` | Schrodinger TD (1D/2D) | split-operator spectral TDSE | density/phase evolution, expectation values |
| `antenna_radiation` | Antenna Radiation | far-field analytical radiation patterns | gain/directivity maps, polar cuts, E/H fields |
| `aerodynamics_2d` | Aerodynamics 2D | incompressible flow with projection + obstacle penalization | speed/vorticity maps, Cd/Cl curves |
| `pipe_flow` | Pipe Flow | steady Darcy-Weisbach and transient 1D pressure-wave model | pressure/velocity profiles, Reynolds/friction metrics |

## General usage pattern

1. Open `Complex Problems` from main menu.
2. Pick a module from the left list.
3. Review the right-side details panel (type, description, configurable options, outputs).
4. Open the selected module.
5. Expand `How to configure` (collapsed by default) if needed.
6. Configure model-specific parameters.
7. Run solve (background execution + loading dialog).
8. Explore module-specific tabs and diagnostics.

## UI notation and text behavior

- Mathematical labels use Unicode notation when possible (`xₘᵢₙ`, `tₘₐₓ`, `Nₓ`, ...).
- If a true Unicode subscript is not available, use `base_subscript` with Unicode symbol, e.g. `N_θ`, `N_φ`.
- Static explanatory text is shown as non-editable labels (not copy-oriented text boxes).
- Intentional copy-friendly Unicode helper blocks remain in custom-equation / transform workflows.

## Numerical quality checks

Most modules expose magnitude checks in results:

- conservation drift (`energy_drift_rel`, `norm_drift_rel`, etc.)
- stability indicators (CFL-like constraints, divergence, max amplitude)
- physical metrics (gain/directivity peaks, drag/lift summaries)

Use these before trusting conclusions from very aggressive settings.

## Performance guidelines

- Increase grid resolution only after validating lower-cost runs.
- Keep `dt` conservative for nonlinear or wave-dominated models.
- Prefer default integrators first; compare alternatives after baseline validation.
- For 2D models, tune both spatial resolution and output sampling cadence.

## Extending with new plugins

Follow the package pattern:

```text
src/complex_problems/<plugin_name>/
  __init__.py
  problem.py
  ui.py
  solver.py
  result_dialog.py
  model.py (optional)
```

Then register the plugin in `src/complex_problems/problem_registry.py`.

Detailed implementation guidance is in [Developer Guide](developer-guide.md).
