# User Guide

This guide covers the normal workflow: choose or define a problem, configure
parameters, solve, inspect results, and export outputs.

## Main Menu

Buttons:

- `Solve Equation`: general equation workflows
- `Function Transform`: scalar function transform workflows
- `Advanced Problems`: specialized complex-problem models with dedicated UI and visualizations
- `Help & About`: in-app help panel
- `Settings`: edit `.env` values through the GUI
- `Exit`: close the application

## Solve Workflow

### 1. Choose an equation type

The standard solver path currently supports:

- ODE
- Difference equation
- PDE
- Vector ODE
- Vector PDE

You can use the predefined YAML catalog or write custom expressions.
The current catalog has 121 entries:

| Type | Entries |
|---|---:|
| ODE | 48 |
| Vector ODE | 50 |
| Difference equation | 10 |
| PDE | 12 |
| Vector PDE | 1 |

The catalog files live in `src/config/equations/`.

### 2. Define the equation

For custom expressions:

- ODE notation: `f[0]`, `f[1]`, `f[2]`, ...
- Vector notation: `f[i,k]`, where `i` is the component and `k` is the derivative order
- Difference notation: `f[0]` for the current term and `n` for the index
- PDE expressions may use variables such as `x`, `y`, `f`, `fx`, `fy`, `fxx`, `fxy`, and `fyy`
- Vector PDE uses one residual expression per equation and only the explicit state
  notation `f[i]`, `fx[i]`, `fy[i]`, `fxx[i]`, `fxy[i]`, and `fyy[i]`

Typical safe math functions are available (`sin`, `cos`, `exp`, `log`, `sqrt`, etc.).

### 3. Configure numeric parameters

Common settings include:

- domain bounds and output sample points
- initial values or boundary values
- ODE solver method
- statistics to compute
- PDE operator and visualization mode

For 2D PDEs, the current solver supports real scalar linear/affine elliptic
equations on rectangular domains and optional custom mask expressions. Boundary
conditions can be Dirichlet or Neumann through the legacy point-array API.
Programmatic callers can instead use `PDEBoundaryCondition` and
`PDEBoundaryConditions` for Dirichlet, outward-normal Neumann, Robin
(`alpha*u + beta*du/dn = gamma`), and rectangular periodic axes. Periodic axes
use a non-duplicated upper endpoint; they cannot be combined with an arbitrary
mask because wrapped mask topology is not implemented.

The principal matrix must be strictly positive or negative definite at every
assembled point and must keep that orientation within each connected component
of the domain. Disconnected masked components may use independent orientations.
Parabolic, hyperbolic, degenerate, sign-changing-within-a-component, nonlinear,
singular, and non-finite problems are reported as solver failures instead of
returning a plausible-looking field.

Neumann and Robin use a one-sided grid-normal relation. On masked contours this
is a grid normal, not a reconstructed geometric normal, and the limitation is
included in diagnostics. Structured corner semantics are explicit: Dirichlet
anchors a mixed Dirichlet/Neumann or Dirichlet/Robin corner; two Dirichlet data
must agree; two non-Dirichlet edge conditions are rejected as ambiguous. Legacy
corner reconstruction also fails if a unique compatible normal cannot be found;
it never invents a zero boundary value. Mixed-derivative stencils wrap across
periodic seams and reject ambiguous masked/corner substitutions.

Example structured configuration:

```python
from solver import PDEBoundaryCondition, PDEBoundaryConditions, solve_pde_2d

boundaries = PDEBoundaryConditions(
    periodic_x=True,
    bottom=PDEBoundaryCondition.dirichlet(0.0),
    top=PDEBoundaryCondition.robin(alpha=2.0, beta=0.5, gamma=1.0),
)
solution = solve_pde_2d(
    residual,
    0.0,
    1.0,
    0.0,
    1.0,
    64,
    33,
    boundary_conditions=boundaries,
)
```

Programmatic `solve_pde_2d()` results include optional `PDEDiagnostics` with the
discrete L2/L-infinity residual, relative L2 residual, sparse matrix shape and
nonzero count. A condition estimate is included only for small systems; large
sparse systems are never converted to dense form solely for diagnostics.

Vector PDE is part of the same standard `Solve Equation` workflow. The custom
editor supports 1–4 components and interprets every expression as a residual
equal to zero. Indexes are integer literals starting at zero; an out-of-range,
indirect, or unsafe subscript is rejected before solver dispatch. The standard
boundary panel deliberately labels its Dirichlet/Neumann data as shared by all
components.

The programmatic API is `solve_vector_pde_2d()`. It represents

```text
sum_q (Axx[p,q] u_q,xx + Axy[p,q] u_q,xy + Ayy[p,q] u_q,yy
       + Ax[p,q] u_q,x + Ay[p,q] u_q,y + A0[p,q] u_q) + r[p] = 0
```

with six real `(m, m)` matrices and one `(m,)` constant vector in
`VectorPDECoefficients`. Provide either a direct coefficient provider or a
residual returning exactly `(m,)`; the residual path probes every state family
and every source component, then checks two dense all-component probes for
matrix affinity. It does not support nonlinear systems.

```python
import numpy as np

from solver import (
    PDEBoundaryCondition,
    VectorPDEBoundaryConditions,
    VectorPDECoefficients,
    solve_vector_pde_2d,
)

shared_zero = PDEBoundaryCondition.dirichlet(0.0)
boundaries = VectorPDEBoundaryConditions(
    left=shared_zero,
    right=shared_zero,
    bottom=shared_zero,
    top=shared_zero,
)


def coefficients(x: float, y: float, params: dict[str, float]) -> VectorPDECoefficients:
    principal = np.array([[1.0, 0.2], [0.2, 1.0]])
    zero = np.zeros((2, 2))
    coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
    return VectorPDECoefficients(principal, zero, principal, zero, zero, coupling, np.zeros(2))


solution = solve_vector_pde_2d(
    None,
    0.0,
    1.0,
    0.0,
    1.0,
    33,
    33,
    components=2,
    coefficient_provider=coefficients,
    boundary_conditions=boundaries,
)
```

A single `PDEBoundaryCondition` on an edge is explicitly shared. A sequence is
component-specific and must contain exactly `m` entries; a length-one sequence
is rejected rather than implicitly broadcast. Neumann and Robin retain the
physical outward-normal `du/dn` convention. All components share one mask and
periodic topology. Periodic axes reuse the scalar non-duplicated endpoint and
wrapping convention; arbitrary masks plus periodicity are rejected.

The strong-ellipticity screen evaluates the symmetric part of
`Axx*ξx**2 + Axy*ξx*ξy + Ayy*ξy**2` at 32 equally spaced unit directions on
`[0, π)`. Each sampled symbol must be positive definite or negative definite
with eigenvalue magnitude greater than `1e-10` times its infinity-norm scale;
the orientation must agree across directions and throughout each connected
spatial component. This deterministic sampled check is a numerical validation,
not a proof for every direction.

`VectorPDESolution.u` has shape `(m, ny, nx)`. Internally the sparse system is
component-major: `component * n_unknown_points + point_index`. Diagnostics
report global residuals, per-equation residuals, matrix shape and nonzero count.
A dense condition estimate remains limited to systems of at most 256 unknowns.
The result dialog provides every component field plus the Euclidean magnitude;
two-component output is marked as a planar vector field in metadata for later
quiver/stream visualization.

### 4. Solve and inspect

Result dialogs include:

- metadata and solver quality information
- selected statistics
- interactive plots/tabs
- derivative, component, or axis selection without re-solving
- export options for data and figures

## Function Transform Workflow

In `Function Transform`:

1. Provide an `f(x)` expression and domain.
2. Select transform type.
3. Tune transform-specific parameters when shown.
4. Switch between curve view and coefficient view.
5. Export transformed data and plots.

Available transform types:

- Original function sampling
- Fourier (FFT)
- Laplace on the real axis
- Taylor series
- Hilbert (discrete)
- Z-transform (discrete)

## Complex Problems Workflow

`Advanced Problems` opens the complex-problem selector dialog with a left module
list and a right details panel.

Each plugin has its own:

- configuration UI
- specialized solver
- background execution/loading behavior where useful
- dedicated result dialog and diagnostics

Current modules are:

- `coupled_oscillators`
- `membrane_2d`
- `nonlinear_waves`
- `schrodinger_td`
- `antenna_radiation`
- `aerodynamics_2d`
- `pipe_flow`

Most plugin dialogs include a collapsed `How to configure` section with:

- equation summary
- physical interpretation
- parameter meaning
- expected visualizations

For module details, see [Complex Problems Guide](complex-problems.md).

## Exports

By default, solves write generated files under `output/`.

Common outputs:

- `solution_*.csv`
- `solution_*.json`
- `solution_*.png` or another selected static figure format
- MP4 animations when the active result dialog supports animation

## Configuration Workflow

Use `Settings` in the main menu to edit environment-backed settings.

Categories include:

- UI colors, fonts, padding, and tooltips
- plot style, fonts, phase-space markers, contour/surface style, and animation
- solver defaults and tolerances
- logging and update checks

Saving from the dialog restarts the app so settings apply cleanly.

See [Configuration Reference](configuration.md) for the current key list.
