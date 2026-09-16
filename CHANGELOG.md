# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added structured scalar PDE diagnostics for discrete residual norms, sparse matrix size,
  nonzero count, and a bounded small-system condition estimate.
- Added structured scalar-2D Dirichlet, Neumann, and Robin boundary objects plus
  mathematically wrapped one-axis/two-axis periodic grids with non-duplicated endpoints.
- Added linear strongly elliptic Vector PDE systems in 2D with six matrix-valued
  operator coefficients, full coupled residual-affinity probing, sampled principal-symbol
  validation, explicit shared/component boundary rules, component-major sparse assembly,
  global/per-component diagnostics, and one coupled catalog example.
- Integrated `Vector PDE` into the standard equation, parameter, result, and export flow
  with safe indexed residual notation and component/magnitude field views.
- Added a dedicated scalar linear elliptic PDE 3D solver with affine residual and direct
  coefficient paths, strict 3x3 principal-matrix validation, mixed-derivative sparse
  stencils, six-face Dirichlet/Neumann/Robin data, periodic axes, algebraic diagnostics,
  standard-workflow dispatch, pre-run sparse-memory advice, and orthogonal slice access.
- Added typed IVP options for events, analytic Jacobians, vectorized evaluation, and
  first-step control; structured ODE work/status/event diagnostics; and safe terminal or
  directional event expressions in the standard ODE dialog.
- Added a typed SciPy BVP wrapper plus automatic, forced-shooting, and forced-BVP
  multipoint strategies with deterministic bounded initial meshes and explicit
  incompatible-condition failures.

### Security

- Added Dependabot configuration for Python dependencies and GitHub Actions.
- Documented GitHub CodeQL default setup and added a weekly/manual `pip-audit`
  dependency audit workflow.
- Added a repository security policy for private vulnerability reporting.
- Hardened expression parameter handling so unsafe parameter names are rejected and
  user parameters cannot shadow approved math functions during expression evaluation.
- Restricted the update checker to HTTPS version URLs.

### Changed

- Preserved zero-valued ODE solver options through explicit `None` resolution, including
  the `max_step=0` infinity sentinel when callers supply their own evaluation grid.
- Corrected automatic multipoint routing so every compatible endpoint-only condition set
  after the all-at-start IVP shortcut uses BVP, including conditions entirely at `x_max`.
- Reject ODE event expressions during parsing unless their test evaluation produces
  exactly one finite real scalar.

- Hardened the scalar 2D PDE solver with strict domain, grid, mask, boundary-data,
  coefficient, affine-residual, ellipticity, sparse-solve, and finiteness validation.
- Corrected nonzero outward-normal Neumann signs on all rectangular edges, made
  mixed-derivative boundary elimination geometry-aware, and rejected spatially
  inconsistent positive/negative ellipticity orientation.
- Split scalar-2D PDE validation, boundary substitution, sparse assembly, and public
  types into focused modules while preserving the `solve_pde_2d()` legacy signature.
- Made corner behavior explicit, removed the zero-valued Neumann reconstruction fallback,
  and validate ellipticity orientation independently in each connected mask component.
- Detect a constant-field numerical nullspace before sparse PDE solves, including large
  periodic Laplacian systems whose zero right-hand side can otherwise hide singularity.

- Expanded CI to run on `main` and `dev` for Python 3.12 and 3.13, with ruff linting,
  ruff format checks, pytest, and pyright.
- Configured pytest to use the repo-local ignored `.pytest-temp` directory for
  temporary files, avoiding Windows temp-permission issues.
- Refreshed repository documentation to match the current 0.4.1 codebase, including
  setup/run modes, solver coverage, complex-problem plugins, configuration defaults,
  logging behavior, API docs, and the pytest/ruff/pyright workflow.
- Aligned `.env.example` with the current `ENV_SCHEMA` defaults, including log
  rotation settings.
- Ignored the local `.tmp/` scratch directory used by sandboxed test runs.
- Improved the logging setup with rotating log files, safer handler reconfiguration, nested log-path support, and graceful console fallback when file logging cannot start.
- Removed the unused internal `_is_uniform` helper from the coupled oscillators model.
- Extracted shared internal complex-problem dialog helpers to reduce UI duplication across simpler solver dialogs.
- Extracted shared internal result-dialog helpers to centralize embedded figure cleanup and animation reset logic.
- Refactored coupled oscillators and standard parameter dialogs to use typed input-collection helpers before solver dispatch.
- Added explicit function annotations across internal source modules.
- Reused a shared background-task helper for solver dialogs and split the standard parameters dialog UI into focused private builders.
- Reduced high-resolution PDE overhead by switching masked-domain classification and sparse assembly to compact integer grids with preallocated buffers.
- Reduced ODE residual/PDE/aerodynamics/pipe-flow overhead by reusing precomputed RHS values, adding a direct PDE coefficient fast path for coordinate-only right-hand sides, avoiding extra periodic derivative temporaries, and stabilizing transient pipe-flow defaults.
- Deferred large Schrödinger and nonlinear-wave `magnitude`/`phase` arrays until first access, and added pre-run performance advisories for expensive solver configurations.
- Hardened expression validation against unsafe attribute access and added opt-in solver controls for lower-memory Schrödinger histories and skipped ODE RHS post-processing.
- Reduced transform import overhead, reused exponential-rate statistics work, and added opt-in lower-memory nonlinear-wave histories.
- Reused result-plot canvases on view changes, debounced resize redraws, and collapsed membrane history summaries into a single energy/extrema pass.
- Added a repo-local `pyright` configuration, tightened Tk/matplotlib typing in shared UI helpers, and cleaned transform/solver typing hotspots until `pyright` reports zero errors.
- Refreshed user-facing dialog copy across the main UI, equation setup, transforms, results, settings, performance warnings, and advanced-problem workflows.

## [0.3.2] - 2026-03-05

### Added

- **Complex Problems** module *(experimental)*: special cases with custom UIs (coupled harmonic oscillators). Still in development; may contain bugs.
- Main menu: new "Complex Problems" button
- Configuration: UI Tooltips (delay, wraplength, padding)
- Configuration: Plot Phase-Space (start/end marker colours and size)
- Configuration: Plot 3D / Contour (colormap, contour levels, alpha, colorbar shrink)
- Loading dialog for long-running solves
