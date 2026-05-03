# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
