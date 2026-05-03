# Logging Improvement Design

**Date:** 2026-05-03

**Goal:** Improve the current logging system so it is more robust in normal use and more helpful during debugging, while preserving the existing `get_logger(name: str) -> logging.Logger` API used across the project.

## Current State

The project already centralizes logging in `src/utils/logger.py`. It reads `LOG_LEVEL`, `LOG_FILE`, and `LOG_CONSOLE` from the environment and configures a namespaced logger under `differential_lab`.

This is a good base, but it still has a few practical limitations:

- The log file grows without rotation.
- The log path is treated as a simple file name and does not create missing parent directories.
- File-handler setup can fail hard if the path is invalid or unavailable.
- There is no supported way to reset logging state in tests.
- There are no dedicated tests for logger configuration behavior.

## Scope

This change should improve logging internals and configuration only. It should not require broad edits to solver, frontend, plotting, or complex-problem modules that already call `get_logger(__name__)`.

Out of scope:

- structured JSON logging,
- external logging libraries,
- per-module custom log levels,
- a UI log viewer.

## Recommended Approach

Keep Python's built-in `logging` module and the current namespaced API, but strengthen the implementation with safer configuration and bounded log-file growth.

### 1. Safer logger configuration

`src/utils/logger.py` should:

- keep `get_logger(name: str) -> logging.Logger` unchanged,
- configure the `differential_lab` root logger only once during normal runtime,
- avoid duplicate handlers if configuration is triggered again,
- expose a small internal reset/reconfigure hook for tests,
- disable propagation to unrelated ancestor loggers to avoid duplicate output.

### 2. Rotating file logs

Replace the current plain `FileHandler` with a rotating file handler.

New environment settings:

- `LOG_MAX_BYTES`: maximum size of the active log file before rotation,
- `LOG_BACKUP_COUNT`: number of rotated log files to keep.

Default behavior should remain friendly for desktop usage: one active log plus a small number of backups, enough for debugging recent sessions without growing forever.

### 3. Better failure handling

If file logging cannot be initialized:

- the application should not crash during logger setup,
- the failure should fall back to console logging when possible,
- if console logging is already enabled, the setup error should still be visible there.

This is especially useful on systems with permission issues, invalid paths, or read-only locations.

### 4. More useful log format

Keep the format readable for humans, but include a bit more execution context.

The format should include:

- timestamp,
- level,
- logger name,
- thread name,
- message.

This adds value for background tasks and UI-triggered operations without making logs hard to scan.

### 5. Configuration and UI exposure

Update the configuration schema and configuration dialog so users can edit the new rotation settings through the same mechanism as the current logging settings.

Files affected:

- `src/config/env.py`
- `src/config/__init__.py`
- `src/frontend/ui_dialogs/config_dialog.py`
- `docs/configuration.md`

## Testing Strategy

Add focused tests for logger setup behavior instead of relying only on indirect coverage.

Recommended test coverage:

- logger creation returns namespaced logger objects,
- repeated setup does not duplicate handlers,
- rotating file handler is configured with the expected limits,
- missing parent directories are created for nested log paths,
- configuration can fall back safely when file handler creation fails,
- reset/reconfigure support works for isolated tests.

These tests should live in a dedicated logger test module under `tests/utils/`.

## Risks and Controls

### Risk: duplicate logs

If propagation or handler reuse is wrong, messages may appear more than once.

Control:

- explicitly manage handler replacement,
- test repeated configuration paths.

### Risk: test fragility

Global logging state is process-wide, so tests can interfere with each other.

Control:

- provide a small internal reset utility,
- keep tests isolated with `monkeypatch` and temporary paths.

### Risk: breaking user expectations

Users may already rely on the current default log filename and readable text format.

Control:

- keep `LOG_FILE` and `LOG_LEVEL` semantics unchanged,
- keep plain text output,
- only extend configuration with rotation settings.

## Expected Outcome

After this change, DifferentialLab should keep a bounded set of readable log files, behave better when log-file setup fails, and provide more useful execution context during debugging, all without changing how the rest of the codebase requests loggers.
