# Logging Improvement Design

**Date:** 2026-05-03

**Status:** Implemented in the current unreleased code.

**Goal:** Improve the logging system so it is robust in normal use and helpful
during debugging, while preserving the existing
`get_logger(name: str) -> logging.Logger` API used across the project.

## Implemented State

The project centralizes logging in `src/utils/logger.py`. It reads `LOG_LEVEL`,
`LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`, and `LOG_CONSOLE` from the
environment and configures a namespaced logger under `differential_lab`.

Current behavior:

- Log files use `RotatingFileHandler`.
- Relative log paths are resolved from the project root.
- Missing parent directories are created automatically.
- File-handler setup failures fall back to console logging.
- Reconfiguration support exists for tests through `_reset_logging_state()`.
- Logger propagation is disabled to avoid duplicate output.
- Log records include timestamp, level, thread name, logger name, and message.

## Scope

This change improved logging internals and configuration only. It did not
require broad edits to solver, frontend, plotting, or complex-problem modules
that already call `get_logger(__name__)`.

Out of scope:

- structured JSON logging
- external logging libraries
- per-module custom log levels
- a UI log viewer

## Configuration

Logging is controlled by these `.env` keys:

- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`
- `LOG_FILE`: filename, relative path, or absolute path for the active log
- `LOG_MAX_BYTES`: maximum active log size before rotation
- `LOG_BACKUP_COUNT`: number of rotated files to keep
- `LOG_CONSOLE`: whether to also print logs to the terminal

The canonical schema lives in `src/config/env.py`.

## Testing Coverage

Focused tests live under `tests/utils/test_logger.py`.

The suite covers:

- logger creation under the `differential_lab` namespace
- repeated configuration without duplicate handlers
- rotating file handler limits
- nested log-path directory creation
- safe fallback when file logging cannot initialize
- reset/reconfigure behavior for isolated tests

## Risks and Controls

### Duplicate logs

Control:

- root logger handlers are explicitly managed
- propagation is disabled
- repeated setup is tested

### Test fragility

Control:

- `_reset_logging_state()` clears global logging state for tests
- tests use temporary paths and monkeypatched environment values

### User expectations

Control:

- `LOG_FILE` and `LOG_LEVEL` semantics remain unchanged
- plain text output remains readable
- rotation settings extend configuration without changing how modules request loggers

## Outcome

DifferentialLab now keeps a bounded set of readable log files, behaves better
when log-file setup fails, and provides more useful execution context during
debugging without changing the public logger API.
