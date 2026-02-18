@echo off
REM ============================================================================
REM DifferentialLab - Clean temporary files (Windows)
REM Removes __pycache__, .pyc, .pyo, .log, and other temp files.
REM Skips the .venv directory.
REM ============================================================================

cd /d "%~dp0.."

echo Cleaning temporary files...

set "count=0"

REM Remove __pycache__ directories (excluding .venv)
for /f "delims=" %%d in ('dir /s /b /ad __pycache__ 2^>nul ^| findstr /v /i "\\.venv\\"') do (
    rmdir /s /q "%%d" 2>nul
    set /a count+=1
)

REM Remove .pyc and .pyo files (excluding .venv)
for %%e in (pyc pyo) do (
    for /f "delims=" %%f in ('dir /s /b "*.%%e" 2^>nul ^| findstr /v /i "\\.venv\\"') do (
        del /q "%%f" 2>nul
        set /a count+=1
    )
)

REM Remove .mypy_cache, .pytest_cache, .ruff_cache (excluding .venv)
for %%d in (.mypy_cache .pytest_cache .ruff_cache) do (
    for /f "delims=" %%p in ('dir /s /b /ad "%%d" 2^>nul ^| findstr /v /i "\\.venv\\"') do (
        rmdir /s /q "%%p" 2>nul
        set /a count+=1
    )
)

REM Remove egg-info directories
for /f "delims=" %%d in ('dir /s /b /ad "*.egg-info" 2^>nul ^| findstr /v /i "\\.venv\\"') do (
    rmdir /s /q "%%d" 2>nul
    set /a count+=1
)

REM Remove log files from root
if exist *.log (
    del /q *.log
    set /a count+=1
)

echo Done. Cleaned %count% items.
