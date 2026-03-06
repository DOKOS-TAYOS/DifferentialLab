@echo off
REM ============================================================================
REM DifferentialLab - Setup Script for Windows
REM ============================================================================

REM Change to project root directory (parent of bin)
cd /d "%~dp0.."

set "MODE=%~1"
if "%MODE%"=="" set "MODE=--prod"

set "INSTALL_TARGET=."
set "INSTALL_MODE_LABEL=production"

if /I "%MODE%"=="--dev" goto mode_dev
if /I "%MODE%"=="-d" goto mode_dev
if /I "%MODE%"=="--prod" goto mode_prod
if /I "%MODE%"=="-p" goto mode_prod

echo Usage: bin\setup.bat [--prod ^| -p ^| --dev ^| -d]
exit /b 1

:mode_dev
set "INSTALL_TARGET=.[dev]"
set "INSTALL_MODE_LABEL=development"
goto mode_ok

:mode_prod
goto mode_ok

:mode_ok

echo.
echo ====================================
echo  DifferentialLab Setup (Windows)
echo ====================================
echo  Mode: %INSTALL_MODE_LABEL%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.12 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
python --version

REM Check Python version is 3.12 or higher
python -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 or higher is required
    pause
    exit /b 1
)
echo        Python version OK

echo.
echo [2/6] Creating virtual environment...
if exist .venv (
    echo        Virtual environment already exists, skipping creation
) else (
    python -m venv .venv
    echo        Virtual environment created
)

echo.
echo [3/6] Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [5/6] Installing dependencies...
python -m pip install -e "%INSTALL_TARGET%"

echo.
echo [6/6] Setting up environment file...
if exist .env (
    echo        .env file already exists, skipping
) else (
    if exist .env.example (
        copy .env.example .env >nul
        echo        .env file created from .env.example
    ) else (
        echo        Warning: .env.example not found, skipping .env creation
    )
)

echo.
echo ====================================
echo  Setup Complete!
echo ====================================
echo.
echo To run DifferentialLab:
echo   bin\run.bat
echo.
echo To configure the application:
echo   Edit .env with your preferences
echo.

pause
