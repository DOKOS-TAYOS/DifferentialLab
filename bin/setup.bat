@echo off
REM ============================================================================
REM ODE Solver - Setup Script for Windows
REM ============================================================================

REM Change to project root directory (parent of bin)
cd /d "%~dp0.."

echo.
echo ====================================
echo  ODE Solver Setup (Windows)
echo ====================================
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
pip install -r requirements.txt

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
echo To run ODE Solver:
echo   bin\run.bat
echo.
echo To configure the application:
echo   Edit .env with your preferences
echo.

pause
