@echo off
REM ============================================================================
REM ODE Solver - Quick Launch Script for Windows
REM ============================================================================

REM Change to project root directory (parent of bin)
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist .venv (
    echo ERROR: Virtual environment not found
    echo Please run bin\setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment and run the program
call .venv\Scripts\activate.bat
start "" pythonw src\main_program.py
