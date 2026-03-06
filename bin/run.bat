@echo off
setlocal
REM ============================================================================
REM DifferentialLab - Quick Launch Script for Windows
REM ============================================================================

REM Change to project root directory (parent of bin)
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist .venv (
    echo ERROR: Virtual environment not found
    echo Please run bin\setup.bat first
    exit /b 1
)

REM Activate virtual environment and run the program
call .venv\Scripts\activate.bat
if errorlevel 1 exit /b %errorlevel%

set "MODE=%~1"
if "%MODE%"=="" set "MODE=--prod"

if /I "%MODE%"=="--dev" goto dev
if /I "%MODE%"=="-d" goto dev
if /I "%MODE%"=="--background" goto background
if /I "%MODE%"=="-b" goto background
if /I "%MODE%"=="--prod" goto prod
if /I "%MODE%"=="-p" goto prod

echo Usage: bin\run.bat [--dev ^| -d ^| --background ^| -b ^| --prod ^| -p]
exit /b 1

:dev
python src\main_program.py
exit /b %errorlevel%

:background
if not exist logs mkdir logs
start "DifferentialLab" /B cmd /c "python src\main_program.py >> logs\run.log 2>&1"
if errorlevel 1 exit /b %errorlevel%
echo DifferentialLab started in background
echo Logs: logs\run.log
exit /b 0

:prod
start "DifferentialLab" pythonw src\main_program.py
if errorlevel 1 exit /b %errorlevel%
echo DifferentialLab started in production mode
exit /b 0
