@echo off
REM ============================================================================
REM DifferentialLab - Installation Script for Windows
REM ============================================================================

echo.
echo ====================================
echo  DifferentialLab Installation
echo ====================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo [1/4] Git found:
git --version

set "REPO_URL=https://github.com/DOKOS-TAYOS/DifferentialLab.git"
set "REPO_NAME=DifferentialLab"

if exist "%REPO_NAME%" (
    echo.
    echo WARNING: Directory '%REPO_NAME%' already exists
    set /p OVERWRITE="Do you want to remove it and clone again? (y/N): "
    if /i "%OVERWRITE%"=="y" (
        echo Removing existing directory...
        rmdir /s /q "%REPO_NAME%" 2>nul
    ) else (
        echo Using existing directory...
        cd "%REPO_NAME%"
        goto :run_setup
    )
)

echo.
echo [2/4] Cloning repository...
git clone "%REPO_URL%" "%REPO_NAME%"
if errorlevel 1 (
    echo ERROR: Failed to clone repository
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo        Repository cloned successfully

cd "%REPO_NAME%"
if errorlevel 1 (
    echo ERROR: Failed to change to repository directory
    pause
    exit /b 1
)

:run_setup
echo.
echo [3/4] Running setup...
echo.
call bin\setup.bat
if errorlevel 1 (
    echo.
    echo ERROR: Setup failed
    pause
    exit /b 1
)

echo.
echo [4/4] Creating desktop shortcut...
set "DESKTOP=%USERPROFILE%\Desktop"
if not exist "%DESKTOP%" set "DESKTOP=%USERPROFILE%\Escritorio"
set "SHORTCUT=%DESKTOP%\DifferentialLab.lnk"
set "PROJECT_DIR=%CD%"

powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; ^
     $sc = $ws.CreateShortcut($args[0]); ^
     $sc.TargetPath = $args[1]; ^
     $sc.WorkingDirectory = $args[2]; ^
     $sc.Description = 'Launch DifferentialLab'; ^
     $sc.Save()" "%SHORTCUT%" "%PROJECT_DIR%\bin\run.bat" "%PROJECT_DIR%" 2>nul

if exist "%SHORTCUT%" (
    echo        Desktop shortcut created: %SHORTCUT%
) else (
    echo        WARNING: Could not create desktop shortcut
)

echo.
echo ====================================
echo  Installation Complete!
echo ====================================
echo.
echo DifferentialLab has been cloned and set up.
echo You can now run the application from: %CD%
echo Desktop shortcut: %SHORTCUT%
echo.
pause
