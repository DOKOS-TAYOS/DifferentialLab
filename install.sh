#!/bin/bash
# ============================================================================
# ODE Solver - Installation Script for Unix/Mac
# ============================================================================

set -e

echo ""
echo "===================================="
echo " ODE Solver Installation"
echo "===================================="
echo ""

if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed"
    echo "Please install Git:"
    echo "  - Ubuntu/Debian: sudo apt-get install git"
    echo "  - macOS: git is included with Xcode Command Line Tools"
    echo "  - Or download from: https://git-scm.com/downloads"
    exit 1
fi

echo "[1/3] Git found:"
git --version

REPO_URL="https://github.com/DOKOS-TAYOS/ode-solver.git"
REPO_NAME="ode_solver"

if [ -d "$REPO_NAME" ]; then
    echo ""
    echo "WARNING: Directory '$REPO_NAME' already exists"
    read -p "Do you want to remove it and clone again? (y/N): " OVERWRITE
    if [[ "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "       Removing existing directory..."
        rm -rf "$REPO_NAME"
    else
        echo "       Using existing directory..."
        cd "$REPO_NAME"
        chmod +x bin/setup.sh
        ./bin/setup.sh
        exit 0
    fi
fi

echo ""
echo "[2/3] Cloning repository..."
if ! git clone "$REPO_URL" "$REPO_NAME"; then
    echo "ERROR: Failed to clone repository"
    echo "Please check your internet connection and try again"
    exit 1
fi

echo "       Repository cloned successfully"

cd "$REPO_NAME" || {
    echo "ERROR: Failed to change to repository directory"
    exit 1
}

echo ""
echo "[3/3] Running setup..."
echo ""

chmod +x bin/setup.sh
./bin/setup.sh

echo ""
echo "===================================="
echo " Installation Complete!"
echo "===================================="
echo ""
echo "The ODE Solver has been cloned and set up."
echo "You can now run the application from: $(pwd)"
echo ""
