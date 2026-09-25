#!/bin/bash
# ============================================================================
# DifferentialLab - Installation Script for Unix/Mac
# ============================================================================

set -e

echo ""
echo "===================================="
echo " DifferentialLab Installation"
echo "===================================="
echo ""

install_git() {
    if command -v apt-get &> /dev/null; then
        echo "       Detected apt. Installing git..."
        sudo apt-get update && sudo apt-get install -y git
    elif command -v dnf &> /dev/null; then
        echo "       Detected dnf. Installing git..."
        sudo dnf install -y git
    elif command -v yum &> /dev/null; then
        echo "       Detected yum. Installing git..."
        sudo yum install -y git
    elif command -v pacman &> /dev/null; then
        echo "       Detected pacman. Installing git..."
        sudo pacman -S --noconfirm git
    elif command -v zypper &> /dev/null; then
        echo "       Detected zypper. Installing git..."
        sudo zypper install -y git
    elif command -v brew &> /dev/null; then
        echo "       Detected Homebrew. Installing git..."
        brew install git
    else
        echo "       Could not detect package manager. Please install Git manually:"
        echo "  - Ubuntu/Debian: sudo apt-get install git"
        echo "  - Fedora/RHEL: sudo dnf install git"
        echo "  - Arch: sudo pacman -S git"
        echo "  - macOS: brew install git  or  xcode-select --install"
        echo "  - Or download from: https://git-scm.com/downloads"
        return 1
    fi
}

if ! command -v git &> /dev/null; then
    echo "Git is not installed."
    read -p "Do you want to install it now? (y/N): " INSTALL_GIT
    if [[ "$INSTALL_GIT" =~ ^[Yy]$ ]]; then
        if install_git; then
            echo "       Git installed successfully."
        else
            exit 1
        fi
    else
        echo ""
        echo "Please install Git manually:"
        echo "  - Ubuntu/Debian: sudo apt-get install git"
        echo "  - macOS: git is included with Xcode Command Line Tools"
        echo "  - Or download from: https://git-scm.com/downloads"
        exit 1
    fi
fi

echo "[1/3] Git found:"
git --version

REPO_URL="https://github.com/DOKOS-TAYOS/DifferentialLab.git"
REPO_NAME="DifferentialLab"

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
        chmod +x bin/setup.sh bin/run.sh
        ./bin/setup.sh
        PROJECT_DIR="$(pwd)"
        source "$PROJECT_DIR/bin/linux_desktop.sh"
        chmod +x install.sh bin/setup.sh bin/run.sh
        if [ "$(uname -s)" = "Linux" ]; then
            create_linux_launchers "$PROJECT_DIR"
        fi
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
echo "[3/4] Running setup..."
echo ""

chmod +x install.sh bin/setup.sh bin/run.sh
./bin/setup.sh

PROJECT_DIR="$(pwd)"
source "$PROJECT_DIR/bin/linux_desktop.sh"
chmod +x install.sh bin/setup.sh bin/run.sh
if [ "$(uname -s)" = "Linux" ]; then
    create_linux_launchers "$PROJECT_DIR"
fi

echo ""
echo "===================================="
echo " Installation Complete!"
echo "===================================="
echo ""
echo "DifferentialLab has been cloned and set up."
echo "You can now run the application from: $PROJECT_DIR"
echo "Application-menu launcher is available on Linux."
echo ""
