#!/bin/bash
# ============================================================================
# DifferentialLab - Setup Script for Unix/Mac
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

is_linux_with_pkg_manager() {
    [ "$(uname -s)" = "Linux" ] || return 1
    command -v apt-get &> /dev/null || command -v dnf &> /dev/null || \
    command -v pacman &> /dev/null
}

install_python() {
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3 python3-pip
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm python python-pip
    elif command -v brew &> /dev/null; then
        brew install python3
    else
        return 1
    fi
}

ask_install_python() {
    echo "Python 3 is not installed."
    echo ""
    read -p "Do you want to install Python 3 now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if install_python; then
            hash -r 2>/dev/null || true
            echo "       Python installed successfully"
            return 0
        else
            echo "ERROR: Could not install Python automatically."
            echo "Please install Python 3.12 or higher manually."
            return 1
        fi
    else
        echo "Please install Python 3.12 or higher and run setup again."
        return 1
    fi
}

install_tkinter_linux() {
    if command -v apt-get &> /dev/null; then
        PYTHON_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" || PYTHON_VER="3.12"
        sudo apt-get install -y "python${PYTHON_VER}-tk" 2>/dev/null || sudo apt-get install -y python3-tk
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-tkinter 2>/dev/null || sudo dnf install -y python3.12-tkinter
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm tk
    else
        return 1
    fi
}

echo ""
echo "===================================="
echo " DifferentialLab Setup (Unix/Mac)"
echo "===================================="
echo ""

if ! command -v python3 &> /dev/null; then
    if ! ask_install_python; then
        exit 1
    fi
fi

echo "[1/7] Checking Python version..."
python3 --version

python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" || {
    echo "ERROR: Python 3.12 or higher is required"
    exit 1
}
echo "       Python version OK"

echo ""
echo "[2/7] Ensuring tkinter is available..."
if ! python3 -c "import tkinter" 2>/dev/null; then
    if is_linux_with_pkg_manager; then
        echo "       Tkinter not found. Installing system package..."
        if install_tkinter_linux; then
            echo "       Tkinter installed successfully"
        else
            echo "       WARNING: Could not install tkinter automatically."
            echo "       On Ubuntu/Debian: sudo apt-get install python3-tk"
            echo "       On Fedora: sudo dnf install python3-tkinter"
            echo "       On Arch: sudo pacman -S tk"
        fi
    else
        echo "       WARNING: Tkinter not found. The GUI requires it."
    fi
else
    echo "       Tkinter already available"
fi

echo ""
echo "[3/7] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "       Virtual environment already exists, skipping creation"
else
    python3 -m venv .venv
    echo "       Virtual environment created"
fi

echo ""
echo "[4/7] Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "[5/7] Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "[6/7] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[7/7] Setting up environment file..."
if [ -f ".env" ]; then
    echo "       .env file already exists, skipping"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "       .env file created from .env.example"
    else
        echo "       Warning: .env.example not found, skipping .env creation"
    fi
fi

echo ""
echo "===================================="
echo " Setup Complete!"
echo "===================================="
echo ""
echo "To run DifferentialLab:"
echo "  ./bin/run.sh"
echo ""
echo "To configure the application:"
echo "  Edit .env with your preferences"
echo ""
