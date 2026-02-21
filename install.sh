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
        chmod +x bin/setup.sh
        ./bin/setup.sh
        PROJECT_DIR="$(pwd)"
        DESKTOP="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
        SHORTCUT="$DESKTOP/DifferentialLab.desktop"
        ICON_PATH="$PROJECT_DIR/images/DifferentialLab_icon.ico"
        if [ -d "$DESKTOP" ]; then
            if [ -f "$ICON_PATH" ]; then
                ICON_LINE="Icon=$ICON_PATH"
            else
                ICON_LINE="Icon=utilities-terminal"
            fi
            cat > "$SHORTCUT" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DifferentialLab
Comment=Launch DifferentialLab
Exec=$PROJECT_DIR/bin/run.sh
Path=$PROJECT_DIR
$ICON_LINE
Terminal=false
EOF
            chmod +x "$SHORTCUT"
            echo ""
            echo "Desktop shortcut created: $SHORTCUT"
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

chmod +x bin/setup.sh
./bin/setup.sh

echo ""
echo "[4/4] Creating desktop shortcut..."

PROJECT_DIR="$(pwd)"
DESKTOP="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
SHORTCUT="$DESKTOP/DifferentialLab.desktop"
ICON_PATH="$PROJECT_DIR/images/DifferentialLab_icon.ico"

if [ -d "$DESKTOP" ]; then
    if [ -f "$ICON_PATH" ]; then
        ICON_LINE="Icon=$ICON_PATH"
    else
        ICON_LINE="Icon=utilities-terminal"
    fi
    cat > "$SHORTCUT" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DifferentialLab
Comment=Launch DifferentialLab
Exec=$PROJECT_DIR/bin/run.sh
Path=$PROJECT_DIR
$ICON_LINE
Terminal=false
EOF
    chmod +x "$SHORTCUT"
    echo "       Desktop shortcut created: $SHORTCUT"
else
    echo "       WARNING: Desktop directory not found, skipping shortcut"
fi

echo ""
echo "===================================="
echo " Installation Complete!"
echo "===================================="
echo ""
echo "DifferentialLab has been cloned and set up."
echo "You can now run the application from: $PROJECT_DIR"
echo "Desktop shortcut: $SHORTCUT"
echo ""
