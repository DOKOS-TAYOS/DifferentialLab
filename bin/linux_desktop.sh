#!/usr/bin/env bash
# Create Linux application-menu and optional Desktop launchers.

create_linux_launchers() {
    local project_dir="$1"
    local data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
    local applications_dir="$data_home/applications"
    local app_launcher="$applications_dir/DifferentialLab.desktop"
    local desktop_dir=""
    local entry_point="$project_dir/.venv/bin/differential-lab"
    local icon_path="$project_dir/images/DifferentialLab_icon.png"

    mkdir -p "$applications_dir"
    cat > "$app_launcher" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DifferentialLab
Comment=Launch DifferentialLab
Exec="$entry_point"
Path=$project_dir
Icon=$icon_path
Terminal=false
EOF
    chmod +x "$app_launcher"
    echo "Application-menu launcher created: $app_launcher"

    if [ -n "${XDG_DESKTOP_DIR:-}" ] && [ -d "$XDG_DESKTOP_DIR" ]; then
        desktop_dir="$XDG_DESKTOP_DIR"
    elif command -v xdg-user-dir >/dev/null 2>&1; then
        desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
        [ -d "$desktop_dir" ] || desktop_dir=""
    fi
    if [ -z "$desktop_dir" ] && [ -d "$HOME/Desktop" ]; then
        desktop_dir="$HOME/Desktop"
    fi

    if [ -n "$desktop_dir" ]; then
        cp "$app_launcher" "$desktop_dir/DifferentialLab.desktop"
        chmod +x "$desktop_dir/DifferentialLab.desktop"
        echo "Desktop launcher created: $desktop_dir/DifferentialLab.desktop"
    else
        echo "No usable Desktop directory found; application-menu launcher is available."
    fi
}
