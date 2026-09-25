"""Focused coverage for Linux venv launch and desktop integration."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def test_run_script_uses_installed_venv_entry_point() -> None:
    """The launcher does not depend on shell activation or system Python."""
    script = (ROOT / "bin" / "run.sh").read_text(encoding="utf-8")

    assert 'ENTRY_POINT=".venv/bin/differential-lab"' in script
    assert '"$ENTRY_POINT"' in script
    assert "source .venv/bin/activate" not in script
    assert "python src/main_program.py" not in script


@pytest.mark.skipif(
    BASH is None or os.name != "posix",
    reason="The Linux desktop helper can only run with POSIX paths",
)
def test_desktop_helper_keeps_menu_launcher_without_desktop(
    tmp_path: Path,
) -> None:
    """A missing Desktop directory does not prevent application-menu setup."""
    project = tmp_path / "Project With Spaces"
    home = tmp_path / "home"
    data_home = tmp_path / "xdg data"
    project.mkdir()
    home.mkdir()
    environment = os.environ.copy()
    environment.update({"HOME": str(home), "XDG_DATA_HOME": str(data_home)})
    environment.pop("XDG_DESKTOP_DIR", None)

    subprocess.run(
        [
            BASH or "bash",
            "-c",
            'source "$1"; create_linux_launchers "$2"',
            "bash",
            str(ROOT / "bin" / "linux_desktop.sh"),
            str(project),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    launcher = data_home / "applications" / "DifferentialLab.desktop"
    contents = launcher.read_text(encoding="utf-8")
    assert f'Exec="{project}/.venv/bin/differential-lab"' in contents
    assert f"Icon={project}/images/DifferentialLab_icon.png" in contents
    assert not (home / "DifferentialLab.desktop").exists()


@pytest.mark.skipif(
    BASH is None or os.name != "posix",
    reason="The Linux desktop helper can only run with POSIX paths",
)
def test_desktop_helper_uses_xdg_desktop_directory(tmp_path: Path) -> None:
    """An explicit XDG Desktop path receives a copy of the menu launcher."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    desktop = tmp_path / "custom desktop"
    project.mkdir()
    home.mkdir()
    desktop.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_DESKTOP_DIR": str(desktop),
        }
    )

    subprocess.run(
        [
            BASH or "bash",
            "-c",
            'source "$1"; create_linux_launchers "$2"',
            "bash",
            str(ROOT / "bin" / "linux_desktop.sh"),
            str(project),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert (desktop / "DifferentialLab.desktop").is_file()
