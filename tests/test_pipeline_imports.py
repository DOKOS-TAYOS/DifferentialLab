"""Regression tests for lightweight standard ODE pipeline startup."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_import_does_not_load_pde_backends() -> None:
    """A fresh standard pipeline import must leave PDE solver backends unloaded."""
    repository_root = Path(__file__).resolve().parents[1]
    source_dir = repository_root / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(source_dir), *filter(None, [env.get("PYTHONPATH")])])

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import pipeline\n"
                "heavy = {'solver.pde_solver', 'solver.pde_3d_solver', "
                "'solver.pde_system_solver'}\n"
                "unexpected = sorted(heavy & sys.modules.keys())\n"
                "if unexpected:\n"
                "    raise SystemExit(', '.join(unexpected))\n"
            ),
        ],
        cwd=repository_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
