"""Verify an installed DifferentialLab wheel without starting its GUI."""

from __future__ import annotations

from importlib.metadata import distribution, entry_points
from importlib.util import find_spec


def main() -> None:
    """Check distribution metadata, entry-point loading, modules, and packaged YAML data."""
    package = distribution("differential-lab")
    assert package.metadata["Name"] == "differential-lab"
    assert package.version

    console_entry = next(
        entry for entry in entry_points(group="console_scripts") if entry.name == "differential-lab"
    )
    assert callable(console_entry.load())

    import main_program
    from solver.predefined import load_predefined_equations

    assert callable(main_program.main)
    assert find_spec("pipeline") is not None
    assert load_predefined_equations()


if __name__ == "__main__":
    main()
