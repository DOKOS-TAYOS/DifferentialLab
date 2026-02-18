"""DifferentialLab — application entry point."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

# Ensure src/ is on the import path when run directly
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def main() -> None:
    """Initialize configuration, logging, and launch the main menu."""
    from config.env import initialize_and_validate_config
    from config.paths import get_output_dir
    from utils.logger import get_logger

    initialize_and_validate_config()
    logger = get_logger(__name__)
    logger.info("DifferentialLab starting")

    get_output_dir()

    try:
        root = tk.Tk()

        from frontend.ui_main_menu import MainMenu

        _app = MainMenu(root)
        root.mainloop()
    except Exception:
        logger.critical("Unhandled exception", exc_info=True)
        raise
    finally:
        logger.info("DifferentialLab shutting down")


if __name__ == "__main__":
    main()
