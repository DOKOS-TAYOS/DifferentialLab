"""DifferentialLab — application entry point."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Ensure src/ is on the import path when run directly
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def _check_for_updates() -> None:
    """Check for updates once a week. If a newer version is available and
    CHECK_UPDATES is enabled, show a dialog asking if the user wants to update.
    If yes, perform git pull (preserves input/, output/, .env).
    """
    from config import APP_VERSION
    from utils.update_checker import (
        is_update_available,
        perform_git_pull,
        record_check_done,
        should_run_check,
    )
    from utils import get_logger

    logger = get_logger(__name__)

    if not should_run_check():
        logger.debug("Update check skipped (CHECK_UPDATES disabled or checked recently)")
        return

    latest = is_update_available(APP_VERSION)
    # Only record when no update found — if there is one, keep reminding every startup
    if not latest:
        record_check_done()

    if latest:
        logger.info("Update available: %s (current: %s)", latest, APP_VERSION)
    else:
        logger.debug("Update check done: no newer version (current: %s)", APP_VERSION)

    if not latest:
        return

    wants_update = messagebox.askyesno(
        "Update available",
        f"A new version ({latest}) is available. You have {APP_VERSION}.\n\n"
        "Do you want to update now? (input/, output/ and .env will be preserved)",
        default=messagebox.YES,
    )
    if not wants_update:
        return

    success, msg = perform_git_pull()
    if success:
        messagebox.showinfo("Update", msg)
    else:
        messagebox.showerror("Update failed", msg)


def main() -> None:
    """Initialize configuration, logging, and launch the main menu."""
    from config import APP_VERSION, get_output_dir, initialize_and_validate_config
    from utils import get_logger

    initialize_and_validate_config()
    logger = get_logger(__name__)
    logger.info("DifferentialLab starting (v%s)", APP_VERSION)

    _check_for_updates()

    get_output_dir()

    try:
        root = tk.Tk()

        from frontend import MainMenu

        _app = MainMenu(root)
        root.mainloop()
    except Exception:
        logger.critical("Unhandled exception", exc_info=True)
        raise
    finally:
        import matplotlib.pyplot as _plt
        _plt.close("all")
        logger.info("DifferentialLab shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
