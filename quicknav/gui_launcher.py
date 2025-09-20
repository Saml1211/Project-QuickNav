#!/usr/bin/env python3
"""
Project QuickNav GUI Launcher

Simple launcher script for the Tkinter GUI application.
Handles initialization, error recovery, and provides a clean entry point.
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path.home() / '.quicknav' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'gui.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []

    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")

    # Check optional dependencies
    optional_deps = []

    try:
        import keyboard
    except ImportError:
        optional_deps.append("keyboard (for global hotkeys)")

    try:
        import pystray
    except ImportError:
        optional_deps.append("pystray (for system tray)")

    if missing_deps:
        print("Error: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install the missing dependencies and try again.")
        return False

    if optional_deps:
        print("Warning: Optional dependencies not available:")
        for dep in optional_deps:
            print(f"  - {dep}")
        print("Some features may not be available.\n")

    return True

def main():
    """Main entry point for the GUI launcher."""
    print("Project QuickNav - Enhanced GUI")
    print("================================")

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)

        print("Starting GUI application...")
        logger.info("Starting Project QuickNav GUI")

        # Import and run the GUI
        from quicknav.gui import ProjectQuickNavGUI

        app = ProjectQuickNavGUI()
        app.run()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        logger.info("Application interrupted by user")

    except ImportError as e:
        error_msg = f"Import error: {e}"
        print(f"Error: {error_msg}")
        logger.error(error_msg)

        if "quicknav" in str(e):
            print("\nTrying to import from local directory...")
            try:
                # Try importing from current directory
                sys.path.insert(0, str(Path(__file__).parent))
                from gui import ProjectQuickNavGUI

                app = ProjectQuickNavGUI()
                app.run()

            except Exception as local_e:
                print(f"Local import also failed: {local_e}")
                logger.error(f"Local import failed: {local_e}")
                print("\nPlease ensure you're running from the correct directory.")
                sys.exit(1)
        else:
            sys.exit(1)

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"Error: {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        print("\nPlease check the log file for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()