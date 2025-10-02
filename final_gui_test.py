#!/usr/bin/env python3
"""
Final test script for GUI fixes - height and theming
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the GUI to demonstrate all fixes"""
    print("Project QuickNav - Final GUI Test")
    print("=================================")
    print("Testing fixes:")
    print("- Window height now shows all components")
    print("- Improved light theme with better contrast")
    print("- Enhanced dark theme with consistent buttons")
    print("- Responsive sizing for 2K displays")
    print("- Proper spacing and padding")
    print()

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create and run the GUI
        app = ProjectQuickNavGUI()

        # Force proper geometry if needed
        if app.root.geometry() == "1x1+0+0":
            app.root.geometry(app._get_responsive_geometry())

        # Update and show info
        app.root.update_idletasks()
        geometry = app.root.geometry()

        print(f"Window geometry: {geometry}")
        print(f"Current theme: {app.theme.get_current_theme_name()}")
        print(f"DPI scale: {app.dpi_scale:.2f}")
        print()
        print("GUI Features:")
        print("- Use Ctrl+Alt+Q for global hotkey")
        print("- Toggle themes via View menu")
        print("- Try both folder and document modes")
        print("- All components should be visible")
        print()
        print("Press Ctrl+C to exit when done testing...")

        # Start the GUI
        app.run()

    except KeyboardInterrupt:
        print("\nTest completed.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())