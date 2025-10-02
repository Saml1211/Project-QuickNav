#!/usr/bin/env python3
"""
Demo script to show the fixed GUI in action
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the GUI to demonstrate fixes"""
    print("Project QuickNav - GUI Fixes Demo")
    print("==================================")
    print("Launching GUI with the following fixes:")
    print("✓ Responsive window sizing for 2K displays")
    print("✓ DPI-aware scaling")
    print("✓ Added missing Open/Navigate buttons")
    print("✓ Proper button layout and functionality")
    print("✓ Screen resolution adaptive layout")
    print()
    print("The GUI should now:")
    print("- Be appropriately sized for your 2K screen")
    print("- Have working navigation buttons")
    print("- Support both folder and document modes")
    print("- Scale properly with your display DPI")
    print()
    print("Press Ctrl+C to exit when done testing...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create and run the GUI
        app = ProjectQuickNavGUI()

        # Print some info about the GUI configuration
        print(f"Window size: {app.root.geometry()}")
        print(f"DPI scale factor: {app.dpi_scale:.2f}")
        print(f"Screen resolution: {app.root.winfo_screenwidth()}x{app.root.winfo_screenheight()}")
        print()

        # Start the GUI
        app.run()

    except KeyboardInterrupt:
        print("\nDemo ended by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())