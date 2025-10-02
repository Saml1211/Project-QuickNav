#!/usr/bin/env python3
"""
Demo script showing the fixed GUI with proper sizing and theming
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the fixed GUI"""
    print("Project QuickNav - Fixed GUI Demo")
    print("=================================")
    print()
    print("FIXES APPLIED:")
    print("1. Window Size:")
    print("   - Proper sizing for 2K displays (640x720px)")
    print("   - All components visible without manual resizing")
    print("   - Window geometry reset (no longer remembers old size)")
    print()
    print("2. Dark Theme Fixes:")
    print("   - Entry fields: Dark background (#1e1e1e) with white text")
    print("   - Buttons: Dark background (#404040) with white text")
    print("   - Consistent theming throughout")
    print()
    print("3. Light Theme Improvements:")
    print("   - Better contrast and visual hierarchy")
    print("   - Improved button styling")
    print("   - Professional appearance")
    print()
    print("TESTING INSTRUCTIONS:")
    print("- The window should open at the correct size")
    print("- Try View > Toggle Theme to switch between light/dark")
    print("- All text should be readable in both themes")
    print("- Entry fields should have proper colors")
    print("- Buttons should show correct text colors")
    print()
    print("Press Ctrl+C to exit...")
    print()

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create and run the GUI
        app = ProjectQuickNavGUI()

        # Show initial info
        geometry = app.root.geometry()
        theme_name = app.theme.get_current_theme_name()

        print(f"Window opened at: {geometry}")
        print(f"Current theme: {theme_name}")
        print("GUI is ready for testing!")

        # Start the GUI
        app.run()

    except KeyboardInterrupt:
        print("\nDemo ended.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())