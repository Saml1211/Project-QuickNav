#!/usr/bin/env python3
"""
Final demonstration of theme fixes with forced visual changes
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch GUI with enhanced theme demonstration"""
    print("PROJECT QUICKNAV - THEME FIXES DEMONSTRATION")
    print("=" * 50)
    print()
    print("FIXES APPLIED:")
    print("1. Window size reset (should open at 640x720 on 2K displays)")
    print("2. Dark theme fixed:")
    print("   - Entry fields: Dark bg (#1e1e1e) with white text")
    print("   - Buttons: Dark bg (#404040) with white text")
    print("   - Window: Dark bg (#1e1e1e)")
    print("3. Light theme improved:")
    print("   - Better contrast and professional styling")
    print("   - Improved button colors (#f6f6f6)")
    print("4. Theme switching now forces widget refresh")
    print()
    print("TESTING INSTRUCTIONS:")
    print("- The GUI should open with proper size and colors")
    print("- Use View menu > Toggle Theme to switch themes")
    print("- You should see IMMEDIATE and OBVIOUS color changes")
    print("- Text should be readable in both themes")
    print()
    print("If you don't see changes:")
    print("- Try toggling theme multiple times")
    print("- Close and reopen the GUI")
    print("- The theme system has been enhanced with forced refresh")
    print()

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create the GUI with enhanced theme handling
        app = ProjectQuickNavGUI()

        # Show current state
        geometry = app.root.geometry()
        current_theme = app.theme.get_current_theme_name()

        print(f"GUI opened:")
        print(f"  Size: {geometry}")
        print(f"  Theme: {current_theme}")

        # Get theme colors to verify they're correct
        theme_obj = app.theme.get_current_theme()
        if theme_obj:
            entry_colors = theme_obj.get_style("entry")["colors"]
            button_colors = theme_obj.get_style("button")["colors"]

            if entry_colors and button_colors:
                entry_normal = entry_colors.get("normal", {})
                button_normal = button_colors.get("normal", {})

                print(f"  Entry colors: bg={entry_normal.get('bg')}, fg={entry_normal.get('fg')}")
                print(f"  Button colors: bg={button_normal.get('bg')}, fg={button_normal.get('fg')}")

                # Verify dark theme colors
                if current_theme == "dark":
                    entry_bg = entry_normal.get('bg')
                    button_bg = button_normal.get('bg')

                    if entry_bg == "#1e1e1e" and button_bg == "#404040":
                        print("  ✓ Dark theme colors are CORRECT")
                    else:
                        print(f"  ⚠ Dark theme colors may be wrong: entry={entry_bg}, button={button_bg}")

        print()
        print("GUI is ready! Try toggling themes with View menu.")
        print("Close window when done testing.")

        # Start the GUI
        app.run()

    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())