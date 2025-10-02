#!/usr/bin/env python3
"""
Test script to demonstrate VISIBLE theme changes in the GUI
"""

import sys
import os
import tkinter as tk
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch GUI and demonstrate visible theme changes"""
    print("Testing VISIBLE Theme Changes")
    print("=============================")
    print()
    print("This test will:")
    print("1. Launch the GUI")
    print("2. Show current theme colors")
    print("3. Wait 3 seconds, then switch themes")
    print("4. You should see OBVIOUS color changes")
    print()
    print("WHAT TO LOOK FOR:")
    print("- Light theme: Light backgrounds, dark text")
    print("- Dark theme: Dark backgrounds (#2d2d30), white text")
    print("- Entry fields should change from white to dark (#1e1e1e)")
    print("- Buttons should change from light to dark (#404040)")
    print()

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create the GUI
        app = ProjectQuickNavGUI()

        # Get initial theme info
        current_theme = app.theme.get_current_theme_name()
        print(f"Starting with theme: {current_theme}")

        # Function to show theme info
        def show_theme_info():
            theme_obj = app.theme.get_current_theme()
            if theme_obj:
                print(f"\nCurrent theme: {app.theme.get_current_theme_name()}")

                # Show key colors
                entry_colors = theme_obj.get_style("entry")["colors"]
                if entry_colors:
                    normal = entry_colors.get("normal", {})
                    print(f"Entry field: bg={normal.get('bg')}, fg={normal.get('fg')}")

                button_colors = theme_obj.get_style("button")["colors"]
                if button_colors:
                    normal = button_colors.get("normal", {})
                    print(f"Button: bg={normal.get('bg')}, fg={normal.get('fg')}")

                window_bg = theme_obj.get_color("bg")
                window_fg = theme_obj.get_color("fg")
                print(f"Window: bg={window_bg}, fg={window_fg}")

        # Show initial theme
        show_theme_info()

        # Schedule theme switch after 3 seconds
        def switch_theme():
            print("\n" + "="*50)
            print("SWITCHING THEME NOW - WATCH FOR CHANGES!")
            print("="*50)

            app.toggle_theme()

            # Show new theme info
            app.root.after(500, show_theme_info)

            # Schedule another switch after 5 more seconds
            app.root.after(5000, switch_theme_back)

        def switch_theme_back():
            print("\n" + "="*50)
            print("SWITCHING BACK - WATCH FOR CHANGES!")
            print("="*50)

            app.toggle_theme()
            app.root.after(500, show_theme_info)

        # Schedule the first theme switch
        app.root.after(3000, switch_theme)

        print("\nGUI launched! Watch for theme changes in 3 seconds...")
        print("Close the window when done testing.")

        # Start the GUI
        app.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())