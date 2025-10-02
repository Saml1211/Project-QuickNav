#!/usr/bin/env python3
"""
Test script for radio button and checkbox visibility fixes
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Test radio button and checkbox fixes"""
    print("Radio Button & Checkbox Visibility Test")
    print("=======================================")
    print()
    print("FIXES APPLIED:")
    print("1. Enhanced radio button styling with proper colors")
    print("2. Enhanced checkbox styling with proper colors")
    print("3. Added hover states for better visibility")
    print("4. Fixed text visibility on hover")
    print()
    print("DARK THEME COLORS:")
    print("- Background: #2d2d30 (dark gray)")
    print("- Text: #ffffff (white)")
    print("- Hover: #3c3c3c (lighter gray)")
    print("- Selected indicator: #3399ff (blue)")
    print()
    print("TESTING:")
    print("- All radio button text should be visible")
    print("- Hover states should show lighter background")
    print("- Selected options should show blue indicators")
    print("- No invisible text on hover")
    print()

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create the GUI
        app = ProjectQuickNavGUI()

        # Show theme info
        current_theme = app.theme.get_current_theme_name()
        print(f"GUI opened with theme: {current_theme}")

        # Get radio button colors
        theme_obj = app.theme.get_current_theme()
        if theme_obj:
            radio_style = theme_obj.get_style("radiobutton")
            radio_colors = radio_style.get("colors", {})

            if radio_colors:
                normal = radio_colors.get("normal", {})
                hover = radio_colors.get("hover", {})
                selected = radio_colors.get("selected", {})

                print(f"\nRadio button colors:")
                print(f"  Normal: bg={normal.get('bg')}, fg={normal.get('fg')}")
                print(f"  Hover: bg={hover.get('bg')}, fg={hover.get('fg')}")
                print(f"  Selected: indicator={selected.get('indicator')}")

            check_style = theme_obj.get_style("checkbutton")
            check_colors = check_style.get("colors", {})

            if check_colors:
                normal = check_colors.get("normal", {})
                hover = check_colors.get("hover", {})
                selected = check_colors.get("selected", {})

                print(f"\nCheckbox colors:")
                print(f"  Normal: bg={normal.get('bg')}, fg={normal.get('fg')}")
                print(f"  Hover: bg={hover.get('bg')}, fg={hover.get('fg')}")
                print(f"  Selected: indicator={selected.get('indicator')}")

        print(f"\nTest Instructions:")
        print(f"1. Try hovering over radio buttons - text should remain visible")
        print(f"2. Click different options - selected indicator should be blue")
        print(f"3. Hover over checkboxes - same visibility test")
        print(f"4. Toggle themes to test both light and dark modes")
        print(f"\nClose window when testing is complete.")

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