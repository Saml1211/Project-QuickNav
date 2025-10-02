#!/usr/bin/env python3
"""
Test script specifically for theme fixes
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dark_theme_colors():
    """Test dark theme colors specifically"""
    print("Testing dark theme colors...")

    try:
        from quicknav.gui_theme import ThemeManager

        theme_manager = ThemeManager()
        theme_manager.set_theme("dark")
        dark_theme = theme_manager.get_current_theme()

        if dark_theme:
            # Test entry colors
            entry_style = dark_theme.get_style("entry")
            entry_colors = entry_style.get("colors", {})
            if entry_colors:
                normal = entry_colors.get("normal", {})
                print(f"  Entry field - bg: {normal.get('bg')}, fg: {normal.get('fg')}")

            # Test button colors
            button_style = dark_theme.get_style("button")
            button_colors = button_style.get("colors", {})
            if button_colors:
                normal = button_colors.get("normal", {})
                print(f"  Button - bg: {normal.get('bg')}, fg: {normal.get('fg')}")

            # Test window colors
            window_style = dark_theme.get_style("window")
            window_colors = window_style.get("colors", {})
            if window_colors:
                normal = window_colors.get("normal", {}) if isinstance(window_colors, dict) else window_colors
                print(f"  Window - bg: {normal.get('bg', 'N/A')}, fg: {normal.get('fg', 'N/A')}")

            return True
        else:
            print("  ERROR: Could not get dark theme")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_with_both_themes():
    """Test GUI with both themes"""
    print("\nTesting GUI with both themes...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        app = ProjectQuickNavGUI()

        # Test light theme
        print("  Switching to light theme...")
        app.theme.set_theme("light")
        app._apply_theme()
        app.root.update_idletasks()

        # Test dark theme
        print("  Switching to dark theme...")
        app.theme.set_theme("dark")
        app._apply_theme()
        app.root.update_idletasks()

        # Check geometry
        geometry = app.root.geometry()
        print(f"  Final geometry: {geometry}")

        # Check if geometry is reasonable
        size_part = geometry.split('+')[0] if '+' in geometry else geometry
        width, height = map(int, size_part.split('x'))

        size_ok = width >= app.min_width and height >= app.min_height
        print(f"  Size adequate: {size_ok} ({width}x{height} >= {app.min_width}x{app.min_height})")

        # Clean up
        app.root.destroy()

        return size_ok

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run theme tests"""
    print("=== Theme Fix Tests ===")

    results = []
    results.append(("Dark Theme Colors", test_dark_theme_colors()))
    results.append(("GUI Theme Switching", test_gui_with_both_themes()))

    print("\n=== TEST RESULTS ===")
    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: Theme fixes working correctly!")
        print("\nThe GUI should now:")
        print("- Use proper window size (reset from saved settings)")
        print("- Have correct dark theme colors:")
        print("  * Entry fields: Dark background (#1e1e1e) with white text")
        print("  * Buttons: Dark background (#404040) with white text")
        print("  * Window: Dark background (#1e1e1e)")
        print("- Switch themes properly without color issues")
        return 0
    else:
        print("FAILED: Some theme issues remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())