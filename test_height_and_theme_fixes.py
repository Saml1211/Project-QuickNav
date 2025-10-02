#!/usr/bin/env python3
"""
Test script for height and theme fixes
"""

import sys
import os
import tkinter as tk
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_window_height():
    """Test that window height shows all components"""
    print("Testing window height...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create app instance
        app = ProjectQuickNavGUI()

        # Get window geometry
        geometry = app.root.geometry()
        print(f"Window geometry: {geometry}")

        # Parse window size
        size_part = geometry.split('+')[0] if '+' in geometry else geometry
        width, height = map(int, size_part.split('x'))

        print(f"Window size: {width}x{height}")
        print(f"Minimum height: {app.min_height}")

        # Check if height meets minimum requirements
        height_ok = height >= app.min_height
        print(f"Height adequate: {height_ok}")

        # Check screen dimensions
        screen_width = app.root.winfo_screenwidth()
        screen_height = app.root.winfo_screenheight()
        print(f"Screen resolution: {screen_width}x{screen_height}")

        # Calculate percentages
        width_pct = (width / screen_width) * 100
        height_pct = (height / screen_height) * 100

        print(f"Window uses {width_pct:.1f}% of screen width")
        print(f"Window uses {height_pct:.1f}% of screen height")

        # Test responsive geometry function
        responsive_geom = app._get_responsive_geometry()
        print(f"Responsive geometry: {responsive_geom}")

        # Clean up
        app.root.destroy()

        return height_ok

    except Exception as e:
        print(f"ERROR: Height test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_theme_improvements():
    """Test theme improvements"""
    print("\nTesting theme improvements...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        app = ProjectQuickNavGUI()

        # Test both themes
        themes = ["light", "dark"]
        theme_results = {}

        for theme_name in themes:
            print(f"  Testing {theme_name} theme...")

            # Switch to theme
            app.theme.set_theme(theme_name)
            app._apply_theme()

            # Get current theme
            current_theme = app.theme.get_current_theme()

            if current_theme:
                # Test button colors
                button_colors = current_theme.get_style("button")["colors"]
                if button_colors:
                    normal = button_colors.get("normal", {})
                    hover = button_colors.get("hover", {})

                    print(f"    Normal button: bg={normal.get('bg')}, fg={normal.get('fg')}")
                    print(f"    Hover button: bg={hover.get('bg')}, fg={hover.get('fg')}")

                # Test spacing
                geometry = current_theme.get_geometry("element")
                if geometry:
                    padding = geometry.get("padding", 5)
                    button_padding = geometry.get("button_padding", 8)
                    section_spacing = geometry.get("section_spacing", 12)

                    print(f"    Padding: {padding}, Button padding: {button_padding}, Section spacing: {section_spacing}")

                theme_results[theme_name] = True
            else:
                print(f"    ERROR: Could not get {theme_name} theme")
                theme_results[theme_name] = False

        # Clean up
        app.root.destroy()

        return all(theme_results.values())

    except Exception as e:
        print(f"ERROR: Theme test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_layout():
    """Test UI layout and spacing"""
    print("\nTesting UI layout...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        app = ProjectQuickNavGUI()

        # Update window to ensure all widgets are rendered
        app.root.update()

        # Check for main components
        components = [
            ('main_frame', app.main_frame),
            ('project_entry', app.project_entry),
            ('folder_radio', app.folder_radio),
            ('doc_radio', app.doc_radio),
            ('open_button', app.open_button),
            ('find_button', app.find_button),
            ('navigate_button', getattr(app, 'navigate_button', None))
        ]

        missing_components = []
        for name, component in components:
            if component is None:
                missing_components.append(name)
            else:
                print(f"  ✓ {name}: Present")

        if missing_components:
            print(f"  ✗ Missing components: {missing_components}")

        # Test frame spacing
        frames = [
            app.main_frame,
            getattr(app, 'folder_frame', None),
            getattr(app, 'doc_frame', None)
        ]

        visible_frames = [f for f in frames if f is not None]
        print(f"  Visible frames: {len(visible_frames)}")

        # Clean up
        app.root.destroy()

        return len(missing_components) == 0

    except Exception as e:
        print(f"ERROR: UI layout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Height and Theme Fix Tests ===")

    # Test results
    results = []

    # Run tests
    results.append(("Window Height", test_window_height()))
    results.append(("Theme Improvements", test_theme_improvements()))
    results.append(("UI Layout", test_ui_layout()))

    # Summary
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
        print("SUCCESS: All height and theme fixes working correctly!")
        return 0
    else:
        print("WARNING: Some tests failed - check output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())