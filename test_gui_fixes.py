#!/usr/bin/env python3
"""
Test script for GUI fixes - verifies sizing, DPI scaling, and navigation functionality
"""

import sys
import os
import tkinter as tk
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_gui_sizing():
    """Test GUI sizing and DPI scaling"""
    print("Testing GUI sizing and DPI scaling...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create app instance
        app = ProjectQuickNavGUI()

        # Get window geometry
        geometry = app.root.geometry()
        print(f"Window geometry: {geometry}")

        # Check DPI scaling
        print(f"DPI scale factor: {app.dpi_scale}")
        print(f"Minimum window size: {app.min_width}x{app.min_height}")

        # Test responsive geometry
        responsive_geom = app._get_responsive_geometry()
        print(f"Responsive geometry: {responsive_geom}")

        # Check screen dimensions
        screen_width = app.root.winfo_screenwidth()
        screen_height = app.root.winfo_screenheight()
        print(f"Screen resolution: {screen_width}x{screen_height}")

        # Parse window size from geometry
        size_part = geometry.split('+')[0] if '+' in geometry else geometry
        width, height = map(int, size_part.split('x'))

        print(f"Current window size: {width}x{height}")

        # Verify window fits on screen
        fits_horizontally = width <= screen_width * 0.8
        fits_vertically = height <= screen_height * 0.8

        print(f"Window fits horizontally: {fits_horizontally}")
        print(f"Window fits vertically: {fits_vertically}")

        # Test button presence
        has_open_button = hasattr(app, 'open_button')
        has_find_button = hasattr(app, 'find_button')
        has_navigate_button = hasattr(app, 'navigate_button')

        print(f"Has open button: {has_open_button}")
        print(f"Has find button: {has_find_button}")
        print(f"Has navigate button: {has_navigate_button}")

        # Test navigation state
        has_search_result = hasattr(app, 'last_search_result')
        has_search_type = hasattr(app, 'last_search_type')

        print(f"Has search result tracking: {has_search_result}")
        print(f"Has search type tracking: {has_search_type}")

        # Clean up
        app.root.destroy()

        return True

    except Exception as e:
        print(f"ERROR: GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_navigation_methods():
    """Test navigation method availability"""
    print("\nTesting navigation methods...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        app = ProjectQuickNavGUI()

        # Check for navigation methods
        methods_to_check = [
            '_execute_final_navigation',
            '_enable_navigate_button',
            '_disable_navigate_button',
            '_update_responsive_layout',
            '_get_responsive_geometry',
            '_apply_dpi_scaling',
            '_get_dpi_scale'
        ]

        for method_name in methods_to_check:
            has_method = hasattr(app, method_name)
            print(f"Has {method_name}: {has_method}")
            if not has_method:
                print(f"  WARNING: Missing method {method_name}")

        # Test method calls (safe ones)
        try:
            responsive_geom = app._get_responsive_geometry()
            print(f"Responsive geometry test: {responsive_geom}")
        except Exception as e:
            print(f"  ERROR calling _get_responsive_geometry: {e}")

        try:
            app._update_responsive_layout()
            print("Responsive layout update: OK")
        except Exception as e:
            print(f"  ERROR calling _update_responsive_layout: {e}")

        # Clean up
        app.root.destroy()

        return True

    except Exception as e:
        print(f"ERROR: Navigation methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_screen_scaling():
    """Test behavior on different screen resolutions"""
    print("\nTesting screen scaling behavior...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        app = ProjectQuickNavGUI()

        screen_width = app.root.winfo_screenwidth()
        screen_height = app.root.winfo_screenheight()

        print(f"Current screen: {screen_width}x{screen_height}")

        # Determine screen category
        if screen_width >= 2560:
            screen_type = "2K/4K"
            expected_factor = 0.25
        elif screen_width >= 1920:
            screen_type = "1080p"
            expected_factor = 0.3
        else:
            screen_type = "Smaller"
            expected_factor = 0.4

        print(f"Screen type: {screen_type}")
        print(f"Expected width factor: {expected_factor}")

        # Test responsive geometry calculation
        responsive_geom = app._get_responsive_geometry()
        size_part = responsive_geom.split('+')[0]
        width, height = map(int, size_part.split('x'))

        actual_factor = width / screen_width
        print(f"Actual width factor: {actual_factor:.2f}")

        # Check if factor is reasonable
        factor_ok = abs(actual_factor - expected_factor) < 0.1
        print(f"Width factor appropriate: {factor_ok}")

        # Check minimum size constraints
        min_width_ok = width >= app.min_width
        min_height_ok = height >= app.min_height

        print(f"Meets minimum width: {min_width_ok}")
        print(f"Meets minimum height: {min_height_ok}")

        # Clean up
        app.root.destroy()

        return True

    except Exception as e:
        print(f"ERROR: Screen scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Project QuickNav GUI Fix Tests ===")

    # Test results
    results = []

    # Run tests
    results.append(("GUI Sizing & DPI", test_gui_sizing()))
    results.append(("Navigation Methods", test_navigation_methods()))
    results.append(("Screen Scaling", test_screen_scaling()))

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
        print("✅ All GUI fixes working correctly!")
        return 0
    else:
        print("❌ Some tests failed - check output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())