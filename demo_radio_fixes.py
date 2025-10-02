#!/usr/bin/env python3
"""
Demo script showing the fixed radio button and checkbox visibility
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch GUI to demonstrate radio button fixes"""
    print("RADIO BUTTON & CHECKBOX FIXES - DEMONSTRATION")
    print("=" * 50)
    print()
    print("✅ ISSUES FIXED:")
    print("1. Radio button text visible on hover")
    print("2. Checkbox text visible on hover")
    print("3. Proper dark theme colors")
    print("4. Better hover states")
    print()
    print("🎨 DARK THEME STYLING:")
    print("- Normal: Dark background (#2d2d30) with white text")
    print("- Hover: Lighter background (#3c3c3c) with white text")
    print("- Selected: Blue indicator (#3399ff)")
    print()
    print("🧪 WHAT TO TEST:")
    print("- Hover over 'Open Project Folder' and 'Find Documents'")
    print("- Hover over subfolder options (System Designs, etc.)")
    print("- Hover over checkboxes (Show Debug Output, etc.)")
    print("- Text should ALWAYS be visible")
    print("- No more invisible text on hover!")
    print()
    print("Press Ctrl+C to exit when done testing...")

    try:
        from quicknav.gui import ProjectQuickNavGUI

        # Create and run the GUI
        app = ProjectQuickNavGUI()

        print(f"GUI launched successfully!")
        print(f"Current theme: {app.theme.get_current_theme_name()}")
        print("Test the radio buttons and checkboxes now!")

        # Start the GUI
        app.run()

    except KeyboardInterrupt:
        print("\nTesting completed.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())