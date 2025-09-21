#!/usr/bin/env python3
"""
Test script to demonstrate the modern styling improvements in the settings dialog.

This script creates a temporary settings dialog to showcase the enhanced UI.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the quicknav module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from quicknav.gui_settings import SettingsManager, SettingsDialog
    from quicknav.gui_theme import ThemeManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def test_settings_dialog():
    """Test the modernized settings dialog."""
    print("Testing modernized settings dialog...")

    # Create root window
    root = tk.Tk()
    root.title("Settings Dialog Test")
    root.geometry("400x300")

    # Create settings manager
    settings = SettingsManager()

    # Apply modern font to root
    try:
        root.option_add('*Font', 'Arial 9')
    except:
        pass

    # Create test button
    test_frame = ttk.Frame(root)
    test_frame.pack(expand=True, fill='both', padx=20, pady=20)

    title_label = ttk.Label(
        test_frame,
        text="QuickNav Settings Dialog Test",
        font=("Arial", 14, "bold")
    )
    title_label.pack(pady=(0, 20))

    info_label = ttk.Label(
        test_frame,
        text="Click the button below to open the modernized settings dialog\nwith enhanced styling, icons, and improved layout.",
        justify=tk.CENTER,
        wraplength=350
    )
    info_label.pack(pady=(0, 30))

    def open_settings():
        dialog = SettingsDialog(root, settings)
        dialog.show()

    # Styled button
    settings_button = ttk.Button(
        test_frame,
        text="Open Settings Dialog",
        command=open_settings,
        width=25
    )
    settings_button.pack(pady=10)

    improvements_text = """Modern UI Improvements Applied:

- Enhanced typography (Segoe UI font family)
- Icons throughout the interface for better visual hierarchy
- Consistent 16px padding and 24px margins
- Card-style LabelFrames with proper spacing
- Grid-based layouts for better alignment
- Enhanced button styling with emojis
- Improved color schemes for all themes
- Better visual organization of settings sections
- Consistent spacing (12px, 16px, 20px system)
- Modern dialog sizing and responsiveness"""

    improvements_label = ttk.Label(
        test_frame,
        text=improvements_text,
        justify=tk.LEFT,
        font=("Arial", 8)
    )
    improvements_label.pack(pady=(20, 0), anchor="w")

    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    print("Settings dialog test window created successfully")
    print("Click 'Open Settings Dialog' to see the modern styling")

    root.mainloop()


if __name__ == "__main__":
    test_settings_dialog()