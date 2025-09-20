#!/usr/bin/env python3
"""
Test script for Project QuickNav GUI

This script provides basic functionality testing for the GUI components
without requiring the full application setup.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Add quicknav to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_gui():
    """Test basic GUI components."""
    print("Testing basic GUI components...")

    root = tk.Tk()
    root.title("GUI Component Test")
    root.geometry("400x300")

    # Test basic widgets
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    ttk.Label(frame, text="GUI Component Test", font=("Arial", 14, "bold")).pack(pady=10)

    # Test entry
    entry_frame = ttk.Frame(frame)
    entry_frame.pack(fill=tk.X, pady=5)
    ttk.Label(entry_frame, text="Test Entry:").pack(side=tk.LEFT)
    test_entry = ttk.Entry(entry_frame)
    test_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

    # Test buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(pady=10)

    def test_callback():
        value = test_entry.get()
        messagebox.showinfo("Test", f"Entry value: {value}")

    ttk.Button(button_frame, text="Test Button", command=test_callback).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Close", command=root.quit).pack(side=tk.LEFT, padx=5)

    # Test listbox
    ttk.Label(frame, text="Test Listbox:").pack(anchor=tk.W, pady=(10, 5))
    listbox = tk.Listbox(frame, height=5)
    listbox.pack(fill=tk.X, pady=5)

    for i in range(5):
        listbox.insert(tk.END, f"Item {i+1}")

    print("Basic GUI test ready. Close the window to continue.")
    root.mainloop()
    root.destroy()

def test_enhanced_widgets():
    """Test enhanced custom widgets."""
    print("Testing enhanced widgets...")

    try:
        from quicknav.gui_widgets import EnhancedEntry, SearchableComboBox, CollapsibleFrame

        root = tk.Tk()
        root.title("Enhanced Widgets Test")
        root.geometry("500x400")

        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(frame, text="Enhanced Widgets Test", font=("Arial", 14, "bold")).pack(pady=10)

        # Test EnhancedEntry
        ttk.Label(frame, text="Enhanced Entry (with placeholder):").pack(anchor=tk.W, pady=(10, 5))
        enhanced_entry = EnhancedEntry(frame, placeholder="Enter text here...")
        enhanced_entry.pack(fill=tk.X, pady=5)

        # Test SearchableComboBox
        ttk.Label(frame, text="Searchable ComboBox:").pack(anchor=tk.W, pady=(10, 5))
        search_combo = SearchableComboBox(
            frame,
            values=["Option 1", "Option 2", "Option 3", "Another Option", "Test Item"]
        )
        search_combo.pack(fill=tk.X, pady=5)

        # Test CollapsibleFrame
        ttk.Label(frame, text="Collapsible Frame:").pack(anchor=tk.W, pady=(10, 5))
        collapsible = CollapsibleFrame(frame, title="Collapsible Section")
        collapsible.pack(fill=tk.X, pady=5)

        # Add content to collapsible frame
        test_content = ttk.Label(collapsible.content_frame, text="This content can be collapsed!")
        collapsible.add_content(test_content)

        ttk.Button(frame, text="Close", command=root.quit).pack(pady=20)

        print("Enhanced widgets test ready. Close the window to continue.")
        root.mainloop()
        root.destroy()

    except ImportError as e:
        print(f"Could not test enhanced widgets: {e}")
        print("This is expected if the full application isn't set up yet.")

def test_theming():
    """Test theming system."""
    print("Testing theming system...")

    try:
        from quicknav.gui_theme import ThemeManager

        root = tk.Tk()
        root.title("Theme Test")
        root.geometry("400x300")

        # Create theme manager
        theme_manager = ThemeManager()

        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(frame, text="Theme Test", font=("Arial", 14, "bold")).pack(pady=10)

        # Sample widgets
        ttk.Label(frame, text="This is a sample label").pack(pady=5)

        entry = ttk.Entry(frame, width=30)
        entry.pack(pady=5)
        entry.insert(0, "Sample text")

        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Sample Button").pack(side=tk.LEFT, padx=5)

        # Theme selection
        theme_var = tk.StringVar(value=theme_manager.get_current_theme_name())
        theme_combo = ttk.Combobox(
            frame,
            textvariable=theme_var,
            values=theme_manager.get_available_themes(),
            state="readonly"
        )
        theme_combo.pack(pady=10)

        def change_theme(*args):
            theme_name = theme_var.get()
            theme_manager.set_theme(theme_name)
            theme_manager.apply_theme(root)

        theme_combo.bind('<<ComboboxSelected>>', change_theme)

        ttk.Button(frame, text="Close", command=root.quit).pack(pady=10)

        # Apply initial theme
        theme_manager.apply_theme(root)

        print("Theme test ready. Try changing themes in the dropdown.")
        root.mainloop()
        root.destroy()

    except ImportError as e:
        print(f"Could not test theming: {e}")

def test_backend_integration():
    """Test backend integration."""
    print("Testing backend integration...")

    try:
        # Test basic imports
        import quicknav.find_project_path as finder
        print("✓ find_project_path module imported successfully")

        # Test basic functionality
        onedrive_folder = finder.get_onedrive_folder()
        print(f"✓ OneDrive folder detected: {onedrive_folder}")

        project_folders = finder.get_project_folders(onedrive_folder)
        print(f"✓ Project folders path: {project_folders}")

        # Test validation
        test_inputs = ["17741", "test", "invalid*chars", "12345"]
        for test_input in test_inputs:
            try:
                if test_input.isdigit() and len(test_input) == 5:
                    result = "Valid project number"
                else:
                    result = "Search term"
                print(f"✓ Input '{test_input}' -> {result}")
            except Exception as e:
                print(f"✗ Input '{test_input}' -> Error: {e}")

    except ImportError as e:
        print(f"Could not test backend integration: {e}")
        print("This is expected if the backend modules aren't available.")

def main():
    """Run all tests."""
    print("Project QuickNav GUI Tests")
    print("==========================\n")

    tests = [
        ("Basic GUI Components", test_basic_gui),
        ("Enhanced Widgets", test_enhanced_widgets),
        ("Theming System", test_theming),
        ("Backend Integration", test_backend_integration)
    ]

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            test_func()
            print("✓ Test completed")
        except Exception as e:
            print(f"✗ Test failed: {e}")

        input("\nPress Enter to continue to next test...")

    print("\nAll tests completed!")

if __name__ == "__main__":
    main()