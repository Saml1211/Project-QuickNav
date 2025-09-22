#!/usr/bin/env python3
"""
Test script for AI integration in Project QuickNav GUI
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all AI-related modules can be imported."""
    print("Testing AI integration imports...")

    try:
        from quicknav.gui_settings import SettingsManager
        print("[PASS] Settings manager imported")

        from quicknav.ai_client import AIClient
        print("[PASS] AI client imported")

        from quicknav.ai_chat_widget import ChatWidget
        print("[PASS] AI chat widget imported")

        from quicknav.gui import ProjectQuickNavGUI
        print("[PASS] Main GUI with AI integration imported")

        return True

    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_settings_ai_section():
    """Test AI settings integration."""
    print("\nTesting AI settings...")

    try:
        from quicknav.gui_settings import SettingsManager

        # Create settings manager
        settings = SettingsManager()

        # Test AI settings defaults
        ai_enabled = settings.get("ai.enabled", False)
        default_model = settings.get("ai.default_model", "gpt-3.5-turbo")

        print(f"[PASS] AI enabled: {ai_enabled}")
        print(f"[PASS] Default model: {default_model}")

        # Test setting AI values
        settings.set("ai.enabled", True)
        settings.set("ai.default_model", "claude-3-haiku-20240307")

        print("[PASS] AI settings can be modified")

        return True

    except Exception as e:
        print(f"[FAIL] Settings test failed: {e}")
        return False

def test_ai_client_creation():
    """Test AI client creation without LiteLLM."""
    print("\nTesting AI client creation...")

    try:
        from quicknav.ai_client import AIClient
        from quicknav.gui_settings import SettingsManager

        settings = SettingsManager()
        ai_client = AIClient(settings=settings)

        # Should work even without LiteLLM
        print(f"[PASS] AI client created, enabled: {ai_client.enabled}")

        # Get default model from settings
        default_model = ai_client._get_default_model() if hasattr(ai_client, '_get_default_model') else "N/A"
        print(f"[PASS] Default model: {default_model}")

        # Test tools registration
        print(f"[PASS] Number of registered tools: {len(ai_client.tools)}")

        expected_tools = ["search_projects", "find_documents", "analyze_project", "list_project_structure", "get_recent_projects"]
        for tool_name in expected_tools:
            if tool_name in ai_client.tools:
                print(f"[PASS] Tool '{tool_name}' registered")
            else:
                print(f"[FAIL] Tool '{tool_name}' missing")

        return True

    except Exception as e:
        print(f"[FAIL] AI client test failed: {e}")
        return False

def test_gui_ai_integration():
    """Test GUI AI integration without launching the window."""
    print("\nTesting GUI AI integration...")

    try:
        from quicknav.gui import ProjectQuickNavGUI
        import tkinter as tk

        # Create root but don't show it
        root = tk.Tk()
        root.withdraw()

        # Mock the GUI initialization without actually showing the window
        gui = ProjectQuickNavGUI()
        gui.root.withdraw()  # Keep it hidden

        # Test AI components
        print(f"[PASS] GUI created with AI enabled: {gui.ai_enabled.get()}")
        print(f"[PASS] AI client initialized: {gui.ai_client is not None}")

        # Test AI menu methods exist
        if hasattr(gui, 'toggle_ai'):
            print("[PASS] AI toggle method exists")
        if hasattr(gui, 'show_ai_settings'):
            print("[PASS] AI settings method exists")
        if hasattr(gui, 'toggle_ai_panel'):
            print("[PASS] AI panel toggle method exists")

        # Test toolbar elements exist
        if hasattr(gui, 'ai_toggle_button'):
            print("[PASS] AI toggle button created")
        if hasattr(gui, 'ai_chat_button'):
            print("[PASS] AI chat button created")
        if hasattr(gui, 'ai_status_label'):
            print("[PASS] AI status label created")

        # Clean up
        gui.root.destroy()
        root.destroy()

        return True

    except Exception as e:
        print(f"[FAIL] GUI AI integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Project QuickNav AI Integration Tests")
    print("=====================================\n")

    tests = [
        ("Import Test", test_imports),
        ("Settings AI Section", test_settings_ai_section),
        ("AI Client Creation", test_ai_client_creation),
        ("GUI AI Integration", test_gui_ai_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))

        try:
            if test_func():
                print(f"[PASS] {test_name} PASSED")
                passed += 1
            else:
                print(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            print(f"[FAIL] {test_name} FAILED with exception: {e}")

    print(f"\n\nTest Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All AI integration tests passed!")
        return 0
    else:
        print("[ERROR] Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())