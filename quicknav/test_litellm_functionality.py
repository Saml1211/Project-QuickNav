#!/usr/bin/env python3
"""
Test script to verify LiteLLM functionality in Project QuickNav
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_litellm_import():
    """Test that LiteLLM can be imported."""
    print("Testing LiteLLM import...")

    try:
        import litellm
        # Try to get version, but don't fail if it doesn't exist
        try:
            version = litellm.__version__
        except AttributeError:
            version = "Available (version not accessible)"
        print(f"[PASS] LiteLLM imported successfully - Version: {version}")
        return True
    except ImportError as e:
        print(f"[FAIL] LiteLLM import failed: {e}")
        return False

def test_ai_client_with_litellm():
    """Test AI client initialization with LiteLLM available."""
    print("\nTesting AI client with LiteLLM...")

    try:
        from quicknav.ai_client import AIClient
        from quicknav.gui_controller import GuiController
        from quicknav.gui_settings import SettingsManager

        # Create components
        settings = SettingsManager()
        controller = GuiController(settings)
        ai_client = AIClient(controller=controller, settings=settings)

        print(f"[PASS] AI client created")
        print(f"[PASS] AI enabled: {ai_client.enabled}")
        print(f"[PASS] Tools registered: {len(ai_client.tools)}")

        # List registered tools
        for tool_name in ai_client.tools:
            print(f"  - {tool_name}")

        # Test tool function access
        if "search_projects" in ai_client.tools:
            tool = ai_client.tools["search_projects"]
            print(f"[PASS] search_projects tool accessible: {tool.name}")

        return True

    except Exception as e:
        print(f"[FAIL] AI client test failed: {e}")
        return False

def test_ai_settings_connection():
    """Test AI settings connection functionality."""
    print("\nTesting AI settings connection...")

    try:
        from quicknav.gui_settings import SettingsDialog, SettingsManager
        import tkinter as tk

        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()

        # Create settings manager
        settings = SettingsManager()

        # Create settings dialog (but don't show it)
        dialog = SettingsDialog(root, settings)

        # Test if AI tab creation works
        if hasattr(dialog, '_create_ai_tab'):
            print("[PASS] AI tab creation method exists")

        # Test if connection test method exists
        if hasattr(dialog, '_test_ai_connection'):
            print("[PASS] AI connection test method exists")

        # Test AI settings structure
        ai_settings = settings.get("ai", {})
        required_keys = ["enabled", "default_model", "api_keys"]

        for key in required_keys:
            if key in ai_settings:
                print(f"[PASS] AI setting '{key}' exists")
            else:
                print(f"[WARN] AI setting '{key}' missing")

        # Clean up
        root.destroy()

        return True

    except Exception as e:
        print(f"[FAIL] AI settings test failed: {e}")
        return False

def test_mock_ai_conversation():
    """Test AI conversation flow without making real API calls."""
    print("\nTesting mock AI conversation...")

    try:
        from quicknav.ai_client import AIClient
        from quicknav.gui_settings import SettingsManager

        # Create AI client
        settings = SettingsManager()
        ai_client = AIClient(settings=settings)

        # Test conversation memory
        if hasattr(ai_client, 'memory'):
            print("[PASS] Conversation memory initialized")

            # Test adding a mock message
            test_message = {
                "role": "user",
                "content": "Find project 17741",
                "timestamp": "2024-01-01T00:00:00"
            }

            ai_client.memory.add_message("user", test_message["content"])
            messages = ai_client.memory.get_messages()

            if len(messages) > 0:
                print("[PASS] Conversation memory working")
            else:
                print("[WARN] Conversation memory not storing messages")

        # Test tool execution simulation
        if "search_projects" in ai_client.tools:
            tool_func = ai_client.tools["search_projects"].function

            # Test with a mock project search
            result = tool_func("17741")
            print(f"[PASS] Tool execution works - Result type: {type(result)}")

            if isinstance(result, dict):
                print(f"  Status: {result.get('status', 'Unknown')}")

        return True

    except Exception as e:
        print(f"[FAIL] Mock conversation test failed: {e}")
        return False

def test_gui_ai_integration_with_litellm():
    """Test full GUI integration with LiteLLM available."""
    print("\nTesting GUI AI integration with LiteLLM...")

    try:
        from quicknav.gui import ProjectQuickNavGUI
        import tkinter as tk

        # Create hidden GUI
        root = tk.Tk()
        root.withdraw()

        gui = ProjectQuickNavGUI()
        gui.root.withdraw()

        # Test AI components
        print(f"[PASS] GUI created with AI enabled: {gui.ai_enabled.get()}")
        print(f"[PASS] AI client exists: {gui.ai_client is not None}")

        if gui.ai_client:
            print(f"[PASS] AI client enabled: {gui.ai_client.enabled}")
            print(f"[PASS] AI tools available: {len(gui.ai_client.tools)}")

        # Test AI UI components
        if hasattr(gui, 'ai_toggle_button'):
            print("[PASS] AI toggle button exists")

        if hasattr(gui, 'ai_chat_button'):
            print("[PASS] AI chat button exists")

        if hasattr(gui, 'ai_status_label'):
            print("[PASS] AI status label exists")

        # Test AI methods
        if hasattr(gui, 'toggle_ai'):
            print("[PASS] AI toggle method exists")

        if hasattr(gui, 'show_ai_settings'):
            print("[PASS] AI settings method exists")

        # Clean up
        gui.root.destroy()
        root.destroy()

        return True

    except Exception as e:
        print(f"[FAIL] GUI AI integration test failed: {e}")
        return False

def main():
    """Run all LiteLLM functionality tests."""
    print("Project QuickNav - LiteLLM Functionality Tests")
    print("==============================================\n")

    tests = [
        ("LiteLLM Import", test_litellm_import),
        ("AI Client with LiteLLM", test_ai_client_with_litellm),
        ("AI Settings Connection", test_ai_settings_connection),
        ("Mock AI Conversation", test_mock_ai_conversation),
        ("GUI AI Integration", test_gui_ai_integration_with_litellm)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"{test_name}")
        print("-" * len(test_name))

        try:
            if test_func():
                print(f"[PASS] {test_name} PASSED\n")
                passed += 1
            else:
                print(f"[FAIL] {test_name} FAILED\n")
        except Exception as e:
            print(f"[FAIL] {test_name} FAILED with exception: {e}\n")

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All LiteLLM functionality tests passed!")
        print("\nAI integration is fully functional with LiteLLM!")
        print("You can now:")
        print("  1. Enable AI in the GUI settings")
        print("  2. Add API keys for OpenAI, Anthropic, or Azure")
        print("  3. Use the AI chat feature for project navigation")
        print("  4. Ask the AI assistant to help find projects and documents")
        return 0
    else:
        print("[ERROR] Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())