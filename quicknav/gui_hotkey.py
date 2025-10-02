"""
Global Hotkey Support for Project QuickNav

This module provides cross-platform global hotkey registration and handling.
It attempts to use platform-specific libraries for optimal performance and
falls back to polling methods when necessary.

Supported platforms:
- Windows: Uses win32 API
- macOS: Uses Cocoa/AppKit
- Linux: Uses X11 with fallback to polling

Features:
- Global hotkey registration
- System tray integration
- Hotkey conflict detection
- Graceful fallbacks
"""

import os
import sys
import threading
import time
import logging
from typing import Dict, Callable, Optional, List, Tuple
import tkinter as tk
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class HotkeyHandler(ABC):
    """Abstract base class for hotkey handlers."""

    @abstractmethod
    def register_hotkey(self, hotkey: str, callback: Callable) -> bool:
        """Register a global hotkey."""
        pass

    @abstractmethod
    def unregister_hotkey(self, hotkey: str) -> bool:
        """Unregister a global hotkey."""
        pass

    @abstractmethod
    def start(self):
        """Start the hotkey handler."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the hotkey handler."""
        pass


class WindowsHotkeyHandler(HotkeyHandler):
    """Windows-specific hotkey handler using win32 API."""

    def __init__(self):
        self.hotkeys = {}
        self.running = False
        self.thread = None
        self._next_id = 1

        try:
            import win32api
            import win32con
            import win32gui
            self.win32api = win32api
            self.win32con = win32con
            self.win32gui = win32gui
            self.available = True
            logger.debug("Windows hotkey handler initialized with win32 API")
        except ImportError:
            logger.warning("win32 API not available, Windows hotkeys disabled")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Windows hotkey handler: {e}")
            self.available = False

    def register_hotkey(self, hotkey: str, callback: Callable) -> bool:
        """Register a global hotkey on Windows."""
        if not self.available:
            return False

        try:
            modifiers, key_code = self._parse_hotkey(hotkey)
            hotkey_id = self._next_id
            self._next_id += 1

            # Register the hotkey
            result = self.win32gui.RegisterHotKey(
                None, hotkey_id, modifiers, key_code
            )

            if result:
                self.hotkeys[hotkey] = {
                    'id': hotkey_id,
                    'callback': callback,
                    'modifiers': modifiers,
                    'key_code': key_code
                }
                logger.info(f"Registered Windows hotkey: {hotkey}")
                return True
            else:
                logger.error(f"Failed to register Windows hotkey: {hotkey}")
                return False

        except Exception as e:
            logger.error(f"Error registering Windows hotkey {hotkey}: {e}")
            return False

    def unregister_hotkey(self, hotkey: str) -> bool:
        """Unregister a global hotkey on Windows."""
        if not self.available or hotkey not in self.hotkeys:
            return False

        try:
            hotkey_info = self.hotkeys[hotkey]
            result = self.win32gui.UnregisterHotKey(None, hotkey_info['id'])

            if result:
                del self.hotkeys[hotkey]
                logger.info(f"Unregistered Windows hotkey: {hotkey}")
                return True
            else:
                logger.error(f"Failed to unregister Windows hotkey: {hotkey}")
                return False

        except Exception as e:
            logger.error(f"Error unregistering Windows hotkey {hotkey}: {e}")
            return False

    def start(self):
        """Start the Windows hotkey message loop."""
        if not self.available or self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._message_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the Windows hotkey message loop."""
        if not self.running:
            return

        self.running = False

        # Unregister all hotkeys
        for hotkey in list(self.hotkeys.keys()):
            self.unregister_hotkey(hotkey)

        if self.thread:
            self.thread.join(timeout=1.0)

    def _message_loop(self):
        """Windows message loop for hotkey events."""
        try:
            while self.running:
                msg = self.win32gui.PeekMessage(None, 0, 0, 1)  # PM_REMOVE
                if msg:
                    message = msg[1]
                    if message == self.win32con.WM_HOTKEY:
                        hotkey_id = msg[3]  # wParam
                        self._handle_hotkey(hotkey_id)

                time.sleep(0.01)  # Small delay to prevent excessive CPU usage

        except Exception as e:
            logger.error(f"Error in Windows hotkey message loop: {e}")

    def _handle_hotkey(self, hotkey_id: int):
        """Handle a hotkey event."""
        for hotkey, info in self.hotkeys.items():
            if info['id'] == hotkey_id:
                try:
                    info['callback']()
                except Exception as e:
                    logger.error(f"Error in hotkey callback for {hotkey}: {e}")
                break

    def _parse_hotkey(self, hotkey: str) -> Tuple[int, int]:
        """Parse hotkey string into Windows modifiers and key code."""
        parts = [part.strip().lower() for part in hotkey.split('+')]
        modifiers = 0
        key = None

        for part in parts:
            if part in ['ctrl', 'control']:
                modifiers |= self.win32con.MOD_CONTROL
            elif part in ['alt']:
                modifiers |= self.win32con.MOD_ALT
            elif part in ['shift']:
                modifiers |= self.win32con.MOD_SHIFT
            elif part in ['win', 'windows']:
                modifiers |= self.win32con.MOD_WIN
            else:
                key = part

        if not key:
            raise ValueError("No key specified in hotkey")

        # Convert key to virtual key code
        key_code = self._get_virtual_key_code(key)
        return modifiers, key_code

    def _get_virtual_key_code(self, key: str) -> int:
        """Get virtual key code for a key string."""
        # Common key mappings
        key_map = {
            'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
            'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
            'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
            'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
            'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59,
            'z': 0x5A,
            '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
            '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
            'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
            'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
            'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
            'space': 0x20, 'return': 0x0D, 'enter': 0x0D,
            'escape': 0x1B, 'esc': 0x1B, 'tab': 0x09,
            'backspace': 0x08, 'delete': 0x2E, 'insert': 0x2D,
            'home': 0x24, 'end': 0x23, 'pageup': 0x21, 'pagedown': 0x22,
            'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
            'comma': 0xBC, 'period': 0xBE, 'slash': 0xBF,
            'semicolon': 0xBA, 'quote': 0xDE, 'backslash': 0xDC,
            'minus': 0xBD, 'equal': 0xBB
        }

        if key in key_map:
            return key_map[key]
        elif len(key) == 1 and key.isalpha():
            return ord(key.upper())
        else:
            raise ValueError(f"Unknown key: {key}")


class MacOSHotkeyHandler(HotkeyHandler):
    """macOS-specific hotkey handler using Cocoa."""

    def __init__(self):
        self.hotkeys = {}
        self.available = False

        try:
            # Try to import macOS-specific modules
            import Cocoa
            import Carbon
            self.Cocoa = Cocoa
            self.Carbon = Carbon
            self.available = True
        except ImportError:
            logger.warning("Cocoa/Carbon not available, macOS hotkeys disabled")

    def register_hotkey(self, hotkey: str, callback: Callable) -> bool:
        """Register a global hotkey on macOS."""
        if not self.available:
            return False

        try:
            # This is a simplified implementation
            # A full implementation would use Carbon Event Manager
            logger.info(f"macOS hotkey registration not fully implemented: {hotkey}")
            return False

        except Exception as e:
            logger.error(f"Error registering macOS hotkey {hotkey}: {e}")
            return False

    def unregister_hotkey(self, hotkey: str) -> bool:
        """Unregister a global hotkey on macOS."""
        return False

    def start(self):
        """Start the macOS hotkey handler."""
        pass

    def stop(self):
        """Stop the macOS hotkey handler."""
        pass


class LinuxHotkeyHandler(HotkeyHandler):
    """Linux-specific hotkey handler using X11."""

    def __init__(self):
        self.hotkeys = {}
        self.available = False

        try:
            # Try to import X11 modules
            from Xlib import display, X
            from Xlib.ext import record
            self.display = display
            self.X = X
            self.record = record
            self.available = True
        except ImportError:
            logger.warning("Xlib not available, Linux hotkeys disabled")

    def register_hotkey(self, hotkey: str, callback: Callable) -> bool:
        """Register a global hotkey on Linux."""
        if not self.available:
            return False

        try:
            # This is a simplified implementation
            # A full implementation would use XGrabKey
            logger.info(f"Linux hotkey registration not fully implemented: {hotkey}")
            return False

        except Exception as e:
            logger.error(f"Error registering Linux hotkey {hotkey}: {e}")
            return False

    def unregister_hotkey(self, hotkey: str) -> bool:
        """Unregister a global hotkey on Linux."""
        return False

    def start(self):
        """Start the Linux hotkey handler."""
        pass

    def stop(self):
        """Stop the Linux hotkey handler."""
        pass


class FallbackHotkeyHandler(HotkeyHandler):
    """Fallback hotkey handler using keyboard polling."""

    def __init__(self):
        self.hotkeys = {}
        self.running = False
        self.thread = None
        self.available = False

        try:
            import keyboard
            self.keyboard = keyboard
            self.available = True
        except ImportError:
            logger.warning("keyboard module not available, hotkey support disabled")

    def register_hotkey(self, hotkey: str, callback: Callable) -> bool:
        """Register a hotkey using keyboard module."""
        if not self.available:
            return False

        try:
            # Convert hotkey format
            keyboard_hotkey = self._convert_hotkey_format(hotkey)
            self.keyboard.add_hotkey(keyboard_hotkey, callback)
            self.hotkeys[hotkey] = callback
            logger.info(f"Registered fallback hotkey: {hotkey}")
            return True

        except Exception as e:
            logger.error(f"Error registering fallback hotkey {hotkey}: {e}")
            return False

    def unregister_hotkey(self, hotkey: str) -> bool:
        """Unregister a hotkey using keyboard module."""
        if not self.available or hotkey not in self.hotkeys:
            return False

        try:
            keyboard_hotkey = self._convert_hotkey_format(hotkey)
            self.keyboard.remove_hotkey(keyboard_hotkey)
            del self.hotkeys[hotkey]
            logger.info(f"Unregistered fallback hotkey: {hotkey}")
            return True

        except Exception as e:
            logger.error(f"Error unregistering fallback hotkey {hotkey}: {e}")
            return False

    def start(self):
        """Start the fallback hotkey handler."""
        # keyboard module starts automatically
        self.running = True

    def stop(self):
        """Stop the fallback hotkey handler."""
        if not self.running:
            return

        # Unregister all hotkeys
        for hotkey in list(self.hotkeys.keys()):
            self.unregister_hotkey(hotkey)

        self.running = False

    def _convert_hotkey_format(self, hotkey: str) -> str:
        """Convert our hotkey format to keyboard module format."""
        # keyboard module uses '+' for combinations
        return hotkey.replace('ctrl', 'ctrl').replace('alt', 'alt')


class HotkeyManager:
    """Cross-platform hotkey manager."""

    def __init__(self):
        self.handler = self._get_platform_handler()
        self.registered_hotkeys = {}

    def _get_platform_handler(self) -> HotkeyHandler:
        """Get the appropriate hotkey handler for the current platform."""
        if sys.platform == 'win32':
            handler = WindowsHotkeyHandler()
            if handler.available:
                return handler

        elif sys.platform == 'darwin':
            handler = MacOSHotkeyHandler()
            if handler.available:
                return handler

        elif sys.platform.startswith('linux'):
            handler = LinuxHotkeyHandler()
            if handler.available:
                return handler

        # Fallback to keyboard module
        return FallbackHotkeyHandler()

    def register_hotkey(self, hotkey: str, callback: Callable) -> bool:
        """
        Register a global hotkey.

        Args:
            hotkey: Hotkey string (e.g., "ctrl+alt+q")
            callback: Function to call when hotkey is pressed

        Returns:
            True if successful, False otherwise
        """
        if hotkey in self.registered_hotkeys:
            logger.warning(f"Hotkey {hotkey} already registered")
            return False

        if self.handler.register_hotkey(hotkey, callback):
            self.registered_hotkeys[hotkey] = callback
            return True

        return False

    def unregister_hotkey(self, hotkey: str) -> bool:
        """
        Unregister a global hotkey.

        Args:
            hotkey: Hotkey string to unregister

        Returns:
            True if successful, False otherwise
        """
        if hotkey not in self.registered_hotkeys:
            return False

        if self.handler.unregister_hotkey(hotkey):
            del self.registered_hotkeys[hotkey]
            return True

        return False

    def start(self):
        """Start the hotkey manager."""
        self.handler.start()

    def stop(self):
        """Stop the hotkey manager."""
        self.handler.stop()

    def cleanup(self):
        """Clean up resources."""
        self.stop()

    def get_registered_hotkeys(self) -> List[str]:
        """Get list of registered hotkeys."""
        return list(self.registered_hotkeys.keys())

    def is_hotkey_available(self, hotkey: str) -> bool:
        """Check if a hotkey is available for registration."""
        return hotkey not in self.registered_hotkeys


class SystemTrayManager:
    """System tray manager for cross-platform tray icon support."""

    def __init__(self, app_name: str = "Project QuickNav"):
        self.app_name = app_name
        self.menu_items = []
        self.tray_icon = None
        self.available = False

        self._init_tray()

    def _init_tray(self):
        """Initialize system tray support."""
        try:
            if sys.platform == 'win32':
                self._init_windows_tray()
            elif sys.platform == 'darwin':
                self._init_macos_tray()
            else:
                self._init_linux_tray()

        except Exception as e:
            logger.warning(f"System tray not available: {e}")

    def _init_windows_tray(self):
        """Initialize Windows system tray."""
        try:
            import pystray
            from PIL import Image
            import io

            # Create a simple icon
            icon_data = self._create_icon_data()
            image = Image.open(io.BytesIO(icon_data))

            self.pystray = pystray
            self.tray_icon = pystray.Icon(
                self.app_name,
                image,
                menu=self._create_menu()
            )
            self.available = True

        except ImportError:
            logger.warning("pystray not available for system tray")

    def _init_macos_tray(self):
        """Initialize macOS system tray."""
        # macOS tray implementation would go here
        logger.info("macOS system tray not implemented")

    def _init_linux_tray(self):
        """Initialize Linux system tray."""
        # Linux tray implementation would go here
        logger.info("Linux system tray not implemented")

    def _create_icon_data(self) -> bytes:
        """Create icon data for the tray icon."""
        # Simple 16x16 icon data (placeholder)
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d\x00\x00\x00\x18tEXtSoftware\x00paint.net 4.0.6\x85\x15\x1e\x00\x00\x00!IDAT8O\x85\x91\xc1\r\x00 \x0c\x03\x9f\xf9\xff?\x8e<B\x84`B\xa8E\x8c\x02\x89"L\x0c\xe9\n!\x1e\x7f\xc8z\x90\x85\xd4\xbb\xfe\x00\x00\x00\x00IEND\xaeB`\x82'

    def _create_menu(self):
        """Create the system tray menu."""
        if not hasattr(self, 'pystray'):
            return None

        return self.pystray.Menu(
            self.pystray.MenuItem("Show QuickNav", self._show_app),
            self.pystray.MenuItem("Settings", self._show_settings),
            self.pystray.Menu.SEPARATOR,
            self.pystray.MenuItem("Exit", self._exit_app)
        )

    def _show_app(self):
        """Show the main application."""
        # This should be connected to the main app's show method
        pass

    def _show_settings(self):
        """Show the settings dialog."""
        # This should be connected to the main app's settings method
        pass

    def _exit_app(self):
        """Exit the application."""
        # This should be connected to the main app's exit method
        pass

    def show(self, show_callback=None, settings_callback=None, exit_callback=None):
        """
        Show the system tray icon.

        Args:
            show_callback: Callback for show action
            settings_callback: Callback for settings action
            exit_callback: Callback for exit action
        """
        if not self.available:
            return False

        # Update callbacks
        if show_callback:
            self._show_app = show_callback
        if settings_callback:
            self._show_settings = settings_callback
        if exit_callback:
            self._exit_app = exit_callback

        try:
            if hasattr(self, 'pystray') and self.tray_icon:
                # Run in separate thread to avoid blocking
                thread = threading.Thread(target=self.tray_icon.run, daemon=True)
                thread.start()
                return True

        except Exception as e:
            logger.error(f"Failed to show system tray: {e}")

        return False

    def hide(self):
        """Hide the system tray icon."""
        try:
            if self.tray_icon and hasattr(self.tray_icon, 'stop'):
                self.tray_icon.stop()

        except Exception as e:
            logger.error(f"Failed to hide system tray: {e}")

    def update_menu(self, menu_items: List[Dict]):
        """Update the system tray menu."""
        # This would update the menu items
        self.menu_items = menu_items

    def set_tooltip(self, tooltip: str):
        """Set the system tray tooltip."""
        if self.tray_icon and hasattr(self.tray_icon, 'tooltip'):
            self.tray_icon.tooltip = tooltip


def test_hotkey_manager():
    """Test the hotkey manager functionality."""
    manager = HotkeyManager()

    def test_callback():
        print("Test hotkey pressed!")

    # Register test hotkey
    if manager.register_hotkey("ctrl+alt+t", test_callback):
        print("Test hotkey registered successfully")
        manager.start()

        try:
            print("Press Ctrl+Alt+T to test the hotkey. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping hotkey manager...")
            manager.cleanup()
    else:
        print("Failed to register test hotkey")


if __name__ == "__main__":
    test_hotkey_manager()