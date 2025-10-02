"""
System Integration Module for Project QuickNav

This module provides comprehensive system integration features including:
- File Explorer context menu integration
- Clipboard monitoring for project numbers
- URL protocol support (quicknav://project_number)
- Browser integration capabilities
- Drag & drop support
- Enhanced system tray functionality

Features are designed to be cross-platform with platform-specific optimizations.
"""

import os
import sys
import re
import time
import json
import threading
import subprocess
import webbrowser
import logging
import winreg
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import messagebox
import tempfile

logger = logging.getLogger(__name__)

# Platform detection
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# Optional dependencies
OPTIONAL_DEPS = {
    'pyperclip': False,
    'watchdog': False,
    'win32clipboard': False,
    'pywin32': False
}

try:
    import pyperclip
    OPTIONAL_DEPS['pyperclip'] = True
except ImportError:
    pyperclip = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    OPTIONAL_DEPS['watchdog'] = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None

if IS_WINDOWS:
    try:
        import win32clipboard
        import win32con
        import win32api
        import win32gui
        OPTIONAL_DEPS['win32clipboard'] = True
        OPTIONAL_DEPS['pywin32'] = True
    except ImportError:
        win32clipboard = None
        win32con = None
        win32api = None
        win32gui = None


@dataclass
class ProjectInfo:
    """Container for project information."""
    number: str
    name: Optional[str] = None
    path: Optional[str] = None
    source: str = "unknown"  # clipboard, drag_drop, url, context_menu


class ClipboardMonitor:
    """Cross-platform clipboard monitoring for project numbers."""

    def __init__(self, callback: Callable[[ProjectInfo], None]):
        self.callback = callback
        self.running = False
        self.thread = None
        self.last_content = ""
        self.project_pattern = re.compile(r'\b(\d{5})\b')

    def start(self):
        """Start clipboard monitoring."""
        if self.running:
            return

        self.running = True
        if IS_WINDOWS and OPTIONAL_DEPS['win32clipboard']:
            self.thread = threading.Thread(target=self._windows_monitor, daemon=True)
        elif OPTIONAL_DEPS['pyperclip']:
            self.thread = threading.Thread(target=self._generic_monitor, daemon=True)
        else:
            logger.warning("No clipboard monitoring support available")
            return

        self.thread.start()
        logger.info("Clipboard monitoring started")

    def stop(self):
        """Stop clipboard monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Clipboard monitoring stopped")

    def _windows_monitor(self):
        """Windows-specific clipboard monitoring using win32 API."""
        try:
            while self.running:
                try:
                    win32clipboard.OpenClipboard()
                    if win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
                        content = win32clipboard.GetClipboardData()
                        if content != self.last_content:
                            self.last_content = content
                            self._process_clipboard_content(content)
                    win32clipboard.CloseClipboard()
                except Exception as e:
                    logger.debug(f"Clipboard access error: {e}")

                time.sleep(0.5)  # Check every 500ms

        except Exception as e:
            logger.error(f"Windows clipboard monitoring error: {e}")

    def _generic_monitor(self):
        """Generic clipboard monitoring using pyperclip."""
        try:
            while self.running:
                try:
                    content = pyperclip.paste()
                    if content != self.last_content:
                        self.last_content = content
                        self._process_clipboard_content(content)
                except Exception as e:
                    logger.debug(f"Clipboard access error: {e}")

                time.sleep(0.5)  # Check every 500ms

        except Exception as e:
            logger.error(f"Generic clipboard monitoring error: {e}")

    def _process_clipboard_content(self, content: str):
        """Process clipboard content for project numbers."""
        if not content or len(content) > 1000:  # Ignore very long content
            return

        matches = self.project_pattern.findall(content)
        if matches:
            # Take the first 5-digit number found
            project_number = matches[0]
            project_info = ProjectInfo(
                number=project_number,
                source="clipboard"
            )

            try:
                self.callback(project_info)
            except Exception as e:
                logger.error(f"Error in clipboard callback: {e}")


class URLProtocolHandler:
    """URL protocol handler for quicknav:// URLs."""

    def __init__(self, app_path: str):
        self.app_path = app_path
        self.protocol_name = "quicknav"

    def register_protocol(self) -> bool:
        """Register the quicknav:// URL protocol."""
        try:
            if IS_WINDOWS:
                return self._register_windows_protocol()
            elif IS_MACOS:
                return self._register_macos_protocol()
            elif IS_LINUX:
                return self._register_linux_protocol()
        except Exception as e:
            logger.error(f"Failed to register URL protocol: {e}")
            return False

    def unregister_protocol(self) -> bool:
        """Unregister the quicknav:// URL protocol."""
        try:
            if IS_WINDOWS:
                return self._unregister_windows_protocol()
            elif IS_MACOS:
                return self._unregister_macos_protocol()
            elif IS_LINUX:
                return self._unregister_linux_protocol()
        except Exception as e:
            logger.error(f"Failed to unregister URL protocol: {e}")
            return False

    def _register_windows_protocol(self) -> bool:
        """Register URL protocol on Windows."""
        try:
            # Create registry entries for quicknav:// protocol
            key_path = f"Software\\Classes\\{self.protocol_name}"

            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:QuickNav Protocol")
                winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")

            # Set the command
            command_path = f"{key_path}\\shell\\open\\command"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, command_path) as key:
                command = f'"{self.app_path}" --url "%1"'
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)

            logger.info("Windows URL protocol registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register Windows URL protocol: {e}")
            return False

    def _unregister_windows_protocol(self) -> bool:
        """Unregister URL protocol on Windows."""
        try:
            key_path = f"Software\\Classes\\{self.protocol_name}"
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
            logger.info("Windows URL protocol unregistered successfully")
            return True
        except FileNotFoundError:
            return True  # Already unregistered
        except Exception as e:
            logger.error(f"Failed to unregister Windows URL protocol: {e}")
            return False

    def _register_macos_protocol(self) -> bool:
        """Register URL protocol on macOS."""
        try:
            # Create a .plist file for the protocol
            plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>QuickNav Protocol Handler</string>
    <key>CFBundleURLTypes</key>
    <array>
        <dict>
            <key>CFBundleURLName</key>
            <string>QuickNav URL</string>
            <key>CFBundleURLSchemes</key>
            <array>
                <string>{self.protocol_name}</string>
            </array>
        </dict>
    </array>
</dict>
</plist>'''

            # This would require creating a proper app bundle
            logger.warning("macOS URL protocol registration requires app bundle setup")
            return False

        except Exception as e:
            logger.error(f"Failed to register macOS URL protocol: {e}")
            return False

    def _unregister_macos_protocol(self) -> bool:
        """Unregister URL protocol on macOS."""
        logger.info("macOS URL protocol unregistration not implemented")
        return False

    def _register_linux_protocol(self) -> bool:
        """Register URL protocol on Linux."""
        try:
            # Create .desktop file
            desktop_content = f'''[Desktop Entry]
Name=QuickNav
Exec={self.app_path} --url %u
Icon={self.app_path}
StartupNotify=true
NoDisplay=true
MimeType=x-scheme-handler/{self.protocol_name};
'''

            desktop_dir = Path.home() / ".local/share/applications"
            desktop_dir.mkdir(parents=True, exist_ok=True)

            desktop_file = desktop_dir / f"{self.protocol_name}-handler.desktop"
            desktop_file.write_text(desktop_content)

            # Register with xdg-mime
            subprocess.run([
                "xdg-mime", "default",
                f"{self.protocol_name}-handler.desktop",
                f"x-scheme-handler/{self.protocol_name}"
            ], check=True)

            logger.info("Linux URL protocol registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register Linux URL protocol: {e}")
            return False

    def _unregister_linux_protocol(self) -> bool:
        """Unregister URL protocol on Linux."""
        try:
            desktop_dir = Path.home() / ".local/share/applications"
            desktop_file = desktop_dir / f"{self.protocol_name}-handler.desktop"

            if desktop_file.exists():
                desktop_file.unlink()

            logger.info("Linux URL protocol unregistered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister Linux URL protocol: {e}")
            return False

    def parse_url(self, url: str) -> Optional[ProjectInfo]:
        """Parse a quicknav:// URL and extract project information."""
        try:
            if not url.startswith(f"{self.protocol_name}://"):
                return None

            # Extract project number from URL
            # Format: quicknav://17741 or quicknav://project/17741
            path = url[len(f"{self.protocol_name}://"):]

            # Simple project number
            if path.isdigit() and len(path) == 5:
                return ProjectInfo(number=path, source="url")

            # Project path format
            if path.startswith("project/"):
                project_number = path[8:]  # Remove "project/"
                if project_number.isdigit() and len(project_number) == 5:
                    return ProjectInfo(number=project_number, source="url")

            return None

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return None


class ContextMenuHandler:
    """File Explorer context menu integration."""

    def __init__(self, app_path: str):
        self.app_path = app_path

    def install_context_menu(self) -> bool:
        """Install context menu integration."""
        try:
            if IS_WINDOWS:
                return self._install_windows_context_menu()
            elif IS_MACOS:
                return self._install_macos_context_menu()
            elif IS_LINUX:
                return self._install_linux_context_menu()
        except Exception as e:
            logger.error(f"Failed to install context menu: {e}")
            return False

    def uninstall_context_menu(self) -> bool:
        """Uninstall context menu integration."""
        try:
            if IS_WINDOWS:
                return self._uninstall_windows_context_menu()
            elif IS_MACOS:
                return self._uninstall_macos_context_menu()
            elif IS_LINUX:
                return self._uninstall_linux_context_menu()
        except Exception as e:
            logger.error(f"Failed to uninstall context menu: {e}")
            return False

    def _install_windows_context_menu(self) -> bool:
        """Install Windows Explorer context menu."""
        try:
            # Add context menu for folders
            folder_key = "Software\\Classes\\Directory\\shell\\QuickNav"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, folder_key) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Open in QuickNav")
                winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, self.app_path)

            command_key = f"{folder_key}\\command"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, command_key) as key:
                command = f'"{self.app_path}" --folder "%1"'
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)

            # Add context menu for background
            background_key = "Software\\Classes\\Directory\\Background\\shell\\QuickNav"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, background_key) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Open QuickNav Here")
                winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, self.app_path)

            bg_command_key = f"{background_key}\\command"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, bg_command_key) as key:
                command = f'"{self.app_path}" --pwd "%V"'
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)

            logger.info("Windows context menu installed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to install Windows context menu: {e}")
            return False

    def _uninstall_windows_context_menu(self) -> bool:
        """Uninstall Windows Explorer context menu."""
        try:
            keys_to_remove = [
                "Software\\Classes\\Directory\\shell\\QuickNav",
                "Software\\Classes\\Directory\\Background\\shell\\QuickNav"
            ]

            for key_path in keys_to_remove:
                try:
                    self._delete_registry_key_recursive(winreg.HKEY_CURRENT_USER, key_path)
                except FileNotFoundError:
                    pass  # Key doesn't exist

            logger.info("Windows context menu uninstalled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall Windows context menu: {e}")
            return False

    def _delete_registry_key_recursive(self, hkey, key_path):
        """Recursively delete a registry key and all subkeys."""
        try:
            with winreg.OpenKey(hkey, key_path, 0, winreg.KEY_ALL_ACCESS) as key:
                subkeys = []
                try:
                    i = 0
                    while True:
                        subkeys.append(winreg.EnumKey(key, i))
                        i += 1
                except WindowsError:
                    pass

                for subkey in subkeys:
                    self._delete_registry_key_recursive(hkey, f"{key_path}\\{subkey}")

            winreg.DeleteKey(hkey, key_path)

        except Exception as e:
            logger.debug(f"Error deleting registry key {key_path}: {e}")

    def _install_macos_context_menu(self) -> bool:
        """Install macOS Finder context menu."""
        logger.warning("macOS context menu not implemented")
        return False

    def _uninstall_macos_context_menu(self) -> bool:
        """Uninstall macOS Finder context menu."""
        logger.warning("macOS context menu not implemented")
        return False

    def _install_linux_context_menu(self) -> bool:
        """Install Linux file manager context menu."""
        try:
            # Create Nautilus script for GNOME
            scripts_dir = Path.home() / ".local/share/nautilus/scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            script_content = f'''#!/bin/bash
# QuickNav Context Menu Script
"{self.app_path}" --folder "$NAUTILUS_SCRIPT_SELECTED_FILE_PATHS"
'''

            script_file = scripts_dir / "Open in QuickNav"
            script_file.write_text(script_content)
            script_file.chmod(0o755)

            logger.info("Linux context menu installed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to install Linux context menu: {e}")
            return False

    def _uninstall_linux_context_menu(self) -> bool:
        """Uninstall Linux file manager context menu."""
        try:
            scripts_dir = Path.home() / ".local/share/nautilus/scripts"
            script_file = scripts_dir / "Open in QuickNav"

            if script_file.exists():
                script_file.unlink()

            logger.info("Linux context menu uninstalled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall Linux context menu: {e}")
            return False


class DragDropHandler:
    """Drag and drop support for project numbers and files."""

    def __init__(self, widget: tk.Widget, callback: Callable[[ProjectInfo], None]):
        self.widget = widget
        self.callback = callback
        self.project_pattern = re.compile(r'\b(\d{5})\b')

        self._setup_drag_drop()

    def _setup_drag_drop(self):
        """Set up drag and drop event handlers."""
        try:
            # Bind drag and drop events
            self.widget.drop_target_register('DND_Text')
            self.widget.dnd_bind('<<Drop>>', self._handle_drop)

            # Also bind file drops if supported
            self.widget.drop_target_register('DND_Files')

        except Exception as e:
            logger.warning(f"Advanced drag-drop not available, using basic events: {e}")

            # Fallback to basic Tkinter drag-drop
            self.widget.bind('<Button-1>', self._handle_drag_start)
            self.widget.bind('<B1-Motion>', self._handle_drag_motion)
            self.widget.bind('<ButtonRelease-1>', self._handle_drag_end)

    def _handle_drop(self, event):
        """Handle drop events."""
        try:
            data = event.data

            if isinstance(data, str):
                # Handle text drops
                self._process_text_drop(data)
            elif isinstance(data, (list, tuple)):
                # Handle file drops
                for item in data:
                    if isinstance(item, str):
                        self._process_file_drop(item)

        except Exception as e:
            logger.error(f"Error handling drop event: {e}")

    def _process_text_drop(self, text: str):
        """Process dropped text for project numbers."""
        matches = self.project_pattern.findall(text)
        if matches:
            project_number = matches[0]
            project_info = ProjectInfo(
                number=project_number,
                source="drag_drop"
            )
            self.callback(project_info)

    def _process_file_drop(self, filepath: str):
        """Process dropped files for project information."""
        try:
            # Extract project number from file path
            path_str = str(filepath)
            matches = self.project_pattern.findall(path_str)

            if matches:
                project_number = matches[0]
                project_info = ProjectInfo(
                    number=project_number,
                    path=filepath,
                    source="drag_drop"
                )
                self.callback(project_info)

        except Exception as e:
            logger.error(f"Error processing file drop {filepath}: {e}")

    def _handle_drag_start(self, event):
        """Handle drag start (fallback implementation)."""
        pass

    def _handle_drag_motion(self, event):
        """Handle drag motion (fallback implementation)."""
        pass

    def _handle_drag_end(self, event):
        """Handle drag end (fallback implementation)."""
        pass


class BrowserIntegration:
    """Browser integration for QuickNav functionality."""

    def __init__(self):
        self.extension_data = None

    def install_browser_extension(self) -> bool:
        """Install browser extension for QuickNav integration."""
        try:
            if IS_WINDOWS:
                return self._install_chrome_extension()
            else:
                logger.warning("Browser extension not implemented for this platform")
                return False

        except Exception as e:
            logger.error(f"Failed to install browser extension: {e}")
            return False

    def _install_chrome_extension(self) -> bool:
        """Install Chrome extension on Windows."""
        try:
            # Create native messaging host manifest
            manifest = {
                "name": "com.quicknav.native",
                "description": "QuickNav Native Messaging Host",
                "path": self._get_native_host_path(),
                "type": "stdio",
                "allowed_origins": [
                    "chrome-extension://quicknav-extension-id/"
                ]
            }

            # Write manifest to Chrome's native messaging hosts directory
            chrome_dir = Path.home() / "AppData/Local/Google/Chrome/User Data/NativeMessagingHosts"
            chrome_dir.mkdir(parents=True, exist_ok=True)

            manifest_file = chrome_dir / "com.quicknav.native.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            logger.info("Chrome extension host installed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to install Chrome extension: {e}")
            return False

    def _get_native_host_path(self) -> str:
        """Get the path to the native messaging host executable."""
        # This would be a separate executable for native messaging
        return str(Path(__file__).parent / "quicknav_native_host.exe")

    def open_project_in_browser(self, project_number: str, url_template: str = None):
        """Open project in browser using URL template."""
        if not url_template:
            url_template = f"quicknav://{project_number}"

        try:
            url = url_template.format(project_number=project_number)
            webbrowser.open(url)
            logger.info(f"Opened project {project_number} in browser")

        except Exception as e:
            logger.error(f"Failed to open project in browser: {e}")


class EnhancedSystemTray:
    """Enhanced system tray with QuickNav integration."""

    def __init__(self, app_callback: Callable = None):
        self.app_callback = app_callback
        self.tray_icon = None
        self.available = False
        self.recent_projects = []

        self._init_tray()

    def _init_tray(self):
        """Initialize enhanced system tray."""
        try:
            import pystray
            from PIL import Image, ImageDraw

            # Create a more sophisticated icon
            image = self._create_app_icon()

            self.tray_icon = pystray.Icon(
                "QuickNav",
                image,
                "QuickNav - Project Navigation",
                menu=self._create_enhanced_menu()
            )
            self.available = True

        except ImportError:
            logger.warning("Enhanced system tray requires pystray and PIL")

    def _create_app_icon(self):
        """Create application icon for system tray."""
        try:
            from PIL import Image, ImageDraw

            # Create 64x64 icon with "QN" text
            image = Image.new('RGBA', (64, 64), (0, 120, 212, 255))
            draw = ImageDraw.Draw(image)

            # Draw "QN" text
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            text = "QN"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = (64 - text_width) // 2
            y = (64 - text_height) // 2

            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

            return image

        except Exception as e:
            logger.warning(f"Failed to create custom icon: {e}")
            # Fallback to simple colored square
            from PIL import Image
            return Image.new('RGBA', (64, 64), (0, 120, 212, 255))

    def _create_enhanced_menu(self):
        """Create enhanced system tray menu."""
        try:
            import pystray

            menu_items = [
                pystray.MenuItem("Show QuickNav", self._show_quicknav),
                pystray.Menu.SEPARATOR,
            ]

            # Add recent projects submenu
            if self.recent_projects:
                recent_menu = pystray.Menu(*[
                    pystray.MenuItem(
                        f"{proj['number']} - {proj.get('name', 'Unknown')}",
                        lambda proj=proj: self._open_recent_project(proj)
                    )
                    for proj in self.recent_projects[:5]  # Show last 5 projects
                ])
                menu_items.append(pystray.MenuItem("Recent Projects", recent_menu))
                menu_items.append(pystray.Menu.SEPARATOR)

            # Add quick actions
            menu_items.extend([
                pystray.MenuItem("Quick Search", self._quick_search),
                pystray.MenuItem("Settings", self._show_settings),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("About", self._show_about),
                pystray.MenuItem("Exit", self._exit_app)
            ])

            return pystray.Menu(*menu_items)

        except Exception as e:
            logger.error(f"Failed to create enhanced menu: {e}")
            return None

    def update_recent_projects(self, projects: List[Dict]):
        """Update the recent projects list."""
        self.recent_projects = projects[:5]  # Keep last 5

        # Update menu if tray is active
        if self.tray_icon and hasattr(self.tray_icon, 'menu'):
            self.tray_icon.menu = self._create_enhanced_menu()

    def show(self):
        """Show the enhanced system tray."""
        if not self.available:
            return False

        try:
            # Run in separate thread
            thread = threading.Thread(target=self.tray_icon.run, daemon=True)
            thread.start()
            return True

        except Exception as e:
            logger.error(f"Failed to show enhanced system tray: {e}")
            return False

    def hide(self):
        """Hide the system tray."""
        if self.tray_icon:
            try:
                self.tray_icon.stop()
            except Exception as e:
                logger.error(f"Failed to hide system tray: {e}")

    def _show_quicknav(self):
        """Show the QuickNav application."""
        if self.app_callback:
            self.app_callback('show')

    def _quick_search(self):
        """Show quick search dialog."""
        if self.app_callback:
            self.app_callback('quick_search')

    def _open_recent_project(self, project):
        """Open a recent project."""
        if self.app_callback:
            self.app_callback('open_project', project)

    def _show_settings(self):
        """Show settings dialog."""
        if self.app_callback:
            self.app_callback('settings')

    def _show_about(self):
        """Show about dialog."""
        if self.app_callback:
            self.app_callback('about')

    def _exit_app(self):
        """Exit the application."""
        if self.app_callback:
            self.app_callback('exit')


class SystemIntegrationManager:
    """Main system integration manager coordinating all features."""

    def __init__(self, app_path: str = None, app_callback: Callable = None):
        self.app_path = app_path or sys.executable
        self.app_callback = app_callback

        # Initialize components
        self.clipboard_monitor = None
        self.url_handler = URLProtocolHandler(self.app_path)
        self.context_menu = ContextMenuHandler(self.app_path)
        self.browser_integration = BrowserIntegration()
        self.enhanced_tray = EnhancedSystemTray(app_callback)

        # State
        self.integration_enabled = False
        self.project_callbacks = []

    def enable_system_integration(self) -> Dict[str, bool]:
        """Enable all system integration features."""
        results = {}

        try:
            # Enable clipboard monitoring
            results['clipboard'] = self._enable_clipboard_monitoring()

            # Register URL protocol
            results['url_protocol'] = self.url_handler.register_protocol()

            # Install context menu
            results['context_menu'] = self.context_menu.install_context_menu()

            # Install browser integration
            results['browser'] = self.browser_integration.install_browser_extension()

            # Show enhanced system tray
            results['system_tray'] = self.enhanced_tray.show()

            self.integration_enabled = any(results.values())

            if self.integration_enabled:
                logger.info("System integration enabled")
            else:
                logger.warning("No system integration features could be enabled")

        except Exception as e:
            logger.error(f"Error enabling system integration: {e}")

        return results

    def disable_system_integration(self) -> Dict[str, bool]:
        """Disable all system integration features."""
        results = {}

        try:
            # Stop clipboard monitoring
            results['clipboard'] = self._disable_clipboard_monitoring()

            # Unregister URL protocol
            results['url_protocol'] = self.url_handler.unregister_protocol()

            # Uninstall context menu
            results['context_menu'] = self.context_menu.uninstall_context_menu()

            # Hide system tray
            results['system_tray'] = self._disable_system_tray()

            self.integration_enabled = False
            logger.info("System integration disabled")

        except Exception as e:
            logger.error(f"Error disabling system integration: {e}")

        return results

    def _enable_clipboard_monitoring(self) -> bool:
        """Enable clipboard monitoring."""
        try:
            if not OPTIONAL_DEPS.get('pyperclip') and not OPTIONAL_DEPS.get('win32clipboard'):
                logger.warning("Clipboard monitoring requires pyperclip or win32clipboard")
                return False

            self.clipboard_monitor = ClipboardMonitor(self._handle_project_detected)
            self.clipboard_monitor.start()
            return True

        except Exception as e:
            logger.error(f"Failed to enable clipboard monitoring: {e}")
            return False

    def _disable_clipboard_monitoring(self) -> bool:
        """Disable clipboard monitoring."""
        try:
            if self.clipboard_monitor:
                self.clipboard_monitor.stop()
                self.clipboard_monitor = None
            return True

        except Exception as e:
            logger.error(f"Failed to disable clipboard monitoring: {e}")
            return False

    def _disable_system_tray(self) -> bool:
        """Disable system tray."""
        try:
            self.enhanced_tray.hide()
            return True
        except Exception as e:
            logger.error(f"Failed to disable system tray: {e}")
            return False

    def setup_drag_drop(self, widget: tk.Widget):
        """Set up drag and drop for a widget."""
        try:
            return DragDropHandler(widget, self._handle_project_detected)
        except Exception as e:
            logger.error(f"Failed to setup drag and drop: {e}")
            return None

    def handle_url_launch(self, url: str) -> bool:
        """Handle application launch from URL protocol."""
        try:
            project_info = self.url_handler.parse_url(url)
            if project_info:
                self._handle_project_detected(project_info)
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to handle URL launch: {e}")
            return False

    def handle_context_menu_launch(self, folder_path: str = None, pwd: str = None) -> bool:
        """Handle application launch from context menu."""
        try:
            # Extract project number from folder path
            if folder_path:
                project_pattern = re.compile(r'\b(\d{5})\b')
                matches = project_pattern.findall(folder_path)
                if matches:
                    project_info = ProjectInfo(
                        number=matches[0],
                        path=folder_path,
                        source="context_menu"
                    )
                    self._handle_project_detected(project_info)
                    return True

            # If no project found, just show the application
            if self.app_callback:
                self.app_callback('show')
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to handle context menu launch: {e}")
            return False

    def add_project_callback(self, callback: Callable[[ProjectInfo], None]):
        """Add a callback for when projects are detected."""
        self.project_callbacks.append(callback)

    def remove_project_callback(self, callback: Callable[[ProjectInfo], None]):
        """Remove a project callback."""
        if callback in self.project_callbacks:
            self.project_callbacks.remove(callback)

    def _handle_project_detected(self, project_info: ProjectInfo):
        """Handle when a project is detected from any source."""
        try:
            logger.info(f"Project detected: {project_info.number} from {project_info.source}")

            # Notify all callbacks
            for callback in self.project_callbacks:
                try:
                    callback(project_info)
                except Exception as e:
                    logger.error(f"Error in project callback: {e}")

        except Exception as e:
            logger.error(f"Error handling project detection: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integration features."""
        return {
            'enabled': self.integration_enabled,
            'clipboard_monitoring': self.clipboard_monitor is not None and self.clipboard_monitor.running,
            'url_protocol_available': True,  # Always available to register
            'context_menu_available': True,  # Always available to install
            'browser_integration_available': IS_WINDOWS,  # Currently Windows only
            'system_tray_available': OPTIONAL_DEPS.get('pystray', False),
            'drag_drop_available': True,  # Basic support always available
            'dependencies': OPTIONAL_DEPS.copy()
        }

    def update_recent_projects(self, projects: List[Dict]):
        """Update recent projects in system tray."""
        if self.enhanced_tray:
            self.enhanced_tray.update_recent_projects(projects)


def get_app_path() -> str:
    """Get the application executable path."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return sys.executable
    else:
        # Running as script
        return sys.executable + " " + os.path.abspath(__file__)


def test_system_integration():
    """Test system integration functionality."""
    def test_callback(project_info: ProjectInfo):
        print(f"Project detected: {project_info.number} from {project_info.source}")

    def app_callback(action, data=None):
        print(f"App callback: {action} with data: {data}")

    manager = SystemIntegrationManager(
        app_path=get_app_path(),
        app_callback=app_callback
    )

    manager.add_project_callback(test_callback)

    print("System Integration Test")
    print("=" * 30)

    status = manager.get_integration_status()
    print("Integration status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\nEnabling system integration...")
    results = manager.enable_system_integration()
    print("Results:")
    for feature, success in results.items():
        print(f"  {feature}: {'✓' if success else '✗'}")

    if results.get('clipboard'):
        print("\nClipboard monitoring enabled. Copy a 5-digit number to test.")

    try:
        input("Press Enter to disable and exit...")
    except KeyboardInterrupt:
        pass

    print("\nDisabling system integration...")
    manager.disable_system_integration()
    print("Test completed.")


if __name__ == "__main__":
    test_system_integration()