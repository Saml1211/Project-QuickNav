"""
Theming System for Project QuickNav

This module provides comprehensive theming support including:
- Built-in themes (Light, Dark, High Contrast)
- Custom theme creation and management
- Dynamic theme switching
- Component-specific styling
- DPI-aware scaling
- System theme detection
"""

import tkinter as tk
from tkinter import ttk
import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import platform

logger = logging.getLogger(__name__)


class Theme:
    """Represents a single theme with all styling information."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    def get_color(self, element: str, state: str = "normal") -> str:
        """Get color for an element in a specific state."""
        colors = self.config.get("colors", {})
        element_colors = colors.get(element, {})

        if isinstance(element_colors, str):
            return element_colors
        elif isinstance(element_colors, dict):
            return element_colors.get(state, element_colors.get("normal", "#000000"))

        return "#000000"

    def get_font(self, element: str = "default") -> Dict[str, Any]:
        """Get font configuration for an element."""
        fonts = self.config.get("fonts", {})
        return fonts.get(element, fonts.get("default", {}))

    def get_geometry(self, element: str) -> Dict[str, Any]:
        """Get geometry configuration for an element."""
        geometry = self.config.get("geometry", {})
        return geometry.get(element, {})

    def get_style(self, element: str) -> Dict[str, Any]:
        """Get complete style configuration for an element."""
        return {
            "colors": self.config.get("colors", {}).get(element, {}),
            "fonts": self.get_font(element),
            "geometry": self.get_geometry(element)
        }


class ThemeManager:
    """Manages themes and applies them to Tkinter applications."""

    def __init__(self, settings_manager=None):
        self.settings = settings_manager
        self.current_theme = None
        self.themes = {}
        self.style = None
        self.theme_change_callbacks = []

        # Load built-in themes
        self._load_builtin_themes()

        # Set initial theme
        initial_theme = self._get_initial_theme()
        self.set_theme(initial_theme)

    def _get_initial_theme(self) -> str:
        """Get the initial theme to use."""
        if self.settings:
            theme_name = self.settings.get_theme()
            if theme_name == "system":
                return self._detect_system_theme()
            return theme_name

        return self._detect_system_theme()

    def _detect_system_theme(self) -> str:
        """Detect the system's preferred theme."""
        try:
            if platform.system() == "Windows":
                return self._detect_windows_theme()
            elif platform.system() == "Darwin":
                return self._detect_macos_theme()
            else:
                return self._detect_linux_theme()

        except Exception as e:
            logger.warning(f"Failed to detect system theme: {e}")
            return "light"

    def _detect_windows_theme(self) -> str:
        """Detect Windows theme preference."""
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            ) as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                return "light" if value else "dark"

        except Exception:
            return "light"

    def _detect_macos_theme(self) -> str:
        """Detect macOS theme preference."""
        try:
            import subprocess
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True
            )
            return "dark" if "Dark" in result.stdout else "light"

        except Exception:
            return "light"

    def _detect_linux_theme(self) -> str:
        """Detect Linux theme preference."""
        try:
            # Check GTK theme
            gtk_settings = os.path.expanduser("~/.config/gtk-3.0/settings.ini")
            if os.path.exists(gtk_settings):
                with open(gtk_settings, 'r') as f:
                    content = f.read()
                    if "dark" in content.lower():
                        return "dark"

            # Check GNOME settings
            try:
                import subprocess
                result = subprocess.run([
                    "gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"
                ], capture_output=True, text=True)
                if "dark" in result.stdout.lower():
                    return "dark"
            except:
                pass

        except Exception:
            pass

        return "light"

    def _load_builtin_themes(self):
        """Load built-in themes."""
        # Light theme
        light_theme = {
            "name": "Light",
            "colors": {
                "bg": "#ffffff",
                "fg": "#000000",
                "select_bg": "#0078d4",
                "select_fg": "#ffffff",
                "active_bg": "#e1ecf4",
                "active_fg": "#000000",
                "disabled_bg": "#f0f0f0",
                "disabled_fg": "#a0a0a0",
                "border": "#d0d0d0",
                "focus": "#0078d4",
                "error": "#d13438",
                "warning": "#ff8c00",
                "success": "#107c10",
                "info": "#0078d4",
                "window": {
                    "bg": "#ffffff",
                    "fg": "#000000"
                },
                "frame": {
                    "bg": "#f8f9fa",
                    "fg": "#000000"
                },
                "button": {
                    "normal": {"bg": "#ffffff", "fg": "#000000"},
                    "hover": {"bg": "#e1ecf4", "fg": "#000000"},
                    "pressed": {"bg": "#cce4f7", "fg": "#000000"},
                    "disabled": {"bg": "#f0f0f0", "fg": "#a0a0a0"}
                },
                "entry": {
                    "normal": {"bg": "#ffffff", "fg": "#000000"},
                    "focus": {"bg": "#ffffff", "fg": "#000000"},
                    "disabled": {"bg": "#f0f0f0", "fg": "#a0a0a0"}
                },
                "listbox": {
                    "normal": {"bg": "#ffffff", "fg": "#000000"},
                    "select": {"bg": "#0078d4", "fg": "#ffffff"}
                },
                "label": {
                    "normal": {"bg": "#ffffff", "fg": "#000000"},
                    "disabled": {"bg": "#ffffff", "fg": "#a0a0a0"}
                },
                "text": {
                    "normal": {"bg": "#ffffff", "fg": "#000000"},
                    "select": {"bg": "#0078d4", "fg": "#ffffff"}
                },
                "menu": {
                    "normal": {"bg": "#ffffff", "fg": "#000000"},
                    "active": {"bg": "#e1ecf4", "fg": "#000000"}
                },
                "progressbar": {
                    "bg": "#e1ecf4",
                    "fg": "#0078d4"
                },
                "scrollbar": {
                    "bg": "#f0f0f0",
                    "fg": "#c0c0c0",
                    "active": "#a0a0a0"
                }
            },
            "fonts": {
                "default": {
                    "family": "Segoe UI",
                    "size": 10,
                    "weight": "normal"
                },
                "heading": {
                    "family": "Segoe UI",
                    "size": 12,
                    "weight": "bold"
                },
                "small": {
                    "family": "Segoe UI",
                    "size": 8,
                    "weight": "normal"
                },
                "monospace": {
                    "family": "Consolas",
                    "size": 10,
                    "weight": "normal"
                }
            },
            "geometry": {
                "border_width": 1,
                "relief": "flat",
                "padding": 5,
                "button_padding": 8
            }
        }

        # Dark theme
        dark_theme = {
            "name": "Dark",
            "colors": {
                "bg": "#2d2d30",
                "fg": "#ffffff",
                "select_bg": "#3399ff",
                "select_fg": "#ffffff",
                "active_bg": "#3c3c3c",
                "active_fg": "#ffffff",
                "disabled_bg": "#404040",
                "disabled_fg": "#808080",
                "border": "#404040",
                "focus": "#3399ff",
                "error": "#f85149",
                "warning": "#d18616",
                "success": "#56d364",
                "info": "#58a6ff",
                "window": {
                    "bg": "#1e1e1e",
                    "fg": "#ffffff"
                },
                "frame": {
                    "bg": "#2d2d30",
                    "fg": "#ffffff"
                },
                "button": {
                    "normal": {"bg": "#3c3c3c", "fg": "#ffffff"},
                    "hover": {"bg": "#464647", "fg": "#ffffff"},
                    "pressed": {"bg": "#525252", "fg": "#ffffff"},
                    "disabled": {"bg": "#404040", "fg": "#808080"}
                },
                "entry": {
                    "normal": {"bg": "#1e1e1e", "fg": "#ffffff"},
                    "focus": {"bg": "#1e1e1e", "fg": "#ffffff"},
                    "disabled": {"bg": "#404040", "fg": "#808080"}
                },
                "listbox": {
                    "normal": {"bg": "#1e1e1e", "fg": "#ffffff"},
                    "select": {"bg": "#3399ff", "fg": "#ffffff"}
                },
                "label": {
                    "normal": {"bg": "#2d2d30", "fg": "#ffffff"},
                    "disabled": {"bg": "#2d2d30", "fg": "#808080"}
                },
                "text": {
                    "normal": {"bg": "#1e1e1e", "fg": "#ffffff"},
                    "select": {"bg": "#3399ff", "fg": "#ffffff"}
                },
                "menu": {
                    "normal": {"bg": "#2d2d30", "fg": "#ffffff"},
                    "active": {"bg": "#3c3c3c", "fg": "#ffffff"}
                },
                "progressbar": {
                    "bg": "#3c3c3c",
                    "fg": "#3399ff"
                },
                "scrollbar": {
                    "bg": "#3c3c3c",
                    "fg": "#606060",
                    "active": "#808080"
                }
            },
            "fonts": {
                "default": {
                    "family": "Segoe UI",
                    "size": 10,
                    "weight": "normal"
                },
                "heading": {
                    "family": "Segoe UI",
                    "size": 12,
                    "weight": "bold"
                },
                "small": {
                    "family": "Segoe UI",
                    "size": 8,
                    "weight": "normal"
                },
                "monospace": {
                    "family": "Consolas",
                    "size": 10,
                    "weight": "normal"
                }
            },
            "geometry": {
                "border_width": 1,
                "relief": "flat",
                "padding": 5,
                "button_padding": 8
            }
        }

        # High contrast theme
        high_contrast_theme = {
            "name": "High Contrast",
            "colors": {
                "bg": "#000000",
                "fg": "#ffffff",
                "select_bg": "#ffffff",
                "select_fg": "#000000",
                "active_bg": "#ffffff",
                "active_fg": "#000000",
                "disabled_bg": "#404040",
                "disabled_fg": "#808080",
                "border": "#ffffff",
                "focus": "#ffff00",
                "error": "#ff0000",
                "warning": "#ffff00",
                "success": "#00ff00",
                "info": "#00ffff",
                "window": {
                    "bg": "#000000",
                    "fg": "#ffffff"
                },
                "frame": {
                    "bg": "#000000",
                    "fg": "#ffffff"
                },
                "button": {
                    "normal": {"bg": "#000000", "fg": "#ffffff"},
                    "hover": {"bg": "#ffffff", "fg": "#000000"},
                    "pressed": {"bg": "#808080", "fg": "#ffffff"},
                    "disabled": {"bg": "#404040", "fg": "#808080"}
                },
                "entry": {
                    "normal": {"bg": "#000000", "fg": "#ffffff"},
                    "focus": {"bg": "#000000", "fg": "#ffffff"},
                    "disabled": {"bg": "#404040", "fg": "#808080"}
                },
                "listbox": {
                    "normal": {"bg": "#000000", "fg": "#ffffff"},
                    "select": {"bg": "#ffffff", "fg": "#000000"}
                },
                "label": {
                    "normal": {"bg": "#000000", "fg": "#ffffff"},
                    "disabled": {"bg": "#000000", "fg": "#808080"}
                },
                "text": {
                    "normal": {"bg": "#000000", "fg": "#ffffff"},
                    "select": {"bg": "#ffffff", "fg": "#000000"}
                },
                "menu": {
                    "normal": {"bg": "#000000", "fg": "#ffffff"},
                    "active": {"bg": "#ffffff", "fg": "#000000"}
                },
                "progressbar": {
                    "bg": "#404040",
                    "fg": "#ffffff"
                },
                "scrollbar": {
                    "bg": "#404040",
                    "fg": "#ffffff",
                    "active": "#ffffff"
                }
            },
            "fonts": {
                "default": {
                    "family": "Segoe UI",
                    "size": 12,
                    "weight": "bold"
                },
                "heading": {
                    "family": "Segoe UI",
                    "size": 14,
                    "weight": "bold"
                },
                "small": {
                    "family": "Segoe UI",
                    "size": 10,
                    "weight": "bold"
                },
                "monospace": {
                    "family": "Consolas",
                    "size": 12,
                    "weight": "bold"
                }
            },
            "geometry": {
                "border_width": 2,
                "relief": "solid",
                "padding": 8,
                "button_padding": 12
            }
        }

        # Register themes
        self.themes["light"] = Theme("Light", light_theme)
        self.themes["dark"] = Theme("Dark", dark_theme)
        self.themes["high_contrast"] = Theme("High Contrast", high_contrast_theme)

    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self.themes.keys())

    def get_theme(self, name: str) -> Optional[Theme]:
        """Get a theme by name."""
        return self.themes.get(name)

    def set_theme(self, name: str) -> bool:
        """
        Set the current theme.

        Args:
            name: Theme name to set

        Returns:
            True if successful, False otherwise
        """
        if name not in self.themes:
            logger.error(f"Theme '{name}' not found")
            return False

        self.current_theme = name
        theme = self.themes[name]

        # Update settings if available
        if self.settings:
            self.settings.set_theme(name)

        # Notify callbacks
        self._notify_theme_change(theme)

        logger.info(f"Theme changed to: {name}")
        return True

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        if self.current_theme == "light":
            self.set_theme("dark")
        else:
            self.set_theme("light")

    def apply_theme(self, root: tk.Tk):
        """
        Apply the current theme to a Tkinter root window.

        Args:
            root: Tkinter root window
        """
        if not self.current_theme:
            return

        theme = self.themes[self.current_theme]

        # Configure ttk style
        if not self.style:
            self.style = ttk.Style(root)

        self._configure_ttk_styles(theme)
        self._configure_tk_defaults(root, theme)

        # Configure root window
        window_colors = theme.get_style("window")["colors"]
        if window_colors:
            if isinstance(window_colors, dict):
                bg = window_colors.get("normal", {}).get("bg", window_colors.get("bg"))
                fg = window_colors.get("normal", {}).get("fg", window_colors.get("fg"))
            else:
                bg = theme.get_color("window", "bg")
                fg = theme.get_color("window", "fg")

            if bg:
                root.configure(bg=bg)

    def _configure_ttk_styles(self, theme: Theme):
        """Configure ttk widget styles."""
        if not self.style:
            return

        # Configure TLabel
        label_style = theme.get_style("label")
        label_colors = label_style["colors"]
        if label_colors:
            normal_colors = label_colors.get("normal", {})
            disabled_colors = label_colors.get("disabled", {})

            self.style.configure("TLabel",
                background=normal_colors.get("bg", theme.get_color("bg")),
                foreground=normal_colors.get("fg", theme.get_color("fg"))
            )

            self.style.map("TLabel",
                foreground=[
                    ("disabled", disabled_colors.get("fg", theme.get_color("disabled_fg")))
                ]
            )

        # Configure TButton
        button_style = theme.get_style("button")
        button_colors = button_style["colors"]
        if button_colors:
            normal = button_colors.get("normal", {})
            hover = button_colors.get("hover", {})
            pressed = button_colors.get("pressed", {})
            disabled = button_colors.get("disabled", {})

            self.style.configure("TButton",
                background=normal.get("bg", theme.get_color("bg")),
                foreground=normal.get("fg", theme.get_color("fg")),
                borderwidth=theme.get_geometry("button").get("border_width", 1),
                relief=theme.get_geometry("button").get("relief", "flat")
            )

            self.style.map("TButton",
                background=[
                    ("active", hover.get("bg", normal.get("bg"))),
                    ("pressed", pressed.get("bg", normal.get("bg"))),
                    ("disabled", disabled.get("bg", normal.get("bg")))
                ],
                foreground=[
                    ("active", hover.get("fg", normal.get("fg"))),
                    ("pressed", pressed.get("fg", normal.get("fg"))),
                    ("disabled", disabled.get("fg", normal.get("fg")))
                ]
            )

        # Configure TEntry
        entry_style = theme.get_style("entry")
        entry_colors = entry_style["colors"]
        if entry_colors:
            normal = entry_colors.get("normal", {})
            focus = entry_colors.get("focus", {})
            disabled = entry_colors.get("disabled", {})

            self.style.configure("TEntry",
                fieldbackground=normal.get("bg", theme.get_color("bg")),
                foreground=normal.get("fg", theme.get_color("fg")),
                borderwidth=theme.get_geometry("entry").get("border_width", 1),
                insertcolor=normal.get("fg", theme.get_color("fg"))
            )

            self.style.map("TEntry",
                fieldbackground=[
                    ("focus", focus.get("bg", normal.get("bg"))),
                    ("disabled", disabled.get("bg", normal.get("bg")))
                ],
                foreground=[
                    ("focus", focus.get("fg", normal.get("fg"))),
                    ("disabled", disabled.get("fg", normal.get("fg")))
                ]
            )

        # Configure TFrame
        frame_style = theme.get_style("frame")
        frame_colors = frame_style["colors"]
        if frame_colors:
            normal = frame_colors.get("normal", {}) if isinstance(frame_colors, dict) else frame_colors

            self.style.configure("TFrame",
                background=normal.get("bg", theme.get_color("bg"))
            )

        # Configure TLabelFrame
        self.style.configure("TLabelframe",
            background=theme.get_color("frame", "bg"),
            borderwidth=theme.get_geometry("frame").get("border_width", 1)
        )

        self.style.configure("TLabelframe.Label",
            background=theme.get_color("frame", "bg"),
            foreground=theme.get_color("frame", "fg")
        )

        # Configure TCombobox
        self.style.configure("TCombobox",
            fieldbackground=theme.get_color("entry", "bg"),
            foreground=theme.get_color("entry", "fg"),
            borderwidth=theme.get_geometry("entry").get("border_width", 1)
        )

        # Configure TCheckbutton
        self.style.configure("TCheckbutton",
            background=theme.get_color("bg"),
            foreground=theme.get_color("fg")
        )

        # Configure TRadiobutton
        self.style.configure("TRadiobutton",
            background=theme.get_color("bg"),
            foreground=theme.get_color("fg")
        )

        # Configure Progressbar
        progressbar_style = theme.get_style("progressbar")
        progressbar_colors = progressbar_style["colors"]
        if progressbar_colors:
            self.style.configure("TProgressbar",
                background=progressbar_colors.get("fg", theme.get_color("select_bg")),
                troughcolor=progressbar_colors.get("bg", theme.get_color("bg")),
                borderwidth=0,
                lightcolor=progressbar_colors.get("fg", theme.get_color("select_bg")),
                darkcolor=progressbar_colors.get("fg", theme.get_color("select_bg"))
            )

        # Configure Treeview
        self.style.configure("Treeview",
            background=theme.get_color("listbox", "bg"),
            foreground=theme.get_color("listbox", "fg"),
            fieldbackground=theme.get_color("listbox", "bg")
        )

        self.style.map("Treeview",
            background=[("selected", theme.get_color("listbox", "select_bg"))],
            foreground=[("selected", theme.get_color("listbox", "select_fg"))]
        )

        # Configure Notebook
        self.style.configure("TNotebook",
            background=theme.get_color("bg"),
            borderwidth=0
        )

        self.style.configure("TNotebook.Tab",
            background=theme.get_color("frame", "bg"),
            foreground=theme.get_color("frame", "fg"),
            padding=[12, 8]
        )

        self.style.map("TNotebook.Tab",
            background=[
                ("selected", theme.get_color("bg")),
                ("active", theme.get_color("active_bg"))
            ],
            foreground=[
                ("selected", theme.get_color("fg")),
                ("active", theme.get_color("active_fg"))
            ]
        )

    def _configure_tk_defaults(self, root: tk.Tk, theme: Theme):
        """Configure default Tk widget options."""
        # Get default colors
        bg = theme.get_color("bg")
        fg = theme.get_color("fg")
        select_bg = theme.get_color("select_bg")
        select_fg = theme.get_color("select_fg")
        active_bg = theme.get_color("active_bg")
        active_fg = theme.get_color("active_fg")
        disabled_fg = theme.get_color("disabled_fg")

        # Configure default options
        root.option_add("*Background", bg)
        root.option_add("*Foreground", fg)
        root.option_add("*SelectBackground", select_bg)
        root.option_add("*SelectForeground", select_fg)
        root.option_add("*DisabledForeground", disabled_fg)
        root.option_add("*ActiveBackground", active_bg)
        root.option_add("*ActiveForeground", active_fg)

        # Configure specific widget defaults
        root.option_add("*Listbox.Background", theme.get_color("listbox", "bg"))
        root.option_add("*Listbox.Foreground", theme.get_color("listbox", "fg"))
        root.option_add("*Listbox.SelectBackground", theme.get_color("listbox", "select_bg"))
        root.option_add("*Listbox.SelectForeground", theme.get_color("listbox", "select_fg"))

        root.option_add("*Text.Background", theme.get_color("text", "bg"))
        root.option_add("*Text.Foreground", theme.get_color("text", "fg"))
        root.option_add("*Text.SelectBackground", theme.get_color("text", "select_bg"))
        root.option_add("*Text.SelectForeground", theme.get_color("text", "select_fg"))

        root.option_add("*Menu.Background", theme.get_color("menu", "bg"))
        root.option_add("*Menu.Foreground", theme.get_color("menu", "fg"))
        root.option_add("*Menu.ActiveBackground", theme.get_color("menu", "active_bg"))
        root.option_add("*Menu.ActiveForeground", theme.get_color("menu", "active_fg"))

    def add_theme_change_callback(self, callback: Callable[[Theme], None]):
        """Add a callback to be called when theme changes."""
        self.theme_change_callbacks.append(callback)

    def remove_theme_change_callback(self, callback: Callable[[Theme], None]):
        """Remove a theme change callback."""
        if callback in self.theme_change_callbacks:
            self.theme_change_callbacks.remove(callback)

    def _notify_theme_change(self, theme: Theme):
        """Notify all callbacks of theme change."""
        for callback in self.theme_change_callbacks:
            try:
                callback(theme)
            except Exception as e:
                logger.error(f"Error in theme change callback: {e}")

    def create_custom_theme(self, name: str, base_theme: str = "light") -> bool:
        """
        Create a new custom theme based on an existing theme.

        Args:
            name: Name for the new theme
            base_theme: Base theme to copy from

        Returns:
            True if successful, False otherwise
        """
        if base_theme not in self.themes:
            logger.error(f"Base theme '{base_theme}' not found")
            return False

        if name in self.themes:
            logger.warning(f"Theme '{name}' already exists")
            return False

        # Copy base theme configuration
        base_config = self.themes[base_theme].config.copy()
        base_config["name"] = name

        # Create new theme
        self.themes[name] = Theme(name, base_config)

        logger.info(f"Created custom theme: {name}")
        return True

    def save_custom_theme(self, name: str, file_path: str) -> bool:
        """
        Save a custom theme to a file.

        Args:
            name: Theme name to save
            file_path: Path to save the theme file

        Returns:
            True if successful, False otherwise
        """
        if name not in self.themes:
            logger.error(f"Theme '{name}' not found")
            return False

        try:
            theme_data = {
                "name": name,
                "config": self.themes[name].config
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(theme_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved theme '{name}' to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save theme '{name}': {e}")
            return False

    def load_custom_theme(self, file_path: str) -> bool:
        """
        Load a custom theme from a file.

        Args:
            file_path: Path to the theme file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                theme_data = json.load(f)

            name = theme_data.get("name")
            config = theme_data.get("config")

            if not name or not config:
                logger.error("Invalid theme file format")
                return False

            # Create theme
            self.themes[name] = Theme(name, config)

            logger.info(f"Loaded custom theme: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load theme from {file_path}: {e}")
            return False

    def delete_custom_theme(self, name: str) -> bool:
        """
        Delete a custom theme.

        Args:
            name: Theme name to delete

        Returns:
            True if successful, False otherwise
        """
        # Don't allow deletion of built-in themes
        if name in ["light", "dark", "high_contrast"]:
            logger.error(f"Cannot delete built-in theme: {name}")
            return False

        if name not in self.themes:
            logger.error(f"Theme '{name}' not found")
            return False

        # Switch to default theme if currently using the theme being deleted
        if self.current_theme == name:
            self.set_theme("light")

        del self.themes[name]

        logger.info(f"Deleted custom theme: {name}")
        return True

    def get_current_theme_name(self) -> str:
        """Get the name of the current theme."""
        return self.current_theme or "light"

    def get_current_theme(self) -> Optional[Theme]:
        """Get the current theme object."""
        if self.current_theme:
            return self.themes.get(self.current_theme)
        return None


def demo_themes():
    """Demo function to show theme switching."""
    import time

    root = tk.Tk()
    root.title("Theme Demo")
    root.geometry("400x300")

    # Create theme manager
    theme_manager = ThemeManager()

    # Create some widgets to show theming
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    ttk.Label(frame, text="Theme Demo", font=("Arial", 16, "bold")).pack(pady=10)

    ttk.Label(frame, text="This is a sample label").pack(pady=5)

    entry = ttk.Entry(frame, width=30)
    entry.pack(pady=5)
    entry.insert(0, "Sample text")

    button_frame = ttk.Frame(frame)
    button_frame.pack(pady=10)

    ttk.Button(button_frame, text="Sample Button").pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Another Button").pack(side=tk.LEFT, padx=5)

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

    # Apply initial theme
    theme_manager.apply_theme(root)

    root.mainloop()


if __name__ == "__main__":
    demo_themes()