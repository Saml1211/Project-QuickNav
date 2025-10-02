"""
Settings Management for Project QuickNav GUI

This module provides comprehensive settings management including:
- JSON-based configuration storage
- Settings dialog with tabbed interface
- Custom root path management
- Theme preferences
- Hotkey customization
- Import/Export functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manages application settings with JSON persistence."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self.settings = self._load_default_settings()
        self.load()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Use AppData on Windows, ~/.config on Unix-like systems
        if os.name == 'nt':  # Windows
            config_dir = os.path.join(os.environ.get('APPDATA', ''), 'QuickNav')
        else:  # Unix-like
            config_dir = os.path.join(os.path.expanduser('~'), '.config', 'quicknav')

        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, 'settings.json')

    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default settings."""
        return {
            "version": "2.0.0",
            "custom_roots": [],
            "theme": {
                "current_theme": "system",  # system, light, dark
                "custom_themes": {}
            },
            "hotkeys": {
                "toggle_window": "ctrl+alt+q",
                "execute_action": "ctrl+return",
                "hide_window": "escape",
                "show_help": "f1",
                "show_settings": "ctrl+comma"
            },
            "ui": {
                "window_geometry": "520x820",
                "always_on_top": False,
                "auto_hide_delay": 1500,
                "show_recent_projects": True,
                "max_recent_projects": 10
            },
            "navigation": {
                "default_mode": "folder",
                "default_folder": "4. System Designs",
                "default_doc_type": "lld",
                "remember_last_selections": True
            },
            "advanced": {
                "cache_enabled": True,
                "cache_timeout_minutes": 5,
                "debug_mode": False,
                "training_data_enabled": False,
                "auto_backup_settings": True
            },
            "ai": {
                "enabled": True,
                "default_model": "gpt-3.5-turbo",
                "api_keys": {},
                "max_conversation_history": 50,
                "temperature": 0.7,
                "max_tokens": 1000,
                "tool_execution_enabled": True,
                "auto_suggestions": True,
                "conversation_memory": True,
                "custom_endpoints": {}
            }
        }

    def load(self) -> bool:
        """
        Load settings from file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)

                # Merge with defaults to ensure all keys exist
                self.settings = self._merge_settings(self.settings, loaded_settings)

                # Validate settings
                self._validate_settings()

                logger.info(f"Settings loaded from {self.config_file}")
                return True
            else:
                logger.info("No settings file found, using defaults")
                return False

        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            # Keep defaults on error
            return False

    def save(self) -> bool:
        """
        Save settings to file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create backup if enabled
            if self.settings.get("advanced", {}).get("auto_backup_settings", True):
                self._create_backup()

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            # Save settings
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)

            logger.info(f"Settings saved to {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False

    def _merge_settings(self, defaults: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded settings with defaults."""
        result = defaults.copy()

        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_settings(self):
        """Validate and fix any invalid settings."""
        # Validate theme
        valid_themes = ["system", "light", "dark"]
        current_theme = self.settings.get("theme", {}).get("current_theme", "system")
        if current_theme not in valid_themes:
            self.settings["theme"]["current_theme"] = "system"

        # Validate custom roots
        custom_roots = self.settings.get("custom_roots", [])
        valid_roots = []
        for root in custom_roots:
            if isinstance(root, str) and root.strip():
                valid_roots.append(root.strip())
        self.settings["custom_roots"] = valid_roots

        # Validate window geometry
        geometry = self.settings.get("ui", {}).get("window_geometry", "520x820")
        if not self._is_valid_geometry(geometry):
            self.settings["ui"]["window_geometry"] = "520x820"

    def _is_valid_geometry(self, geometry: str) -> bool:
        """Check if geometry string is valid."""
        try:
            # Format: "WIDTHxHEIGHT" or "WIDTHxHEIGHT+X+Y"
            parts = geometry.split('+')[0].split('x')
            if len(parts) == 2:
                width, height = int(parts[0]), int(parts[1])
                return 200 <= width <= 3000 and 200 <= height <= 2000
        except:
            pass
        return False

    def _create_backup(self):
        """Create a backup of the current settings file."""
        try:
            if os.path.exists(self.config_file):
                backup_dir = os.path.join(os.path.dirname(self.config_file), 'backups')
                os.makedirs(backup_dir, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = os.path.join(backup_dir, f'settings_backup_{timestamp}.json')

                shutil.copy2(self.config_file, backup_file)

                # Keep only last 10 backups
                self._cleanup_old_backups(backup_dir)

        except Exception as e:
            logger.warning(f"Failed to create settings backup: {e}")

    def _cleanup_old_backups(self, backup_dir: str, max_backups: int = 10):
        """Clean up old backup files."""
        try:
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.startswith('settings_backup_') and file.endswith('.json'):
                    full_path = os.path.join(backup_dir, file)
                    backup_files.append((full_path, os.path.getmtime(full_path)))

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)

            # Remove excess backups
            for file_path, _ in backup_files[max_backups:]:
                os.remove(file_path)

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    # Getter methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value using dot notation."""
        keys = key.split('.')
        value = self.settings

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_custom_roots(self) -> List[str]:
        """Get list of custom root directories."""
        return self.settings.get("custom_roots", [])

    def get_theme(self) -> str:
        """Get current theme name."""
        return self.get("theme.current_theme", "system")

    def get_hotkey(self, action: str) -> str:
        """Get hotkey for an action."""
        return self.get(f"hotkeys.{action}", "")

    def get_window_geometry(self) -> str:
        """Get window geometry."""
        return self.get("ui.window_geometry", "")

    def get_default_mode(self) -> str:
        """Get default navigation mode."""
        return self.get("navigation.default_mode", "folder")

    # Setter methods
    def set(self, key: str, value: Any):
        """Set a setting value using dot notation."""
        keys = key.split('.')
        setting = self.settings

        for k in keys[:-1]:
            if k not in setting:
                setting[k] = {}
            setting = setting[k]

        setting[keys[-1]] = value

    def set_custom_roots(self, roots: List[str]):
        """Set custom root directories."""
        self.settings["custom_roots"] = [root for root in roots if root.strip()]

    def set_theme(self, theme: str):
        """Set current theme."""
        if theme in ["system", "light", "dark"]:
            self.set("theme.current_theme", theme)

    def set_hotkey(self, action: str, hotkey: str):
        """Set hotkey for an action."""
        self.set(f"hotkeys.{action}", hotkey)

    def set_window_geometry(self, geometry: str):
        """Set window geometry."""
        if self._is_valid_geometry(geometry):
            self.set("ui.window_geometry", geometry)

    # Import/Export methods
    def export_settings(self, file_path: str) -> bool:
        """Export settings to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            return False

    def import_settings(self, file_path: str, merge: bool = True) -> bool:
        """Import settings from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_settings = json.load(f)

            if merge:
                self.settings = self._merge_settings(self.settings, imported_settings)
            else:
                self.settings = imported_settings

            self._validate_settings()
            return True

        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            return False

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = self._load_default_settings()


class SettingsDialog:
    """Settings dialog with tabbed interface."""

    def __init__(self, parent, settings_manager: SettingsManager):
        self.parent = parent
        self.settings = settings_manager
        self.dialog = None
        self.changes_made = False

        # Track original values for cancellation
        self.original_settings = json.loads(json.dumps(settings_manager.settings))

    def show(self):
        """Show the settings dialog."""
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.lift()
            return

        self._create_dialog()

    def _create_dialog(self):
        """Create the settings dialog."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Settings")
        self.dialog.geometry("600x500")
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.transient(self.parent)

        self._center_on_parent()
        self._create_content()

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        self.dialog.geometry(f"+{x}+{y}")

    def _create_content(self):
        """Create dialog content."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.dialog)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self._create_general_tab()
        self._create_paths_tab()
        self._create_theme_tab()
        self._create_hotkeys_tab()
        self._create_ai_tab()
        self._create_advanced_tab()

        # Button frame
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Import/Export buttons
        ttk.Button(
            button_frame,
            text="Import...",
            command=self._import_settings,
            width=10
        ).pack(side=tk.LEFT)

        ttk.Button(
            button_frame,
            text="Export...",
            command=self._export_settings,
            width=10
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._reset_to_defaults,
            width=15
        ).pack(side=tk.LEFT, padx=(20, 0))

        # OK/Cancel buttons
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel,
            width=10
        ).pack(side=tk.RIGHT)

        ttk.Button(
            button_frame,
            text="OK",
            command=self._ok,
            width=10
        ).pack(side=tk.RIGHT, padx=(0, 5))

        ttk.Button(
            button_frame,
            text="Apply",
            command=self._apply,
            width=10
        ).pack(side=tk.RIGHT, padx=(0, 5))

    def _create_general_tab(self):
        """Create general settings tab with modern styling."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙️ General")

        # Window settings with card-style frame
        window_group = ttk.LabelFrame(frame, text="🪟 Window Settings")
        window_group.pack(fill=tk.X, padx=16, pady=(16, 20))

        # Always on top with enhanced styling
        self.always_on_top_var = tk.BooleanVar()
        self.always_on_top_var.set(self.settings.get("ui.always_on_top", False))
        ttk.Checkbutton(
            window_group,
            text="📌 Keep window always on top",
            variable=self.always_on_top_var
        ).pack(anchor=tk.W, padx=16, pady=(12, 8))

        # Auto hide delay with better layout
        delay_frame = ttk.Frame(window_group)
        delay_frame.pack(fill=tk.X, padx=16, pady=(8, 12))

        ttk.Label(
            delay_frame,
            text="⏱️ Auto-hide delay (ms):",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)

        self.auto_hide_var = tk.StringVar()
        self.auto_hide_var.set(str(self.settings.get("ui.auto_hide_delay", 1500)))
        entry = ttk.Entry(
            delay_frame,
            textvariable=self.auto_hide_var,
            width=12,
            font=("Segoe UI", 9)
        )
        entry.pack(side=tk.LEFT, padx=(12, 0), ipady=2)

        # Navigation defaults with enhanced styling
        nav_group = ttk.LabelFrame(frame, text="🧭 Navigation Defaults")
        nav_group.pack(fill=tk.X, padx=16, pady=(0, 20))
        nav_group.columnconfigure(1, weight=1)

        # Default mode with improved layout
        ttk.Label(
            nav_group,
            text="🎯 Default mode:",
            font=("Segoe UI", 9)
        ).grid(row=0, column=0, sticky="w", padx=(16, 12), pady=(12, 8))

        self.default_mode_var = tk.StringVar()
        self.default_mode_var.set(self.settings.get("navigation.default_mode", "folder"))
        mode_combo = ttk.Combobox(
            nav_group,
            textvariable=self.default_mode_var,
            values=["folder", "document"],
            state="readonly",
            width=18,
            font=("Segoe UI", 9)
        )
        mode_combo.grid(row=0, column=1, sticky="ew", padx=(0, 16), pady=(12, 8), ipady=2)

        # Remember selections with icon
        self.remember_selections_var = tk.BooleanVar()
        self.remember_selections_var.set(
            self.settings.get("navigation.remember_last_selections", True)
        )
        ttk.Checkbutton(
            nav_group,
            text="💾 Remember last selections",
            variable=self.remember_selections_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=16, pady=(8, 12))

    def _create_paths_tab(self):
        """Create custom paths tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Custom Paths")

        # Instructions
        instructions = ttk.Label(
            frame,
            text="Add custom root directories to search for projects:",
            wraplength=550
        )
        instructions.pack(pady=10)

        # Listbox frame
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.paths_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set
        )
        self.paths_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.paths_listbox.yview)

        # Populate listbox
        for root in self.settings.get_custom_roots():
            self.paths_listbox.insert(tk.END, root)

        # Button frame
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            button_frame,
            text="Add...",
            command=self._add_custom_path,
            width=10
        ).pack(side=tk.LEFT)

        ttk.Button(
            button_frame,
            text="Edit...",
            command=self._edit_custom_path,
            width=10
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Button(
            button_frame,
            text="Remove",
            command=self._remove_custom_path,
            width=10
        ).pack(side=tk.LEFT, padx=(5, 0))

    def _create_theme_tab(self):
        """Create theme settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Theme")

        # Theme selection
        theme_group = ttk.LabelFrame(frame, text="Theme Selection")
        theme_group.pack(fill=tk.X, padx=10, pady=10)

        self.theme_var = tk.StringVar()
        self.theme_var.set(self.settings.get_theme())

        themes = [
            ("System Default", "system"),
            ("Light Theme", "light"),
            ("Dark Theme", "dark")
        ]

        for text, value in themes:
            ttk.Radiobutton(
                theme_group,
                text=text,
                variable=self.theme_var,
                value=value
            ).pack(anchor=tk.W, padx=10, pady=2)

        # Theme preview (placeholder)
        preview_group = ttk.LabelFrame(frame, text="Preview")
        preview_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        preview_label = ttk.Label(
            preview_group,
            text="Theme preview would be shown here",
            anchor=tk.CENTER
        )
        preview_label.pack(expand=True)

    def _create_hotkeys_tab(self):
        """Create hotkey settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Hotkeys")

        # Hotkey settings
        hotkey_group = ttk.LabelFrame(frame, text="Global Hotkeys")
        hotkey_group.pack(fill=tk.X, padx=10, pady=10)

        self.hotkey_vars = {}
        hotkey_actions = [
            ("Toggle Window", "toggle_window"),
            ("Execute Action", "execute_action"),
            ("Hide Window", "hide_window"),
            ("Show Help", "show_help"),
            ("Show Settings", "show_settings")
        ]

        for i, (label, action) in enumerate(hotkey_actions):
            row_frame = ttk.Frame(hotkey_group)
            row_frame.pack(fill=tk.X, padx=10, pady=2)

            ttk.Label(row_frame, text=f"{label}:", width=15).pack(side=tk.LEFT)

            var = tk.StringVar()
            var.set(self.settings.get_hotkey(action))
            self.hotkey_vars[action] = var

            ttk.Entry(
                row_frame,
                textvariable=var,
                width=20
            ).pack(side=tk.LEFT, padx=(10, 0))

        # Warning label
        warning_label = ttk.Label(
            frame,
            text="Note: Global hotkey changes require application restart",
            foreground="orange"
        )
        warning_label.pack(pady=10)

    def _create_ai_tab(self):
        """Create AI settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="AI Assistant")

        # AI Enable/Disable
        ai_group = ttk.LabelFrame(frame, text="AI Assistant Settings")
        ai_group.pack(fill=tk.X, padx=10, pady=10)

        # Enable AI
        self.ai_enabled_var = tk.BooleanVar()
        self.ai_enabled_var.set(self.settings.get("ai.enabled", True))
        ttk.Checkbutton(
            ai_group,
            text="Enable AI Assistant",
            variable=self.ai_enabled_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Model selection
        model_frame = ttk.Frame(ai_group)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(model_frame, text="AI Model:").pack(side=tk.LEFT)
        self.ai_model_var = tk.StringVar()
        self.ai_model_var.set(self.settings.get("ai.default_model", "gpt-3.5-turbo"))

        models = [
            "gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
            "azure/gpt-4", "azure/gpt-35-turbo",
            "ollama/llama2", "ollama/codellama", "ollama/mistral"
        ]

        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.ai_model_var,
            values=models,
            state="readonly",
            width=25
        )
        model_combo.pack(side=tk.LEFT, padx=(10, 0))

        # API Keys section
        api_group = ttk.LabelFrame(frame, text="API Configuration")
        api_group.pack(fill=tk.X, padx=10, pady=10)

        # OpenAI API Key
        openai_frame = ttk.Frame(api_group)
        openai_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(openai_frame, text="OpenAI API Key:", width=15).pack(side=tk.LEFT)
        self.openai_key_var = tk.StringVar()
        self.openai_key_var.set(self.settings.get("ai.api_keys.openai", ""))
        openai_entry = ttk.Entry(
            openai_frame,
            textvariable=self.openai_key_var,
            show="*",
            width=40
        )
        openai_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        # Anthropic API Key
        anthropic_frame = ttk.Frame(api_group)
        anthropic_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(anthropic_frame, text="Anthropic API Key:", width=15).pack(side=tk.LEFT)
        self.anthropic_key_var = tk.StringVar()
        self.anthropic_key_var.set(self.settings.get("ai.api_keys.anthropic", ""))
        anthropic_entry = ttk.Entry(
            anthropic_frame,
            textvariable=self.anthropic_key_var,
            show="*",
            width=40
        )
        anthropic_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        # Azure API Key
        azure_frame = ttk.Frame(api_group)
        azure_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(azure_frame, text="Azure API Key:", width=15).pack(side=tk.LEFT)
        self.azure_key_var = tk.StringVar()
        self.azure_key_var.set(self.settings.get("ai.api_keys.azure", ""))
        azure_entry = ttk.Entry(
            azure_frame,
            textvariable=self.azure_key_var,
            show="*",
            width=40
        )
        azure_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        # Conversation settings
        conv_group = ttk.LabelFrame(frame, text="Conversation Settings")
        conv_group.pack(fill=tk.X, padx=10, pady=10)

        # Max conversation history
        history_frame = ttk.Frame(conv_group)
        history_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(history_frame, text="Max conversation history:").pack(side=tk.LEFT)
        self.max_history_var = tk.StringVar()
        self.max_history_var.set(str(self.settings.get("ai.max_conversation_history", 50)))
        ttk.Entry(
            history_frame,
            textvariable=self.max_history_var,
            width=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Temperature
        temp_frame = ttk.Frame(conv_group)
        temp_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(temp_frame, text="Temperature (0.0-2.0):").pack(side=tk.LEFT)
        self.temperature_var = tk.StringVar()
        self.temperature_var.set(str(self.settings.get("ai.temperature", 0.7)))
        ttk.Entry(
            temp_frame,
            textvariable=self.temperature_var,
            width=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Max tokens
        tokens_frame = ttk.Frame(conv_group)
        tokens_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(tokens_frame, text="Max tokens per response:").pack(side=tk.LEFT)
        self.max_tokens_var = tk.StringVar()
        self.max_tokens_var.set(str(self.settings.get("ai.max_tokens", 1000)))
        ttk.Entry(
            tokens_frame,
            textvariable=self.max_tokens_var,
            width=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Feature toggles
        features_group = ttk.LabelFrame(frame, text="Features")
        features_group.pack(fill=tk.X, padx=10, pady=10)

        # Tool execution
        self.tool_execution_var = tk.BooleanVar()
        self.tool_execution_var.set(self.settings.get("ai.tool_execution_enabled", True))
        ttk.Checkbutton(
            features_group,
            text="Enable tool execution (project search, document analysis)",
            variable=self.tool_execution_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Auto suggestions
        self.auto_suggestions_var = tk.BooleanVar()
        self.auto_suggestions_var.set(self.settings.get("ai.auto_suggestions", True))
        ttk.Checkbutton(
            features_group,
            text="Enable auto-suggestions",
            variable=self.auto_suggestions_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Conversation memory
        self.conversation_memory_var = tk.BooleanVar()
        self.conversation_memory_var.set(self.settings.get("ai.conversation_memory", True))
        ttk.Checkbutton(
            features_group,
            text="Remember conversation context",
            variable=self.conversation_memory_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Test connection button
        test_frame = ttk.Frame(frame)
        test_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            test_frame,
            text="Test AI Connection",
            command=self._test_ai_connection,
            width=20
        ).pack()

    def _test_ai_connection(self):
        """Test AI connection with current settings."""
        try:
            from quicknav.ai_client import AIClient

            # Get current settings from the form
            model = self.ai_model_var.get()
            openai_key = self.openai_key_var.get().strip()
            anthropic_key = self.anthropic_key_var.get().strip()
            azure_key = self.azure_key_var.get().strip()

            if not model:
                messagebox.showwarning("Test Failed", "Please select a model first.")
                return

            # Create temporary AI client with test settings
            test_settings = {
                'ai': {
                    'enabled': True,
                    'default_model': model,
                    'api_keys': {
                        'openai': openai_key,
                        'anthropic': anthropic_key,
                        'azure': azure_key
                    },
                    'temperature': float(self.temperature_var.get()),
                    'max_tokens': int(self.max_tokens_var.get())
                }
            }

            # Create a simple settings object for testing
            class TestSettings:
                def __init__(self, data):
                    self.data = data
                def get(self, key, default=None):
                    keys = key.split('.')
                    value = self.data
                    for k in keys:
                        if isinstance(value, dict) and k in value:
                            value = value[k]
                        else:
                            return default
                    return value

            test_settings_obj = TestSettings(test_settings)
            ai_client = AIClient(settings=test_settings_obj)

            # Test with a simple message
            import asyncio
            response = asyncio.run(ai_client.chat("Hello, this is a connection test."))

            if response and len(response.strip()) > 0:
                messagebox.showinfo(
                    "Test Successful",
                    f"AI connection test successful!\n\nModel: {model}\nResponse: {response[:100]}..."
                )
            else:
                messagebox.showwarning("Test Failed", "AI responded but with empty content.")

        except ImportError:
            messagebox.showerror(
                "Test Failed",
                "AI client module not available. Please ensure all dependencies are installed."
            )
        except Exception as e:
            messagebox.showerror("Test Failed", f"Connection test failed:\n{str(e)}")

    def _on_model_change(self, *args):
        """Handle model selection change."""
        if hasattr(self, 'ai_model_var'):
            model = self.ai_model_var.get()
            # Enable/disable API key fields based on model
            if model.startswith('gpt-') or model.startswith('o1-'):
                if hasattr(self, 'openai_key_entry'):
                    self.openai_key_entry.config(state='normal')
            elif model.startswith('claude-'):
                if hasattr(self, 'anthropic_key_entry'):
                    self.anthropic_key_entry.config(state='normal')
            elif model.startswith('azure-'):
                if hasattr(self, 'azure_key_entry'):
                    self.azure_key_entry.config(state='normal')

    def _create_advanced_tab(self):
        """Create advanced settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Advanced")

        # Performance settings
        perf_group = ttk.LabelFrame(frame, text="Performance")
        perf_group.pack(fill=tk.X, padx=10, pady=10)

        # Cache enabled
        self.cache_enabled_var = tk.BooleanVar()
        self.cache_enabled_var.set(self.settings.get("advanced.cache_enabled", True))
        ttk.Checkbutton(
            perf_group,
            text="Enable result caching",
            variable=self.cache_enabled_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Cache timeout
        cache_frame = ttk.Frame(perf_group)
        cache_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(cache_frame, text="Cache timeout (minutes):").pack(side=tk.LEFT)
        self.cache_timeout_var = tk.StringVar()
        self.cache_timeout_var.set(str(self.settings.get("advanced.cache_timeout_minutes", 5)))
        ttk.Entry(
            cache_frame,
            textvariable=self.cache_timeout_var,
            width=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Debug settings
        debug_group = ttk.LabelFrame(frame, text="Debug & Development")
        debug_group.pack(fill=tk.X, padx=10, pady=10)

        # Debug mode
        self.debug_mode_var = tk.BooleanVar()
        self.debug_mode_var.set(self.settings.get("advanced.debug_mode", False))
        ttk.Checkbutton(
            debug_group,
            text="Enable debug mode",
            variable=self.debug_mode_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Training data
        self.training_data_var = tk.BooleanVar()
        self.training_data_var.set(self.settings.get("advanced.training_data_enabled", False))
        ttk.Checkbutton(
            debug_group,
            text="Enable training data generation by default",
            variable=self.training_data_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Backup settings
        backup_group = ttk.LabelFrame(frame, text="Backup")
        backup_group.pack(fill=tk.X, padx=10, pady=10)

        # Auto backup
        self.auto_backup_var = tk.BooleanVar()
        self.auto_backup_var.set(self.settings.get("advanced.auto_backup_settings", True))
        ttk.Checkbutton(
            backup_group,
            text="Automatically backup settings",
            variable=self.auto_backup_var
        ).pack(anchor=tk.W, padx=10, pady=2)

    # Event handlers
    def _add_custom_path(self):
        """Add a new custom path."""
        path = filedialog.askdirectory(
            title="Select Project Root Directory",
            initialdir=os.path.expanduser("~")
        )

        if path:
            self.paths_listbox.insert(tk.END, path)
            self.changes_made = True

    def _edit_custom_path(self):
        """Edit selected custom path."""
        selection = self.paths_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a path to edit.")
            return

        current_path = self.paths_listbox.get(selection[0])
        path = filedialog.askdirectory(
            title="Edit Project Root Directory",
            initialdir=current_path
        )

        if path:
            self.paths_listbox.delete(selection[0])
            self.paths_listbox.insert(selection[0], path)
            self.changes_made = True

    def _remove_custom_path(self):
        """Remove selected custom path."""
        selection = self.paths_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a path to remove.")
            return

        result = messagebox.askyesno(
            "Confirm Removal",
            "Remove the selected path?"
        )

        if result:
            self.paths_listbox.delete(selection[0])
            self.changes_made = True

    def _import_settings(self):
        """Import settings from file."""
        file_path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            result = messagebox.askyesnocancel(
                "Import Settings",
                "Merge with current settings (Yes) or replace all settings (No)?"
            )

            if result is not None:
                merge = result  # Yes = True (merge), No = False (replace)
                if self.settings.import_settings(file_path, merge):
                    messagebox.showinfo("Success", "Settings imported successfully.")
                    self._refresh_dialog()
                else:
                    messagebox.showerror("Error", "Failed to import settings.")

    def _export_settings(self):
        """Export settings to file."""
        file_path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            if self.settings.export_settings(file_path):
                messagebox.showinfo("Success", "Settings exported successfully.")
            else:
                messagebox.showerror("Error", "Failed to export settings.")

    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        result = messagebox.askyesno(
            "🔄 Reset Settings",
            "⚠️ Reset all settings to default values?\n\nThis action cannot be undone and will restore all settings to their original state."
        )

        if result:
            self.settings.reset_to_defaults()
            self._refresh_dialog()
            messagebox.showinfo("✅ Reset Complete", "🎉 All settings have been successfully reset to defaults.")

    def _refresh_dialog(self):
        """Refresh dialog with current settings."""
        # Close and recreate dialog
        if self.dialog:
            self.dialog.destroy()
        self._create_dialog()

    def _apply_settings(self):
        """Apply settings without closing dialog."""
        # Update custom roots
        custom_roots = []
        for i in range(self.paths_listbox.size()):
            custom_roots.append(self.paths_listbox.get(i))
        self.settings.set_custom_roots(custom_roots)

        # Update other settings
        self.settings.set("ui.always_on_top", self.always_on_top_var.get())
        self.settings.set("ui.auto_hide_delay", int(self.auto_hide_var.get() or 1500))
        self.settings.set("navigation.default_mode", self.default_mode_var.get())
        self.settings.set("navigation.remember_last_selections", self.remember_selections_var.get())
        self.settings.set_theme(self.theme_var.get())

        # Update hotkeys
        for action, var in self.hotkey_vars.items():
            self.settings.set_hotkey(action, var.get())

        # Update advanced settings
        self.settings.set("advanced.cache_enabled", self.cache_enabled_var.get())
        self.settings.set("advanced.cache_timeout_minutes", int(self.cache_timeout_var.get() or 5))
        self.settings.set("advanced.debug_mode", self.debug_mode_var.get())
        self.settings.set("advanced.training_data_enabled", self.training_data_var.get())
        self.settings.set("advanced.auto_backup_settings", self.auto_backup_var.get())

        # Update AI settings if they exist (AI tab may not be created yet)
        if hasattr(self, 'ai_enabled_var'):
            self.settings.set("ai.enabled", self.ai_enabled_var.get())
            self.settings.set("ai.default_model", self.ai_model_var.get())
            self.settings.set("ai.api_keys.openai", self.openai_key_var.get())
            self.settings.set("ai.api_keys.anthropic", self.anthropic_key_var.get())
            self.settings.set("ai.api_keys.azure", self.azure_key_var.get())
            self.settings.set("ai.max_conversation_history", int(self.max_history_var.get() or 50))
            self.settings.set("ai.temperature", float(self.temperature_var.get() or 0.7))
            self.settings.set("ai.max_tokens", int(self.max_tokens_var.get() or 1000))
            self.settings.set("ai.enable_tool_execution", self.tool_execution_var.get())
            self.settings.set("ai.enable_auto_suggestions", self.auto_suggestions_var.get())
            self.settings.set("ai.enable_conversation_memory", self.conversation_memory_var.get())

        self.changes_made = True

    def _apply(self):
        """Apply settings."""
        self._apply_settings()
        self.settings.save()
        messagebox.showinfo("Settings Applied", "Settings have been applied and saved.")

    def _ok(self):
        """Apply settings and close dialog."""
        self._apply_settings()
        self.settings.save()
        self.dialog.destroy()

    def _cancel(self):
        """Cancel changes and close dialog."""
        if self.changes_made:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?"
            )

            if result is True:  # Yes - save
                self._apply_settings()
                self.settings.save()
            elif result is False:  # No - discard
                # Restore original settings
                self.settings.settings = self.original_settings
            else:  # Cancel - don't close
                return

        self.dialog.destroy()

    def destroy(self):
        """Destroy the settings dialog and clean up resources."""
        if hasattr(self, 'dialog') and self.dialog and self.dialog.winfo_exists():
            try:
                # Release grab and destroy dialog
                self.dialog.grab_release()
                self.dialog.destroy()
            except Exception as e:
                logger.warning(f"Error during settings dialog cleanup: {e}")
            finally:
                self.dialog = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.destroy()
        except Exception:
            pass  # Ignore errors during destruction