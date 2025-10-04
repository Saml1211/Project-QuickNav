#!/usr/bin/env python3
"""
GUI State Management Module

Centralized state management for the Project QuickNav GUI application.
Manages all StringVar, BooleanVar, IntVar variables and provides clean API
for state access and modification.
"""

import tkinter as tk
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class GUIState:
    """
    Centralized state management for GUI application.

    Manages all Tkinter variables and provides observers for state changes.
    This follows the Observer pattern for reactive state management.
    """

    def __init__(self):
        """Initialize all state variables."""
        # Navigation mode state
        self.current_mode = tk.StringVar(value="folder")

        # Project input state
        self.project_input = tk.StringVar()

        # Operational modes
        self.debug_mode = tk.BooleanVar()
        self.training_mode = tk.BooleanVar()

        # Status display
        self.status_text = tk.StringVar(value="Ready - Select navigation mode and enter project number")

        # Document mode state
        self.doc_type = tk.StringVar()  # Selected document type
        self.version_filter = tk.StringVar(value="Auto (Latest/Best)")
        self.room_filter = tk.StringVar()
        self.co_filter = tk.StringVar()  # Change Order filter
        self.include_archive = tk.BooleanVar()

        # Folder mode state
        self.selected_folder = tk.StringVar(value="4. System Designs")

        # AI state
        self.ai_enabled = tk.BooleanVar()
        self.ai_panel_visible = tk.BooleanVar(value=False)

        # UI state
        self.is_resizing = False

        # State change observers
        self._observers: Dict[str, list[Callable]] = {}

        logger.info("GUIState initialized")

    def get_current_mode(self) -> str:
        """Get current navigation mode."""
        return self.current_mode.get()

    def set_current_mode(self, mode: str):
        """Set current navigation mode and notify observers."""
        old_mode = self.current_mode.get()
        self.current_mode.set(mode)
        if old_mode != mode:
            self._notify_observers('mode_changed', old_mode, mode)

    def get_project_input(self) -> str:
        """Get project input text."""
        return self.project_input.get()

    def set_project_input(self, value: str):
        """Set project input text."""
        self.project_input.set(value)

    def get_selected_folder(self) -> str:
        """Get currently selected folder."""
        return self.selected_folder.get()

    def set_selected_folder(self, folder: str):
        """Set selected folder."""
        self.selected_folder.set(folder)

    def get_doc_type(self) -> str:
        """Get selected document type."""
        return self.doc_type.get()

    def set_doc_type(self, doc_type: str):
        """Set document type."""
        self.doc_type.set(doc_type)

    def get_status_text(self) -> str:
        """Get current status text."""
        return self.status_text.get()

    def set_status_text(self, text: str):
        """Set status text."""
        self.status_text.set(text)
        self._notify_observers('status_changed', text)

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug_mode.get()

    def set_debug_mode(self, enabled: bool):
        """Set debug mode."""
        self.debug_mode.set(enabled)

    def is_training_mode(self) -> bool:
        """Check if training mode is enabled."""
        return self.training_mode.get()

    def set_training_mode(self, enabled: bool):
        """Set training mode."""
        self.training_mode.set(enabled)

    def is_ai_enabled(self) -> bool:
        """Check if AI is enabled."""
        return self.ai_enabled.get()

    def set_ai_enabled(self, enabled: bool):
        """Set AI enabled state."""
        old_value = self.ai_enabled.get()
        self.ai_enabled.set(enabled)
        if old_value != enabled:
            self._notify_observers('ai_enabled_changed', enabled)

    def is_ai_panel_visible(self) -> bool:
        """Check if AI panel is visible."""
        return self.ai_panel_visible.get()

    def set_ai_panel_visible(self, visible: bool):
        """Set AI panel visibility."""
        self.ai_panel_visible.set(visible)

    def get_version_filter(self) -> str:
        """Get version filter setting."""
        return self.version_filter.get()

    def get_room_filter(self) -> str:
        """Get room filter setting."""
        return self.room_filter.get()

    def get_co_filter(self) -> str:
        """Get change order filter setting."""
        return self.co_filter.get()

    def is_include_archive(self) -> bool:
        """Check if archive should be included."""
        return self.include_archive.get()

    def reset_form(self):
        """Reset all form inputs to default values."""
        self.project_input.set("")
        self.doc_type.set("")
        self.version_filter.set("Auto (Latest/Best)")
        self.room_filter.set("")
        self.co_filter.set("")
        self.include_archive.set(False)
        self.status_text.set("Ready - Select navigation mode and enter project number")
        logger.info("Form state reset to defaults")

    def get_document_filters(self) -> Dict[str, Any]:
        """
        Get all document filter settings as a dictionary.

        Returns:
            Dictionary containing all document filter settings
        """
        return {
            'doc_type': self.doc_type.get(),
            'version_filter': self.version_filter.get(),
            'room_filter': self.room_filter.get(),
            'co_filter': self.co_filter.get(),
            'include_archive': self.include_archive.get()
        }

    def add_observer(self, event: str, callback: Callable):
        """
        Add an observer for state change events.

        Args:
            event: Event name to observe (e.g., 'mode_changed', 'status_changed')
            callback: Function to call when event occurs
        """
        if event not in self._observers:
            self._observers[event] = []
        self._observers[event].append(callback)
        logger.debug(f"Observer added for event: {event}")

    def remove_observer(self, event: str, callback: Callable):
        """
        Remove an observer for state change events.

        Args:
            event: Event name
            callback: Function to remove
        """
        if event in self._observers and callback in self._observers[event]:
            self._observers[event].remove(callback)
            logger.debug(f"Observer removed for event: {event}")

    def _notify_observers(self, event: str, *args, **kwargs):
        """
        Notify all observers of a state change event.

        Args:
            event: Event name
            *args: Positional arguments to pass to observers
            **kwargs: Keyword arguments to pass to observers
        """
        if event in self._observers:
            for callback in self._observers[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error notifying observer for {event}: {e}")

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of all current state values.

        Returns:
            Dictionary containing all state values
        """
        return {
            'current_mode': self.current_mode.get(),
            'project_input': self.project_input.get(),
            'debug_mode': self.debug_mode.get(),
            'training_mode': self.training_mode.get(),
            'selected_folder': self.selected_folder.get(),
            'doc_type': self.doc_type.get(),
            'version_filter': self.version_filter.get(),
            'room_filter': self.room_filter.get(),
            'co_filter': self.co_filter.get(),
            'include_archive': self.include_archive.get(),
            'ai_enabled': self.ai_enabled.get(),
            'ai_panel_visible': self.ai_panel_visible.get(),
            'status_text': self.status_text.get()
        }
