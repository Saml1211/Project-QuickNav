#!/usr/bin/env python3
"""
GUI Event Coordinator Module

Centralizes event handling and keyboard shortcuts for the GUI application.
"""

import tkinter as tk
from typing import Dict, Callable, Any
import logging

logger = logging.getLogger(__name__)


class EventCoordinator:
    """
    Centralized event handling coordinator.

    Responsibilities:
    - Window event handling
    - Keyboard shortcuts
    - Input validation
    - Event routing to appropriate handlers
    """

    def __init__(self, root: tk.Tk, state, sections: Dict[str, Any]):
        """
        Initialize the event coordinator.

        Args:
            root: The Tkinter root window
            state: GUIState instance
            sections: Dictionary of UI sections
        """
        self.root = root
        self.state = state
        self.sections = sections

        # Event handler callbacks - will be set by main GUI coordinator
        self.handlers: Dict[str, Callable] = {}

        logger.info("EventCoordinator initialized")

    def register_handler(self, event_name: str, handler: Callable):
        """
        Register an event handler.

        Args:
            event_name: Name of the event
            handler: Callback function to handle the event
        """
        self.handlers[event_name] = handler
        logger.debug(f"Registered handler for event: {event_name}")

    def setup_events(self):
        """Set up all event bindings with comprehensive keyboard shortcuts."""
        # Window events
        self.root.bind('<Configure>', self._on_window_configure)
        self.root.bind('<KeyPress>', self._on_key_press)

        # Global shortcuts
        self.root.bind('<Control-Return>', lambda e: self._call_handler('execute_action'))
        self.root.bind('<Return>', lambda e: self._call_handler('execute_action'))
        self.root.bind('<Escape>', lambda e: self._call_handler('hide_window'))
        self.root.bind('<F1>', lambda e: self._call_handler('show_help'))
        self.root.bind('<Control-comma>', lambda e: self._call_handler('show_settings'))

        # Quick navigation shortcuts
        self.root.bind('<Control-f>', lambda e: self._call_handler('focus_search'))
        self.root.bind('<Control-1>', lambda e: self._call_handler('set_folder_mode'))
        self.root.bind('<Control-2>', lambda e: self._call_handler('set_document_mode'))
        self.root.bind('<Control-d>', lambda e: self._call_handler('toggle_theme'))
        self.root.bind('<Control-t>', lambda e: self._call_handler('toggle_always_on_top'))
        self.root.bind('<Control-r>', lambda e: self._call_handler('clear_and_reset'))
        self.root.bind('<Control-Shift-R>', lambda e: self._call_handler('reset_window'))

        # Enhanced functionality shortcuts
        self.root.bind('<Control-Shift-C>', lambda e: self._call_handler('copy_path'))
        self.root.bind('<F11>', lambda e: self._call_handler('toggle_fullscreen'))
        self.root.bind('<Control-Shift-F>', lambda e: self._call_handler('focus_and_select_all'))

        # AI shortcuts
        self.root.bind('<Control-space>', lambda e: self._call_handler('toggle_ai_panel'))
        self.root.bind('<Control-slash>', lambda e: self._call_handler('toggle_ai'))

        # Settings shortcuts
        self.root.bind('<Control-s>', lambda e: self._call_handler('show_settings'))
        self.root.bind('<Control-h>', lambda e: self._call_handler('show_help'))

        # Focus events
        self.root.bind('<FocusIn>', self._on_focus_in)

        logger.info("Event bindings configured")

    def _call_handler(self, event_name: str, *args, **kwargs):
        """
        Call a registered event handler.

        Args:
            event_name: Name of the event to handle
            *args: Positional arguments for handler
            **kwargs: Keyword arguments for handler
        """
        if event_name in self.handlers:
            try:
                return self.handlers[event_name](*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in handler for {event_name}: {e}")
        else:
            logger.warning(f"No handler registered for event: {event_name}")

    def _on_window_configure(self, event):
        """Handle window configuration changes (resize, move)."""
        if event.widget == self.root:
            self._call_handler('window_configure', event)

    def _on_key_press(self, event):
        """Handle key press events for accessibility."""
        self._call_handler('key_press', event)

    def _on_focus_in(self, event):
        """Handle focus in events."""
        self._call_handler('focus_in', event)

    def validate_project_input(self, value: str) -> bool:
        """
        Validate project input.

        Args:
            value: Input value to validate

        Returns:
            True if valid, False otherwise
        """
        # Allow empty string
        if not value:
            return True

        # Allow partial 5-digit numbers
        if value.isdigit() and len(value) <= 5:
            return True

        # Allow search terms (alphanumeric with spaces and basic punctuation)
        if len(value) <= 100:  # Reasonable max length
            return all(c.isalnum() or c.isspace() or c in '-_.,()' for c in value)

        return False

    def validate_numeric(self, value: str) -> bool:
        """
        Validate numeric input.

        Args:
            value: Input value to validate

        Returns:
            True if valid, False otherwise
        """
        # Allow empty string
        if not value:
            return True

        # Allow only digits
        return value.isdigit()

    def setup_validation(self, project_entry, room_entry, co_entry):
        """
        Set up input validation for entry fields.

        Args:
            project_entry: Project input entry widget
            room_entry: Room filter entry widget
            co_entry: Change order entry widget
        """
        # Register validation commands
        vcmd = (self.root.register(self.validate_project_input), '%P')
        project_entry.config(validate='key', validatecommand=vcmd)

        # Numeric validation for filters
        vcmd_num = (self.root.register(self.validate_numeric), '%P')
        room_entry.config(validate='key', validatecommand=vcmd_num)
        co_entry.config(validate='key', validatecommand=vcmd_num)

        logger.info("Input validation configured")

    def validate_inputs(self) -> tuple[bool, str]:
        """
        Validate all inputs before navigation.

        Returns:
            Tuple of (is_valid, error_message)
        """
        project_input = self.state.get_project_input().strip()

        if not project_input:
            return False, "Please enter a project number or search term"

        # Basic validation passed
        return True, ""
