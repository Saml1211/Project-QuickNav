#!/usr/bin/env python3
"""
Action Buttons Section Module

Handles the main action buttons UI section (Open Folder, Find Documents, etc.).
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class ActionButtonsSection:
    """
    Action buttons UI component.

    Responsibilities:
    - Primary context-aware action button.
    - Optional choose/list document action.
    - Button state management (enabled/disabled).
    """

    def __init__(self, parent, state, layout, tooltip_callback: Optional[Callable] = None):
        """
        Initialize the action buttons section.

        Args:
            parent: Parent widget (main_frame)
            state: GUIState instance
            layout: LayoutManager instance
            tooltip_callback: Optional callback for adding tooltips
        """
        self.parent = parent
        self.state = state
        self.layout = layout
        self.add_tooltip = tooltip_callback

        # UI elements
        self.frame = None
        self.action_button = None
        self.choose_button = None

        # Callbacks
        self.on_action = None
        self.on_choose = None

        self.choose_button_visible = False

        # Create the UI
        self._create_ui()

        logger.info("ActionButtonsSection initialized")

    def _create_ui(self):
        """Create the action buttons section UI."""
        self.frame = ttk.Frame(self.parent)
        spacing = self.layout.get_consistent_spacing()
        self.frame.grid(
            row=5, column=0, sticky="ew",
            pady=(spacing // 2, 0)
        )
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)

        self.action_button = ttk.Button(
            self.frame,
            text="Go",
            command=self._on_action,
            state=tk.DISABLED,
            width=20
        )
        self.action_button.grid(row=0, column=0, sticky="ew")

        self.choose_button = ttk.Button(
            self.frame,
            text="Choose From List",
            command=self._on_choose,
            state=tk.DISABLED,
            width=20
        )
        self.choose_button.grid(row=0, column=1, sticky="ew", padx=(spacing // 2, 0))
        self.choose_button.grid_remove()

        if self.add_tooltip:
            self.add_tooltip(self.action_button, "Execute the selected action")
            self.add_tooltip(self.choose_button, "Show all matching documents to choose from")

    def set_on_action(self, callback: Callable):
        """Set callback for the action button."""
        self.on_action = callback

    def _on_action(self):
        """Handle action button click."""
        if self.on_action:
            try:
                self.on_action()
            except Exception as e:
                logger.error(f"Error in action callback: {e}")

    def _on_choose(self):
        """Handle choose button click."""
        if self.on_choose:
            try:
                self.on_choose()
            except Exception as e:
                logger.error(f"Error in choose callback: {e}")

    def enable_action_button(self):
        """Enable the action button."""
        self.action_button.config(state=tk.NORMAL)
        self.enable_choose_button()

    def disable_action_button(self):
        """Disable the action button."""
        self.action_button.config(state=tk.DISABLED)
        self.disable_choose_button()

    def set_action_button_text(self, text: str):
        """Set the text of the action button."""
        self.action_button.config(text=text)

    def set_on_choose(self, callback: Optional[Callable]):
        """Set callback for the choose button."""
        self.on_choose = callback
        if callback is None:
            self.hide_choose_button()

    def enable_choose_button(self):
        """Enable the choose button if visible and configured."""
        if self.choose_button_visible and self.on_choose:
            self.choose_button.config(state=tk.NORMAL)

    def disable_choose_button(self):
        """Disable the choose button."""
        if self.choose_button is not None:
            self.choose_button.config(state=tk.DISABLED)

    def set_choose_button_text(self, text: str):
        """Set the text of the choose button."""
        if self.choose_button is not None:
            self.choose_button.config(text=text)

    def show_choose_button(self):
        """Show the choose button."""
        if self.choose_button is None or self.on_choose is None:
            return
        if not self.choose_button_visible:
            self.choose_button.grid()
            self.choose_button_visible = True
        self.choose_button.config(state=self.action_button["state"])

    def hide_choose_button(self):
        """Hide the choose button."""
        if self.choose_button is None:
            return
        if self.choose_button_visible:
            self.choose_button.grid_remove()
            self.choose_button_visible = False
        self.choose_button.config(state=tk.DISABLED)

    def disable_navigate_button(self):
        """Backward-compatible alias for disabling the navigation button."""
        self.disable_action_button()

    def enable_navigate_button(self):
        """Backward-compatible alias for enabling the navigation button."""
        self.enable_action_button()

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame

