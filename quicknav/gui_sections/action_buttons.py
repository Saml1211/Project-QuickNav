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
    - Folder mode action button
    - Document mode action buttons
    - Button visibility based on mode
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
        self.button_container = None
        self.open_button = None
        self.find_button = None
        self.choose_button = None
        self.navigate_button = None

        # Callbacks
        self.on_folder_navigation = None
        self.on_document_navigation = None
        self.on_document_choose = None
        self.on_final_navigation = None

        # Create the UI
        self._create_ui()

        logger.info("ActionButtonsSection initialized")

    def _create_ui(self):
        """Create the action buttons section UI."""
        # Button frame with improved spacing
        self.frame = ttk.Frame(self.parent)
        self.frame.grid(
            row=5, column=0, sticky="ew",
            pady=(self.layout.get_consistent_spacing() // 2, 0)
        )

        # Center the buttons with improved spacing and grid layout
        self.button_container = ttk.Frame(self.frame)
        self.button_container.grid(row=0, column=0)
        self.button_container.columnconfigure(0, weight=1)

        # Folder mode button with primary styling and icon
        self.open_button = ttk.Button(
            self.button_container,
            text="📂 Open Folder",
            command=self._on_folder_navigation,
            width=18
        )
        self.open_button.grid(
            row=0, column=0,
            padx=(0, self.layout.get_consistent_spacing() // 2)
        )

        if self.add_tooltip:
            self.add_tooltip(self.open_button, "Open the selected project subfolder in file explorer")

        # Document mode buttons with improved styling and icons (initially hidden)
        self.find_button = ttk.Button(
            self.button_container,
            text="🔍 Find Documents",
            command=self._on_document_navigation,
            width=16
        )
        self.find_button.grid(
            row=0, column=1,
            padx=(0, self.layout.get_consistent_spacing() // 2)
        )

        if self.add_tooltip:
            self.add_tooltip(self.find_button, "Search for the best matching document")

        self.choose_button = ttk.Button(
            self.button_container,
            text="📋 Choose From List",
            command=self._on_document_choose,
            width=16
        )
        self.choose_button.grid(
            row=0, column=2,
            padx=(0, self.layout.get_consistent_spacing() // 2)
        )

        if self.add_tooltip:
            self.add_tooltip(self.choose_button, "Show all matching documents to choose from")

        # Open/Navigate button (appears after successful search)
        self.navigate_button = ttk.Button(
            self.button_container,
            text="✅ Open Selected",
            command=self._on_final_navigation,
            width=16,
            state=tk.DISABLED
        )
        self.navigate_button.grid(row=0, column=3)

        if self.add_tooltip:
            self.add_tooltip(self.navigate_button, "Open the selected result")

    def set_on_folder_navigation(self, callback: Callable):
        """Set callback for folder navigation."""
        self.on_folder_navigation = callback

    def set_on_document_navigation(self, callback: Callable):
        """Set callback for document navigation."""
        self.on_document_navigation = callback

    def set_on_document_choose(self, callback: Callable):
        """Set callback for document choose mode."""
        self.on_document_choose = callback

    def set_on_final_navigation(self, callback: Callable):
        """Set callback for final navigation."""
        self.on_final_navigation = callback

    def _on_folder_navigation(self):
        """Handle folder navigation button click."""
        if self.on_folder_navigation:
            try:
                self.on_folder_navigation()
            except Exception as e:
                logger.error(f"Error in folder navigation callback: {e}")

    def _on_document_navigation(self):
        """Handle document navigation button click."""
        if self.on_document_navigation:
            try:
                self.on_document_navigation(choose_mode=False)
            except Exception as e:
                logger.error(f"Error in document navigation callback: {e}")

    def _on_document_choose(self):
        """Handle document choose button click."""
        if self.on_document_choose:
            try:
                self.on_document_choose(choose_mode=True)
            except Exception as e:
                logger.error(f"Error in document choose callback: {e}")

    def _on_final_navigation(self):
        """Handle final navigation button click."""
        if self.on_final_navigation:
            try:
                self.on_final_navigation()
            except Exception as e:
                logger.error(f"Error in final navigation callback: {e}")

    def update_button_visibility(self, mode: str):
        """
        Update button visibility based on navigation mode.

        Args:
            mode: Current navigation mode ('folder' or 'document')
        """
        if mode == "folder":
            # Show folder button, hide document buttons
            self.open_button.grid()
            self.find_button.grid_remove()
            self.choose_button.grid_remove()
            self.navigate_button.grid_remove()
        else:
            # Hide folder button, show document buttons
            self.open_button.grid_remove()
            self.find_button.grid()
            self.choose_button.grid()
            # navigate_button stays hidden until a search succeeds

    def enable_navigate_button(self):
        """Enable the navigate button."""
        self.navigate_button.config(state=tk.NORMAL)
        self.navigate_button.grid()

    def disable_navigate_button(self):
        """Disable the navigate button."""
        self.navigate_button.config(state=tk.DISABLED)

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
