#!/usr/bin/env python3
"""
Navigation Mode Section Module

Handles the navigation mode selection UI (folder mode vs document mode).
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class NavigationModeSection:
    """
    Navigation mode selection UI component.

    Responsibilities:
    - Folder/Document mode radio buttons
    - Mode change callbacks
    - Tooltips for mode selection
    """

    def __init__(self, parent, state, layout, tooltip_callback: Optional[Callable] = None):
        """
        Initialize the navigation mode section.

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
        self.folder_radio = None
        self.doc_radio = None

        # Mode change callback
        self.on_mode_change_callback = None

        # Create the UI
        self._create_ui()

        logger.info("NavigationModeSection initialized")

    def _create_ui(self):
        """Create the navigation mode section UI."""
        # Navigation mode frame with improved styling
        self.frame = ttk.LabelFrame(self.parent, text="Navigation Mode")
        self.frame.grid(
            row=1, column=0, sticky="ew",
            pady=(0, self.layout.get_consistent_spacing() // 2)
        )
        self.frame.columnconfigure(0, weight=1)

        # Mode selection frame with improved spacing and grid layout
        mode_frame = ttk.Frame(self.frame)
        mode_frame.grid(
            row=0, column=0, sticky="ew",
            padx=self.layout.get_consistent_padding(),
            pady=self.layout.get_consistent_padding()
        )
        mode_frame.columnconfigure(0, weight=1)

        # Radio buttons for navigation mode with better styling and icons using grid layout
        self.folder_radio = ttk.Radiobutton(
            mode_frame,
            text="📁 Open Project Folder",
            variable=self.state.current_mode,
            value="folder",
            command=self._on_mode_change
        )
        self.folder_radio.grid(
            row=0, column=0, sticky="w",
            pady=(0, self.layout.get_consistent_spacing() // 2)
        )

        if self.add_tooltip:
            self.add_tooltip(self.folder_radio, "Navigate directly to a project subfolder")

        self.doc_radio = ttk.Radiobutton(
            mode_frame,
            text="📄 Find Documents",
            variable=self.state.current_mode,
            value="document",
            command=self._on_mode_change
        )
        self.doc_radio.grid(row=1, column=0, sticky="w")

        if self.add_tooltip:
            self.add_tooltip(self.doc_radio, "Search for specific documents within projects")

    def set_on_mode_change(self, callback: Callable):
        """
        Set callback for mode change events.

        Args:
            callback: Function to call when mode changes
        """
        self.on_mode_change_callback = callback

    def _on_mode_change(self):
        """Handle mode change event."""
        if self.on_mode_change_callback:
            mode = self.state.get_current_mode()
            try:
                self.on_mode_change_callback(mode)
            except Exception as e:
                logger.error(f"Error in mode change callback: {e}")

    def get_current_mode(self) -> str:
        """Get currently selected mode."""
        return self.state.get_current_mode()

    def set_mode(self, mode: str):
        """
        Set navigation mode programmatically.

        Args:
            mode: Mode to set ('folder' or 'document')
        """
        self.state.set_current_mode(mode)

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
