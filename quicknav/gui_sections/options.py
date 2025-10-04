#!/usr/bin/env python3
"""
Options Section Module

Handles the options/settings UI section (debug mode, training mode, etc.).
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class OptionsSection:
    """
    Options section UI component.

    Responsibilities:
    - Debug mode checkbox
    - Training mode checkbox
    - Other operational settings
    """

    def __init__(self, parent, state, layout, tooltip_callback: Optional[Callable] = None):
        """
        Initialize the options section.

        Args:
            parent: Parent widget (sidebar_frame)
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
        self.debug_check = None
        self.training_check = None

        # Create the UI
        self._create_ui()

        logger.info("OptionsSection initialized")

    def _create_ui(self):
        """Create the options section UI."""
        # Options frame
        self.frame = ttk.LabelFrame(self.parent, text="Options")
        self.frame.grid(
            row=0, column=0, sticky="ew",
            padx=(0, self.layout.get_consistent_spacing() // 2)
        )
        self.frame.columnconfigure(0, weight=1)

        # Options container with grid layout
        opts_container = ttk.Frame(self.frame)
        opts_container.grid(
            row=0, column=0, sticky="ew",
            padx=self.layout.get_consistent_padding(),
            pady=self.layout.get_consistent_padding()
        )
        opts_container.columnconfigure(0, weight=1)

        # Debug mode checkbox with consistent spacing
        self.debug_check = ttk.Checkbutton(
            opts_container,
            text="🔧 Show Debug Output",
            variable=self.state.debug_mode
        )
        self.debug_check.grid(row=0, column=0, sticky="w")

        if self.add_tooltip:
            self.add_tooltip(self.debug_check, "Show detailed debug information in results")

        # Training data checkbox with consistent spacing
        self.training_check = ttk.Checkbutton(
            opts_container,
            text="📊 Generate Training Data",
            variable=self.state.training_mode
        )
        self.training_check.grid(
            row=0, column=1, sticky="w",
            padx=(self.layout.get_consistent_spacing(), 0)
        )

        if self.add_tooltip:
            self.add_tooltip(self.training_check, "Generate JSON training data for AI analysis")

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.state.is_debug_mode()

    def is_training_mode(self) -> bool:
        """Check if training mode is enabled."""
        return self.state.is_training_mode()

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
