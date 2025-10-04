#!/usr/bin/env python3
"""
Status Section Module

Handles the status display UI section.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StatusSection:
    """
    Status display UI component.

    Responsibilities:
    - Status text label
    - Progress bar
    - Loading indicator
    """

    def __init__(self, parent, state, layout):
        """
        Initialize the status section.

        Args:
            parent: Parent widget (main_frame)
            state: GUIState instance
            layout: LayoutManager instance
        """
        self.parent = parent
        self.state = state
        self.layout = layout

        # UI elements
        self.frame = None
        self.status_label = None
        self.progress_bar = None
        self.loading_label = None

        # Create the UI
        self._create_ui()

        logger.info("StatusSection initialized")

    def _create_ui(self):
        """Create the status section UI."""
        # Status frame
        self.frame = ttk.Frame(self.parent)
        self.frame.grid(
            row=4, column=0, sticky="ew",
            pady=(0, self.layout.get_consistent_spacing() // 2)
        )
        self.frame.columnconfigure(0, weight=1)

        # Status label with responsive wrapping and consistent padding
        wrap_length = max(300, int(self.layout.min_width * 0.8))
        self.status_label = ttk.Label(
            self.frame,
            textvariable=self.state.status_text,
            wraplength=wrap_length,
            justify="left"
        )
        self.status_label.grid(
            row=0, column=0, sticky="ew",
            padx=self.layout.get_consistent_padding()
        )

        # Progress bar with loading spinner (hidden by default)
        self.progress_bar = ttk.Progressbar(
            self.frame,
            mode='indeterminate'
        )
        self.progress_bar.grid(
            row=1, column=0, sticky="ew",
            padx=self.layout.get_consistent_padding(),
            pady=(self.layout.get_consistent_spacing() // 2, 0)
        )
        self.progress_bar.grid_remove()

        # Loading indicator label with consistent padding
        self.loading_label = ttk.Label(
            self.frame,
            text="",
            font=("Segoe UI", 9)
        )
        self.loading_label.grid(
            row=2, column=0, sticky="ew",
            padx=self.layout.get_consistent_padding(),
            pady=(2, 0)
        )
        self.loading_label.grid_remove()

    def set_status(self, text: str):
        """
        Set status text.

        Args:
            text: Status text to display
        """
        self.state.set_status_text(text)

    def show_progress(self, message: str = "Processing..."):
        """
        Show progress indicator.

        Args:
            message: Optional loading message
        """
        self.progress_bar.grid()
        self.progress_bar.start(10)
        if message:
            self.loading_label.config(text=message)
            self.loading_label.grid()

    def hide_progress(self):
        """Hide progress indicator."""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.loading_label.grid_remove()

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
