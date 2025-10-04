#!/usr/bin/env python3
"""
Folder Mode Section Module

Handles the folder selection UI for folder navigation mode.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FolderModeSection:
    """
    Folder selection UI component.

    Responsibilities:
    - Folder/subfolder selection radio buttons
    - Folder descriptions and tooltips
    """

    def __init__(self, parent, state, layout, tooltip_callback: Optional[Callable] = None):
        """
        Initialize the folder mode section.

        Args:
            parent: Parent widget (main content frame)
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
        self.folder_radios = []

        # Create the UI
        self._create_ui()

        logger.info("FolderModeSection initialized")

    def _create_ui(self):
        """Create the folder selection UI."""
        # Folder selection frame
        self.frame = ttk.LabelFrame(self.parent, text="Select Subfolder")
        # Note: Frame created but not gridded - will be shown by parent

        # Folder options with icons
        folder_options: List[Tuple[str, str]] = [
            ("📁 4. System Designs", "4. System Designs"),
            ("📁 1. Sales Handover", "1. Sales Handover"),
            ("📁 2. BOM & Orders", "2. BOM & Orders"),
            ("📁 6. Customer Handover Documents", "6. Customer Handover Documents"),
            ("📁 5. Floor Plans", "5. Floor Plans"),
            ("📁 6. Site Photos", "6. Site Photos")
        ]

        # Folder tooltips
        folder_tooltips = {
            "4. System Designs": "Technical drawings, CAD files, and system specifications",
            "1. Sales Handover": "Initial project information and requirements",
            "2. BOM & Orders": "Bill of materials and purchase orders",
            "6. Customer Handover Documents": "Final delivery documentation",
            "5. Floor Plans": "Architectural drawings and layouts",
            "6. Site Photos": "Project site photographs and documentation"
        }

        # Create container frame with grid layout
        folder_container = ttk.Frame(self.frame)
        folder_container.grid(
            row=0, column=0, sticky="ew",
            padx=self.layout.get_consistent_padding(),
            pady=self.layout.get_consistent_padding()
        )
        folder_container.columnconfigure(0, weight=1)

        # Create radio buttons for folders
        for i, (display_text, folder_value) in enumerate(folder_options):
            radio = ttk.Radiobutton(
                folder_container,
                text=display_text,
                variable=self.state.selected_folder,
                value=folder_value
            )
            radio.grid(row=i, column=0, sticky="w", pady=2)
            self.folder_radios.append(radio)

            if self.add_tooltip:
                tooltip_text = folder_tooltips.get(folder_value, "Project subfolder")
                self.add_tooltip(radio, tooltip_text)

    def show(self):
        """Show the folder mode section."""
        self.frame.grid(row=0, column=0, sticky="ew")

    def hide(self):
        """Hide the folder mode section."""
        self.frame.grid_forget()

    def get_selected_folder(self) -> str:
        """Get currently selected folder."""
        return self.state.get_selected_folder()

    def set_selected_folder(self, folder: str):
        """
        Set selected folder programmatically.

        Args:
            folder: Folder name to select
        """
        self.state.set_selected_folder(folder)

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
