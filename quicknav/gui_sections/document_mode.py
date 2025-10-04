#!/usr/bin/env python3
"""
Document Mode Section Module

Handles the document type and filter UI for document search mode.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List
import logging

logger = logging.getLogger(__name__)


class DocumentModeSection:
    """
    Document mode UI component.

    Responsibilities:
    - Document type selection
    - Version filtering
    - Room/CO/Archive filters
    """

    def __init__(self, parent, state, layout, tooltip_callback: Optional[Callable] = None):
        """
        Initialize the document mode section.

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
        self.doc_type_combo = None
        self.version_combo = None
        self.room_entry = None
        self.co_entry = None
        self.archive_check = None

        # Create the UI
        self._create_ui()

        logger.info("DocumentModeSection initialized")

    def _create_ui(self):
        """Create the document mode UI."""
        # Document mode frame (initially hidden) within main content frame
        self.frame = ttk.Frame(self.parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)

        # Create document type selection
        self._create_document_type_selection()

        # Create version filter
        self._create_version_filter()

        # Create document filters (room, CO, archive)
        self._create_document_filters()

    def _create_document_type_selection(self):
        """Create document type selection controls."""
        # Document type label and options with consistent spacing
        doc_type_label = ttk.Label(self.frame, text="Document Type:")
        doc_type_label.grid(
            row=0, column=0, sticky="w",
            padx=self.layout.get_consistent_padding(),
            pady=(self.layout.get_consistent_padding(), self.layout.get_consistent_spacing() // 2)
        )

        doc_type_options: List[str] = [
            "🔧 Low-Level Design (LLD)",
            "📊 High-Level Design (HLD)",
            "📄 Change Orders",
            "💰 Sales & PO Reports",
            "🏗️ Floor Plans",
            "📋 Scope Documents",
            "✅ QA/ITP Reports",
            "⚠️ SWMS",
            "💰 Supplier Quotes",
            "📷 Site Photos"
        ]

        self.doc_type_combo = ttk.Combobox(
            self.frame,
            textvariable=self.state.doc_type,
            values=doc_type_options,
            state="readonly"
        )
        self.doc_type_combo.grid(
            row=0, column=1, sticky="ew",
            padx=(0, self.layout.get_consistent_padding()),
            pady=(self.layout.get_consistent_padding(), self.layout.get_consistent_spacing() // 2)
        )
        self.doc_type_combo.current(0)

        if self.add_tooltip:
            self.add_tooltip(self.doc_type_combo, "Select the type of document to search for")

    def _create_version_filter(self):
        """Create version filter controls."""
        version_label = ttk.Label(self.frame, text="Version Filter:")
        version_label.grid(
            row=1, column=0, sticky="w",
            padx=self.layout.get_consistent_padding(),
            pady=(self.layout.get_consistent_spacing() // 2, self.layout.get_consistent_spacing() // 2)
        )

        version_options: List[str] = [
            "Auto (Latest/Best)",
            "Latest Version",
            "As-Built Only",
            "Initial Version",
            "All Versions"
        ]

        self.version_combo = ttk.Combobox(
            self.frame,
            textvariable=self.state.version_filter,
            values=version_options,
            state="readonly"
        )
        self.version_combo.grid(
            row=1, column=1, sticky="ew",
            padx=(0, self.layout.get_consistent_padding()),
            pady=(self.layout.get_consistent_spacing() // 2, self.layout.get_consistent_spacing() // 2)
        )

        if self.add_tooltip:
            self.add_tooltip(self.version_combo, "Filter documents by version (latest, as-built, etc.)")

    def _create_document_filters(self):
        """Create document filter controls (room, CO, archive)."""
        # Filters frame with responsive layout and consistent spacing
        filters_frame = ttk.Frame(self.frame)
        filters_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew",
            padx=self.layout.get_consistent_padding(),
            pady=(self.layout.get_consistent_spacing() // 2, self.layout.get_consistent_padding())
        )
        filters_frame.columnconfigure(1, weight=1)
        filters_frame.columnconfigure(3, weight=1)

        # Room filter with consistent spacing
        room_label = ttk.Label(filters_frame, text="🏠 Room:")
        room_label.grid(row=0, column=0, sticky="w", padx=(0, self.layout.get_consistent_spacing() // 2))

        self.room_entry = ttk.Entry(filters_frame, textvariable=self.state.room_filter, width=8)
        self.room_entry.grid(row=0, column=1, sticky="w", padx=(0, self.layout.get_consistent_spacing()))

        if self.add_tooltip:
            self.add_tooltip(self.room_entry, "Filter by room number (e.g., 101, 201)")

        # CO filter with consistent spacing
        co_label = ttk.Label(filters_frame, text="📝 CO:")
        co_label.grid(row=0, column=2, sticky="w", padx=(0, self.layout.get_consistent_spacing() // 2))

        self.co_entry = ttk.Entry(filters_frame, textvariable=self.state.co_filter, width=8)
        self.co_entry.grid(row=0, column=3, sticky="w", padx=(0, self.layout.get_consistent_spacing()))

        if self.add_tooltip:
            self.add_tooltip(self.co_entry, "Filter by Change Order number")

        # Include archive checkbox with consistent spacing
        self.archive_check = ttk.Checkbutton(
            filters_frame,
            text="📦 Include Archive",
            variable=self.state.include_archive
        )
        self.archive_check.grid(row=0, column=4, sticky="w")

        if self.add_tooltip:
            self.add_tooltip(self.archive_check, "Include archived and older versions")

    def show(self):
        """Show the document mode section."""
        self.frame.grid(row=0, column=0, sticky="ew")

    def hide(self):
        """Hide the document mode section."""
        self.frame.grid_forget()

    def get_doc_type(self) -> str:
        """Get selected document type."""
        return self.state.get_doc_type()

    def get_filters(self) -> dict:
        """
        Get all document filters.

        Returns:
            Dictionary containing all filter settings
        """
        return self.state.get_document_filters()

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
