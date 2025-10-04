#!/usr/bin/env python3
"""
Project Input Section Module

Handles the project input UI section including the search entry,
recent projects, and autocomplete functionality.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Import enhanced entry widget
try:
    from ..gui_widgets import EnhancedEntry
except ImportError:
    from gui_widgets import EnhancedEntry


class ProjectInputSection:
    """
    Project input section UI component.

    Responsibilities:
    - Project number/search entry field
    - Recent projects quick access buttons
    - Autocomplete suggestions
    - Input validation callbacks
    """

    def __init__(self, parent, state, layout, settings, tooltip_callback: Optional[Callable] = None):
        """
        Initialize the project input section.

        Args:
            parent: Parent widget (main_frame)
            state: GUIState instance
            layout: LayoutManager instance
            settings: SettingsManager instance
            tooltip_callback: Optional callback for adding tooltips
        """
        self.parent = parent
        self.state = state
        self.layout = layout
        self.settings = settings
        self.add_tooltip = tooltip_callback

        # Autocomplete state
        self.autocomplete_window = None
        self.autocomplete_listbox = None
        self.autocomplete_suggestions = []
        self.autocomplete_index = -1

        # UI elements
        self.frame = None
        self.project_entry = None
        self.recent_buttons_frame = None

        # Create the UI
        self._create_ui()

        logger.info("ProjectInputSection initialized")

    def _create_ui(self):
        """Create the project input section UI."""
        # Project input frame with improved styling
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Project entry with enhanced features and responsive styling
        entry_font_size = max(9, min(12, int(10 * self.layout.dpi_scale)))
        self.project_entry = EnhancedEntry(
            self.frame,
            textvariable=self.state.project_input,
            placeholder="Enter 5-digit project number or search term",
            font=("Segoe UI", entry_font_size)
        )
        entry_padding = self.layout.get_consistent_padding()
        ipady_value = max(3, int(4 * self.layout.dpi_scale))
        self.project_entry.grid(
            row=0, column=0, sticky="ew",
            padx=entry_padding,
            pady=(entry_padding, self.layout.get_consistent_spacing()),
            ipady=ipady_value
        )

        # Recent projects frame with consistent layout
        recent_frame = ttk.Frame(self.frame)
        recent_frame.grid(row=1, column=0, sticky="ew", padx=entry_padding, pady=(0, entry_padding))
        recent_frame.columnconfigure(0, weight=1)

        # Recent projects label with consistent styling
        recent_label = ttk.Label(recent_frame, text="⏱️ Recent:", font=("Segoe UI", 9))
        recent_label.grid(row=0, column=0, sticky="w", padx=(0, self.layout.get_consistent_spacing()))

        # Recent projects buttons container with proper grid layout
        self.recent_buttons_frame = ttk.Frame(recent_frame)
        self.recent_buttons_frame.grid(row=0, column=1, sticky="ew")
        self.recent_buttons_frame.columnconfigure(0, weight=1)

        # Populate recent projects
        self.update_recent_projects()

        # Setup autocomplete
        self._setup_autocomplete()

    def bind_validation(self, callback: Callable):
        """
        Bind validation callback to entry events.

        Args:
            callback: Function to call on input change
        """
        self.project_entry.bind('<KeyRelease>', callback)
        self.project_entry.bind('<FocusOut>', callback)

    def update_recent_projects(self):
        """Update the recent projects quick access buttons."""
        # Clear existing buttons
        for widget in self.recent_buttons_frame.winfo_children():
            widget.destroy()

        # Get recent projects
        recent_projects = self.settings.get_recent_projects()[:5]  # Limit to 5

        if not recent_projects:
            # Show placeholder if no recent projects
            ttk.Label(
                self.recent_buttons_frame,
                text="No recent projects",
                font=("Segoe UI", 9),
                foreground="gray"
            ).pack(side="left")
            return

        # Create button for each recent project
        for project in recent_projects:
            project_num = project.get('project_number', 'Unknown')
            project_name = project.get('project_name', '')

            # Format button text
            button_text = str(project_num)
            if len(button_text) > 8:
                button_text = button_text[:8] + "..."

            # Create button
            btn = ttk.Button(
                self.recent_buttons_frame,
                text=button_text,
                command=lambda p=project_num: self.select_recent_project(p),
                width=8
            )
            btn.pack(side="left", padx=(0, 4))

            # Add tooltip with full project info
            if self.add_tooltip:
                tooltip_text = f"{project_num}"
                if project_name:
                    tooltip_text += f"\n{project_name}"
                self.add_tooltip(btn, tooltip_text)

    def select_recent_project(self, project_number):
        """Select a project from recent projects list."""
        self.state.set_project_input(str(project_number))
        self.project_entry.focus_set()
        self.state.set_status_text(f"✓ Selected recent project: {project_number}")

    def _setup_autocomplete(self):
        """Set up autocomplete functionality for project search."""
        # Bind autocomplete events to project entry
        self.project_entry.bind('<KeyRelease>', self._on_autocomplete_keyrelease)
        self.project_entry.bind('<Down>', self._on_autocomplete_down)
        self.project_entry.bind('<Up>', self._on_autocomplete_up)
        self.project_entry.bind('<Tab>', self._on_autocomplete_tab)

    def _on_autocomplete_keyrelease(self, event):
        """Handle key release for autocomplete suggestions."""
        # Ignore special keys
        if event.keysym in ('Up', 'Down', 'Left', 'Right', 'Return', 'Tab', 'Escape',
                           'Shift_L', 'Shift_R', 'Control_L', 'Control_R'):
            return

        text = self.state.get_project_input()

        # Hide autocomplete if text too short
        if len(text) < 2:
            self._hide_autocomplete()
            return

        # Get suggestions from recent projects
        suggestions = self._get_autocomplete_suggestions(text)

        if suggestions:
            self._show_autocomplete(suggestions)
        else:
            self._hide_autocomplete()

    def _get_autocomplete_suggestions(self, text: str) -> List[str]:
        """Get autocomplete suggestions based on input text."""
        suggestions = []

        # Get recent projects from settings
        recent_projects = self.settings.get_recent_projects()

        # Filter by matching text (case-insensitive)
        text_lower = text.lower()
        for project in recent_projects:
            project_str = str(project.get('project_number', ''))
            if text_lower in project_str.lower():
                suggestions.append(project_str)

        return suggestions[:10]  # Limit to 10 suggestions

    def _show_autocomplete(self, suggestions: List[str]):
        """Show autocomplete suggestions dropdown."""
        if not suggestions:
            self._hide_autocomplete()
            return

        # Store suggestions
        self.autocomplete_suggestions = suggestions
        self.autocomplete_index = -1

        # Create or update autocomplete window
        if self.autocomplete_window is None:
            self.autocomplete_window = tk.Toplevel(self.project_entry)
            self.autocomplete_window.wm_overrideredirect(True)

            # Create listbox
            self.autocomplete_listbox = tk.Listbox(
                self.autocomplete_window,
                height=min(len(suggestions), 5),
                font=("Segoe UI", 9)
            )
            self.autocomplete_listbox.pack(fill=tk.BOTH, expand=True)

            # Bind selection
            self.autocomplete_listbox.bind('<<ListboxSelect>>', self._on_autocomplete_select)
            self.autocomplete_listbox.bind('<Return>', self._on_autocomplete_select)

        # Clear and populate listbox
        self.autocomplete_listbox.delete(0, tk.END)
        for suggestion in suggestions:
            self.autocomplete_listbox.insert(tk.END, suggestion)

        # Position window below entry
        x = self.project_entry.winfo_rootx()
        y = self.project_entry.winfo_rooty() + self.project_entry.winfo_height()
        width = self.project_entry.winfo_width()

        self.autocomplete_window.geometry(f"{width}x{min(len(suggestions) * 20, 100)}+{x}+{y}")

    def _hide_autocomplete(self):
        """Hide autocomplete suggestions dropdown."""
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
            self.autocomplete_window = None
            self.autocomplete_listbox = None
        self.autocomplete_suggestions = []
        self.autocomplete_index = -1

    def _on_autocomplete_down(self, event):
        """Handle down arrow in autocomplete."""
        if self.autocomplete_listbox and self.autocomplete_suggestions:
            self.autocomplete_index = min(
                self.autocomplete_index + 1,
                len(self.autocomplete_suggestions) - 1
            )
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(self.autocomplete_index)
            self.autocomplete_listbox.see(self.autocomplete_index)
            return "break"

    def _on_autocomplete_up(self, event):
        """Handle up arrow in autocomplete."""
        if self.autocomplete_listbox and self.autocomplete_suggestions:
            self.autocomplete_index = max(self.autocomplete_index - 1, 0)
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(self.autocomplete_index)
            self.autocomplete_listbox.see(self.autocomplete_index)
            return "break"

    def _on_autocomplete_tab(self, event):
        """Handle tab key in autocomplete."""
        if self.autocomplete_suggestions and self.autocomplete_index >= 0:
            selected = self.autocomplete_suggestions[self.autocomplete_index]
            self.state.set_project_input(selected)
            self._hide_autocomplete()
            return "break"

    def _on_autocomplete_select(self, event):
        """Handle selection from autocomplete."""
        if self.autocomplete_listbox:
            selection = self.autocomplete_listbox.curselection()
            if selection:
                selected = self.autocomplete_suggestions[selection[0]]
                self.state.set_project_input(selected)
                self._hide_autocomplete()

    def focus(self):
        """Set focus to the project entry."""
        self.project_entry.focus_set()

    def select_all(self):
        """Select all text in the entry."""
        self.project_entry.select_range(0, tk.END)
        self.project_entry.icursor(tk.END)

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
