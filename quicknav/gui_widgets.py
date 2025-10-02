"""
Custom Tkinter Widgets for Project QuickNav

This module provides enhanced custom widgets that extend the standard
Tkinter components with additional functionality and modern styling.

Widgets:
- EnhancedEntry: Entry with placeholder text and validation
- SearchableComboBox: ComboBox with search functionality
- CollapsibleFrame: Frame that can be collapsed/expanded
- ProgressDialog: Modal dialog with progress indication
- SelectionDialog: Dialog for selecting from multiple options
- SearchResultDialog: Dialog for displaying search results
- DocumentPreview: Widget for previewing documents
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import re
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


class EnhancedEntry(ttk.Entry):
    """Enhanced Entry widget with placeholder text and validation."""

    def __init__(self, parent, placeholder: str = "", **kwargs):
        self.placeholder = placeholder
        self.placeholder_color = '#888888'
        self.default_color = '#000000'
        self.showing_placeholder = False

        super().__init__(parent, **kwargs)

        if self.placeholder:
            self._show_placeholder()
            self.bind('<FocusIn>', self._on_focus_in)
            self.bind('<FocusOut>', self._on_focus_out)

    def _show_placeholder(self):
        """Show placeholder text."""
        if not self.showing_placeholder and not self.get():
            self.insert(0, self.placeholder)
            self.showing_placeholder = True
            self.config(foreground=self.placeholder_color)

    def _hide_placeholder(self):
        """Hide placeholder text."""
        if self.showing_placeholder:
            self.delete(0, tk.END)
            self.showing_placeholder = False
            self.config(foreground=self.default_color)

    def _on_focus_in(self, event):
        """Handle focus in event."""
        self._hide_placeholder()

    def _on_focus_out(self, event):
        """Handle focus out event."""
        if not self.get():
            self._show_placeholder()

    def get(self):
        """Get the current value, excluding placeholder text."""
        if self.showing_placeholder:
            return ""
        return super().get()

    def set(self, value):
        """Set the value of the entry."""
        self._hide_placeholder()
        self.delete(0, tk.END)
        self.insert(0, value)
        if not value:
            self._show_placeholder()


class SearchableComboBox(ttk.Frame):
    """ComboBox with search functionality and autocomplete."""

    def __init__(self, parent, values: List[str] = None, **kwargs):
        super().__init__(parent)

        self.values = values or []
        self.filtered_values = self.values.copy()
        self.callback = kwargs.pop('callback', None)

        # Create entry for searching
        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.entry_var)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create dropdown button
        self.dropdown_button = ttk.Button(
            self, text="▼", width=3,
            command=self._toggle_dropdown
        )
        self.dropdown_button.pack(side=tk.RIGHT)

        # Create dropdown listbox (initially hidden)
        self.dropdown = None
        self.dropdown_visible = False

        # Bind events
        self.entry_var.trace('w', self._on_entry_change)
        self.entry.bind('<Down>', self._on_arrow_down)
        self.entry.bind('<Up>', self._on_arrow_up)
        self.entry.bind('<Return>', self._on_return)
        self.entry.bind('<Escape>', self._hide_dropdown)
        self.entry.bind('<FocusOut>', self._on_focus_out)

    def _create_dropdown(self):
        """Create the dropdown listbox."""
        if self.dropdown:
            return

        # Create toplevel window for dropdown
        self.dropdown = tk.Toplevel(self)
        self.dropdown.wm_overrideredirect(True)
        self.dropdown.configure(relief=tk.RAISED, borderwidth=1)

        # Create listbox
        self.listbox = tk.Listbox(self.dropdown, height=8)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        # Bind listbox events
        self.listbox.bind('<Button-1>', self._on_listbox_select)
        self.listbox.bind('<Return>', self._on_listbox_select)
        self.listbox.bind('<Escape>', self._hide_dropdown)

        # Position dropdown
        self._position_dropdown()

    def _position_dropdown(self):
        """Position the dropdown below the entry."""
        if not self.dropdown:
            return

        # Get entry position
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        width = self.winfo_width()

        self.dropdown.geometry(f"{width}x200+{x}+{y}")

    def _toggle_dropdown(self):
        """Toggle dropdown visibility."""
        if self.dropdown_visible:
            self._hide_dropdown()
        else:
            self._show_dropdown()

    def _show_dropdown(self):
        """Show the dropdown."""
        if not self.dropdown:
            self._create_dropdown()

        self._update_listbox()
        self.dropdown.deiconify()
        self.dropdown_visible = True

        # Focus the listbox if it has items
        if self.listbox.size() > 0:
            self.listbox.selection_set(0)
            self.listbox.focus_set()

    def _hide_dropdown(self, event=None):
        """Hide the dropdown."""
        if self.dropdown:
            self.dropdown.withdraw()
        self.dropdown_visible = False

    def _update_listbox(self):
        """Update listbox contents based on filter."""
        if not self.dropdown:
            return

        self.listbox.delete(0, tk.END)
        for value in self.filtered_values:
            self.listbox.insert(tk.END, value)

    def _on_entry_change(self, *args):
        """Handle entry text change."""
        search_text = self.entry_var.get().lower()

        # Filter values
        if search_text:
            self.filtered_values = [
                value for value in self.values
                if search_text in value.lower()
            ]
        else:
            self.filtered_values = self.values.copy()

        # Update dropdown if visible
        if self.dropdown_visible:
            self._update_listbox()

        # Show dropdown if there are matches and entry has focus
        if self.filtered_values and self.entry == self.focus_get():
            if not self.dropdown_visible:
                self._show_dropdown()

    def _on_arrow_down(self, event):
        """Handle down arrow key."""
        if not self.dropdown_visible and self.filtered_values:
            self._show_dropdown()
        elif self.dropdown_visible and self.listbox.size() > 0:
            current = self.listbox.curselection()
            if current:
                next_index = min(current[0] + 1, self.listbox.size() - 1)
            else:
                next_index = 0
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(next_index)
            self.listbox.see(next_index)

    def _on_arrow_up(self, event):
        """Handle up arrow key."""
        if self.dropdown_visible and self.listbox.size() > 0:
            current = self.listbox.curselection()
            if current:
                prev_index = max(current[0] - 1, 0)
            else:
                prev_index = self.listbox.size() - 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(prev_index)
            self.listbox.see(prev_index)

    def _on_return(self, event):
        """Handle return key."""
        if self.dropdown_visible and self.listbox.curselection():
            self._on_listbox_select()
        elif self.callback:
            self.callback()

    def _on_listbox_select(self, event=None):
        """Handle listbox selection."""
        if not self.listbox.curselection():
            return

        selected_value = self.listbox.get(self.listbox.curselection()[0])
        self.entry_var.set(selected_value)
        self._hide_dropdown()

        if self.callback:
            self.callback()

    def _on_focus_out(self, event):
        """Handle focus out event."""
        # Delay hiding to allow listbox clicks
        self.after(100, self._hide_dropdown)

    def get(self):
        """Get the current value."""
        return self.entry_var.get()

    def set(self, value):
        """Set the current value."""
        self.entry_var.set(value)

    def configure_values(self, values: List[str]):
        """Update the list of values."""
        self.values = values
        self.filtered_values = values.copy()
        if self.dropdown_visible:
            self._update_listbox()


class CollapsibleFrame(ttk.Frame):
    """Frame that can be collapsed and expanded."""

    def __init__(self, parent, title: str = "", collapsed: bool = False, **kwargs):
        super().__init__(parent, **kwargs)

        self.title = title
        self.collapsed = tk.BooleanVar(value=collapsed)

        # Create header frame
        self.header_frame = ttk.Frame(self)
        self.header_frame.pack(fill=tk.X, pady=(0, 5))

        # Create toggle button
        self.toggle_button = ttk.Button(
            self.header_frame,
            text="▼" if not collapsed else "▶",
            width=3,
            command=self.toggle
        )
        self.toggle_button.pack(side=tk.LEFT)

        # Create title label
        if title:
            self.title_label = ttk.Label(self.header_frame, text=title)
            self.title_label.pack(side=tk.LEFT, padx=(5, 0))

        # Create content frame
        self.content_frame = ttk.Frame(self)
        if not collapsed:
            self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.collapsed.trace('w', self._on_collapse_change)

    def toggle(self):
        """Toggle collapsed state."""
        self.collapsed.set(not self.collapsed.get())

    def _on_collapse_change(self, *args):
        """Handle collapse state change."""
        if self.collapsed.get():
            self.content_frame.pack_forget()
            self.toggle_button.config(text="▶")
        else:
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            self.toggle_button.config(text="▼")

    def add_content(self, widget):
        """Add widget to content frame."""
        widget.pack(in_=self.content_frame, fill=tk.BOTH, expand=True)


class ProgressDialog:
    """Modal dialog with progress indication."""

    def __init__(self, parent, title: str = "Progress", message: str = "Working..."):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()

        # Center on parent
        self.dialog.transient(parent)
        self._center_on_parent()

        # Create content
        self._create_content(message)

        # Variables
        self.cancelled = False

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        self.dialog.geometry(f"+{x}+{y}")

    def _create_content(self, message: str):
        """Create dialog content."""
        # Message label
        self.message_label = ttk.Label(
            self.dialog,
            text=message,
            wraplength=350
        )
        self.message_label.pack(pady=20)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.dialog,
            mode='indeterminate',
            length=300
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()

        # Cancel button
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel
        )
        self.cancel_button.pack()

    def update_message(self, message: str):
        """Update the progress message."""
        self.message_label.config(text=message)
        self.dialog.update()

    def cancel(self):
        """Cancel the operation."""
        self.cancelled = True
        self.close()

    def close(self):
        """Close the dialog."""
        self.progress_bar.stop()
        self.dialog.grab_release()
        self.dialog.destroy()

    def is_cancelled(self):
        """Check if operation was cancelled."""
        return self.cancelled


class SelectionDialog:
    """Dialog for selecting from multiple options."""

    def __init__(self, parent, title: str = "Select Option", paths: List[str] = None,
                 callback: Callable = None):
        self.parent = parent
        self.paths = paths or []
        self.callback = callback
        self.selected_path = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("600x400")
        self.dialog.grab_set()
        self.dialog.transient(parent)

        self._center_on_parent()
        self._create_content()

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        self.dialog.geometry(f"+{x}+{y}")

    def _create_content(self):
        """Create dialog content."""
        # Instructions
        instruction_label = ttk.Label(
            self.dialog,
            text=f"Multiple options found. Select one:",
            wraplength=550
        )
        instruction_label.pack(pady=10)

        # Create listbox with scrollbar
        list_frame = ttk.Frame(self.dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Listbox
        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Populate listbox
        for path in self.paths:
            display_name = os.path.basename(path)
            self.listbox.insert(tk.END, display_name)

        # Select first item
        if self.paths:
            self.listbox.selection_set(0)

        # Bind double-click
        self.listbox.bind('<Double-1>', self._on_double_click)

        # Button frame
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)

        # Buttons
        ttk.Button(
            button_frame,
            text="OK",
            command=self._on_ok,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=10
        ).pack(side=tk.LEFT, padx=5)

    def _on_double_click(self, event):
        """Handle double-click on listbox item."""
        self._on_ok()

    def _on_ok(self):
        """Handle OK button click."""
        selection = self.listbox.curselection()
        if selection:
            self.selected_path = self.paths[selection[0]]
            if self.callback:
                self.callback(self.selected_path)
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle Cancel button click."""
        self.dialog.destroy()

    def show(self):
        """Show the dialog."""
        self.dialog.wait_window()
        return self.selected_path


class SearchResultDialog(SelectionDialog):
    """Dialog for displaying search results with enhanced information."""

    def _create_content(self):
        """Create enhanced search result content."""
        # Instructions
        instruction_label = ttk.Label(
            self.dialog,
            text=f"Found {len(self.paths)} project folders matching your search:",
            wraplength=550
        )
        instruction_label.pack(pady=10)

        # Create treeview for better display
        tree_frame = ttk.Frame(self.dialog)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create treeview
        columns = ('Project Number', 'Project Name', 'Full Path')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=12)

        # Define headings
        self.tree.heading('Project Number', text='Project Number')
        self.tree.heading('Project Name', text='Project Name')
        self.tree.heading('Full Path', text='Full Path')

        # Configure column widths
        self.tree.column('Project Number', width=120)
        self.tree.column('Project Name', width=200)
        self.tree.column('Full Path', width=280)

        # Add scrollbar
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        # Pack tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate tree
        for path in self.paths:
            project_name = os.path.basename(path)

            # Extract project number and name
            match = re.match(r'^(\d{5}) - (.+)$', project_name)
            if match:
                proj_num = match.group(1)
                proj_name = match.group(2)
            else:
                proj_num = "N/A"
                proj_name = project_name

            self.tree.insert('', tk.END, values=(proj_num, proj_name, path))

        # Select first item
        if self.paths:
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)

        # Bind double-click
        self.tree.bind('<Double-1>', self._on_tree_double_click)

        # Button frame
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)

        # Buttons
        ttk.Button(
            button_frame,
            text="Open",
            command=self._on_ok,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=10
        ).pack(side=tk.LEFT, padx=5)

    def _on_tree_double_click(self, event):
        """Handle double-click on tree item."""
        self._on_ok()

    def _on_ok(self):
        """Handle OK button click."""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            values = self.tree.item(item, 'values')
            self.selected_path = values[2]  # Full path is in third column
            if self.callback:
                self.callback(self.selected_path)
        self.dialog.destroy()


class DocumentPreview(ttk.Frame):
    """Widget for previewing documents with metadata."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.current_document = None
        self._create_ui()

    def _create_ui(self):
        """Create preview UI."""
        # Header frame
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        self.title_label = ttk.Label(
            header_frame,
            text="Document Preview",
            font=('TkDefaultFont', 12, 'bold')
        )
        self.title_label.pack(side=tk.LEFT)

        self.info_label = ttk.Label(
            header_frame,
            text="No document selected",
            foreground='gray'
        )
        self.info_label.pack(side=tk.RIGHT)

        # Content frame
        content_frame = ttk.Frame(self)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Metadata tab
        self.metadata_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metadata_frame, text="Metadata")

        # Preview tab (for images/thumbnails)
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="Preview")

        self._create_metadata_tab()
        self._create_preview_tab()

    def _create_metadata_tab(self):
        """Create metadata display tab."""
        # Scrollable text widget
        text_frame = ttk.Frame(self.metadata_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.metadata_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=10
        )

        metadata_scrollbar = ttk.Scrollbar(
            text_frame,
            orient=tk.VERTICAL,
            command=self.metadata_text.yview
        )
        self.metadata_text.configure(yscrollcommand=metadata_scrollbar.set)

        self.metadata_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metadata_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_preview_tab(self):
        """Create preview display tab."""
        self.preview_label = ttk.Label(
            self.preview_frame,
            text="Preview not available",
            anchor=tk.CENTER
        )
        self.preview_label.pack(expand=True)

    def load_document(self, document_path: str, metadata: Dict[str, Any] = None):
        """Load a document for preview."""
        self.current_document = document_path

        # Update title
        document_name = os.path.basename(document_path)
        self.title_label.config(text=document_name)

        # Update info
        if os.path.exists(document_path):
            file_size = os.path.getsize(document_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(document_path))
            self.info_label.config(
                text=f"Size: {file_size:,} bytes | Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            self.info_label.config(text="File not found", foreground='red')

        # Update metadata
        self._update_metadata(metadata or {})

        # Update preview
        self._update_preview(document_path)

    def _update_metadata(self, metadata: Dict[str, Any]):
        """Update metadata display."""
        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete(1.0, tk.END)

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (list, dict, set)):
                    value_str = str(value)
                else:
                    value_str = str(value)

                self.metadata_text.insert(tk.END, f"{key}: {value_str}\n")
        else:
            self.metadata_text.insert(tk.END, "No metadata available")

        self.metadata_text.config(state=tk.DISABLED)

    def _update_preview(self, document_path: str):
        """Update preview display."""
        # For now, just show file type information
        file_ext = os.path.splitext(document_path)[1].lower()

        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            preview_text = "Image file - Preview would show thumbnail here"
        elif file_ext == '.pdf':
            preview_text = "PDF file - Preview would show first page here"
        elif file_ext in ['.docx', '.doc']:
            preview_text = "Word document - Preview would show document preview here"
        elif file_ext in ['.vsdx', '.vsd']:
            preview_text = "Visio diagram - Preview would show diagram thumbnail here"
        else:
            preview_text = f"File type: {file_ext}\nPreview not supported"

        self.preview_label.config(text=preview_text)

    def clear(self):
        """Clear the preview."""
        self.current_document = None
        self.title_label.config(text="Document Preview")
        self.info_label.config(text="No document selected", foreground='gray')

        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete(1.0, tk.END)
        self.metadata_text.config(state=tk.DISABLED)

        self.preview_label.config(text="Preview not available")


class StatusBar(ttk.Frame):
    """Enhanced status bar with multiple sections."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.configure(relief=tk.SUNKEN, borderwidth=1)

        # Create sections
        self.main_status = tk.StringVar(value="Ready")
        self.secondary_status = tk.StringVar()
        self.progress_value = tk.IntVar()

        self._create_sections()

    def _create_sections(self):
        """Create status bar sections."""
        # Main status label
        self.main_label = ttk.Label(
            self,
            textvariable=self.main_status,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.main_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=1)

        # Progress section
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(side=tk.RIGHT, padx=2, pady=1)

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=100,
            variable=self.progress_value
        )

        self.progress_label = ttk.Label(
            self.progress_frame,
            textvariable=self.secondary_status,
            width=20,
            anchor=tk.E
        )
        self.progress_label.pack(side=tk.RIGHT, padx=(0, 5))

    def set_status(self, message: str):
        """Set main status message."""
        self.main_status.set(message)

    def set_secondary_status(self, message: str):
        """Set secondary status message."""
        self.secondary_status.set(message)

    def show_progress(self, value: int = None):
        """Show progress bar."""
        if value is not None:
            self.progress_value.set(value)
        self.progress_bar.pack(side=tk.RIGHT, padx=(0, 5))

    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.pack_forget()
        self.progress_value.set(0)

    def set_progress(self, value: int):
        """Set progress value (0-100)."""
        self.progress_value.set(max(0, min(100, value)))