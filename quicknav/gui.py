#!/usr/bin/env python3
"""
Project QuickNav - Enhanced Tkinter GUI Application

A comprehensive GUI replacement for the AutoHotkey version with enhanced
features including modern theming, async operations, advanced search,
and cross-platform compatibility.

Features:
- DPI-aware responsive layout
- Global hotkey support
- System tray integration
- Dark/Light themes
- Advanced document search
- Real-time validation
- Async operations
- Settings management
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import threading
import queue
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from .gui_controller import GuiController
    from .gui_settings import SettingsManager
    from .gui_theme import ThemeManager
    from .gui_widgets import *
    from .ai_client import AIClient
    from .ai_chat_widget import ChatWidget as AIChatWidget
except ImportError:
    # Fallback imports for development
    from gui_controller import GuiController
    from gui_settings import SettingsManager
    from gui_theme import ThemeManager
    from gui_widgets import *
    try:
        from ai_client import AIClient
        from ai_chat_widget import ChatWidget as AIChatWidget
    except ImportError:
        AIClient = None
        AIChatWidget = None


class ProjectQuickNavGUI:
    """Main GUI application class."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Project QuickNav - Enhanced")
        self.root.withdraw()  # Hide initially

        # Initialize components
        self.settings = SettingsManager()
        self.theme = ThemeManager(self.settings)
        self.controller = GuiController(self.settings)

        # State variables
        self.current_mode = tk.StringVar(value="folder")
        self.project_input = tk.StringVar()
        self.debug_mode = tk.BooleanVar()
        self.training_mode = tk.BooleanVar()
        self.status_text = tk.StringVar(value="Ready - Select navigation mode and enter project number")

        # Document mode variables
        self.doc_type = tk.StringVar()
        self.version_filter = tk.StringVar(value="Auto (Latest/Best)")
        self.room_filter = tk.StringVar()
        self.co_filter = tk.StringVar()
        self.include_archive = tk.BooleanVar()

        # Folder mode variables
        self.selected_folder = tk.StringVar(value="4. System Designs")

        # UI state
        self.is_resizing = False
        self.min_width = 480
        self.min_height = 650

        # Task queue for async operations
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # AI components
        self.ai_client = None
        self.ai_chat_widget = None
        self.ai_enabled = tk.BooleanVar()
        self.ai_panel_visible = tk.BooleanVar(value=False)

        # Initialize AI if enabled
        self._initialize_ai()

        # Initialize UI
        self._setup_ui()
        self._setup_events()
        self._setup_validation()
        self._apply_theme()

        # Start task processor
        self._start_task_processor()

        # Update AI UI
        self._update_ai_ui()

        logger.info("ProjectQuickNavGUI initialized successfully")

    def _setup_ui(self):
        """Set up the main user interface."""
        # Configure root window with improved dimensions
        self.root.geometry("520x720")
        self.root.minsize(520, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create main container with enhanced padding and styling
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=24)

        # Configure grid weights for responsive layout
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(7, weight=1)  # Status area gets extra space

        self._create_project_input_section()
        self._create_navigation_mode_section()
        self._create_folder_mode_section()
        self._create_document_mode_section()
        self._create_options_section()
        self._create_ai_toolbar()
        self._create_status_section()
        self._create_action_buttons()
        self._create_menu_bar()

    def _create_project_input_section(self):
        """Create project input section with enhanced styling."""
        # Project input frame with improved styling
        input_frame = ttk.LabelFrame(self.main_frame, text="🏢 Project Input")
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        input_frame.columnconfigure(0, weight=1)

        # Project entry with enhanced features and better styling
        self.project_entry = EnhancedEntry(
            input_frame,
            textvariable=self.project_input,
            placeholder="Enter 5-digit project number or search term",
            font=("Segoe UI", 11)
        )
        self.project_entry.grid(row=0, column=0, sticky="ew", padx=16, pady=16, ipady=4)

        # Bind validation events
        self.project_entry.bind('<KeyRelease>', self._on_project_input_change)
        self.project_entry.bind('<FocusOut>', self._on_project_input_change)

    def _create_navigation_mode_section(self):
        """Create navigation mode selection section with enhanced styling."""
        # Navigation mode frame with improved styling
        nav_frame = ttk.LabelFrame(self.main_frame, text="🎯 Navigation Mode")
        nav_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        nav_frame.columnconfigure(0, weight=1)

        # Mode selection frame with improved spacing
        mode_frame = ttk.Frame(nav_frame)
        mode_frame.pack(fill="x", padx=16, pady=16)

        # Radio buttons for navigation mode with better styling
        self.folder_radio = ttk.Radiobutton(
            mode_frame,
            text="📁 Open Project Folder",
            variable=self.current_mode,
            value="folder",
            command=self._on_mode_change
        )
        self.folder_radio.pack(anchor="w", pady=(0, 8))

        self.doc_radio = ttk.Radiobutton(
            mode_frame,
            text="🔍 Find Documents",
            variable=self.current_mode,
            value="document",
            command=self._on_mode_change
        )
        self.doc_radio.pack(anchor="w")

        # Settings button with improved styling
        settings_frame = ttk.Frame(nav_frame)
        settings_frame.pack(fill="x", padx=16, pady=(8, 12))

        ttk.Button(
            settings_frame,
            text="⚙️ Settings",
            command=self.show_settings,
            width=14
        ).pack(side="right")

    def _create_folder_mode_section(self):
        """Create folder selection section for folder mode."""
        # Folder selection frame
        self.folder_frame = ttk.LabelFrame(self.main_frame, text="Select Subfolder")
        self.folder_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))

        # Folder options
        folder_options = [
            "4. System Designs",
            "1. Sales Handover",
            "2. BOM & Orders",
            "6. Customer Handover Documents",
            "5. Floor Plans",
            "6. Site Photos"
        ]

        # Create radio buttons for folders
        self.folder_radios = []
        for i, folder in enumerate(folder_options):
            radio = ttk.Radiobutton(
                self.folder_frame,
                text=folder,
                variable=self.selected_folder,
                value=folder
            )
            radio.pack(anchor="w", padx=15, pady=2)
            self.folder_radios.append(radio)

    def _create_document_mode_section(self):
        """Create document type and filter section for document mode."""
        # Document mode frame (initially hidden)
        self.doc_frame = ttk.LabelFrame(self.main_frame, text="Document Type & Filters")
        self.doc_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        self.doc_frame.columnconfigure(1, weight=1)

        # Document type selection
        ttk.Label(self.doc_frame, text="Document Type:").grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=(10, 5)
        )

        doc_type_options = [
            "Low-Level Design (LLD)",
            "High-Level Design (HLD)",
            "Change Orders",
            "Sales & PO Reports",
            "Floor Plans",
            "Scope Documents",
            "QA/ITP Reports",
            "SWMS",
            "Supplier Quotes",
            "Site Photos"
        ]

        self.doc_type_combo = ttk.Combobox(
            self.doc_frame,
            textvariable=self.doc_type,
            values=doc_type_options,
            state="readonly"
        )
        self.doc_type_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 5))
        self.doc_type_combo.current(0)

        # Version filter
        ttk.Label(self.doc_frame, text="Version Filter:").grid(
            row=1, column=0, sticky="w", padx=(10, 5), pady=(5, 5)
        )

        version_options = [
            "Auto (Latest/Best)",
            "Latest Version",
            "As-Built Only",
            "Initial Version",
            "All Versions"
        ]

        self.version_combo = ttk.Combobox(
            self.doc_frame,
            textvariable=self.version_filter,
            values=version_options,
            state="readonly"
        )
        self.version_combo.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(5, 5))

        # Filters frame
        filters_frame = ttk.Frame(self.doc_frame)
        filters_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))
        filters_frame.columnconfigure(1, weight=1)
        filters_frame.columnconfigure(3, weight=1)

        # Room filter
        ttk.Label(filters_frame, text="Room:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.room_entry = ttk.Entry(filters_frame, textvariable=self.room_filter, width=8)
        self.room_entry.grid(row=0, column=1, sticky="w", padx=(0, 20))

        # CO filter
        ttk.Label(filters_frame, text="CO:").grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.co_entry = ttk.Entry(filters_frame, textvariable=self.co_filter, width=8)
        self.co_entry.grid(row=0, column=3, sticky="w", padx=(0, 20))

        # Include archive checkbox
        self.archive_check = ttk.Checkbutton(
            filters_frame,
            text="Include Archive",
            variable=self.include_archive
        )
        self.archive_check.grid(row=0, column=4, sticky="w")

        # Initially hide document frame
        self.doc_frame.grid_remove()

    def _create_options_section(self):
        """Create options section."""
        # Options frame
        options_frame = ttk.LabelFrame(self.main_frame, text="Options")
        options_frame.grid(row=4, column=0, sticky="ew", pady=(0, 15))

        # Options container
        opts_container = ttk.Frame(options_frame)
        opts_container.pack(fill="x", padx=10, pady=10)

        # Debug mode checkbox
        self.debug_check = ttk.Checkbutton(
            opts_container,
            text="Show Debug Output",
            variable=self.debug_mode
        )
        self.debug_check.pack(side="left")

        # Training data checkbox
        self.training_check = ttk.Checkbutton(
            opts_container,
            text="Generate Training Data",
            variable=self.training_mode
        )
        self.training_check.pack(side="left", padx=(20, 0))

    def _create_ai_toolbar(self):
        """Create AI toolbar with enhanced styling."""
        # AI Toolbar frame with improved styling
        ai_frame = ttk.LabelFrame(self.main_frame, text="🤖 AI Assistant")
        ai_frame.grid(row=5, column=0, sticky="ew", pady=(0, 20))

        # AI controls container
        ai_container = ttk.Frame(ai_frame)
        ai_container.pack(fill="x", padx=16, pady=16)

        # AI toggle button with primary styling
        self.ai_toggle_button = ttk.Button(
            ai_container,
            text="🔌 Enable AI",
            command=self.toggle_ai,
            style="Primary.TButton",
            width=14
        )
        self.ai_toggle_button.pack(side=tk.LEFT, padx=(0, 8))

        # AI chat button
        self.ai_chat_button = ttk.Button(
            ai_container,
            text="💬 AI Chat",
            command=self.toggle_ai_panel,
            width=14,
            state=tk.DISABLED
        )
        self.ai_chat_button.pack(side=tk.LEFT, padx=(0, 16))

        # AI status indicator with improved styling
        self.ai_status_label = ttk.Label(
            ai_container,
            text="Status: Disabled",
            font=("Segoe UI", 9)
        )
        self.ai_status_label.pack(side=tk.LEFT, pady=2)

    def _create_status_section(self):
        """Create status section."""
        # Status frame
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=6, column=0, sticky="ew", pady=(0, 15))
        status_frame.columnconfigure(0, weight=1)

        # Status label
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_text,
            wraplength=400,
            justify="left"
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate'
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.progress_bar.grid_remove()

    def _create_action_buttons(self):
        """Create action buttons with enhanced styling."""
        # Button frame with improved spacing
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=7, column=0, sticky="ew", pady=(20, 0))

        # Center the buttons with improved spacing
        button_container = ttk.Frame(button_frame)
        button_container.pack(expand=True)

        # Folder mode button with primary styling
        self.open_button = ttk.Button(
            button_container,
            text="📁 Open Folder",
            command=self.execute_folder_navigation,
            style="Primary.TButton",
            width=18
        )
        self.open_button.pack(side="left", padx=(0, 12))

        # Document mode buttons with improved styling (initially hidden)
        self.find_button = ttk.Button(
            button_container,
            text="🔍 Find Documents",
            command=self.execute_document_navigation,
            style="Primary.TButton",
            width=18
        )

        self.choose_button = ttk.Button(
            button_container,
            text="📋 Choose From List",
            command=lambda: self.execute_document_navigation(choose_mode=True),
            width=18
        )

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings...", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Theme", command=self.toggle_theme)
        view_menu.add_command(label="Always on Top", command=self.toggle_always_on_top)

        # AI menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_checkbutton(
            label="Enable AI Assistant",
            variable=self.ai_enabled,
            command=self.toggle_ai
        )
        ai_menu.add_checkbutton(
            label="Show AI Chat Panel",
            variable=self.ai_panel_visible,
            command=self.toggle_ai_panel
        )
        ai_menu.add_separator()
        ai_menu.add_command(label="AI Settings...", command=self.show_ai_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_help)

    def _setup_events(self):
        """Set up event bindings."""
        # Window events
        self.root.bind('<Configure>', self._on_window_configure)
        self.root.bind('<KeyPress>', self._on_key_press)

        # Global shortcuts
        self.root.bind('<Control-Return>', lambda e: self._execute_current_action())
        self.root.bind('<Escape>', lambda e: self.hide_window())
        self.root.bind('<F1>', lambda e: self.show_help())
        self.root.bind('<Control-comma>', lambda e: self.show_settings())

        # Focus events
        self.root.bind('<FocusIn>', self._on_focus_in)

    def _setup_validation(self):
        """Set up input validation."""
        # Register validation commands
        vcmd = (self.root.register(self._validate_project_input), '%P')
        self.project_entry.config(validate='key', validatecommand=vcmd)

        # Numeric validation for filters
        vcmd_num = (self.root.register(self._validate_numeric), '%P')
        self.room_entry.config(validate='key', validatecommand=vcmd_num)
        self.co_entry.config(validate='key', validatecommand=vcmd_num)

    def _apply_theme(self):
        """Apply the current theme."""
        self.theme.apply_theme(self.root)

        # Update status based on current theme
        if self.theme.current_theme == "dark":
            self.status_text.set("Dark theme applied - " + self.status_text.get())

    def _start_task_processor(self):
        """Start the background task processor."""
        self._process_results()

    def _process_results(self):
        """Process results from background tasks."""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._handle_task_result(result)
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self._process_results)

    def _handle_task_result(self, result: Dict[str, Any]):
        """Handle a task result."""
        task_type = result.get('type')
        success = result.get('success', False)
        data = result.get('data')
        error = result.get('error')

        # Hide progress bar
        self.progress_bar.grid_remove()

        if success:
            if task_type == 'project_navigation':
                self._handle_project_result(data)
            elif task_type == 'document_navigation':
                self._handle_document_result(data)
            elif task_type == 'training_data':
                self._handle_training_result(data)
        else:
            self._show_error(f"Operation failed: {error}")
            self.status_text.set("Error occurred")

    def _execute_current_action(self):
        """Execute the action for current mode."""
        if self.current_mode.get() == "folder":
            self.execute_folder_navigation()
        else:
            self.execute_document_navigation()

    # Event handlers
    def _on_mode_change(self):
        """Handle navigation mode change."""
        mode = self.current_mode.get()

        if mode == "folder":
            # Show folder frame, hide document frame
            self.folder_frame.grid()
            self.doc_frame.grid_remove()

            # Show/hide appropriate buttons
            self.open_button.pack(side="left", padx=(0, 10))
            self.find_button.pack_forget()
            self.choose_button.pack_forget()

            self.status_text.set("Folder mode - Select a project subfolder to open")

        else:  # document mode
            # Hide folder frame, show document frame
            self.folder_frame.grid_remove()
            self.doc_frame.grid()

            # Show/hide appropriate buttons
            self.open_button.pack_forget()
            self.find_button.pack(side="left", padx=(0, 10))
            self.choose_button.pack(side="left")

            self.status_text.set("Document mode - Find specific documents by type and filters")

    def _on_project_input_change(self, event=None):
        """Handle project input changes."""
        self._validate_inputs()

    def _on_window_configure(self, event=None):
        """Handle window configuration changes."""
        if event and event.widget == self.root:
            # Update minimum size based on content
            if not self.is_resizing:
                self.is_resizing = True
                self.root.after(100, lambda: setattr(self, 'is_resizing', False))

    def _on_key_press(self, event):
        """Handle key press events."""
        # Handle special key combinations
        if event.keysym == 'Return' and event.state & 0x4:  # Ctrl+Return
            self._execute_current_action()

    def _on_focus_in(self, event):
        """Handle window focus events."""
        # Auto-select project input when window gains focus
        if event.widget == self.root and not self.project_input.get():
            self.project_entry.focus_set()

    # Validation methods
    def _validate_project_input(self, value: str) -> bool:
        """Validate project input."""
        # Allow empty or valid input
        if not value:
            return True

        # Check length
        if len(value) > 100:
            return False

        # Check for invalid filesystem characters
        invalid_chars = '<>:"|?*'
        if any(char in value for char in invalid_chars):
            return False

        return True

    def _validate_numeric(self, value: str) -> bool:
        """Validate numeric input."""
        if not value:
            return True
        return value.isdigit() and len(value) <= 6

    def _validate_inputs(self):
        """Validate all inputs and update status."""
        project_input = self.project_input.get().strip()

        if not project_input:
            self.status_text.set("Enter a 5-digit project number or search term")
            return False

        # Check if it's a 5-digit number
        if project_input.isdigit() and len(project_input) == 5:
            self.status_text.set(f"Ready - Project number: {project_input}")
        else:
            self.status_text.set(f"Ready - Search term: {project_input}")

        return True

    # Action methods
    def execute_folder_navigation(self):
        """Execute folder navigation."""
        if not self._validate_inputs():
            return

        project_input = self.project_input.get().strip()
        selected_folder = self.selected_folder.get()

        # Show progress
        self.progress_bar.grid()
        self.progress_bar.start()
        self.status_text.set("Searching for project...")

        # Execute in background
        def task():
            try:
                result = self.controller.navigate_to_project(
                    project_input=project_input,
                    selected_folder=selected_folder,
                    debug_mode=self.debug_mode.get(),
                    training_data=self.training_mode.get()
                )
                self.result_queue.put({
                    'type': 'project_navigation',
                    'success': True,
                    'data': result
                })
            except Exception as e:
                logger.exception("Project navigation failed")
                self.result_queue.put({
                    'type': 'project_navigation',
                    'success': False,
                    'error': str(e)
                })

        threading.Thread(target=task, daemon=True).start()

    def execute_document_navigation(self, choose_mode: bool = False):
        """Execute document navigation."""
        if not self._validate_inputs():
            return

        project_input = self.project_input.get().strip()

        # Get document type key
        doc_type_text = self.doc_type.get()
        doc_type_map = {
            "Low-Level Design (LLD)": "lld",
            "High-Level Design (HLD)": "hld",
            "Change Orders": "change_order",
            "Sales & PO Reports": "sales_po",
            "Floor Plans": "floor_plans",
            "Scope Documents": "scope",
            "QA/ITP Reports": "qa_itp",
            "SWMS": "swms",
            "Supplier Quotes": "supplier_quotes",
            "Site Photos": "photos"
        }

        doc_type_key = doc_type_map.get(doc_type_text)
        if not doc_type_key:
            self._show_error("Please select a document type")
            return

        # Show progress
        self.progress_bar.grid()
        self.progress_bar.start()
        self.status_text.set("Searching for documents...")

        # Execute in background
        def task():
            try:
                result = self.controller.navigate_to_document(
                    project_input=project_input,
                    doc_type=doc_type_key,
                    version_filter=self.version_filter.get(),
                    room_filter=self.room_filter.get(),
                    co_filter=self.co_filter.get(),
                    include_archive=self.include_archive.get(),
                    choose_mode=choose_mode,
                    debug_mode=self.debug_mode.get(),
                    training_data=self.training_mode.get()
                )
                self.result_queue.put({
                    'type': 'document_navigation',
                    'success': True,
                    'data': result
                })
            except Exception as e:
                logger.exception("Document navigation failed")
                self.result_queue.put({
                    'type': 'document_navigation',
                    'success': False,
                    'error': str(e)
                })

        threading.Thread(target=task, daemon=True).start()

    # Result handlers
    def _handle_project_result(self, result: Dict[str, Any]):
        """Handle project navigation result."""
        status = result.get('status')

        if status == 'SUCCESS':
            path = result.get('path')
            folder = result.get('folder')
            self._open_folder(path)
            self.status_text.set(f"Opened folder: {folder}")

            # Auto-hide after success
            self.root.after(1500, self.hide_window)

        elif status == 'SELECT':
            paths = result.get('paths', [])
            folder = result.get('folder')
            self._show_selection_dialog(paths, folder, 'project')

        elif status == 'SEARCH':
            paths = result.get('paths', [])
            folder = result.get('folder')
            self._show_search_dialog(paths, folder)

        elif status == 'ERROR':
            error = result.get('error', 'Unknown error')
            self._show_error(error)
            self.status_text.set("Project not found")

    def _handle_document_result(self, result: Dict[str, Any]):
        """Handle document navigation result."""
        status = result.get('status')

        if status == 'SUCCESS':
            path = result.get('path')
            self._open_file(path)
            self.status_text.set("Opened document")

            # Auto-hide after success
            self.root.after(1500, self.hide_window)

        elif status == 'SELECT':
            paths = result.get('paths', [])
            self._show_selection_dialog(paths, None, 'document')

        elif status == 'ERROR':
            error = result.get('error', 'Unknown error')
            self._show_error(error)
            self.status_text.set("Documents not found")

    def _handle_training_result(self, result: Dict[str, Any]):
        """Handle training data generation result."""
        if result.get('success'):
            count = result.get('count', 0)
            filepath = result.get('filepath', '')
            message = f"Training data generated: {count} documents"
            if filepath:
                message += f"\nSaved to: {filepath}"
            messagebox.showinfo("Training Data", message)
        else:
            self._show_error(f"Training data generation failed: {result.get('error')}")

    # Dialog methods
    def _show_selection_dialog(self, paths: List[str], folder: Optional[str], dialog_type: str):
        """Show selection dialog for multiple matches."""
        dialog = SelectionDialog(
            self.root,
            title="Select Project" if dialog_type == 'project' else "Select Document",
            paths=paths,
            callback=lambda path: self._handle_selection(path, folder, dialog_type)
        )
        dialog.show()

    def _show_search_dialog(self, paths: List[str], folder: Optional[str]):
        """Show search results dialog."""
        dialog = SearchResultDialog(
            self.root,
            title="Search Results",
            paths=paths,
            callback=lambda path: self._handle_selection(path, folder, 'project')
        )
        dialog.show()

    def _handle_selection(self, path: str, folder: Optional[str], selection_type: str):
        """Handle user selection from dialog."""
        if selection_type == 'project':
            self._open_folder_with_subfolder(path, folder)
        else:
            self._open_file(path)

    def _open_folder(self, path: str):
        """Open folder in file explorer."""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif os.name == 'posix':  # macOS and Linux
                if sys.platform == 'darwin':  # macOS
                    os.system(f'open "{path}"')
                else:  # Linux
                    os.system(f'xdg-open "{path}"')
        except Exception as e:
            logger.exception("Failed to open folder")
            self._show_error(f"Failed to open folder: {e}")

    def _open_folder_with_subfolder(self, project_path: str, subfolder: str):
        """Open specific subfolder within project."""
        # Map folder names to actual paths
        folder_mappings = {
            "5. Floor Plans": "1. Sales Handover/Floor Plans",
            "6. Site Photos": "1. Sales Handover/Site Photos"
        }

        if subfolder in folder_mappings:
            full_path = os.path.join(project_path, folder_mappings[subfolder])
        else:
            full_path = os.path.join(project_path, subfolder)

        if os.path.exists(full_path):
            self._open_folder(full_path)
        else:
            self._show_error(f"Subfolder '{subfolder}' not found")

    def _open_file(self, path: str):
        """Open file with default application."""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif os.name == 'posix':  # macOS and Linux
                if sys.platform == 'darwin':  # macOS
                    os.system(f'open "{path}"')
                else:  # Linux
                    os.system(f'xdg-open "{path}"')
        except Exception as e:
            logger.exception("Failed to open file")
            self._show_error(f"Failed to open file: {e}")

    def _show_error(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", message)
        logger.error(f"Error shown to user: {message}")

    # UI state methods
    def show_window(self):
        """Show the main window."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.project_entry.focus_set()

    def hide_window(self):
        """Hide the main window."""
        self.root.withdraw()

    def toggle_window(self):
        """Toggle window visibility."""
        if self.root.state() == 'withdrawn':
            self.show_window()
        else:
            self.hide_window()

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.theme.toggle_theme()
        self._apply_theme()

    def toggle_always_on_top(self):
        """Toggle always on top mode."""
        current = self.root.attributes('-topmost')
        self.root.attributes('-topmost', not current)

    # Settings and dialogs
    def show_settings(self):
        """Show settings dialog."""
        from .gui_settings import SettingsDialog
        dialog = SettingsDialog(self.root, self.settings)
        dialog.show()

    def show_ai_settings(self):
        """Show AI settings dialog."""
        from .gui_settings import SettingsDialog
        dialog = SettingsDialog(self.root, self.settings)
        # Switch to AI tab if available
        dialog.show()

    def _initialize_ai(self):
        """Initialize AI components if enabled."""
        if AIClient is None or AIChatWidget is None:
            self.ai_enabled.set(False)
            return

        # Check if AI is enabled in settings
        ai_enabled = self.settings.get("ai.enabled", False)
        self.ai_enabled.set(ai_enabled)

        if ai_enabled:
            try:
                # Initialize AI client
                self.ai_client = AIClient(controller=self.controller, settings=self.settings)
                logger.info("AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI client: {e}")
                self.ai_enabled.set(False)

    def toggle_ai(self):
        """Toggle AI assistant on/off."""
        if self.ai_enabled.get():
            if self.ai_client is None:
                self._initialize_ai()

            if self.ai_client is None:
                messagebox.showerror(
                    "AI Error",
                    "Failed to initialize AI assistant. Please check your settings."
                )
                self.ai_enabled.set(False)
                self._update_ai_ui()
                return

            # Create AI chat widget if not exists
            if self.ai_chat_widget is None:
                self._create_ai_chat_panel()

            # Update UI
            self._update_ai_ui()
            self.status_text.set("AI Assistant enabled")
        else:
            # Hide AI panel if visible
            if self.ai_panel_visible.get():
                self.toggle_ai_panel()

            # Update UI
            self._update_ai_ui()
            self.status_text.set("AI Assistant disabled")

    def _update_ai_ui(self):
        """Update AI-related UI elements with enhanced styling."""
        if hasattr(self, 'ai_toggle_button'):
            if self.ai_enabled.get():
                self.ai_toggle_button.config(text="🔌 Disable AI")
                self.ai_chat_button.config(state=tk.NORMAL)
                self.ai_status_label.config(text="Status: ✅ Enabled")
            else:
                self.ai_toggle_button.config(text="🔌 Enable AI")
                self.ai_chat_button.config(state=tk.DISABLED)
                self.ai_status_label.config(text="Status: ❌ Disabled")

    def toggle_ai_panel(self):
        """Toggle AI chat panel visibility."""
        if not self.ai_enabled.get():
            messagebox.showwarning(
                "AI Not Enabled",
                "Please enable AI Assistant first from the AI menu."
            )
            self.ai_panel_visible.set(False)
            return

        if self.ai_panel_visible.get():
            self._show_ai_panel()
        else:
            self._hide_ai_panel()

    def _create_ai_chat_panel(self):
        """Create AI chat panel as a separate window."""
        if self.ai_chat_widget is not None:
            return

        # Create chat window
        self.ai_chat_window = tk.Toplevel(self.root)
        self.ai_chat_window.title("AI Assistant")
        self.ai_chat_window.geometry("400x500")
        self.ai_chat_window.transient(self.root)

        # Apply theme to chat window
        self.theme.apply_theme(self.ai_chat_window)

        # Create chat widget
        self.ai_chat_widget = AIChatWidget(
            self.ai_chat_window,
            ai_client=self.ai_client
        )
        self.ai_chat_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Handle window close
        self.ai_chat_window.protocol("WM_DELETE_WINDOW", self._on_chat_window_close)

    def _show_ai_panel(self):
        """Show AI chat panel."""
        if self.ai_chat_widget is None:
            self._create_ai_chat_panel()

        if hasattr(self, 'ai_chat_window'):
            self.ai_chat_window.deiconify()
            self.ai_chat_window.lift()

    def _hide_ai_panel(self):
        """Hide AI chat panel."""
        if hasattr(self, 'ai_chat_window'):
            self.ai_chat_window.withdraw()

    def _on_chat_window_close(self):
        """Handle AI chat window close."""
        self.ai_panel_visible.set(False)
        self._hide_ai_panel()

    def show_about(self):
        """Show about dialog."""
        about_text = """Project QuickNav - Enhanced GUI

Version: 2.0.0
A comprehensive project navigation tool with enhanced
document search, AI assistance, and cross-platform compatibility.

Features:
• Project folder navigation
• Advanced document search
• AI-powered assistance and chat
• Training data generation
• Theme customization
• Global hotkey support

© 2024 Project QuickNav"""

        messagebox.showinfo("About Project QuickNav", about_text)

    def show_help(self):
        """Show help documentation."""
        help_text = """Project QuickNav Help

Quick Start:
1. Enter a 5-digit project number or search term
2. Choose navigation mode (Folder or Document)
3. Select options and click the action button

Keyboard Shortcuts:
• Ctrl+Alt+Q - Show/hide window (global)
• Ctrl+Return - Execute current action
• Escape - Hide window
• F1 - Show this help
• Ctrl+, - Open settings

Navigation Modes:
• Folder Mode: Open project subfolders
• Document Mode: Find specific documents

For more help, visit the project documentation."""

        messagebox.showinfo("Help", help_text)

    def on_closing(self):
        """Handle window closing."""
        # Save settings before closing
        self.settings.save()

        # Clean up resources
        if hasattr(self, 'hotkey_manager'):
            self.hotkey_manager.cleanup()

        self.root.quit()
        self.root.destroy()

    def run(self):
        """Run the GUI application."""
        try:
            # Initialize global hotkey support
            self._setup_global_hotkey()

            # Show window
            self.show_window()

            # Start main loop
            self.root.mainloop()

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            self.on_closing()
        except Exception as e:
            logger.exception("Unexpected error in main loop")
            self._show_error(f"Unexpected error: {e}")
            self.on_closing()

    def _setup_global_hotkey(self):
        """Set up global hotkey support."""
        try:
            from .gui_hotkey import HotkeyManager
            self.hotkey_manager = HotkeyManager()
            self.hotkey_manager.register_hotkey('ctrl+alt+q', self.toggle_window)
            logger.info("Global hotkey registered: Ctrl+Alt+Q")
        except ImportError:
            logger.warning("Global hotkey support not available")
        except Exception as e:
            logger.warning(f"Failed to set up global hotkey: {e}")


def main():
    """Main entry point for the GUI application."""
    try:
        app = ProjectQuickNavGUI()
        app.run()
    except Exception as e:
        logger.exception("Failed to start GUI application")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()