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
import subprocess
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

# Import core components
try:
    from .gui_controller import GuiController
    from .gui_settings import SettingsManager
    from .gui_theme import ThemeManager
    from .gui_widgets import *
except ImportError:
    # Fallback imports for development
    from gui_controller import GuiController
    from gui_settings import SettingsManager
    from gui_theme import ThemeManager
    from gui_widgets import *

# Import optional AI components with proper error handling
AIClient = None
AIChatWidget = None
AI_AVAILABLE = False
AI_IMPORT_ERROR = None

try:
    from .ai_client import AIClient
    from .ai_chat_widget import ChatWidget as AIChatWidget
    AI_AVAILABLE = True
except ImportError as e:
    try:
        # Try development imports
        from ai_client import AIClient
        from ai_chat_widget import ChatWidget as AIChatWidget
        AI_AVAILABLE = True
    except ImportError as fallback_e:
        AIClient = None
        AIChatWidget = None
        AI_AVAILABLE = False
        AI_IMPORT_ERROR = str(fallback_e)
        logger.info(f"AI components not available: {fallback_e}")

# Check for optional dependencies
OPTIONAL_DEPS = {
    'litellm': False,
    'keyboard': False,
    'pystray': False
}

try:
    import litellm
    OPTIONAL_DEPS['litellm'] = True
except ImportError:
    pass

try:
    import keyboard
    OPTIONAL_DEPS['keyboard'] = True
except ImportError:
    pass

try:
    import pystray
    OPTIONAL_DEPS['pystray'] = True
except ImportError:
    pass


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
        self.min_width = 420
        self.min_height = 720  # Increased to show all components

        # DPI awareness
        self.dpi_scale = self._get_dpi_scale()
        self._apply_dpi_scaling()

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

        # Force widget refresh to ensure theme is visible
        self.root.update_idletasks()
        self.root.after(100, self._force_widget_refresh)

        # Restore window state
        self.restore_window_state()

        # Start task processor
        self._start_task_processor()

        # Update AI UI
        self._update_ai_ui()

        # Initialize navigation state
        self.last_search_result = None
        self.last_search_type = None

        logger.info("ProjectQuickNavGUI initialized successfully")

    def _get_dpi_scale(self):
        """Get DPI scaling factor for the display."""
        try:
            # Get DPI from tkinter
            dpi = self.root.winfo_fpixels('1i')
            # Standard DPI is 96
            scale = dpi / 96.0
            # Clamp scale between 1.0 and 3.0
            return max(1.0, min(3.0, scale))
        except Exception:
            return 1.0

    def _apply_dpi_scaling(self):
        """Apply DPI scaling to the interface."""
        if self.dpi_scale > 1.1:  # Only scale if significantly different
            try:
                # Scale fonts
                import tkinter.font as tkFont
                default_font = tkFont.nametofont("TkDefaultFont")
                default_font.configure(size=int(default_font['size'] * self.dpi_scale))

                # Adjust minimum window size for DPI
                self.min_width = int(self.min_width * self.dpi_scale)
                self.min_height = int(self.min_height * self.dpi_scale)
            except Exception as e:
                logger.warning(f"Failed to apply DPI scaling: {e}")

    def _ensure_main_thread(self, func, *args, **kwargs):
        """Ensure a function runs in the main thread."""
        if threading.current_thread() == threading.main_thread():
            return func(*args, **kwargs)
        else:
            self.root.after(0, lambda: func(*args, **kwargs))

    def _setup_ui(self):
        """Set up the main user interface."""
        # Configure root window with responsive geometry
        saved_geometry = self.settings.get_window_geometry()
        if not saved_geometry or not self._is_geometry_on_screen(saved_geometry):
            saved_geometry = self._get_responsive_geometry()
            # Save the new geometry
            self.settings.set_window_geometry(saved_geometry)

        self.root.geometry(saved_geometry)
        self.root.minsize(self.min_width, self.min_height)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Ensure window is properly sized
        self.root.update_idletasks()

        # Create main container with responsive padding
        padding = max(8, int(12 * self.dpi_scale))
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=padding, pady=padding)

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
        input_frame = ttk.LabelFrame(self.main_frame, text="Project Input")
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        input_frame.columnconfigure(0, weight=1)

        # Project entry with enhanced features and responsive styling
        entry_font_size = max(9, min(12, int(10 * self.dpi_scale)))
        self.project_entry = EnhancedEntry(
            input_frame,
            textvariable=self.project_input,
            placeholder="Enter 5-digit project number or search term",
            font=("Segoe UI", entry_font_size)
        )
        entry_padding = max(12, int(16 * self.dpi_scale))
        ipady_value = max(3, int(4 * self.dpi_scale))
        self.project_entry.grid(row=0, column=0, sticky="ew", padx=entry_padding, pady=entry_padding, ipady=ipady_value)

        # Bind validation events
        self.project_entry.bind('<KeyRelease>', self._on_project_input_change)
        self.project_entry.bind('<FocusOut>', self._on_project_input_change)

    def _create_navigation_mode_section(self):
        """Create navigation mode selection section with enhanced styling."""
        # Navigation mode frame with improved styling
        nav_frame = ttk.LabelFrame(self.main_frame, text="Navigation Mode")
        nav_frame.grid(row=1, column=0, sticky="ew", pady=(0, 16))
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
            text="Find Documents",
            variable=self.current_mode,
            value="document",
            command=self._on_mode_change
        )
        self.doc_radio.pack(anchor="w")

        # Remove standalone settings button - available via menu

    def _create_folder_mode_section(self):
        """Create folder selection section for folder mode."""
        # Folder selection frame
        self.folder_frame = ttk.LabelFrame(self.main_frame, text="Select Subfolder")
        self.folder_frame.grid(row=2, column=0, sticky="ew", pady=(0, 16))

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
        self.doc_frame.grid(row=3, column=0, sticky="ew", pady=(0, 16))
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
        options_frame.grid(row=4, column=0, sticky="ew", pady=(0, 16))

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
        ai_frame = ttk.LabelFrame(self.main_frame, text="AI Assistant")
        ai_frame.grid(row=5, column=0, sticky="ew", pady=(0, 16))

        # AI controls container
        ai_container = ttk.Frame(ai_frame)
        ai_container.pack(fill="x", padx=16, pady=16)

        # AI toggle button with primary styling
        self.ai_toggle_button = ttk.Button(
            ai_container,
            text="Enable AI",
            command=self.toggle_ai,
            width=14
        )
        self.ai_toggle_button.pack(side=tk.LEFT, padx=(0, 8))

        # AI chat button
        self.ai_chat_button = ttk.Button(
            ai_container,
            text="AI Chat",
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
        status_frame.grid(row=6, column=0, sticky="ew", pady=(0, 16))
        status_frame.columnconfigure(0, weight=1)

        # Status label with responsive wrapping
        wrap_length = max(300, int(self.min_width * 0.8))
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_text,
            wraplength=wrap_length,
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
        button_frame.grid(row=7, column=0, sticky="ew", pady=(16, 0))

        # Center the buttons with improved spacing
        button_container = ttk.Frame(button_frame)
        button_container.pack(expand=True)

        # Folder mode button with primary styling
        self.open_button = ttk.Button(
            button_container,
            text="Open Folder",
            command=self.execute_folder_navigation,
            width=18
        )
        self.open_button.pack(side="left", padx=(0, 12))

        # Document mode buttons with improved styling (initially hidden)
        self.find_button = ttk.Button(
            button_container,
            text="Find Documents",
            command=self.execute_document_navigation,
            width=16
        )

        self.choose_button = ttk.Button(
            button_container,
            text="Choose From List",
            command=lambda: self.execute_document_navigation(choose_mode=True),
            width=16
        )

        # Open/Navigate button (appears after successful search)
        self.navigate_button = ttk.Button(
            button_container,
            text="Open Selected",
            command=self._execute_final_navigation,
            width=16,
            state=tk.DISABLED
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
        help_menu.add_command(label="Dependency Status...", command=self.show_dependency_status)
        help_menu.add_separator()
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

        # Force update after theme change
        self.root.update_idletasks()

        # Ensure proper window colors
        current_theme_obj = self.theme.get_current_theme()
        if current_theme_obj:
            window_bg = current_theme_obj.get_color("window", "bg") or current_theme_obj.get_color("bg")
            if window_bg:
                self.root.configure(bg=window_bg)
                # Update main frame background too
                if hasattr(self, 'main_frame'):
                    try:
                        self.main_frame.configure(style="TFrame")
                    except:
                        pass

    def _start_task_processor(self):
        """Start the background task processor."""
        self._process_results()

    def _process_results(self):
        """Process results from background tasks safely in main thread."""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._handle_task_result(result)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing task result: {e}")

        # Schedule next check
        self.root.after(100, self._process_results)

    def _handle_task_result(self, result: Dict[str, Any]):
        """Handle a task result safely from main thread."""
        task_type = result.get('type')
        success = result.get('success', False)
        data = result.get('data')
        error = result.get('error')

        # Ensure we're running in the main thread
        if threading.current_thread() != threading.main_thread():
            self.root.after(0, lambda: self._handle_task_result(result))
            return

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
            self.open_button.pack(side="left", padx=(0, 8))
            self.find_button.pack_forget()
            self.choose_button.pack_forget()
            self.navigate_button.pack_forget()

            self.status_text.set("Folder mode - Select a project subfolder to open")

        else:  # document mode
            # Hide folder frame, show document frame
            self.folder_frame.grid_remove()
            self.doc_frame.grid()

            # Show/hide appropriate buttons
            self.open_button.pack_forget()
            self.find_button.pack(side="left", padx=(0, 8))
            self.choose_button.pack(side="left", padx=(0, 8))
            if hasattr(self, 'last_search_result') and self.last_search_result:
                self.navigate_button.pack(side="left")
            else:
                self.navigate_button.pack_forget()

            self.status_text.set("Document mode - Find specific documents by type and filters")

    def _on_project_input_change(self, event=None):
        """Handle project input changes."""
        self._validate_inputs()

    def _on_window_configure(self, event=None):
        """Handle window configuration changes."""
        if event and event.widget == self.root:
            # Update responsive elements based on window size
            if not self.is_resizing:
                self.is_resizing = True
                self._update_responsive_layout()
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

        # Capture current values to avoid thread safety issues
        debug_mode_value = self.debug_mode.get()
        training_mode_value = self.training_mode.get()

        # Execute in background
        def task():
            try:
                result = self.controller.navigate_to_project(
                    project_input=project_input,
                    selected_folder=selected_folder,
                    debug_mode=debug_mode_value,
                    training_data=training_mode_value
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

        # Capture current values to avoid thread safety issues
        version_filter_value = self.version_filter.get()
        room_filter_value = self.room_filter.get()
        co_filter_value = self.co_filter.get()
        include_archive_value = self.include_archive.get()
        debug_mode_value = self.debug_mode.get()
        training_mode_value = self.training_mode.get()

        # Execute in background
        def task():
            try:
                result = self.controller.navigate_to_document(
                    project_input=project_input,
                    doc_type=doc_type_key,
                    version_filter=version_filter_value,
                    room_filter=room_filter_value,
                    co_filter=co_filter_value,
                    include_archive=include_archive_value,
                    choose_mode=choose_mode,
                    debug_mode=debug_mode_value,
                    training_data=training_mode_value
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
            # Store result for potential navigation
            self.last_search_result = {'path': path, 'folder': folder}
            self.last_search_type = 'project'
            # Directly open in folder mode
            self._open_folder_with_subfolder(path, folder)
            self.status_text.set(f"Opened folder: {folder}")

            # Auto-hide after success
            self.root.after(1500, self.hide_window)

        elif status == 'SELECT':
            paths = result.get('paths', [])
            folder = result.get('folder')
            self.last_search_result = {'paths': paths, 'folder': folder}
            self.last_search_type = 'project_select'
            self._show_selection_dialog(paths, folder, 'project')
            self._enable_navigate_button()

        elif status == 'SEARCH':
            paths = result.get('paths', [])
            folder = result.get('folder')
            self.last_search_result = {'paths': paths, 'folder': folder}
            self.last_search_type = 'project_search'
            self._show_search_dialog(paths, folder)
            self._enable_navigate_button()

        elif status == 'ERROR':
            error = result.get('error', 'Unknown error')
            self._show_error(error)
            self.status_text.set("Project not found")
            self.last_search_result = None
            self._disable_navigate_button()

    def _handle_document_result(self, result: Dict[str, Any]):
        """Handle document navigation result."""
        status = result.get('status')

        if status == 'SUCCESS':
            path = result.get('path')
            # Store result for potential navigation
            self.last_search_result = {'path': path}
            self.last_search_type = 'document'
            # Directly open the document
            self._open_file(path)
            self.status_text.set("Opened document")

            # Auto-hide after success
            self.root.after(1500, self.hide_window)

        elif status == 'SELECT':
            paths = result.get('paths', [])
            self.last_search_result = {'paths': paths}
            self.last_search_type = 'document_select'
            self._show_selection_dialog(paths, None, 'document')
            self._enable_navigate_button()

        elif status == 'ERROR':
            error = result.get('error', 'Unknown error')
            self._show_error(error)
            self.status_text.set("Documents not found")
            self.last_search_result = None
            self._disable_navigate_button()

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
        # Update search result with selected path
        if selection_type == 'project':
            self.last_search_result = {'path': path, 'folder': folder}
            self.last_search_type = 'project'
            self._open_folder_with_subfolder(path, folder)
        else:
            self.last_search_result = {'path': path}
            self.last_search_type = 'document'
            self._open_file(path)

        self.status_text.set("Selection opened")
        # Auto-hide after success
        self.root.after(1500, self.hide_window)

    def _open_folder(self, path: str):
        """Open folder in file explorer."""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif os.name == 'posix':  # macOS and Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', path], check=False)
                else:  # Linux
                    subprocess.run(['xdg-open', path], check=False)
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
                    subprocess.run(['open', path], check=False)
                else:  # Linux
                    subprocess.run(['xdg-open', path], check=False)
        except (OSError, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.exception("Failed to open file")
            self._show_error(f"Failed to open file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error opening file {path}: {e}")
            self._show_error(f"Failed to open file: {e}")

    def _show_error(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", message)
        logger.error(f"Error shown to user: {message}")

    def _execute_final_navigation(self):
        """Execute final navigation using stored search result."""
        if not self.last_search_result:
            self.status_text.set("No search result to navigate to")
            return

        try:
            if self.last_search_type in ['project', 'project_select', 'project_search']:
                path = self.last_search_result.get('path')
                folder = self.last_search_result.get('folder', self.selected_folder.get())
                if path:
                    self._open_folder_with_subfolder(path, folder)
                    self.status_text.set(f"Opened folder: {folder}")
                    self.root.after(1500, self.hide_window)
                else:
                    self.status_text.set("No valid path selected")
            elif self.last_search_type in ['document', 'document_select']:
                path = self.last_search_result.get('path')
                if path:
                    self._open_file(path)
                    self.status_text.set("Opened document")
                    self.root.after(1500, self.hide_window)
                else:
                    self.status_text.set("No valid document selected")
        except Exception as e:
            logger.exception("Final navigation failed")
            self._show_error(f"Failed to open: {e}")

    def _enable_navigate_button(self):
        """Enable the navigate button when results are available."""
        if hasattr(self, 'navigate_button'):
            self.navigate_button.config(state=tk.NORMAL)
            if self.current_mode.get() == "document":
                self.navigate_button.pack(side="left")

    def _disable_navigate_button(self):
        """Disable the navigate button when no results are available."""
        if hasattr(self, 'navigate_button'):
            self.navigate_button.config(state=tk.DISABLED)
            self.navigate_button.pack_forget()

    def _update_responsive_layout(self):
        """Update layout elements for current window size."""
        try:
            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()

            # Update status label wrap length based on window width
            if hasattr(self, 'status_label'):
                wrap_length = max(300, int(current_width * 0.8))
                self.status_label.config(wraplength=wrap_length)

            # Adjust button layout for narrow windows
            if hasattr(self, 'open_button') and current_width < 500:
                # Stack buttons vertically for narrow windows
                button_width = max(12, int(current_width * 0.25))
                self.open_button.config(width=button_width)
                if hasattr(self, 'find_button'):
                    self.find_button.config(width=button_width)
                if hasattr(self, 'choose_button'):
                    self.choose_button.config(width=button_width)
                if hasattr(self, 'navigate_button'):
                    self.navigate_button.config(width=button_width)
            else:
                # Normal horizontal layout
                self.open_button.config(width=16)
                if hasattr(self, 'find_button'):
                    self.find_button.config(width=16)
                if hasattr(self, 'choose_button'):
                    self.choose_button.config(width=16)
                if hasattr(self, 'navigate_button'):
                    self.navigate_button.config(width=16)

        except Exception as e:
            logger.warning(f"Failed to update responsive layout: {e}")

    # UI state methods
    def show_window(self):
        """Show the main window."""
        # Ensure proper geometry before showing
        if self.root.geometry() == "1x1+0+0" or self.root.winfo_width() < self.min_width:
            self.root.geometry(self._get_responsive_geometry())

        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.project_entry.focus_set()

        # Update layout after showing
        self.root.update_idletasks()

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
        # Force widget refresh to ensure theme changes are visible
        self.root.after(50, self._force_widget_refresh)

    def toggle_always_on_top(self):
        """Toggle always on top mode."""
        current = self.root.attributes('-topmost')
        new_state = not current
        self.root.attributes('-topmost', new_state)

        # Save the preference
        self.settings.set("ui.always_on_top", new_state)

    def restore_window_state(self):
        """Restore window state from settings."""
        try:
            # Restore always on top setting
            always_on_top = self.settings.get("ui.always_on_top", False)
            self.root.attributes('-topmost', always_on_top)

            # Ensure window is visible and properly positioned
            geometry = self.settings.get_window_geometry()
            if geometry:
                # Parse geometry to ensure it's on screen
                if self._is_geometry_on_screen(geometry):
                    self.root.geometry(geometry)
                else:
                    # Reset to responsive default if off-screen
                    self.root.geometry(self._get_responsive_geometry())
                    logger.info("Window was off-screen, reset to responsive geometry")

        except (tk.TclError, ValueError, KeyError) as e:
            logger.warning(f"Window state restoration error: {e}")
            # Fallback to responsive geometry
            self.root.geometry(self._get_responsive_geometry())
        except Exception as e:
            logger.error(f"Unexpected error restoring window state: {e}")
            # Fallback to minimum size
            self.root.geometry(f"{self.min_width}x{self.min_height}")

    def _force_widget_refresh(self):
        """Force refresh of all widgets to apply theme changes."""
        try:
            # Force re-application of styles to all ttk widgets
            def refresh_widget(widget):
                try:
                    if isinstance(widget, ttk.Widget):
                        # Force widget to re-read its style by temporarily changing and restoring it
                        try:
                            original_style = widget.cget('style')
                            widget.configure(style='')  # Clear style
                            self.root.update_idletasks()
                            widget.configure(style=original_style)  # Restore style
                        except:
                            # If no style, apply default for widget type
                            widget_type = widget.__class__.__name__
                            if widget_type.startswith('Ttk'):
                                default_style = widget_type[3:]  # Remove 'Ttk' prefix
                            else:
                                default_style = widget_type
                            try:
                                widget.configure(style=f"T{default_style}")
                            except:
                                pass

                    # Recurse to children
                    for child in widget.winfo_children():
                        refresh_widget(child)
                except Exception:
                    pass

            # Refresh all widgets starting from root
            refresh_widget(self.root)

            # Force final update
            self.root.update_idletasks()

        except Exception as e:
            logger.warning(f"Error refreshing widgets: {e}")

    def _get_responsive_geometry(self):
        """Get responsive geometry string based on screen size."""
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Calculate size based on screen resolution and DPI
            if screen_width >= 2560:  # 2K/4K displays
                width_factor = 0.25
                height_factor = 0.45  # Increased for better fit
            elif screen_width >= 1920:  # 1080p displays
                width_factor = 0.3
                height_factor = 0.5   # Increased for better fit
            else:  # Smaller displays
                width_factor = 0.4
                height_factor = 0.6   # Increased for better fit

            width = max(self.min_width, int(screen_width * width_factor))
            height = max(self.min_height, int(screen_height * height_factor))

            # Center on screen
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

            return f"{width}x{height}+{x}+{y}"
        except Exception:
            return f"{self.min_width}x{self.min_height}"

    def _is_geometry_on_screen(self, geometry: str) -> bool:
        """Check if geometry would place window on screen."""
        try:
            # Parse geometry string: "WxH+X+Y" or "WxH"
            if '+' in geometry:
                size_part, pos_part = geometry.split('+', 1)
                width, height = map(int, size_part.split('x'))
                pos_parts = pos_part.split('+')
                x = int(pos_parts[0]) if pos_parts[0] else 0
                y = int(pos_parts[1]) if len(pos_parts) > 1 and pos_parts[1] else 0

                # Get screen dimensions
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()

                # Check if window would be mostly visible
                return (x >= -width//2 and y >= -height//2 and
                       x < screen_width - width//2 and y < screen_height - height//2)
            else:
                # No position specified, just size - that's fine
                return True

        except (ValueError, IndexError, AttributeError) as e:
            logger.debug(f"Geometry parsing error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking geometry: {e}")
            return False

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
        """Initialize AI components if enabled with improved error handling."""
        if not AI_AVAILABLE:
            self.ai_enabled.set(False)
            if AI_IMPORT_ERROR:
                logger.info(f"AI features disabled due to import error: {AI_IMPORT_ERROR}")
            return

        # Check if AI is enabled in settings
        ai_enabled = self.settings.get("ai.enabled", False)
        self.ai_enabled.set(ai_enabled)

        if ai_enabled:
            try:
                # Check for required dependencies
                if not OPTIONAL_DEPS['litellm']:
                    logger.warning("AI enabled but LiteLLM not available. Install with: pip install litellm")
                    self.ai_enabled.set(False)
                    return

                # Initialize AI client
                self.ai_client = AIClient(controller=self.controller, settings=self.settings)

                # Verify client is functional
                if self.ai_client and self.ai_client.is_available():
                    logger.info("AI client initialized successfully")
                else:
                    logger.warning("AI client initialized but not available")
                    self.ai_enabled.set(False)

            except (ImportError, AttributeError, TypeError) as e:
                logger.error(f"AI client initialization error: {e}")
                self.ai_enabled.set(False)
            except Exception as e:
                logger.error(f"Unexpected error initializing AI client: {e}")
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
                self.ai_toggle_button.config(text="Disable AI")
                self.ai_chat_button.config(state=tk.NORMAL)
                self.ai_status_label.config(text="Status: Enabled")
            else:
                self.ai_toggle_button.config(text="Enable AI")
                self.ai_chat_button.config(state=tk.DISABLED)
                self.ai_status_label.config(text="Status: Disabled")

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
        # Save current window geometry
        try:
            current_geometry = self.root.geometry()
            if current_geometry:
                self.settings.set_window_geometry(current_geometry)
                logger.debug(f"Saved window geometry: {current_geometry}")
        except Exception as e:
            logger.warning(f"Failed to save window geometry: {e}")

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
            if self.hotkey_manager.register_hotkey('ctrl+alt+q', self.toggle_window):
                logger.info("Global hotkey registered: Ctrl+Alt+Q")
            else:
                logger.warning("Failed to register global hotkey Ctrl+Alt+Q")
        except ImportError:
            logger.warning("Global hotkey support not available")
        except Exception as e:
            logger.warning(f"Failed to set up global hotkey: {e}")

    def get_hotkey_manager(self):
        """Get the hotkey manager instance."""
        return getattr(self, 'hotkey_manager', None)

    def register_hotkey(self, hotkey: str, callback) -> bool:
        """Register a new global hotkey."""
        if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
            try:
                return self.hotkey_manager.register_hotkey(hotkey, callback)
            except Exception as e:
                logger.error(f"Failed to register hotkey {hotkey}: {e}")
                return False
        return False

    def unregister_hotkey(self, hotkey: str) -> bool:
        """Unregister a global hotkey."""
        if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
            try:
                return self.hotkey_manager.unregister_hotkey(hotkey)
            except Exception as e:
                logger.error(f"Failed to unregister hotkey {hotkey}: {e}")
                return False
        return False

    def get_registered_hotkeys(self) -> List[str]:
        """Get list of registered hotkeys."""
        if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
            try:
                return self.hotkey_manager.get_registered_hotkeys()
            except Exception as e:
                logger.error(f"Failed to get registered hotkeys: {e}")
                return []
        return []

    def is_hotkey_available(self, hotkey: str) -> bool:
        """Check if a hotkey is available for registration."""
        if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
            try:
                return self.hotkey_manager.is_hotkey_available(hotkey)
            except Exception as e:
                logger.error(f"Failed to check hotkey availability: {e}")
                return False
        return False

    def get_dependency_status(self) -> Dict[str, Any]:
        """Get status of optional dependencies."""
        return {
            'ai_available': AI_AVAILABLE,
            'ai_error': AI_IMPORT_ERROR,
            'optional_deps': OPTIONAL_DEPS.copy(),
            'ai_client_ready': hasattr(self, 'ai_client') and self.ai_client and self.ai_client.is_available(),
            'hotkey_manager_ready': hasattr(self, 'hotkey_manager') and self.hotkey_manager is not None
        }

    def show_dependency_status(self):
        """Show a dialog with dependency status."""
        status = self.get_dependency_status()

        message = "Dependency Status:\n\n"

        # Core components
        message += "Core Components:\n"
        message += f"✓ GUI Controller: Available\n"
        message += f"✓ Settings Manager: Available\n"
        message += f"✓ Theme Manager: Available\n\n"

        # AI Components
        message += "AI Components:\n"
        if status['ai_available']:
            message += f"✓ AI Client: Available\n"
            if status['ai_client_ready']:
                message += f"✓ AI Client Ready: Yes\n"
            else:
                message += f"⚠ AI Client Ready: No\n"
        else:
            message += f"✗ AI Client: Not Available\n"
            if status['ai_error']:
                message += f"  Error: {status['ai_error']}\n"

        # Optional Dependencies
        message += "\nOptional Dependencies:\n"
        for dep, available in status['optional_deps'].items():
            icon = "✓" if available else "✗"
            message += f"{icon} {dep}: {'Available' if available else 'Not Available'}\n"

        # Installation suggestions
        missing_deps = [dep for dep, available in status['optional_deps'].items() if not available]
        if missing_deps:
            message += f"\nTo install missing dependencies:\n"
            message += f"pip install {' '.join(missing_deps)}\n"

        # Hardware features
        message += "\nFeature Status:\n"
        message += f"{'✓' if status['hotkey_manager_ready'] else '✗'} Global Hotkeys: {'Available' if status['hotkey_manager_ready'] else 'Not Available'}\n"

        messagebox.showinfo("Dependency Status", message)


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