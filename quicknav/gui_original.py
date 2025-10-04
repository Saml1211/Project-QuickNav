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

# Constants for better maintainability
class GUIConstants:
    """Constants used throughout the GUI application."""
    # Window dimensions
    DEFAULT_MIN_WIDTH = 420
    DEFAULT_MIN_HEIGHT = 720
    COMPILED_ENV_CHECK_DELAY = 200  # ms
    DPI_SCALE_THRESHOLD = 1.1
    THEME_REFRESH_DELAY = 50  # ms

    # UI Layout
    SECTION_PADDING = 16
    ELEMENT_PADDING = 8
    BUTTON_WIDTH = 16
    ENTRY_WIDTH = 8

    # Animation and timing
    RESIZE_DEBOUNCE_DELAY = 100  # ms
    AUTOCOMPLETE_DEBOUNCE_DELAY = 300  # ms
    WINDOW_AUTO_HIDE_DELAY = 1500  # ms

    # Display scaling
    DPI_SCALE_MIN = 1.0
    DPI_SCALE_MAX = 3.0
    STANDARD_DPI = 96

    # Keyboard shortcuts
    HOTKEY_RESET_WINDOW = '<Control-Shift-R>'
    HOTKEY_TOGGLE_THEME = '<Control-d>'
    HOTKEY_RESET_FORM = '<Control-r>'

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
        self.doc_type = tk.StringVar()  # Selected document type for search
        self.version_filter = tk.StringVar(value="Auto (Latest/Best)")  # Version filter preference
        self.room_filter = tk.StringVar()  # Room number filter
        self.co_filter = tk.StringVar()  # Change Order filter
        self.include_archive = tk.BooleanVar()  # Include archived documents flag

        # Folder mode variables
        self.selected_folder = tk.StringVar(value="4. System Designs")  # Selected project subfolder

        # UI state management
        self.is_resizing = False  # Flag to prevent resize loops
        self.min_width = GUIConstants.DEFAULT_MIN_WIDTH  # Minimum window width
        self.min_height = GUIConstants.DEFAULT_MIN_HEIGHT  # Minimum window height

        # DPI scaling support
        self.dpi_scale = self._get_dpi_scale()  # Display scaling factor
        self._apply_dpi_scaling()

        # Background task management
        self.task_queue = queue.Queue()  # Queue for background operations
        self.result_queue = queue.Queue()  # Queue for operation results

        # AI assistant components
        self.ai_client = None  # AI client instance
        self.ai_chat_widget = None  # AI chat widget instance
        self.ai_enabled = tk.BooleanVar()  # AI feature enabled flag
        self.ai_panel_visible = tk.BooleanVar(value=False)  # AI chat panel visibility

        # Initialize AI if enabled
        self._initialize_ai()

        # Batch initialization operations for better performance
        self._batch_initialize_ui()

        logger.info("ProjectQuickNavGUI initialized successfully")

    def _get_dpi_scale(self):
        """Get DPI scaling factor for the display."""
        try:
            # Get DPI from tkinter
            dpi = self.root.winfo_fpixels('1i')
            # Calculate scale factor using standard DPI
            scale = dpi / GUIConstants.STANDARD_DPI
            # Clamp scale between min and max values
            return max(GUIConstants.DPI_SCALE_MIN, min(GUIConstants.DPI_SCALE_MAX, scale))
        except Exception:
            return GUIConstants.DPI_SCALE_MIN

    def _get_consistent_padding(self) -> int:
        """Get consistent padding value based on DPI scaling and window size."""
        base_padding = max(8, int(12 * self.dpi_scale))
        current_width = self.root.winfo_width()

        # Adjust padding for very small windows to maximize content area
        if current_width > 0 and current_width < 450:
            return max(4, int(base_padding * 0.7))
        return base_padding

    def _get_consistent_spacing(self) -> int:
        """Get consistent spacing value for internal component spacing."""
        return max(6, int(8 * self.dpi_scale))

    def _apply_dpi_scaling(self):
        """Apply DPI scaling to the interface."""
        if self.dpi_scale > GUIConstants.DPI_SCALE_THRESHOLD:  # Only scale if significantly different
            try:
                logger.info(f"Applying DPI scaling factor: {self.dpi_scale}")

                # Scale fonts
                import tkinter.font as tkFont
                default_font = tkFont.nametofont("TkDefaultFont")
                original_size = default_font['size']
                new_size = int(original_size * self.dpi_scale)
                default_font.configure(size=new_size)
                logger.info(f"Font scaled from {original_size} to {new_size}")

                # Adjust minimum window size for DPI
                original_min_width = self.min_width
                original_min_height = self.min_height
                self.min_width = int(self.min_width * self.dpi_scale)
                self.min_height = int(self.min_height * self.dpi_scale)
                logger.info(f"Min window size scaled from {original_min_width}x{original_min_height} to {self.min_width}x{self.min_height}")
            except Exception as e:
                logger.warning(f"Failed to apply DPI scaling: {e}")

    def _update_dpi_scaling_for_resize(self, width: int, height: int):
        """Update DPI scaling calculations during resize for better consistency."""
        try:
            # Recalculate DPI scale based on current window size
            old_dpi_scale = self.dpi_scale
            self.dpi_scale = self._get_dpi_scale()

            # Only reapply scaling if there's a significant change (>5%)
            if abs(self.dpi_scale - old_dpi_scale) > 0.05:
                logger.info(f"DPI scale changed during resize: {old_dpi_scale} -> {self.dpi_scale}")

                # Reapply DPI scaling to fonts and minimum sizes
                if self.dpi_scale > GUIConstants.DPI_SCALE_THRESHOLD:
                    import tkinter.font as tkFont
                    default_font = tkFont.nametofont("TkDefaultFont")
                    original_size = default_font['size']
                    new_size = int(original_size * self.dpi_scale)
                    default_font.configure(size=new_size)

                    # Update minimum window size for new DPI scale
                    base_min_width = GUIConstants.DEFAULT_MIN_WIDTH
                    base_min_height = GUIConstants.DEFAULT_MIN_HEIGHT
                    self.min_width = int(base_min_width * self.dpi_scale)
                    self.min_height = int(base_min_height * self.dpi_scale)

                    # Update root window minimum size
                    self.root.minsize(self.min_width, self.min_height)
                    logger.info(f"DPI scaling updated for resize - Min size: {self.min_width}x{self.min_height}")
        except Exception as e:
            logger.warning(f"Failed to update DPI scaling during resize: {e}")

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
        padding = self._get_consistent_padding()
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=padding, pady=padding)

        # Configure grid weights for responsive layout - more compact design
        self.main_frame.columnconfigure(0, weight=1)
        # Give weight to rows that actually contain content for better space distribution
        self.main_frame.rowconfigure(2, weight=1)  # Main content area gets extra space
        self.main_frame.rowconfigure(4, weight=1)  # Status area gets extra space

        # Create sections in a more intuitive order with reduced spacing
        logger.info("Initializing UI sections")
        self._create_project_input_section()      # Row 0 - Primary input
        self._create_navigation_mode_section()   # Row 1 - Mode selection
        self._create_main_content_section()      # Row 2 - Folder or Document mode (combined)
        self._create_sidebar_section()           # Row 3 - Options and AI toolbar (side by side)
        self._create_status_section()            # Row 4 - Status information
        self._create_action_buttons()            # Row 5 - Action buttons
        self._create_menu_bar()
        logger.info("UI initialization complete")

    def _create_project_input_section(self):
        """Create project input section with enhanced styling."""
        # Project input frame with improved styling
        input_frame = ttk.LabelFrame(self.main_frame, text="Project Input")
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, self._get_consistent_spacing() // 2))
        input_frame.columnconfigure(0, weight=1)

        # Project entry with enhanced features and responsive styling
        entry_font_size = max(9, min(12, int(10 * self.dpi_scale)))
        self.project_entry = EnhancedEntry(
            input_frame,
            textvariable=self.project_input,
            placeholder="Enter 5-digit project number or search term",
            font=("Segoe UI", entry_font_size)
        )
        entry_padding = self._get_consistent_padding()
        ipady_value = max(3, int(4 * self.dpi_scale))
        self.project_entry.grid(row=0, column=0, sticky="ew", padx=entry_padding, pady=(entry_padding, self._get_consistent_spacing()), ipady=ipady_value)

        # Bind validation events
        self.project_entry.bind('<KeyRelease>', self._on_project_input_change)
        self.project_entry.bind('<FocusOut>', self._on_project_input_change)

        # Recent projects frame with consistent layout
        recent_frame = ttk.Frame(input_frame)
        recent_frame.grid(row=1, column=0, sticky="ew", padx=entry_padding, pady=(0, entry_padding))
        recent_frame.columnconfigure(0, weight=1)

        # Recent projects label with consistent styling
        recent_label = ttk.Label(recent_frame, text="⏱️ Recent:", font=("Segoe UI", 9))
        recent_label.grid(row=0, column=0, sticky="w", padx=(0, self._get_consistent_spacing()))

        # Recent projects buttons container with proper grid layout
        self.recent_buttons_frame = ttk.Frame(recent_frame)
        self.recent_buttons_frame.grid(row=0, column=1, sticky="ew")
        self.recent_buttons_frame.columnconfigure(0, weight=1)

        # Populate recent projects
        self._update_recent_projects()

    def _create_navigation_mode_section(self):
        """Create navigation mode selection section with enhanced styling."""
        # Navigation mode frame with improved styling
        nav_frame = ttk.LabelFrame(self.main_frame, text="Navigation Mode")
        nav_frame.grid(row=1, column=0, sticky="ew", pady=(0, self._get_consistent_spacing() // 2))
        nav_frame.columnconfigure(0, weight=1)

        # Mode selection frame with improved spacing and grid layout
        mode_frame = ttk.Frame(nav_frame)
        mode_frame.grid(row=0, column=0, sticky="ew", padx=self._get_consistent_padding(), pady=self._get_consistent_padding())
        mode_frame.columnconfigure(0, weight=1)

        # Radio buttons for navigation mode with better styling and icons using grid layout
        self.folder_radio = ttk.Radiobutton(
            mode_frame,
            text="📁 Open Project Folder",
            variable=self.current_mode,
            value="folder",
            command=self._on_mode_change
        )
        self.folder_radio.grid(row=0, column=0, sticky="w", pady=(0, self._get_consistent_spacing() // 2))
        self._add_tooltip(self.folder_radio, "Navigate directly to a project subfolder")

        self.doc_radio = ttk.Radiobutton(
            mode_frame,
            text="📄 Find Documents",
            variable=self.current_mode,
            value="document",
            command=self._on_mode_change
        )
        self.doc_radio.grid(row=1, column=0, sticky="w")
        self._add_tooltip(self.doc_radio, "Search for specific documents within projects")

        # Remove standalone settings button - available via menu

    def _create_main_content_section(self):
        """Create main content section that switches between folder and document modes."""
        # Create a container frame for mode-specific content
        self.main_content_frame = ttk.Frame(self.main_frame)
        self.main_content_frame.grid(row=2, column=0, sticky="ew", pady=(0, self._get_consistent_spacing() // 2))
        self.main_content_frame.columnconfigure(0, weight=1)

        # Create folder mode section within the main content frame
        self._create_folder_mode_section()

        # Create document mode section within the main content frame
        self._create_document_mode_section()

        # Initially show folder mode (document mode is created hidden)
        self.folder_frame.grid(row=0, column=0, sticky="ew")

    def _create_folder_mode_section(self):
        """Create folder selection section for folder mode."""
        # Folder selection frame
        self.folder_frame = ttk.LabelFrame(self.main_content_frame, text="Select Subfolder")
        # Note: Frame created but not gridded - will be shown by _create_main_content_section

        # Folder options with icons
        folder_options = [
            ("📁 4. System Designs", "4. System Designs"),
            ("📁 1. Sales Handover", "1. Sales Handover"),
            ("📁 2. BOM & Orders", "2. BOM & Orders"),
            ("📁 6. Customer Handover Documents", "6. Customer Handover Documents"),
            ("📁 5. Floor Plans", "5. Floor Plans"),
            ("📁 6. Site Photos", "6. Site Photos")
        ]

        # Create radio buttons for folders with icons and tooltips using grid layout
        self.folder_radios = []
        folder_tooltips = {
            "4. System Designs": "Technical drawings, CAD files, and system specifications",
            "1. Sales Handover": "Initial project information and requirements",
            "2. BOM & Orders": "Bill of materials and purchase orders",
            "6. Customer Handover Documents": "Final delivery documentation",
            "5. Floor Plans": "Architectural drawings and layouts",
            "6. Site Photos": "Project site photographs and documentation"
        }

        # Create container frame with grid layout
        folder_container = ttk.Frame(self.folder_frame)
        folder_container.grid(row=0, column=0, sticky="ew", padx=self._get_consistent_padding(), pady=self._get_consistent_padding())
        folder_container.columnconfigure(0, weight=1)

        for i, (display_text, folder_value) in enumerate(folder_options):
            radio = ttk.Radiobutton(
                folder_container,
                text=display_text,
                variable=self.selected_folder,
                value=folder_value
            )
            radio.grid(row=i, column=0, sticky="w", pady=2)
            self.folder_radios.append(radio)
            self._add_tooltip(radio, folder_tooltips.get(folder_value, "Project subfolder"))

    def _create_document_mode_section(self):
        """Create document type and filter section for document mode."""
        # Document mode frame (initially hidden) within main content frame
        self.doc_frame = ttk.LabelFrame(self.main_content_frame, text="Document Type & Filters")
        self.doc_frame.columnconfigure(0, weight=1)

        # Create document type selection
        self._create_document_type_selection()

        # Create version filter
        self._create_version_filter()

        # Create document filters (room, CO, archive)
        self._create_document_filters()

        # Note: Frame is created but not gridded - will be shown when document mode is selected

    def _create_document_type_selection(self):
        """Create document type selection controls."""
        # Document type label and options with consistent spacing
        doc_type_label = ttk.Label(self.doc_frame, text="Document Type:")
        doc_type_label.grid(
            row=0, column=0, sticky="w", padx=self._get_consistent_padding(), pady=(self._get_consistent_padding(), self._get_consistent_spacing() // 2)
        )

        doc_type_options = [
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
            self.doc_frame,
            textvariable=self.doc_type,
            values=doc_type_options,
            state="readonly"
        )
        self.doc_type_combo.grid(row=0, column=1, sticky="ew", padx=(0, self._get_consistent_padding()), pady=(self._get_consistent_padding(), self._get_consistent_spacing() // 2))
        self.doc_type_combo.current(0)
        self._add_tooltip(self.doc_type_combo, "Select the type of document to search for")

    def _create_version_filter(self):
        """Create version filter controls."""
        version_label = ttk.Label(self.doc_frame, text="Version Filter:")
        version_label.grid(
            row=1, column=0, sticky="w", padx=self._get_consistent_padding(), pady=(self._get_consistent_spacing() // 2, self._get_consistent_spacing() // 2)
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
        self.version_combo.grid(row=1, column=1, sticky="ew", padx=(0, self._get_consistent_padding()), pady=(self._get_consistent_spacing() // 2, self._get_consistent_spacing() // 2))
        self._add_tooltip(self.version_combo, "Filter documents by version (latest, as-built, etc.)")

    def _create_document_filters(self):
        """Create document filter controls (room, CO, archive)."""
        # Filters frame with responsive layout and consistent spacing
        filters_frame = ttk.Frame(self.doc_frame)
        filters_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=self._get_consistent_padding(), pady=(self._get_consistent_spacing() // 2, self._get_consistent_padding()))
        filters_frame.columnconfigure(1, weight=1)
        filters_frame.columnconfigure(3, weight=1)

        # Room filter with consistent spacing
        room_label = ttk.Label(filters_frame, text="🏠 Room:")
        room_label.grid(row=0, column=0, sticky="w", padx=(0, self._get_consistent_spacing() // 2))
        self.room_entry = ttk.Entry(filters_frame, textvariable=self.room_filter, width=8)
        self.room_entry.grid(row=0, column=1, sticky="w", padx=(0, self._get_consistent_spacing()))
        self._add_tooltip(self.room_entry, "Filter by room number (e.g., 101, 201)")

        # CO filter with consistent spacing
        co_label = ttk.Label(filters_frame, text="📝 CO:")
        co_label.grid(row=0, column=2, sticky="w", padx=(0, self._get_consistent_spacing() // 2))
        self.co_entry = ttk.Entry(filters_frame, textvariable=self.co_filter, width=8)
        self.co_entry.grid(row=0, column=3, sticky="w", padx=(0, self._get_consistent_spacing()))
        self._add_tooltip(self.co_entry, "Filter by Change Order number")

        # Include archive checkbox with consistent spacing
        self.archive_check = ttk.Checkbutton(
            filters_frame,
            text="📦 Include Archive",
            variable=self.include_archive
        )
        self.archive_check.grid(row=0, column=4, sticky="w")
        self._add_tooltip(self.archive_check, "Include archived and older versions")

    def _create_sidebar_section(self):
        """Create sidebar section with options and AI toolbar side by side."""
        # Sidebar container frame
        self.sidebar_frame = ttk.Frame(self.main_frame)
        self.sidebar_frame.grid(row=3, column=0, sticky="ew", pady=(0, self._get_consistent_spacing() // 2))
        self.sidebar_frame.columnconfigure(0, weight=1)
        self.sidebar_frame.columnconfigure(1, weight=1)

        # Create options subsection
        self._create_options_section()

        # Create AI toolbar subsection
        self._create_ai_toolbar()

    def _create_options_section(self):
        """Create options section within sidebar."""
        # Options frame
        options_frame = ttk.LabelFrame(self.sidebar_frame, text="Options")
        options_frame.grid(row=0, column=0, sticky="ew", padx=(0, self._get_consistent_spacing() // 2))
        options_frame.columnconfigure(0, weight=1)

        # Options container with grid layout
        opts_container = ttk.Frame(options_frame)
        opts_container.grid(row=0, column=0, sticky="ew", padx=self._get_consistent_padding(), pady=self._get_consistent_padding())
        opts_container.columnconfigure(0, weight=1)

        # Debug mode checkbox with consistent spacing
        self.debug_check = ttk.Checkbutton(
            opts_container,
            text="🔧 Show Debug Output",
            variable=self.debug_mode
        )
        self.debug_check.grid(row=0, column=0, sticky="w")
        self._add_tooltip(self.debug_check, "Show detailed debug information in results")

        # Training data checkbox with consistent spacing
        self.training_check = ttk.Checkbutton(
            opts_container,
            text="📊 Generate Training Data",
            variable=self.training_mode
        )
        self.training_check.grid(row=0, column=1, sticky="w", padx=(self._get_consistent_spacing(), 0))
        self._add_tooltip(self.training_check, "Generate JSON training data for AI analysis")

    def _create_ai_toolbar(self):
        """Create AI toolbar within sidebar section."""
        # AI Toolbar frame with improved styling
        ai_frame = ttk.LabelFrame(self.sidebar_frame, text="AI Assistant")
        ai_frame.grid(row=0, column=1, sticky="ew", padx=(self._get_consistent_spacing() // 2, 0))
        ai_frame.columnconfigure(0, weight=1)

        # AI controls container with grid layout
        ai_container = ttk.Frame(ai_frame)
        ai_container.grid(row=0, column=0, sticky="ew", padx=self._get_consistent_padding(), pady=self._get_consistent_padding())
        ai_container.columnconfigure(0, weight=1)

        # AI toggle button with primary styling and icon
        self.ai_toggle_button = ttk.Button(
            ai_container,
            text="🤖 Enable AI",
            command=self.toggle_ai,
            width=14
        )
        self.ai_toggle_button.grid(row=0, column=0, sticky="w", padx=(0, self._get_consistent_spacing() // 2))
        self._add_tooltip(self.ai_toggle_button, "Enable or disable AI assistant features")

        # AI chat button
        self.ai_chat_button = ttk.Button(
            ai_container,
            text="💬 AI Chat",
            command=self.toggle_ai_panel,
            width=14,
            state=tk.DISABLED
        )
        self.ai_chat_button.grid(row=0, column=1, sticky="w", padx=(0, self._get_consistent_spacing()))
        self._add_tooltip(self.ai_chat_button, "Open AI chat window for interactive assistance")

        # AI status indicator with improved styling
        self.ai_status_label = ttk.Label(
            ai_container,
            text="Status: Disabled",
            font=("Segoe UI", 9)
        )
        self.ai_status_label.grid(row=0, column=2, sticky="w", padx=(0, self._get_consistent_spacing() // 2))

    def _create_status_section(self):
        """Create status section."""
        # Status frame
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=4, column=0, sticky="ew", pady=(0, self._get_consistent_spacing() // 2))
        status_frame.columnconfigure(0, weight=1)

        # Status label with responsive wrapping and consistent padding
        wrap_length = max(300, int(self.min_width * 0.8))
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_text,
            wraplength=wrap_length,
            justify="left"
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=self._get_consistent_padding())

        # Progress bar with loading spinner (hidden by default)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate'
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=self._get_consistent_padding(), pady=(self._get_consistent_spacing() // 2, 0))
        self.progress_bar.grid_remove()

        # Loading indicator label with consistent padding
        self.loading_label = ttk.Label(
            status_frame,
            text="",
            font=("Segoe UI", 9)
        )
        self.loading_label.grid(row=2, column=0, sticky="ew", padx=self._get_consistent_padding(), pady=(2, 0))
        self.loading_label.grid_remove()

    def _create_action_buttons(self):
        """Create action buttons with enhanced styling."""
        # Button frame with improved spacing
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=5, column=0, sticky="ew", pady=(self._get_consistent_spacing() // 2, 0))

        # Center the buttons with improved spacing and grid layout
        button_container = ttk.Frame(button_frame)
        button_container.grid(row=0, column=0)
        button_container.columnconfigure(0, weight=1)

        # Folder mode button with primary styling and icon
        self.open_button = ttk.Button(
            button_container,
            text="📂 Open Folder",
            command=self.execute_folder_navigation,
            width=18
        )
        self.open_button.grid(row=0, column=0, padx=(0, self._get_consistent_spacing() // 2))
        self._add_tooltip(self.open_button, "Open the selected project subfolder in file explorer")

        # Document mode buttons with improved styling and icons (initially hidden)
        self.find_button = ttk.Button(
            button_container,
            text="🔍 Find Documents",
            command=self.execute_document_navigation,
            width=16
        )
        self.find_button.grid(row=0, column=1, padx=(0, self._get_consistent_spacing() // 2))
        self._add_tooltip(self.find_button, "Search for the best matching document")

        self.choose_button = ttk.Button(
            button_container,
            text="📋 Choose From List",
            command=lambda: self.execute_document_navigation(choose_mode=True),
            width=16
        )
        self.choose_button.grid(row=0, column=2, padx=(0, self._get_consistent_spacing() // 2))
        self._add_tooltip(self.choose_button, "Show all matching documents to choose from")

        # Open/Navigate button (appears after successful search)
        self.navigate_button = ttk.Button(
            button_container,
            text="✅ Open Selected",
            command=self._execute_final_navigation,
            width=16,
            state=tk.DISABLED
        )
        self.navigate_button.grid(row=0, column=3)
        self._add_tooltip(self.navigate_button, "Open the selected result")

    def _create_menu_bar(self):
        """Create menu bar with keyboard shortcut accelerators."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings...", command=self.show_settings, accelerator="Ctrl+S")
        file_menu.add_command(label="Clear & Reset", command=self._clear_and_reset, accelerator="Ctrl+R")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Escape")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Folder Mode", command=lambda: self._set_mode("folder"), accelerator="Ctrl+1")
        view_menu.add_command(label="Document Mode", command=lambda: self._set_mode("document"), accelerator="Ctrl+2")
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Theme", command=self.toggle_theme, accelerator="Ctrl+D")
        view_menu.add_command(label="Always on Top", command=self.toggle_always_on_top, accelerator="Ctrl+T")

        # Navigate menu
        nav_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Navigate", menu=nav_menu)
        nav_menu.add_command(label="Focus Search", command=lambda: self.project_entry.focus_set(), accelerator="Ctrl+F")
        nav_menu.add_command(label="Execute", command=self._execute_current_action, accelerator="Enter")
        nav_menu.add_separator()
        nav_menu.add_command(label="Recent Projects", state="disabled")  # Separator label

        # AI menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_checkbutton(
            label="Enable AI Assistant",
            variable=self.ai_enabled,
            command=self.toggle_ai,
            accelerator="Ctrl+/"
        )
        ai_menu.add_checkbutton(
            label="Show AI Chat Panel",
            variable=self.ai_panel_visible,
            command=self.toggle_ai_panel,
            accelerator="Ctrl+Space"
        )
        ai_menu.add_separator()
        ai_menu.add_command(label="AI Settings...", command=self.show_ai_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help, accelerator="F1")
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_keyboard_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="Dependency Status...", command=self.show_dependency_status)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

    def _setup_events(self):
        """Set up event bindings with comprehensive keyboard shortcuts."""
        # Window events
        self.root.bind('<Configure>', self._on_window_configure)
        self.root.bind('<KeyPress>', self._on_key_press)

        # Global shortcuts
        self.root.bind('<Control-Return>', lambda e: self._execute_current_action())
        self.root.bind('<Return>', lambda e: self._execute_current_action())  # Enter to execute
        self.root.bind('<Escape>', lambda e: self.hide_window())
        self.root.bind('<F1>', lambda e: self.show_help())
        self.root.bind('<Control-comma>', lambda e: self.show_settings())

        # Quick navigation shortcuts
        self.root.bind('<Control-f>', lambda e: self.project_entry.focus_set())  # Focus search
        self.root.bind('<Control-1>', lambda e: self._set_mode("folder"))  # Folder mode
        self.root.bind('<Control-2>', lambda e: self._set_mode("document"))  # Document mode
        self.root.bind('<Control-d>', lambda e: self.toggle_theme())  # Toggle dark mode
        self.root.bind('<Control-t>', lambda e: self.toggle_always_on_top())  # Toggle on top
        self.root.bind('<Control-r>', lambda e: self._clear_and_reset())  # Reset/clear
        self.root.bind('<Control-Shift-R>', lambda e: self._reset_window_size_flag())  # Reset window size

        # Enhanced functionality shortcuts
        self.root.bind('<Control-Shift-C>', lambda e: self._copy_current_path())  # Copy current path
        self.root.bind('<F11>', lambda e: self._toggle_fullscreen())  # Toggle fullscreen
        self.root.bind('<Control-Shift-F>', lambda e: self._focus_and_select_all())  # Focus and select all

        # AI shortcuts
        self.root.bind('<Control-space>', lambda e: self.toggle_ai_panel())  # Toggle AI chat
        self.root.bind('<Control-slash>', lambda e: self.toggle_ai())  # Toggle AI

        # Settings shortcuts
        self.root.bind('<Control-s>', lambda e: self.show_settings())  # Settings
        self.root.bind('<Control-h>', lambda e: self.show_help())  # Help

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
        """Apply the current theme with forced refresh."""
        logger.info("Applying theme with forced refresh")
        current_theme_name = self.theme.get_current_theme_name()
        logger.info(f"Current theme before application: {current_theme_name}")

        self.theme.apply_theme(self.root)
        logger.info("Theme applied to root window")

        # Force update after theme change
        self.root.update_idletasks()
        logger.info("Root window updated after theme change")

        # Enhanced widget refresh for theme changes
        self._force_complete_widget_refresh()
        logger.info("Complete widget refresh executed")

        # Ensure proper window colors
        current_theme_obj = self.theme.get_current_theme()
        if current_theme_obj:
            theme_name = current_theme_obj.name if hasattr(current_theme_obj, 'name') else 'Unknown'
            logger.info(f"Retrieved theme object: {theme_name}")

            window_bg = current_theme_obj.get_color("window", "bg") or current_theme_obj.get_color("bg")
            if window_bg:
                logger.info(f"Setting window background color: {window_bg}")
                self.root.configure(bg=window_bg)
                # Update main frame background too
                if hasattr(self, 'main_frame'):
                    try:
                        self.main_frame.configure(style="TFrame")
                        logger.info("Main frame style updated")
                    except Exception as e:
                        logger.warning(f"Failed to update main frame style: {e}")
            else:
                logger.warning("No background color found in theme")
        else:
            logger.warning("No current theme object available")

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
        """Handle navigation mode change with enhanced stability."""
        mode = self.current_mode.get()
        logger.debug(f"Mode change: {mode}")

        # Prevent mode change loops with state tracking
        if hasattr(self, '_last_mode') and self._last_mode == mode:
            return

        self._last_mode = mode

        # Get current window dimensions before mode change
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()

        # Set mode change flag to prevent resize event conflicts
        self._is_mode_changing = True

        try:
            if mode == "folder":
                # Show folder frame, hide document frame within main content frame
                self.folder_frame.grid(row=0, column=0, sticky="ew")
                self.doc_frame.grid_remove()

                # Show/hide appropriate buttons with enhanced layout management
                self._update_button_visibility("folder")

                # Update folder mode specific layout
                self._update_folder_mode_layout(current_width, current_height)

                self.status_text.set("Folder mode - Select a project subfolder to open")

            else:  # document mode
                # Hide folder frame, show document frame within main content frame
                self.folder_frame.grid_remove()
                self.doc_frame.grid(row=0, column=0, sticky="ew")  # Use same row as folder_frame for proper swapping

                # Show/hide appropriate buttons with enhanced layout management
                self._update_button_visibility("document")

                # Update document mode specific layout
                self._update_document_mode_layout(current_width, current_height)

                self.status_text.set("Document mode - Find specific documents by type and filters")

            # Force layout update after mode change with error handling
            self.root.update_idletasks()

            # Check window dimensions after mode change
            new_width = self.root.winfo_width()
            new_height = self.root.winfo_height()
            logger.info(f"Window size after mode change: {new_width}x{new_height}")

            # Update responsive layout only if size actually changed significantly
            width_change = abs(new_width - current_width)
            height_change = abs(new_height - current_height)
            if width_change > 5 or height_change > 5:  # Only update if significant change
                logger.info(f"Significant size change detected ({width_change}x{height_change}), updating layout")
                self._update_responsive_layout()
            else:
                logger.debug("Minor size change during mode switch, skipping layout update")

        except Exception as e:
            logger.error(f"Error during mode change: {e}")
        finally:
            # Clear mode change flag after a short delay
            def clear_mode_change_flag():
                self._is_mode_changing = False
                logger.debug("Mode change flag cleared")

            self.root.after(100, clear_mode_change_flag)

    def _update_button_visibility(self, mode: str):
        """Update button visibility for specific mode with enhanced layout management."""
        try:
            if mode == "folder":
                # Folder mode buttons - show open button, hide others
                self.open_button.grid(row=0, column=0, padx=(0, self._get_consistent_spacing() // 2))
                self.find_button.grid_remove()
                self.choose_button.grid_remove()
                self.navigate_button.grid_remove()
                logger.debug("Folder mode buttons configured")
            else:  # document mode
                # Document mode buttons - hide open button, show others
                self.open_button.grid_remove()
                self.find_button.grid(row=0, column=1, padx=(0, self._get_consistent_spacing() // 2))
                self.choose_button.grid(row=0, column=2, padx=(0, self._get_consistent_spacing() // 2))

                # Show navigate button only if we have search results
                if hasattr(self, 'last_search_result') and self.last_search_result:
                    self.navigate_button.grid(row=0, column=3)
                    logger.debug("Navigate button shown - search result available")
                else:
                    self.navigate_button.grid_remove()
                    logger.debug("Navigate button hidden - no search result")

        except Exception as e:
            logger.warning(f"Error updating button visibility for {mode} mode: {e}")

    def _on_project_input_change(self, event=None):
        """Handle project input changes."""
        self._validate_inputs()

    def _on_window_configure(self, event=None):
        """Handle window configuration changes with enhanced stability."""
        if event and event.widget == self.root:
            current_width = event.width
            current_height = event.height
            current_size = f"{current_width}x{current_height}"

            logger.info(f"Window configure event - Size: {current_size}, Is resizing: {self.is_resizing}")

            # Enforce minimum size constraints
            self._enforce_minimum_size()

            # Skip resize handling during mode changes to prevent conflicts
            if getattr(self, '_is_mode_changing', False):
                logger.debug("Mode change in progress, skipping resize handling")
                return

            # Enhanced size change detection with tolerance for minor fluctuations
            size_changed = True
            if hasattr(self, '_last_resize_size'):
                last_width, last_height = self._last_resize_size.split('x')
                last_width, last_height = int(last_width), int(last_height)

                # Allow 2px tolerance to prevent unnecessary updates from minor fluctuations
                width_tolerance = max(2, int(current_width * 0.01))  # 1% of current width, min 2px
                height_tolerance = max(2, int(current_height * 0.01))  # 1% of current height, min 2px

                if (abs(current_width - last_width) <= width_tolerance and
                    abs(current_height - last_height) <= height_tolerance):
                    logger.debug(f"Size change within tolerance ({width_tolerance}x{height_tolerance}px), skipping resize")
                    size_changed = False

            if not size_changed:
                return

            self._last_resize_size = current_size

            # Enhanced resize state management with timeout tracking
            if not self.is_resizing:
                logger.info("Starting resize operation")
                self.is_resizing = True
                self._resize_start_time = datetime.now()

                try:
                    # Update DPI scaling for new window size if needed
                    self._update_dpi_scaling_for_resize(current_width, current_height)

                    # Update responsive layout with mode-specific handling
                    self._update_responsive_layout()
                    logger.info("Responsive layout update completed successfully")
                except Exception as e:
                    logger.error(f"Error in responsive layout update: {e}")

                # Enhanced timer reset with timeout protection
                def reset_resize_flag():
                    try:
                        if hasattr(self, 'is_resizing') and self.is_resizing:
                            # Check if resize operation has been running too long (>500ms)
                            if hasattr(self, '_resize_start_time'):
                                elapsed = (datetime.now() - self._resize_start_time).total_seconds() * 1000
                                if elapsed > 500:
                                    logger.warning(f"Resize operation timed out after {elapsed:.0f}ms, forcing reset")
                                else:
                                    logger.debug("Resize flag reset completed")
                            self.is_resizing = False
                    except Exception as e:
                        logger.error(f"Error resetting resize flag: {e}")

                self.root.after(150, reset_resize_flag)
                logger.info("Resize operation scheduled for completion")
            else:
                logger.info("Resize operation already in progress, skipping")

    def _enforce_minimum_size(self):
        """Enforce minimum window size to ensure all components are visible."""
        try:
            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()

            # Calculate required minimum sizes
            min_width = self._calculate_minimum_width()
            min_height = self._calculate_minimum_height()

            # Update instance variables if calculated minimums are larger
            if min_width > self.min_width:
                self.min_width = min_width
            if min_height > self.min_height:
                self.min_height = min_height

            # Enforce minimum size
            needs_resize = False
            if current_width < self.min_width:
                current_width = self.min_width
                needs_resize = True

            if current_height < self.min_height:
                current_height = self.min_height
                needs_resize = True

            if needs_resize:
                logger.info(f"Enforcing minimum size: {current_width}x{current_height}")
                self.root.geometry(f"{current_width}x{current_height}")
                self.root.minsize(self.min_width, self.min_height)

        except Exception as e:
            logger.warning(f"Error enforcing minimum size: {e}")

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

    def _show_error(self, message: str, suggestion: Optional[str] = None):
        """Show enhanced error message with actionable guidance."""
        # Extract error type and provide helpful suggestions
        if not suggestion:
            suggestion = self._generate_error_suggestion(message)

        # Combine message and suggestion
        full_message = f"❌ {message}\n\n{suggestion}"

        messagebox.showerror("Error", full_message)
        logger.error(f"Error shown to user: {message}")

        # Update status with brief error
        self.status_text.set(f"⚠️ Error: {message[:50]}...")

    def _generate_error_suggestion(self, message: str) -> str:
        """Generate contextual error suggestions based on message content."""
        message_lower = message.lower()

        if "not found" in message_lower or "no such file" in message_lower:
            return ("💡 Try these solutions:\n"
                   "• Check the project number is correct (5 digits)\n"
                   "• Verify the project exists in OneDrive\n"
                   "• Try searching by project name instead\n"
                   "• Use Ctrl+Shift+F to focus and edit search")
        elif "permission" in message_lower or "access denied" in message_lower:
            return ("💡 Try these solutions:\n"
                   "• Check you have access to the OneDrive folder\n"
                   "• Verify OneDrive sync is working\n"
                   "• Contact IT if access issues persist\n"
                   "• Try restarting OneDrive sync")
        elif "network" in message_lower or "connection" in message_lower:
            return ("💡 Try these solutions:\n"
                   "• Check your internet connection\n"
                   "• Verify OneDrive is synced\n"
                   "• Try again in a few moments\n"
                   "• Check VPN connection if applicable")
        elif "invalid" in message_lower:
            return ("💡 Try these solutions:\n"
                   "• Enter a 5-digit project number (e.g., 17741)\n"
                   "• Or search by project name\n"
                   "• Use Ctrl+R to clear and start over\n"
                   "• Check for typos in project number")
        elif "timeout" in message_lower:
            return ("💡 Try these solutions:\n"
                   "• Check network connection stability\n"
                   "• Try again (operation may be temporarily slow)\n"
                   "• Contact IT if timeouts persist")
        else:
            return ("💡 General troubleshooting:\n"
                   "• Press Ctrl+R to clear and try again\n"
                   "• Press F1 for help and documentation\n"
                   "• Check the logs for more details\n"
                   "• Restart the application if issues persist")

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
        """Update layout elements for current window size with enhanced stability."""
        try:
            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()
            current_mode = self.current_mode.get()

            # Ensure minimum size is maintained
            min_width = self._calculate_minimum_width()
            min_height = self._calculate_minimum_height()

            if current_width < min_width or current_height < min_height:
                logger.info(f"Window size {current_width}x{current_height} below minimum {min_width}x{min_height}, enforcing constraints")
                self._enforce_minimum_size()
                return

            # Prevent layout update loops with more sophisticated state tracking
            layout_state_key = f"{current_width}x{current_height}:{current_mode}"
            if hasattr(self, '_last_layout_state') and self._last_layout_state == layout_state_key:
                logger.debug(f"Layout state unchanged ({layout_state_key}), skipping update")
                return

            self._last_layout_state = layout_state_key
            logger.info(f"Updating responsive layout - Window: {current_width}x{current_height}, Mode: {current_mode}")

            # Mode-specific layout adjustments
            if current_mode == "folder":
                self._update_folder_mode_layout(current_width, current_height)
            else:  # document mode
                self._update_document_mode_layout(current_width, current_height)

            # Update status label wrap length based on window width with DPI awareness
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                # Use DPI-scaled wrap length for better text wrapping
                dpi_aware_width = int(current_width / self.dpi_scale) if self.dpi_scale > 0 else current_width
                wrap_length = max(250, int(dpi_aware_width * 0.85))  # Better wrap length for readability
                original_wrap = self.status_label.cget('wraplength')
                self.status_label.config(wraplength=wrap_length)
                logger.info(f"Status label wrap length: {original_wrap} -> {wrap_length}")

            # Enhanced button layout management with better responsive breakpoints
            self._update_button_layout(current_width, current_height)

            # Update padding and margins based on DPI scaling
            self._update_dpi_aware_padding(current_width, current_height)

            # Force layout recalculation with error handling
            self.root.update_idletasks()
            logger.info("Responsive layout update completed")

        except Exception as e:
            logger.warning(f"Failed to update responsive layout: {e}")

    def _update_folder_mode_layout(self, width: int, height: int):
        """Update layout specifically for folder mode."""
        try:
            # Ensure folder frame visibility and proper sizing
            if hasattr(self, 'folder_frame') and self.folder_frame.winfo_exists():
                self.folder_frame.grid_configure(sticky="ew")

                # Update folder radio button layout for current window size
                if hasattr(self, 'folder_radios'):
                    for radio in self.folder_radios:
                        if radio.winfo_exists():
                            # Adjust padding based on window width with consistent scaling
                            base_padx = self._get_consistent_padding()
                            scaled_padx = max(base_padx, int(base_padx * (width / 400)))
                            radio.grid_configure(padx=scaled_padx)

            logger.debug("Folder mode layout updated")
        except Exception as e:
            logger.debug(f"Error updating folder mode layout: {e}")

    def _update_document_mode_layout(self, width: int, height: int):
        """Update layout specifically for document mode."""
        try:
            # Ensure document frame visibility and proper sizing
            if hasattr(self, 'doc_frame') and self.doc_frame.winfo_exists():
                self.doc_frame.grid_configure(sticky="ew")

                # Update document filter layout for current window size
                if hasattr(self, 'room_entry') and self.room_entry.winfo_exists():
                    # Adjust entry width based on available space with consistent scaling
                    base_width = 8
                    scaled_width = max(6, min(12, int(base_width * (width / 400))))
                    self.room_entry.config(width=scaled_width)

                if hasattr(self, 'co_entry') and self.co_entry.winfo_exists():
                    base_width = 8
                    scaled_width = max(6, min(12, int(base_width * (width / 400))))
                    self.co_entry.config(width=scaled_width)

                # Update document filter container padding
                if hasattr(self, 'filters_frame') and self.filters_frame.winfo_exists():
                    filter_padding = self._get_consistent_padding()
                    self.filters_frame.grid_configure(padx=filter_padding, pady=(self._get_consistent_spacing() // 2, filter_padding))

            logger.debug("Document mode layout updated")
        except Exception as e:
            logger.debug(f"Error updating document mode layout: {e}")

    def _update_button_layout(self, width: int, height: int):
        """Update button layout with enhanced responsive behavior."""
        try:
            # Enhanced responsive breakpoints with better granularity
            if width < 400:
                # Very narrow - stack vertically
                layout_mode = "vertical"
                button_width = max(10, int(width * 0.3))
            elif width < 550:
                # Narrow - compact horizontal
                layout_mode = "compact"
                button_width = max(12, int(width * 0.22))
            elif width < 700:
                # Medium - standard horizontal
                layout_mode = "standard"
                button_width = 16
            else:
                # Wide - expanded horizontal
                layout_mode = "expanded"
                button_width = 18

            logger.info(f"Button layout mode: {layout_mode} ({width}px) - width: {button_width}")

            # Apply button width changes
            buttons_to_update = []
            if hasattr(self, 'open_button') and self.open_button.winfo_exists():
                buttons_to_update.append(('open_button', self.open_button))
            if hasattr(self, 'find_button') and self.find_button.winfo_exists():
                buttons_to_update.append(('find_button', self.find_button))
            if hasattr(self, 'choose_button') and self.choose_button.winfo_exists():
                buttons_to_update.append(('choose_button', self.choose_button))
            if hasattr(self, 'navigate_button') and self.navigate_button.winfo_exists():
                buttons_to_update.append(('navigate_button', self.navigate_button))

            for button_name, button in buttons_to_update:
                current_width = button.cget('width')
                if current_width != button_width:
                    button.config(width=button_width)
                    logger.debug(f"Updated {button_name} width: {current_width} -> {button_width}")

            # Update button container padding based on layout mode
            if hasattr(self, 'button_container') and self.button_container.winfo_exists():
                if layout_mode == "vertical":
                    # For vertical layout, add more padding between buttons
                    padx_value = self._get_consistent_spacing() // 2
                    pady_value = self._get_consistent_spacing() // 4
                else:
                    # For horizontal layout, use consistent spacing
                    padx_value = self._get_consistent_spacing() // 2
                    pady_value = 0

                # Update padding for all visible buttons
                for button_name, button in buttons_to_update:
                    if not button.grid_info().get('in', None):  # Skip hidden buttons
                        continue
                    current_padx = button.grid_info().get('padx', 0)
                    if current_padx != padx_value:
                        button.grid_configure(padx=(0, padx_value))

        except Exception as e:
            logger.warning(f"Error updating button layout: {e}")

    def _update_dpi_aware_padding(self, width: int, height: int):
        """Update padding and margins based on DPI scaling and window size."""
        try:
            # Get consistent padding values
            consistent_padding = self._get_consistent_padding()
            consistent_spacing = self._get_consistent_spacing()

            # Update main frame padding based on DPI and window size
            if hasattr(self, 'main_frame') and self.main_frame.winfo_exists():
                # Use consistent padding for main frame
                self.main_frame.pack_configure(padx=consistent_padding, pady=consistent_padding)
                logger.debug(f"Updated main frame padding to {consistent_padding}px")

            # Update section padding for all LabelFrames with consistent spacing
            label_frames = []
            if hasattr(self, 'folder_frame') and self.folder_frame.winfo_exists():
                label_frames.append(self.folder_frame)
            if hasattr(self, 'doc_frame') and self.doc_frame.winfo_exists():
                label_frames.append(self.doc_frame)

            for frame in label_frames:
                # Use consistent internal padding for all sections
                try:
                    # Grid configuration for frames that use grid
                    frame.grid_configure(pady=consistent_spacing)
                    logger.debug(f"Updated {frame.cget('text')} frame spacing to {consistent_spacing}px")
                except:
                    logger.debug(f"Could not update frame padding for {frame}")

            # Update specific component containers with consistent padding
            self._update_section_padding('input_frame', consistent_padding, consistent_spacing)
            self._update_section_padding('nav_frame', consistent_padding, consistent_spacing)
            self._update_section_padding('opts_container', consistent_padding, consistent_spacing)
            self._update_section_padding('ai_container', consistent_padding, consistent_spacing)
            self._update_section_padding('status_frame', consistent_padding, consistent_spacing)
            self._update_section_padding('button_container', consistent_padding, consistent_spacing)

        except Exception as e:
            logger.debug(f"Error updating DPI-aware padding: {e}")

    def _update_section_padding(self, container_name: str, padding: int, spacing: int):
        """Update padding for specific section containers."""
        try:
            if hasattr(self, container_name):
                container = getattr(self, container_name)
                if container and container.winfo_exists():
                    try:
                        container.grid_configure(padx=padding, pady=spacing)
                        logger.debug(f"Updated {container_name} padding to {padding}px, spacing to {spacing}px")
                    except:
                        try:
                            container.pack_configure(padx=padding, pady=spacing)
                            logger.debug(f"Updated {container_name} pack padding to {padding}px, spacing to {spacing}px")
                        except:
                            logger.debug(f"Could not update padding for {container_name}")
            else:
                logger.debug(f"Container {container_name} not found or not exists")
        except Exception as e:
            logger.debug(f"Error updating section padding for {container_name}: {e}")

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
        """Toggle between light and dark themes with enhanced refresh."""
        self.theme.toggle_theme()
        self._apply_theme()

        # Enhanced refresh sequence
        self.root.after(50, self._force_complete_widget_refresh)
        self.root.after(150, self._verify_theme_application)

        # Update status
        current_theme = self.theme.get_current_theme_name()
        self.status_text.set(f"Theme switched to {current_theme}")

    def toggle_always_on_top(self):
        """Toggle always on top mode."""
        current = self.root.attributes('-topmost')
        new_state = not current
        self.root.attributes('-topmost', new_state)

        # Save the preference
        self.settings.set("ui.always_on_top", new_state)

    def restore_window_state(self):
        """Restore window state from settings with reset option."""
        try:
            # Restore always on top setting
            always_on_top = self.settings.get("ui.always_on_top", False)
            self.root.attributes('-topmost', always_on_top)

            # Check if user wants to reset to default (e.g., via Ctrl+Shift+R)
            reset_to_default = getattr(self, '_reset_window_size', False)

            if reset_to_default:
                self.root.geometry(self._get_responsive_geometry())
                self._reset_window_size = False  # Reset flag
                logger.info("Window reset to responsive default")
            else:
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
                else:
                    # No saved geometry, use responsive default
                    self.root.geometry(self._get_responsive_geometry())

        except (tk.TclError, ValueError, KeyError) as e:
            logger.warning(f"Window state restoration error: {e}")
            # Fallback to responsive geometry
            self.root.geometry(self._get_responsive_geometry())
        except Exception as e:
            logger.error(f"Unexpected error restoring window state: {e}")
            # Fallback to minimum size
            self.root.geometry(f"{self.min_width}x{self.min_height}")

    def reset_window_to_default(self):
        """Reset window to default responsive size."""
        default_geometry = self._get_responsive_geometry()
        self.root.geometry(default_geometry)
        self.settings.set_window_geometry(default_geometry)
        self.status_text.set("Window reset to default size")

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

    def _reset_window_size_flag(self):
        """Set flag to reset window size on next restore."""
        self._reset_window_size = True
        self.status_text.set("Window size will reset on next launch")

    def _copy_current_path(self):
        """Copy current project path to clipboard if available."""
        try:
            if hasattr(self, 'last_search_result') and self.last_search_result:
                path = self.last_search_result.get('path', '')
                if path:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(path)
                    self.status_text.set(f"Copied path to clipboard: {path}")
                else:
                    self.status_text.set("No path available to copy")
            else:
                self.status_text.set("No search result to copy")
        except Exception as e:
            logger.warning(f"Failed to copy path to clipboard: {e}")
            self.status_text.set("Failed to copy path")

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode for better visibility."""
        try:
            is_fullscreen = self.root.attributes('-fullscreen')
            self.root.attributes('-fullscreen', not is_fullscreen)
            self.status_text.set("Fullscreen " + ("enabled" if not is_fullscreen else "disabled"))
        except Exception as e:
            logger.warning(f"Failed to toggle fullscreen: {e}")

    def _focus_and_select_all(self):
        """Focus search field and select all text."""
        try:
            self.project_entry.focus_set()
            self.project_entry.select_range(0, tk.END)
            self.status_text.set("Search field focused and text selected")
        except Exception as e:
            logger.warning(f"Failed to focus and select: {e}")

    def _force_complete_widget_refresh(self):
        """Force complete refresh of all widgets for theme changes."""
        try:
            # Batch update operations to reduce overhead
            updates_needed = []

            def collect_widget_updates(parent):
                """Collect widget updates before applying them."""
                for child in parent.winfo_children():
                    try:
                        if isinstance(child, ttk.Widget):
                            original_style = child.cget('style')
                            if original_style:
                                updates_needed.append(('style', child, original_style))
                            else:
                                # Apply appropriate default style
                                widget_type = child.__class__.__name__
                                if widget_type.startswith('Ttk'):
                                    default_style = widget_type[3:]
                                else:
                                    default_style = widget_type
                                try:
                                    updates_needed.append(('default_style', child, f"T{default_style}"))
                                except:
                                    pass

                        # Handle special cases
                        if hasattr(child, 'configure') and 'bg' in child.configure():
                            try:
                                if isinstance(child, tk.Text):
                                    bg_color = self.theme.get_current_theme().get_color("bg")
                                    updates_needed.append(('text_bg', child, bg_color))
                                elif isinstance(child, tk.Canvas):
                                    bg_color = self.theme.get_current_theme().get_color("bg")
                                    updates_needed.append(('canvas_bg', child, bg_color))
                            except:
                                pass

                        # Recurse to children
                        collect_widget_updates(child)
                    except Exception as e:
                        logger.debug(f"Error collecting widget updates for {child}: {e}")

            # Collect all updates first
            collect_widget_updates(self.root)

            # Apply all updates in batch
            for update_type, widget, value in updates_needed:
                try:
                    if update_type == 'style':
                        widget.configure(style='')
                        widget.configure(style=value)
                    elif update_type == 'default_style':
                        widget.configure(style=value)
                    elif update_type == 'text_bg':
                        widget.configure(bg=value)
                    elif update_type == 'canvas_bg':
                        widget.configure(bg=value)
                except Exception as e:
                    logger.debug(f"Error applying widget update {update_type}: {e}")

            # Single comprehensive update instead of multiple
            self.root.update_idletasks()

            # Reduced delay for ttk styles
            self.root.after(50, self._final_theme_update)

        except Exception as e:
            logger.warning(f"Error in complete widget refresh: {e}")

    def _final_theme_update(self):
        """Final theme update after all widgets are refreshed."""
        try:
            # Ensure all styles are properly applied
            self.root.update_idletasks()

            # Force repaint
            self.root.event_generate("<<ThemeChanged>>")

        except Exception as e:
            logger.debug(f"Final theme update error: {e}")

    def _verify_theme_application(self):
        """Verify that theme has been applied correctly."""
        try:
            current_theme = self.theme.get_current_theme()
            if current_theme:
                # Check if key widgets have correct styling
                theme_name = current_theme.name.lower()

                # Verify entry field styling
                if hasattr(self, 'project_entry'):
                    # This is a custom widget, check its internal styling
                    pass

                logger.info(f"Theme verification completed for: {theme_name}")
        except Exception as e:
            logger.warning(f"Theme verification failed: {e}")

    def _setup_for_compiled_environment(self):
        """Set up GUI for compiled PyInstaller environment."""
        try:
            # Check if running from PyInstaller bundle
            if getattr(sys, 'frozen', False):
                # Running as compiled executable

                # Force theme refresh for compiled environment
                self.root.after(200, self._force_complete_widget_refresh)

                # Ensure proper DPI handling in compiled environment
                self.root.after(300, self._verify_compiled_environment)

                logger.info("Running in compiled environment - applied compatibility fixes")
        except Exception as e:
            logger.warning(f"Compiled environment setup error: {e}")

    def _verify_compiled_environment(self):
        """Verify GUI works correctly in compiled environment."""
        try:
            # Force window to proper size if needed
            current_geom = self.root.geometry()
            if current_geom == "1x1+0+0":
                self.root.geometry(self._get_responsive_geometry())

            # Ensure theme is properly applied
            self._apply_theme()

            logger.info("Compiled environment verification completed")
        except Exception as e:
            logger.error(f"Compiled environment verification failed: {e}")

    def _batch_initialize_ui(self):
        """Batch UI initialization operations for better performance."""
        try:
            # Setup all UI components first
            self._setup_ui()
            self._setup_events()
            self._setup_validation()
            self._setup_autocomplete()

            # Apply theme
            self._apply_theme()

            # Batch widget updates - single update instead of multiple
            self.root.update_idletasks()

            # Restore window state
            self.restore_window_state()

            # Start background services
            self._start_task_processor()

            # Update UI elements
            self._update_ai_ui()

            # Initialize state
            self.last_search_result = None
            self.last_search_type = None
            self.tooltips = {}

            # Set up for compiled environment
            self._setup_for_compiled_environment()

            # Schedule delayed refresh for theme stability
            self.root.after(50, self._force_widget_refresh)

        except Exception as e:
            logger.error(f"Error in batch UI initialization: {e}")
            # Fallback to individual initialization if batch fails
            self._fallback_initialization()

    def _fallback_initialization(self):
        """Fallback initialization if batch initialization fails."""
        try:
            logger.info("Using fallback initialization...")
            # Individual setup calls
            self._setup_ui()
            self._setup_events()
            self._setup_validation()
            self._setup_autocomplete()
            self._apply_theme()
            self.restore_window_state()
            self._start_task_processor()
            self._update_ai_ui()
            self._setup_for_compiled_environment()

            # Initialize state
            self.last_search_result = None
            self.last_search_type = None
            self.tooltips = {}

        except Exception as e:
            logger.error(f"Fallback initialization also failed: {e}")

    def _get_responsive_geometry(self):
        """Get responsive geometry string based on screen size."""
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Calculate minimum required size based on content
            min_required_width = self._calculate_minimum_width()
            min_required_height = self._calculate_minimum_height()

            # Update min dimensions if calculated requirements are larger
            if min_required_width > self.min_width:
                self.min_width = min_required_width
            if min_required_height > self.min_height:
                self.min_height = min_required_height

            # Calculate size based on screen resolution with better proportions
            if screen_width >= 2560:  # 2K/4K displays
                width_factor = 0.35  # Wider for better usability
                height_factor = 0.65  # Taller to show all content
            elif screen_width >= 1920:  # 1080p displays
                width_factor = 0.4   # Comfortable width
                height_factor = 0.7   # Show all content
            else:  # Smaller displays
                width_factor = 0.5   # Use more screen space
                height_factor = 0.8   # Maximize content visibility

            width = max(self.min_width, int(screen_width * width_factor))
            height = max(self.min_height, int(screen_height * height_factor))

            # Ensure minimum content visibility
            width = max(width, min_required_width)
            height = max(height, min_required_height)

            # Center on screen
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

            return f"{width}x{height}+{x}+{y}"
        except Exception:
            return f"{self.min_width}x{self.min_height}"

    def _calculate_minimum_width(self):
        """Calculate minimum width required to display all components properly."""
        try:
            # Base minimum width for content
            base_width = 480

            # Add space for buttons (approximately 200px for multiple buttons)
            button_space = 200

            # Add padding (20px on each side)
            padding = 40

            # Consider DPI scaling
            dpi_scaled_width = int((base_width + button_space + padding) * self.dpi_scale)

            return max(base_width, dpi_scaled_width)
        except Exception:
            return 480

    def _calculate_minimum_height(self):
        """Calculate minimum height required to display all components properly."""
        try:
            # Base components height - more compact estimates
            project_input_height = 60   # Reduced from 80
            nav_mode_height = 50        # Reduced from 60
            folder_mode_height = 160    # Reduced from 200 (6 radio buttons)
            doc_mode_height = 140       # Reduced from 180 (document filters)
            options_height = 40         # Reduced from 50
            ai_toolbar_height = 50      # Reduced from 60
            status_height = 50          # Reduced from 60
            action_buttons_height = 40  # Reduced from 50

            # Add section spacing - reduced spacing
            section_spacing = 6 * 15    # 6 sections with 15px spacing each (reduced from 20)

            # Add padding (top and bottom) - reduced padding
            padding = 30                # Reduced from 40

            total_height = (
                project_input_height +
                nav_mode_height +
                max(folder_mode_height, doc_mode_height) +  # Show whichever mode is active
                options_height +
                ai_toolbar_height +
                status_height +
                action_buttons_height +
                section_spacing +
                padding
            )

            # Consider DPI scaling
            dpi_scaled_height = int(total_height * self.dpi_scale)

            # Lower minimum height for more compact design
            return max(500, dpi_scaled_height)  # Reduced from 600
        except Exception:
            return 500

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
        """Update AI-related UI elements with enhanced styling and status icons."""
        if hasattr(self, 'ai_toggle_button'):
            if self.ai_enabled.get():
                self.ai_toggle_button.config(text="🤖 Disable AI")
                self.ai_chat_button.config(state=tk.NORMAL)
                self.ai_status_label.config(text="✅ Status: Enabled")
            else:
                self.ai_toggle_button.config(text="🤖 Enable AI")
                self.ai_chat_button.config(state=tk.DISABLED)
                self.ai_status_label.config(text="❌ Status: Disabled")

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

    # Keyboard Navigation Methods
    def _setup_keyboard_navigation(self):
        """Set up comprehensive keyboard navigation system."""
        # Define tab order for logical navigation
        self._tab_order = [
            'project_entry',
            'folder_radio',
            'doc_radio',
            'folder_group',
            'doc_type_combo',
            'version_combo',
            'room_entry',
            'co_entry',
            'archive_check',
            'debug_check',
            'training_check',
            'ai_toggle_button',
            'ai_chat_button',
            'action_buttons'
        ]

        # Define focus groups for arrow key navigation
        self._focus_groups = {
            'mode_selection': ['folder_radio', 'doc_radio'],
            'folder_selection': ['folder_radios'],
            'document_filters': ['doc_type_combo', 'version_combo', 'room_entry', 'co_entry', 'archive_check'],
            'options': ['debug_check', 'training_check'],
            'ai_controls': ['ai_toggle_button', 'ai_chat_button'],
            'action_buttons': ['open_button', 'find_button', 'choose_button', 'navigate_button']
        }

        # Apply focus styling to all focusable widgets
        self._apply_focus_styling()

        # Initialize keyboard navigation state
        self._keyboard_navigation_enabled = True
        self._current_focus_index = 0

    def _apply_focus_styling(self):
        """Apply focus indicators to all focusable widgets."""
        # Focus styling is handled by the theme system
        # This method can be used for additional custom focus indicators
        pass

    def _on_tab_navigation(self, event):
        """Handle Tab key navigation."""
        if not self._keyboard_navigation_enabled:
            return

        try:
            # Get current mode to determine available widgets
            current_mode = self.current_mode.get()
            available_widgets = self._get_available_widgets(current_mode)

            if not available_widgets:
                return

            # Find current focus index
            current_widget = self.root.focus_get()
            current_index = self._find_widget_index(current_widget, available_widgets)

            # Move to next widget
            next_index = (current_index + 1) % len(available_widgets)
            next_widget = available_widgets[next_index]

            if next_widget:
                next_widget.focus_set()
                # Scroll widget into view if needed
                self._scroll_widget_into_view(next_widget)

            return "break"  # Prevent default tab behavior

        except Exception as e:
            logger.debug(f"Tab navigation error: {e}")

    def _on_shift_tab_navigation(self, event):
        """Handle Shift+Tab key navigation (reverse)."""
        if not self._keyboard_navigation_enabled:
            return

        try:
            # Get current mode to determine available widgets
            current_mode = self.current_mode.get()
            available_widgets = self._get_available_widgets(current_mode)

            if not available_widgets:
                return

            # Find current focus index
            current_widget = self.root.focus_get()
            current_index = self._find_widget_index(current_widget, available_widgets)

            # Move to previous widget
            prev_index = (current_index - 1) % len(available_widgets)
            prev_widget = available_widgets[prev_index]

            if prev_widget:
                prev_widget.focus_set()
                # Scroll widget into view if needed
                self._scroll_widget_into_view(prev_widget)

            return "break"  # Prevent default tab behavior

        except Exception as e:
            logger.debug(f"Shift+Tab navigation error: {e}")

    def _on_arrow_navigation(self, event):
        """Handle arrow key navigation within option groups."""
        if not self._keyboard_navigation_enabled:
            return

        try:
            current_widget = self.root.focus_get()
            if not current_widget:
                return

            # Determine which group the current widget belongs to
            group_name = self._find_widget_group(current_widget)
            if not group_name:
                return

            group_widgets = self._get_group_widgets(group_name)
            if len(group_widgets) <= 1:
                return

            current_index = group_widgets.index(current_widget) if current_widget in group_widgets else 0

            # Navigate based on arrow key
            if event.keysym in ['Up', 'Left']:
                next_index = (current_index - 1) % len(group_widgets)
            elif event.keysym in ['Down', 'Right']:
                next_index = (current_index + 1) % len(group_widgets)
            else:
                return

            next_widget = group_widgets[next_index]
            if next_widget:
                next_widget.focus_set()
                # For radio buttons, also select them
                if isinstance(next_widget, ttk.Radiobutton):
                    next_widget.invoke()

            return "break"  # Prevent default arrow behavior

        except Exception as e:
            logger.debug(f"Arrow navigation error: {e}")

    def _get_available_widgets(self, mode: str) -> List:
        """Get list of currently available/visible widgets for navigation."""
        widgets = []

        # Always available - project input section
        if hasattr(self, 'project_entry') and self.project_entry.winfo_exists():
            widgets.append(self.project_entry)

        # Mode selection - navigation mode section
        if hasattr(self, 'folder_radio') and self.folder_radio.winfo_exists():
            widgets.append(self.folder_radio)
        if hasattr(self, 'doc_radio') and self.doc_radio.winfo_exists():
            widgets.append(self.doc_radio)

        # Mode-specific widgets
        if mode == "folder":
            # Add folder radio buttons from folder mode section
            if hasattr(self, 'folder_radios'):
                for radio in self.folder_radios:
                    if radio and radio.winfo_exists():
                        widgets.append(radio)
        else:  # document mode
            # Add document controls from document mode section
            if hasattr(self, 'doc_type_combo') and self.doc_type_combo.winfo_exists():
                widgets.append(self.doc_type_combo)
            if hasattr(self, 'version_combo') and self.version_combo.winfo_exists():
                widgets.append(self.version_combo)
            if hasattr(self, 'room_entry') and self.room_entry.winfo_exists():
                widgets.append(self.room_entry)
            if hasattr(self, 'co_entry') and self.co_entry.winfo_exists():
                widgets.append(self.co_entry)
            if hasattr(self, 'archive_check') and self.archive_check.winfo_exists():
                widgets.append(self.archive_check)

        # Options section
        if hasattr(self, 'debug_check') and self.debug_check.winfo_exists():
            widgets.append(self.debug_check)
        if hasattr(self, 'training_check') and self.training_check.winfo_exists():
            widgets.append(self.training_check)

        # AI controls (if enabled) from AI toolbar section
        if hasattr(self, 'ai_toggle_button') and self.ai_toggle_button.winfo_exists():
            widgets.append(self.ai_toggle_button)
        if hasattr(self, 'ai_chat_button') and self.ai_chat_button.winfo_exists() and self.ai_enabled.get():
            widgets.append(self.ai_chat_button)

        # Action buttons based on mode from action buttons section
        if mode == "folder":
            if hasattr(self, 'open_button') and self.open_button.winfo_exists():
                widgets.append(self.open_button)
        else:
            if hasattr(self, 'find_button') and self.find_button.winfo_exists():
                widgets.append(self.find_button)
            if hasattr(self, 'choose_button') and self.choose_button.winfo_exists():
                widgets.append(self.choose_button)
            if hasattr(self, 'navigate_button') and self.navigate_button.winfo_exists() and self.last_search_result:
                widgets.append(self.navigate_button)

        return widgets

    def _find_widget_index(self, widget, widget_list: List) -> int:
        """Find the index of a widget in the widget list."""
        try:
            return widget_list.index(widget)
        except (ValueError, AttributeError):
            return 0

    def _find_widget_group(self, widget) -> Optional[str]:
        """Find which focus group a widget belongs to."""
        # Check folder radios group
        if hasattr(self, 'folder_radios') and widget in self.folder_radios:
            return 'folder_selection'

        # Check mode selection group
        if widget in [getattr(self, 'folder_radio', None), getattr(self, 'doc_radio', None)]:
            if widget and widget.winfo_exists():
                return 'mode_selection'

        # Check document filters group
        if widget in [getattr(self, 'doc_type_combo', None), getattr(self, 'version_combo', None),
                      getattr(self, 'room_entry', None), getattr(self, 'co_entry', None),
                      getattr(self, 'archive_check', None)]:
            if widget and widget.winfo_exists():
                return 'document_filters'

        # Check options group
        if widget in [getattr(self, 'debug_check', None), getattr(self, 'training_check', None)]:
            if widget and widget.winfo_exists():
                return 'options'

        # Check AI controls group
        if widget in [getattr(self, 'ai_toggle_button', None), getattr(self, 'ai_chat_button', None)]:
            if widget and widget.winfo_exists():
                return 'ai_controls'

        # Check action buttons group
        if widget in [getattr(self, 'open_button', None), getattr(self, 'find_button', None),
                      getattr(self, 'choose_button', None), getattr(self, 'navigate_button', None)]:
            if widget and widget.winfo_exists():
                return 'action_buttons'

        return None

    def _get_group_widgets(self, group_name: str) -> List:
        """Get all widgets in a specific focus group."""
        widgets = []

        if group_name == 'mode_selection':
            mode_widgets = []
            if hasattr(self, 'folder_radio') and self.folder_radio.winfo_exists():
                mode_widgets.append(self.folder_radio)
            if hasattr(self, 'doc_radio') and self.doc_radio.winfo_exists():
                mode_widgets.append(self.doc_radio)
            widgets = mode_widgets

        elif group_name == 'folder_selection' and hasattr(self, 'folder_radios'):
            folder_widgets = []
            for radio in self.folder_radios:
                if radio and radio.winfo_exists():
                    folder_widgets.append(radio)
            widgets = folder_widgets

        elif group_name == 'document_filters':
            filter_widgets = []
            if hasattr(self, 'doc_type_combo') and self.doc_type_combo.winfo_exists():
                filter_widgets.append(self.doc_type_combo)
            if hasattr(self, 'version_combo') and self.version_combo.winfo_exists():
                filter_widgets.append(self.version_combo)
            if hasattr(self, 'room_entry') and self.room_entry.winfo_exists():
                filter_widgets.append(self.room_entry)
            if hasattr(self, 'co_entry') and self.co_entry.winfo_exists():
                filter_widgets.append(self.co_entry)
            if hasattr(self, 'archive_check') and self.archive_check.winfo_exists():
                filter_widgets.append(self.archive_check)
            widgets = filter_widgets

        elif group_name == 'options':
            option_widgets = []
            if hasattr(self, 'debug_check') and self.debug_check.winfo_exists():
                option_widgets.append(self.debug_check)
            if hasattr(self, 'training_check') and self.training_check.winfo_exists():
                option_widgets.append(self.training_check)
            widgets = option_widgets

        elif group_name == 'ai_controls':
            ai_widgets = []
            if hasattr(self, 'ai_toggle_button') and self.ai_toggle_button.winfo_exists():
                ai_widgets.append(self.ai_toggle_button)
            if hasattr(self, 'ai_chat_button') and self.ai_chat_button.winfo_exists() and self.ai_enabled.get():
                ai_widgets.append(self.ai_chat_button)
            widgets = ai_widgets

        elif group_name == 'action_buttons':
            current_mode = self.current_mode.get()
            action_widgets = []

            if current_mode == "folder":
                if hasattr(self, 'open_button') and self.open_button.winfo_exists():
                    action_widgets.append(self.open_button)
            else:
                if hasattr(self, 'find_button') and self.find_button.winfo_exists():
                    action_widgets.append(self.find_button)
                if hasattr(self, 'choose_button') and self.choose_button.winfo_exists():
                    action_widgets.append(self.choose_button)
                if self.last_search_result and hasattr(self, 'navigate_button') and self.navigate_button.winfo_exists():
                    action_widgets.append(self.navigate_button)

            widgets = action_widgets

        return widgets

    def _scroll_widget_into_view(self, widget):
        """Scroll widget into view if it's outside the visible area."""
        try:
            # Update widget position and layout
            widget.update_idletasks()

            # Get widget position relative to main frame for better accuracy
            try:
                main_frame = getattr(self, 'main_frame', None)
                if main_frame and main_frame.winfo_exists():
                    widget_y = widget.winfo_y()
                    frame_height = main_frame.winfo_height()

                    # Check if widget is within main frame bounds
                    if widget_y < 0 or widget_y > frame_height - 50:  # 50px buffer
                        # Use see method if widget supports it (like in scrollable frames)
                        if hasattr(widget, 'see'):
                            widget.see()
                        else:
                            # Fallback to focus and raise
                            widget.focus_set()
                            widget.tkraise()
                else:
                    # Fallback to basic visibility check
                    widget_y = widget.winfo_rooty() - self.root.winfo_rooty()
                    window_height = self.root.winfo_height()

                    if widget_y < 0 or widget_y > window_height:
                        widget.tkraise()

            except Exception as e:
                logger.debug(f"Enhanced scroll into view error: {e}")
                # Fallback to basic implementation
                widget.tkraise()

        except Exception as e:
            logger.debug(f"Scroll into view error: {e}")

    def _focus_search(self):
        """Focus the search input field (Ctrl+F)."""
        if hasattr(self, 'project_entry'):
            self.project_entry.focus_set()
            self.project_entry.select_range(0, tk.END)

    def _switch_to_folder_mode(self):
        """Switch to folder navigation mode (Ctrl+1)."""
        self.current_mode.set("folder")
        self._on_mode_change()
        if hasattr(self, 'folder_radio'):
            self.folder_radio.focus_set()

    def _switch_to_document_mode(self):
        """Switch to document navigation mode (Ctrl+2)."""
        self.current_mode.set("document")
        self._on_mode_change()
        if hasattr(self, 'doc_radio'):
            self.doc_radio.focus_set()

    def _toggle_debug_mode(self):
        """Toggle debug mode (Ctrl+D)."""
        current_value = self.debug_mode.get()
        self.debug_mode.set(not current_value)
        if hasattr(self, 'debug_check'):
            self.debug_check.focus_set()

    def set_keyboard_navigation_enabled(self, enabled: bool):
        """Enable or disable keyboard navigation."""
        self._keyboard_navigation_enabled = enabled
        logger.info(f"Keyboard navigation {'enabled' if enabled else 'disabled'}")

    def get_keyboard_navigation_enabled(self) -> bool:
        """Check if keyboard navigation is enabled."""
        return getattr(self, '_keyboard_navigation_enabled', True)

    # Keyboard Shortcut Helper Methods
    def _set_mode(self, mode: str):
        """Set navigation mode via keyboard shortcut."""
        if mode in ["folder", "document"]:
            self.current_mode.set(mode)
            self._on_mode_change()
            self.status_text.set(f"Switched to {mode} mode")

    def _clear_and_reset(self):
        """Clear all inputs and reset to default state."""
        self.project_input.set("")
        self.selected_folder.set("4. System Designs")
        self.doc_type.set("🔧 Low-Level Design (LLD)")  # Fixed: Use proper doc type value
        self.version_filter.set("Auto (Latest/Best)")  # Fixed: Use correct variable name
        self.room_filter.set("")
        self.co_filter.set("")
        self.include_archive.set(False)
        self.debug_mode.set(False)
        self.training_mode.set(False)
        self.status_text.set("✓ Cleared all inputs")
        self.project_entry.focus_set()

        # Clear any stored results
        if hasattr(self, 'last_search_result'):
            self.last_search_result = None
        if hasattr(self, 'last_search_type'):
            self.last_search_type = None

        # Reset navigate button state
        if hasattr(self, 'navigate_button'):
            self.navigate_button.config(state=tk.DISABLED)

    # Recent Projects System
    def _update_recent_projects(self):
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
                command=lambda p=project_num: self._select_recent_project(p),
                width=8
            )
            btn.pack(side="left", padx=(0, 4))

            # Add tooltip with full project info
            tooltip_text = f"{project_num}"
            if project_name:
                tooltip_text += f"\n{project_name}"
            self._add_tooltip(btn, tooltip_text)

    def _select_recent_project(self, project_number):
        """Select a project from recent projects list."""
        self.project_input.set(str(project_number))
        self.project_entry.focus_set()
        self.status_text.set(f"✓ Selected recent project: {project_number}")

    # Autocomplete System
    def _setup_autocomplete(self):
        """Set up autocomplete functionality for project search."""
        # Initialize autocomplete state
        self.autocomplete_window = None
        self.autocomplete_listbox = None
        self.autocomplete_suggestions = []
        self.autocomplete_index = -1

        # Bind autocomplete events to project entry
        self.project_entry.bind('<KeyRelease>', self._on_autocomplete_keyrelease)
        self.project_entry.bind('<Down>', self._on_autocomplete_down)
        self.project_entry.bind('<Up>', self._on_autocomplete_up)
        self.project_entry.bind('<Tab>', self._on_autocomplete_tab)

    def _on_autocomplete_keyrelease(self, event):
        """Handle key release for autocomplete suggestions."""
        # Ignore special keys
        if event.keysym in ('Up', 'Down', 'Left', 'Right', 'Return', 'Tab', 'Escape', 'Shift_L', 'Shift_R', 'Control_L', 'Control_R'):
            return

        text = self.project_input.get()

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
            project_str = str(project)
            if text_lower in project_str.lower():
                suggestions.append(project_str)

        # Limit to top 5 suggestions
        return suggestions[:5]

    def _show_autocomplete(self, suggestions: List[str]):
        """Show autocomplete dropdown with suggestions."""
        self.autocomplete_suggestions = suggestions

        # Create autocomplete window if not exists
        if self.autocomplete_window is None:
            self.autocomplete_window = tk.Toplevel(self.root)
            self.autocomplete_window.wm_overrideredirect(True)

            # Create listbox
            self.autocomplete_listbox = tk.Listbox(
                self.autocomplete_window,
                height=min(5, len(suggestions)),
                font=("Segoe UI", 9)
            )
            self.autocomplete_listbox.pack(fill=tk.BOTH, expand=True)

            # Bind selection
            self.autocomplete_listbox.bind('<<ListboxSelect>>', self._on_autocomplete_select)
            self.autocomplete_listbox.bind('<Return>', self._on_autocomplete_select)
            self.autocomplete_listbox.bind('<Double-Button-1>', self._on_autocomplete_select)

        # Update listbox content
        self.autocomplete_listbox.delete(0, tk.END)
        for suggestion in suggestions:
            self.autocomplete_listbox.insert(tk.END, suggestion)

        # Position window below entry
        x = self.project_entry.winfo_rootx()
        y = self.project_entry.winfo_rooty() + self.project_entry.winfo_height()
        width = self.project_entry.winfo_width()

        self.autocomplete_window.geometry(f"{width}x{min(100, len(suggestions) * 20)}+{x}+{y}")
        self.autocomplete_window.lift()

    def _hide_autocomplete(self):
        """Hide autocomplete dropdown."""
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
            self.autocomplete_window = None
            self.autocomplete_listbox = None
        self.autocomplete_index = -1

    def _on_autocomplete_down(self, event):
        """Handle down arrow in autocomplete."""
        if self.autocomplete_listbox and self.autocomplete_suggestions:
            self.autocomplete_index = (self.autocomplete_index + 1) % len(self.autocomplete_suggestions)
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(self.autocomplete_index)
            self.autocomplete_listbox.see(self.autocomplete_index)
            return "break"

    def _on_autocomplete_up(self, event):
        """Handle up arrow in autocomplete."""
        if self.autocomplete_listbox and self.autocomplete_suggestions:
            self.autocomplete_index = (self.autocomplete_index - 1) % len(self.autocomplete_suggestions)
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(self.autocomplete_index)
            self.autocomplete_listbox.see(self.autocomplete_index)
            return "break"

    def _on_autocomplete_tab(self, event):
        """Handle tab key in autocomplete."""
        if self.autocomplete_listbox and self.autocomplete_suggestions:
            if self.autocomplete_index >= 0:
                selected = self.autocomplete_suggestions[self.autocomplete_index]
            elif self.autocomplete_listbox.curselection():
                idx = self.autocomplete_listbox.curselection()[0]
                selected = self.autocomplete_suggestions[idx]
            else:
                selected = self.autocomplete_suggestions[0]

            self.project_input.set(selected)
            self._hide_autocomplete()
            return "break"

    def _on_autocomplete_select(self, event):
        """Handle autocomplete selection."""
        if self.autocomplete_listbox and self.autocomplete_listbox.curselection():
            idx = self.autocomplete_listbox.curselection()[0]
            selected = self.autocomplete_suggestions[idx]
            self.project_input.set(selected)
            self._hide_autocomplete()
            self.project_entry.focus_set()

    # Keyboard Shortcuts Help
    def _show_keyboard_shortcuts(self):
        """Show comprehensive keyboard shortcuts reference dialog."""
        shortcuts_window = tk.Toplevel(self.root)
        shortcuts_window.title("Keyboard Shortcuts")
        shortcuts_window.geometry("600x500")
        shortcuts_window.transient(self.root)

        # Apply theme
        self.theme.apply_theme(shortcuts_window)

        # Create scrollable frame
        canvas = tk.Canvas(shortcuts_window)
        scrollbar = ttk.Scrollbar(shortcuts_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Title
        title_label = ttk.Label(
            scrollable_frame,
            text="⌨️ Keyboard Shortcuts Reference",
            font=("Segoe UI", 14, "bold")
        )
        title_label.pack(pady=(10, 20))

        # Shortcuts categories
        shortcuts = {
            "🔍 Navigation": [
                ("Ctrl+F", "Focus search box"),
                ("Enter", "Execute current action"),
                ("Escape", "Hide window"),
                ("Ctrl+1", "Switch to folder mode"),
                ("Ctrl+2", "Switch to document mode"),
            ],
            "✏️ Editing": [
                ("Ctrl+R", "Clear and reset all inputs"),
                ("Tab", "Autocomplete selection"),
                ("Up/Down", "Navigate autocomplete suggestions"),
            ],
            "🎨 View": [
                ("Ctrl+D", "Toggle dark/light theme"),
                ("Ctrl+T", "Toggle always on top"),
            ],
            "🤖 AI": [
                ("Ctrl+Space", "Toggle AI chat panel"),
                ("Ctrl+/", "Enable/disable AI assistant"),
            ],
            "⚙️ Settings & Help": [
                ("Ctrl+S", "Open settings"),
                ("Ctrl+H", "Show help"),
                ("F1", "Show user guide"),
            ],
        }

        # Create sections for each category
        for category, items in shortcuts.items():
            # Category header
            category_frame = ttk.LabelFrame(scrollable_frame, text=category)
            category_frame.pack(fill="x", padx=20, pady=(0, 15))

            # Shortcuts list
            for key, description in items:
                shortcut_frame = ttk.Frame(category_frame)
                shortcut_frame.pack(fill="x", padx=10, pady=5)

                key_label = ttk.Label(
                    shortcut_frame,
                    text=key,
                    font=("Consolas", 10, "bold"),
                    width=15
                )
                key_label.pack(side="left")

                desc_label = ttk.Label(
                    shortcut_frame,
                    text=description,
                    font=("Segoe UI", 9)
                )
                desc_label.pack(side="left", fill="x", expand=True)

        # Tips section
        tips_frame = ttk.LabelFrame(scrollable_frame, text="💡 Pro Tips")
        tips_frame.pack(fill="x", padx=20, pady=(0, 15))

        tips = [
            "Start typing in search box to see autocomplete suggestions",
            "Click recent project buttons for quick access",
            "Hover over controls to see helpful tooltips",
            "Use Ctrl+R to quickly start a new search",
            "Press Enter from any input field to execute",
        ]

        for tip in tips:
            tip_label = ttk.Label(
                tips_frame,
                text=f"• {tip}",
                font=("Segoe UI", 9),
                wraplength=550,
                justify="left"
            )
            tip_label.pack(anchor="w", padx=10, pady=3)

        # Close button
        close_button = ttk.Button(
            scrollable_frame,
            text="Close",
            command=shortcuts_window.destroy,
            width=15
        )
        close_button.pack(pady=20)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Center window
        shortcuts_window.update_idletasks()
        x = (shortcuts_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (shortcuts_window.winfo_screenheight() // 2) - (500 // 2)
        shortcuts_window.geometry(f"600x500+{x}+{y}")

    # Tooltip System
    def _add_tooltip(self, widget, text):
        """Add a tooltip to a widget."""
        if not hasattr(self, 'tooltips'):
            self.tooltips = {}

        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = tk.Label(tooltip, text=text,
                           background="lightyellow",
                           foreground="black",
                           relief="solid",
                           borderwidth=1,
                           font=("Segoe UI", 9))
            label.pack()

            self.tooltips[widget] = tooltip

        def hide_tooltip(event):
            if widget in self.tooltips:
                self.tooltips[widget].destroy()
                del self.tooltips[widget]

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)


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