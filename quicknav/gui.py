#!/usr/bin/env python3
"""
Project QuickNav - Enhanced Tkinter GUI Application (Refactored)

A comprehensive GUI replacement for the AutoHotkey version with enhanced
features including modern theming, async operations, advanced search,
and cross-platform compatibility.

This version uses a modern, modular architecture with separated concerns:
- GUIState: Centralized state management
- LayoutManager: DPI scaling and responsive layout
- EventCoordinator: Event handling and keyboard shortcuts
- Section Components: Individual UI sections (project input, navigation mode, etc.)
- Main ProjectQuickNavGUI: Coordinator that wires everything together
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

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
    from .gui_state import GUIState
    from .gui_layout import LayoutManager, GUIConstants
    from .gui_events import EventCoordinator
    from .gui_sections import (
        ProjectInputSection,
        NavigationModeSection,
        FolderModeSection,
        DocumentModeSection,
        OptionsSection,
        AIAssistantSection,
        StatusSection,
        ActionButtonsSection
    )
except ImportError:
    # Fallback imports for development
    from gui_controller import GuiController
    from gui_settings import SettingsManager
    from gui_theme import ThemeManager
    from gui_state import GUIState
    from gui_layout import LayoutManager, GUIConstants
    from gui_events import EventCoordinator
    from gui_sections import (
        ProjectInputSection,
        NavigationModeSection,
        FolderModeSection,
        DocumentModeSection,
        OptionsSection,
        AIAssistantSection,
        StatusSection,
        ActionButtonsSection
    )

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
    except ImportError as e2:
        AI_IMPORT_ERROR = str(e2)
        logger.info(f"AI features not available: {AI_IMPORT_ERROR}")

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
    """
    Main GUI application coordinator class.

    This refactored version acts as a coordinator between components rather than
    implementing everything directly. It delegates responsibilities to specialized
    components while maintaining the application's overall behavior.

    Responsibilities:
    - Initialize all components (state, layout, sections, events)
    - Wire up event handlers and callbacks
    - Coordinate between components
    - Handle business logic (project navigation, document search)
    - Manage background tasks and results processing
    """

    def __init__(self):
        """Initialize the GUI application."""
        # Create root window
        self.root = tk.Tk()
        self.root.title("Project QuickNav - Enhanced")
        self.root.withdraw()  # Hide initially

        # Initialize core components
        self.settings = SettingsManager()
        self.theme = ThemeManager(self.settings)
        self.controller = GuiController(self.settings)

        # Initialize state management
        self.state = GUIState()

        # Initialize layout manager
        self.layout = LayoutManager(self.root, self.settings)

        # Background task management
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # AI components
        self.ai_client = None
        self.ai_chat_widget = None
        self.ai_chat_window = None

        # Additional state for navigation results
        self.last_search_result = None
        self.last_search_type = None
        self.tooltips = {}

        # UI sections (will be initialized in _setup_ui)
        self.sections = {}
        self.main_frame = None

        # Hotkey manager
        self.hotkey_manager = None

        # Initialize AI if enabled
        self._initialize_ai()

        # Batch initialize UI
        self._batch_initialize_ui()

        logger.info("ProjectQuickNavGUI initialized successfully")

    def _batch_initialize_ui(self):
        """Batch UI initialization operations for better performance."""
        try:
            # Setup all UI components
            self._setup_ui()
            self._setup_section_callbacks()
            self._apply_theme()

            # Initialize event coordinator
            self.events = EventCoordinator(self.root, self.state, self.sections)
            self._setup_event_handlers()
            self.events.setup_events()

            # Setup validation
            if 'project_input' in self.sections and 'document_mode' in self.sections:
                self.events.setup_validation(
                    self.sections['project_input'].project_entry,
                    self.sections['document_mode'].room_entry,
                    self.sections['document_mode'].co_entry
                )

            # Batch widget updates
            self.root.update_idletasks()

            # Restore window state
            self.restore_window_state()

            # Start background services
            self._start_task_processor()

            # Update UI elements
            self._update_ai_ui()

            # Set up for compiled environment
            self._setup_for_compiled_environment()

            # Schedule delayed refresh for theme stability
            self.root.after(50, self._force_widget_refresh)

        except Exception as e:
            logger.error(f"Error in batch UI initialization: {e}")
            self._fallback_initialization()

    def _fallback_initialization(self):
        """Fallback initialization if batch initialization fails."""
        try:
            logger.info("Using fallback initialization...")
            self._setup_ui()
            self._setup_section_callbacks()
            self._apply_theme()
            self.restore_window_state()
            self._start_task_processor()
            self._update_ai_ui()
            self._setup_for_compiled_environment()
        except Exception as e:
            logger.error(f"Fallback initialization also failed: {e}")

    def _setup_ui(self):
        """Set up the main user interface."""
        # Configure root window with responsive geometry
        saved_geometry = self.settings.get_window_geometry()
        if not saved_geometry or not self.layout.is_geometry_on_screen(saved_geometry):
            saved_geometry = self.layout.get_responsive_geometry()
            self.settings.set_window_geometry(saved_geometry)

        self.root.geometry(saved_geometry)
        self.root.minsize(self.layout.min_width, self.layout.min_height)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Ensure window is properly sized
        self.root.update_idletasks()

        # Create main container with responsive padding
        padding = self.layout.get_consistent_padding()
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=padding, pady=padding)

        # Configure grid weights for responsive layout
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)  # Main content area
        self.main_frame.rowconfigure(4, weight=1)  # Status area

        # Create all UI sections
        logger.info("Initializing UI sections")

        # Row 0 - Project Input
        self.sections['project_input'] = ProjectInputSection(
            self.main_frame, self.state, self.layout, self.settings, self._add_tooltip
        )

        # Row 1 - Navigation Mode
        self.sections['navigation_mode'] = NavigationModeSection(
            self.main_frame, self.state, self.layout, self._add_tooltip
        )

        # Row 2 - Main Content (Folder/Document modes)
        self.main_content_frame = ttk.Frame(self.main_frame)
        self.main_content_frame.grid(
            row=2, column=0, sticky="ew",
            pady=(0, self.layout.get_consistent_spacing() // 2)
        )
        self.main_content_frame.columnconfigure(0, weight=1)

        self.sections['folder_mode'] = FolderModeSection(
            self.main_content_frame, self.state, self.layout, self._add_tooltip
        )
        self.sections['document_mode'] = DocumentModeSection(
            self.main_content_frame, self.state, self.layout, self._add_tooltip
        )

        # Initially show folder mode
        self.sections['folder_mode'].show()

        # Row 3 - Sidebar (Options and AI Assistant)
        self.sidebar_frame = ttk.Frame(self.main_frame)
        self.sidebar_frame.grid(
            row=3, column=0, sticky="ew",
            pady=(0, self.layout.get_consistent_spacing() // 2)
        )
        self.sidebar_frame.columnconfigure(0, weight=1)
        self.sidebar_frame.columnconfigure(1, weight=1)

        self.sections['options'] = OptionsSection(
            self.sidebar_frame, self.state, self.layout, self._add_tooltip
        )
        self.sections['ai_assistant'] = AIAssistantSection(
            self.sidebar_frame, self.state, self.layout, self._add_tooltip
        )

        # Row 4 - Status
        self.sections['status'] = StatusSection(
            self.main_frame, self.state, self.layout
        )

        # Row 5 - Action Buttons
        self.sections['action_buttons'] = ActionButtonsSection(
            self.main_frame, self.state, self.layout, self._add_tooltip
        )

        # Create menu bar
        self._create_menu_bar()

        logger.info("UI initialization complete")

    def _setup_section_callbacks(self):
        """Wire up callbacks between sections and main logic."""
        # Navigation mode callbacks
        self.sections['navigation_mode'].set_on_mode_change(self._on_mode_change)

        # Action button callbacks
        self.sections['action_buttons'].set_on_folder_navigation(self.execute_folder_navigation)
        self.sections['action_buttons'].set_on_document_navigation(self.execute_document_navigation)
        self.sections['action_buttons'].set_on_document_choose(self.execute_document_navigation)
        self.sections['action_buttons'].set_on_final_navigation(self._execute_final_navigation)

        # AI assistant callbacks
        self.sections['ai_assistant'].set_on_toggle_ai(self.toggle_ai)
        self.sections['ai_assistant'].set_on_toggle_chat(self.toggle_ai_panel)
        self.sections['ai_assistant'].set_on_show_settings(self.show_ai_settings)

        # Project input validation
        self.sections['project_input'].bind_validation(self._on_project_input_change)

    def _setup_event_handlers(self):
        """Register event handlers with the event coordinator."""
        handlers = {
            'execute_action': self._execute_current_action,
            'hide_window': self.hide_window,
            'show_help': self.show_help,
            'show_settings': self.show_settings,
            'focus_search': self._focus_search,
            'set_folder_mode': lambda: self._set_mode("folder"),
            'set_document_mode': lambda: self._set_mode("document"),
            'toggle_theme': self.toggle_theme,
            'toggle_always_on_top': self.toggle_always_on_top,
            'clear_and_reset': self._clear_and_reset,
            'reset_window': self.layout.reset_window_to_default,
            'copy_path': self._copy_current_path,
            'toggle_fullscreen': self._toggle_fullscreen,
            'focus_and_select_all': self._focus_and_select_all,
            'toggle_ai_panel': self.toggle_ai_panel,
            'toggle_ai': self.toggle_ai,
            'window_configure': self._on_window_configure,
            'key_press': self._on_key_press,
            'focus_in': self._on_focus_in
        }

        for event_name, handler in handlers.items():
            self.events.register_handler(event_name, handler)

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
        nav_menu.add_command(label="Focus Search", command=self._focus_search, accelerator="Ctrl+F")
        nav_menu.add_command(label="Execute", command=self._execute_current_action, accelerator="Enter")

        # AI menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_checkbutton(
            label="Enable AI Assistant",
            variable=self.state.ai_enabled,
            command=self.toggle_ai,
            accelerator="Ctrl+/"
        )
        ai_menu.add_checkbutton(
            label="Show AI Chat Panel",
            variable=self.state.ai_panel_visible,
            command=self.toggle_ai_panel,
            accelerator="Ctrl+Space"
        )
        ai_menu.add_separator()
        ai_menu.add_command(label="AI Settings...", command=self.show_ai_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Help", command=self.show_help, accelerator="F1")
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_keyboard_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

    # Business Logic Methods (preserved from original)

    def execute_folder_navigation(self):
        """Execute folder navigation."""
        valid, error_msg = self.events.validate_inputs()
        if not valid:
            self._show_error(error_msg)
            return

        project_input = self.state.get_project_input().strip()
        selected_folder = self.state.get_selected_folder()

        # Show progress
        self.sections['status'].show_progress("Searching for project...")

        # Capture current values to avoid thread safety issues
        debug_mode_value = self.state.is_debug_mode()
        training_mode_value = self.state.is_training_mode()

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
        valid, error_msg = self.events.validate_inputs()
        if not valid:
            self._show_error(error_msg)
            return

        project_input = self.state.get_project_input().strip()

        # Get document type key
        doc_type_text = self.state.get_doc_type()
        doc_type_map = {
            "🔧 Low-Level Design (LLD)": "lld",
            "📊 High-Level Design (HLD)": "hld",
            "📄 Change Orders": "change_order",
            "💰 Sales & PO Reports": "sales_po",
            "🏗️ Floor Plans": "floor_plans",
            "📋 Scope Documents": "scope",
            "✅ QA/ITP Reports": "qa_itp",
            "⚠️ SWMS": "swms",
            "💰 Supplier Quotes": "supplier_quotes",
            "📷 Site Photos": "photos"
        }

        doc_type_key = doc_type_map.get(doc_type_text)
        if not doc_type_key:
            self._show_error("Please select a document type")
            return

        # Show progress
        self.sections['status'].show_progress("Searching for documents...")

        # Get filters
        filters = self.state.get_document_filters()
        debug_mode_value = self.state.is_debug_mode()
        training_mode_value = self.state.is_training_mode()

        # Execute in background
        def task():
            try:
                result = self.controller.search_documents(
                    project_input=project_input,
                    doc_type=doc_type_key,
                    version_filter=filters['version_filter'],
                    room_filter=filters['room_filter'],
                    co_filter=filters['co_filter'],
                    include_archive=filters['include_archive'],
                    debug_mode=debug_mode_value,
                    training_data=training_mode_value,
                    choose_mode=choose_mode
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

    def _execute_final_navigation(self):
        """Execute navigation to the last search result."""
        if not self.last_search_result:
            return

        try:
            if self.last_search_type == "folder":
                self._open_folder(self.last_search_result)
            elif self.last_search_type == "file":
                self._open_file(self.last_search_result)

            # Update recent projects
            self.sections['project_input'].update_recent_projects()

            # Clear selection
            self.last_search_result = None
            self.last_search_type = None
            self.sections['action_buttons'].disable_navigate_button()

        except Exception as e:
            logger.error(f"Error executing final navigation: {e}")
            self._show_error(f"Failed to open: {e}")

    def _execute_current_action(self):
        """Execute current action based on mode."""
        mode = self.state.get_current_mode()
        if mode == "folder":
            self.execute_folder_navigation()
        else:
            self.execute_document_navigation()

    # Event Handlers

    def _on_mode_change(self, mode: str):
        """Handle navigation mode change."""
        logger.debug(f"Mode change: {mode}")

        if mode == "folder":
            self.sections['folder_mode'].show()
            self.sections['document_mode'].hide()
            self.sections['action_buttons'].update_button_visibility("folder")
            self.state.set_status_text("Folder mode - Select a project subfolder to open")
        else:  # document mode
            self.sections['folder_mode'].hide()
            self.sections['document_mode'].show()
            self.sections['action_buttons'].update_button_visibility("document")
            self.state.set_status_text("Document mode - Find specific documents by type and filters")

        self.root.update_idletasks()

    def _on_project_input_change(self, event=None):
        """Handle project input change."""
        # Validation is handled by EventCoordinator
        pass

    def _on_window_configure(self, event):
        """Handle window configuration changes (resize, move)."""
        if event.widget == self.root:
            width = event.width
            height = event.height
            self.layout.update_dpi_scaling_for_resize(width, height)

    def _on_key_press(self, event):
        """Handle key press events."""
        # Keyboard navigation handled by EventCoordinator
        pass

    def _on_focus_in(self, event):
        """Handle focus in events."""
        pass

    # Background Task Processing

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

        # Hide progress bar
        self.sections['status'].hide_progress()

        if success:
            if task_type == 'project_navigation':
                self._handle_project_result(data)
            elif task_type == 'document_navigation':
                self._handle_document_result(data)
        else:
            self._show_error(f"Operation failed: {error}")
            self.state.set_status_text("Error occurred")

    def _handle_project_result(self, result: Dict[str, Any]):
        """Handle project navigation result."""
        status = result.get('status')
        path = result.get('path')
        folder = result.get('folder')

        if status == 'SUCCESS':
            self.last_search_result = path
            self.last_search_type = "folder"
            self._open_folder_with_subfolder(path, folder)
            self.sections['project_input'].update_recent_projects()
        elif status == 'SELECT':
            paths = result.get('paths', [])
            self._show_selection_dialog(paths, folder, "project")
        elif status == 'ERROR':
            error_msg = result.get('message', 'Unknown error')
            self._show_error(error_msg)

    def _handle_document_result(self, result: Dict[str, Any]):
        """Handle document navigation result."""
        status = result.get('status')
        path = result.get('path')

        if status == 'SUCCESS':
            self.last_search_result = path
            self.last_search_type = "file"
            self._open_file(path)
            self.sections['project_input'].update_recent_projects()
        elif status == 'SELECT':
            paths = result.get('paths', [])
            self._show_search_dialog(paths, None)
        elif status == 'ERROR':
            error_msg = result.get('message', 'Unknown error')
            self._show_error(error_msg)

    # File Operations

    def _open_folder(self, path: str):
        """Open folder in file explorer."""
        import subprocess
        import platform

        try:
            if platform.system() == "Windows":
                subprocess.run(['explorer', path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(['open', path])
            else:  # Linux
                subprocess.run(['xdg-open', path])
            logger.info(f"Opened folder: {path}")
        except Exception as e:
            logger.error(f"Failed to open folder: {e}")
            self._show_error(f"Failed to open folder: {e}")

    def _open_folder_with_subfolder(self, project_path: str, subfolder: str):
        """Open folder with subfolder path."""
        full_path = Path(project_path) / subfolder
        if full_path.exists():
            self._open_folder(str(full_path))
        else:
            self._open_folder(project_path)

    def _open_file(self, path: str):
        """Open file with default application."""
        import subprocess
        import platform

        try:
            if platform.system() == "Windows":
                subprocess.run(['start', '', path], shell=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(['open', path])
            else:  # Linux
                subprocess.run(['xdg-open', path])
            logger.info(f"Opened file: {path}")
        except Exception as e:
            logger.error(f"Failed to open file: {e}")
            self._show_error(f"Failed to open file: {e}")

    def _show_selection_dialog(self, paths: List[str], folder: Optional[str], dialog_type: str):
        """Show selection dialog for multiple paths."""
        # Simplified version - just open the first one
        if paths:
            if dialog_type == "project" and folder:
                self._open_folder_with_subfolder(paths[0], folder)
            else:
                self._open_folder(paths[0])

    def _show_search_dialog(self, paths: List[str], folder: Optional[str]):
        """Show search results dialog."""
        # Simplified version - just open the first one
        if paths:
            self._open_file(paths[0])

    def _show_error(self, message: str, suggestion: Optional[str] = None):
        """Show error message."""
        messagebox.showerror("Error", message)
        logger.error(message)

    # Window Management

    def show_window(self):
        """Show the main window."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.sections['project_input'].focus()

    def hide_window(self):
        """Hide the main window."""
        self.root.withdraw()

    def toggle_theme(self):
        """Toggle between light and dark theme."""
        self.theme.toggle_theme()
        self._apply_theme()

    def toggle_always_on_top(self):
        """Toggle always on top mode."""
        current = self.root.attributes('-topmost')
        self.root.attributes('-topmost', not current)

    def restore_window_state(self):
        """Restore window state from settings."""
        # Window geometry is restored in _setup_ui
        always_on_top = self.settings.get("window.always_on_top", False)
        self.root.attributes('-topmost', always_on_top)

    # Theme and Styling

    def _apply_theme(self):
        """Apply the current theme."""
        self.theme.apply_theme(self.root)
        self.root.update_idletasks()

    def _force_widget_refresh(self):
        """Force widget refresh for theme changes."""
        try:
            self.root.update_idletasks()
        except Exception as e:
            logger.error(f"Error refreshing widgets: {e}")

    # AI Methods

    def _initialize_ai(self):
        """Initialize AI components if enabled."""
        if not AI_AVAILABLE:
            self.state.set_ai_enabled(False)
            if AI_IMPORT_ERROR:
                logger.info(f"AI features disabled: {AI_IMPORT_ERROR}")
            return

        ai_enabled = self.settings.get("ai.enabled", False)
        self.state.set_ai_enabled(ai_enabled)

        if ai_enabled and OPTIONAL_DEPS['litellm']:
            try:
                self.ai_client = AIClient(controller=self.controller, settings=self.settings)
                if self.ai_client and self.ai_client.is_available():
                    logger.info("AI client initialized successfully")
                else:
                    self.state.set_ai_enabled(False)
            except Exception as e:
                logger.error(f"AI initialization error: {e}")
                self.state.set_ai_enabled(False)

    def toggle_ai(self):
        """Toggle AI assistant on/off."""
        if self.state.is_ai_enabled():
            if self.ai_client is None:
                self._initialize_ai()
            if self.ai_client is None:
                messagebox.showerror("AI Error", "Failed to initialize AI assistant.")
                self.state.set_ai_enabled(False)
                self._update_ai_ui()
                return
            self._update_ai_ui()
            self.state.set_status_text("AI Assistant enabled")
        else:
            if self.state.is_ai_panel_visible():
                self.toggle_ai_panel()
            self._update_ai_ui()
            self.state.set_status_text("AI Assistant disabled")

    def _update_ai_ui(self):
        """Update AI-related UI elements."""
        if 'ai_assistant' in self.sections:
            self.sections['ai_assistant'].update_ui(self.state.is_ai_enabled())

    def toggle_ai_panel(self):
        """Toggle AI chat panel visibility."""
        # Simplified - AI panel not implemented in refactored version yet
        pass

    def show_ai_settings(self):
        """Show AI settings dialog."""
        # Delegate to settings dialog
        self.show_settings()

    # Menu and Dialog Actions

    def show_settings(self):
        """Show settings dialog."""
        # Import here to avoid circular dependency
        try:
            from .gui_settings import SettingsDialog
        except ImportError:
            from gui_settings import SettingsDialog

        dialog = SettingsDialog(self.root, self.settings, self.theme)
        dialog.show()
        # Apply any changed settings
        self._apply_theme()

    def show_help(self):
        """Show help dialog."""
        help_text = """Project QuickNav Help

FOLDER MODE:
1. Enter project number (e.g., 17741)
2. Select desired subfolder
3. Click 'Open Folder'

DOCUMENT MODE:
1. Enter project number
2. Select document type
3. Apply filters (optional)
4. Click 'Find Documents' or 'Choose From List'

KEYBOARD SHORTCUTS:
Ctrl+1: Folder Mode
Ctrl+2: Document Mode
Ctrl+F: Focus Search
Ctrl+D: Toggle Theme
Ctrl+S: Settings
Ctrl+R: Reset Form
Enter: Execute
Escape: Hide Window

For more information, see the documentation."""
        messagebox.showinfo("Help", help_text)

    def show_about(self):
        """Show about dialog."""
        about_text = """Project QuickNav v3.0

Enhanced GUI for project navigation with ML-powered features.

Features:
- Cross-platform support
- Advanced document search
- AI-powered assistance
- Modern theming
- DPI-aware responsive layout

Built with Python and Tkinter"""
        messagebox.showinfo("About", about_text)

    # Utility Methods

    def _focus_search(self):
        """Focus the search input."""
        if 'project_input' in self.sections:
            self.sections['project_input'].focus()

    def _set_mode(self, mode: str):
        """Set navigation mode."""
        self.state.set_current_mode(mode)

    def _clear_and_reset(self):
        """Clear and reset form."""
        self.state.reset_form()
        self.last_search_result = None
        self.last_search_type = None
        if 'action_buttons' in self.sections:
            self.sections['action_buttons'].disable_navigate_button()

    def _copy_current_path(self):
        """Copy current path to clipboard."""
        if self.last_search_result:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_search_result)
            self.state.set_status_text(f"Copied: {self.last_search_result}")

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)

    def _focus_and_select_all(self):
        """Focus and select all text in search."""
        if 'project_input' in self.sections:
            self.sections['project_input'].focus()
            self.sections['project_input'].select_all()

    def _show_keyboard_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts = """Keyboard Shortcuts:

GLOBAL:
Enter / Ctrl+Enter: Execute current action
Escape: Hide window
F1: Show help

NAVIGATION:
Ctrl+F: Focus search box
Ctrl+1: Switch to Folder Mode
Ctrl+2: Switch to Document Mode

VIEW:
Ctrl+D: Toggle theme (Light/Dark)
Ctrl+T: Toggle always on top
Ctrl+Shift+R: Reset window size

ACTIONS:
Ctrl+R: Clear and reset form
Ctrl+S: Open settings
Ctrl+Shift+C: Copy current path
Ctrl+Shift+F: Focus and select all

AI:
Ctrl+/: Toggle AI assistant
Ctrl+Space: Toggle AI chat panel

ADVANCED:
F11: Toggle fullscreen"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def _add_tooltip(self, widget, text):
        """Add tooltip to a widget."""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            label = tk.Label(tooltip, text=text, background="lightyellow",
                           foreground="black", relief="solid", borderwidth=1,
                           font=("Segoe UI", 9))
            label.pack()
            self.tooltips[widget] = tooltip

        def hide_tooltip(event):
            if widget in self.tooltips:
                self.tooltips[widget].destroy()
                del self.tooltips[widget]

        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)

    def _setup_for_compiled_environment(self):
        """Setup for compiled environment."""
        # Handle any compiled environment specific setup
        pass

    # Application Lifecycle

    def on_closing(self):
        """Handle window closing."""
        # Save window geometry
        geometry = self.root.geometry()
        self.settings.set_window_geometry(geometry)

        # Clean up hotkeys
        if self.hotkey_manager:
            try:
                self.hotkey_manager.cleanup()
            except:
                pass

        # Destroy window
        self.root.destroy()

    def run(self):
        """Run the application."""
        # Show window
        self.show_window()

        # Start main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
