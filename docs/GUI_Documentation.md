# Project QuickNav - Tkinter GUI Documentation

## Overview

The Project QuickNav Tkinter GUI is a comprehensive desktop application that fully replicates and enhances the functionality of the original AutoHotkey implementation. Built with Python's Tkinter framework, it provides a modern, cross-platform interface for project navigation with advanced features including AI assistance, theming, and enhanced document search capabilities.

**Latest Update (v2.1):** Major stability improvements including fixes for theme rendering issues, AI chat initialization errors, and enhanced error handling throughout the application.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [AI Integration](#ai-integration)
4. [User Interface](#user-interface)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Configuration](#configuration)
8. [Development](#development)

## Architecture Overview

The GUI application follows a modular MVC (Model-View-Controller) architecture:

```
quicknav/
├── gui.py                  # Main GUI application (View)
├── gui_controller.py       # Business logic (Controller)
├── gui_settings.py         # Settings management (Model)
├── gui_theme.py           # Theme management
├── gui_widgets.py         # Custom UI widgets
├── gui_hotkey.py          # Global hotkey support
├── gui_launcher.py        # Application launcher
├── ai_client.py           # AI integration client
├── ai_chat_widget.py      # AI chat interface
└── test_ai_integration.py # AI integration tests
```

### Key Design Principles

- **Separation of Concerns**: UI, business logic, and data are clearly separated
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux
- **DPI Awareness**: Responsive layout that scales with system DPI settings
- **Async Operations**: Background processing for better user experience
- **Graceful Degradation**: Optional features degrade gracefully when dependencies are missing

## Core Components

### 1. Main GUI Application (`gui.py`)

The `ProjectQuickNavGUI` class is the central application controller that:

- Manages the main window and layout
- Coordinates between different UI sections
- Handles user interactions and events
- Integrates AI functionality
- Manages themes and settings

**Key Features:**
- Project input with validation and autocomplete
- Navigation mode switching (folder/document)
- Document filtering and search
- Real-time status updates
- Global hotkey support (Ctrl+Alt+Q)
- AI assistant integration

### 2. Controller (`gui_controller.py`)

The `GuiController` class handles business logic:

- Project path resolution
- Document search and filtering
- Caching for performance
- Integration with backend Python modules
- Error handling and validation

### 3. Settings Management (`gui_settings.py`)

The `SettingsManager` and `SettingsDialog` classes provide:

- JSON-based configuration persistence
- Comprehensive settings interface with tabs:
  - **General**: Paths, UI preferences, hotkeys
  - **AI**: AI model configuration, API keys, features
  - **Advanced**: Performance, debug, and development options
- Real-time settings validation
- Backup and restore functionality

### 4. Theme System (`gui_theme.py`)

The `ThemeManager` supports:

- Built-in themes: Light, Dark, High Contrast
- System theme detection
- Dynamic theme switching
- Custom color schemes
- DPI-aware styling

### 5. Custom Widgets (`gui_widgets.py`)

Enhanced UI components:

- **EnhancedEntry**: Placeholder text, validation, autocomplete
- **SearchableComboBox**: Filterable dropdown with search
- **CollapsibleFrame**: Expandable/collapsible sections
- **SelectionDialog**: Multi-selection with search
- **SearchResultDialog**: Document search results browser
- **DocumentPreview**: File preview with syntax highlighting

### 6. Global Hotkeys (`gui_hotkey.py`)

Cross-platform hotkey implementation:

- **Windows**: Win32 API RegisterHotKey
- **macOS**: PyObjC Cocoa integration
- **Linux**: X11 key grabbing
- Configurable key combinations
- System tray integration

## AI Integration

### Overview

The AI integration provides intelligent assistance for project navigation through:

- **Natural Language Queries**: Ask questions about projects in plain English
- **Tool Functions**: AI can directly interact with the project backend
- **Chat Interface**: Persistent conversation with context memory
- **Multi-Provider Support**: OpenAI, Anthropic, Azure, and local models

### AI Client (`ai_client.py`)

The `AIClient` class manages AI interactions:

```python
class AIClient:
    def __init__(self, controller=None, settings=None):
        self.controller = controller
        self.settings = settings
        self.enabled = self._check_litellm_availability()
        self.tools = {}
        self.memory = ConversationMemory()
```

**Features:**
- **LiteLLM Integration**: Unified interface for multiple AI providers
- **Tool Registration**: 5 built-in tools for project operations
- **Conversation Memory**: Maintains context across sessions
- **Error Handling**: Graceful fallback when AI services are unavailable

### AI Tools

The AI assistant has access to 5 specialized tools:

1. **`search_projects`**: Find project folders by number or name
2. **`find_documents`**: Locate specific documents within projects
3. **`analyze_project`**: Get detailed project structure and information
4. **`list_project_structure`**: Show folder hierarchy of a project
5. **`get_recent_projects`**: Access recently opened projects

### Chat Widget (`ai_chat_widget.py`)

The `ChatWidget` provides a full-featured chat interface:

```python
class ChatWidget(ttk.Frame):
    def __init__(self, parent, ai_client=None):
        super().__init__(parent)
        self.ai_client = ai_client
        self.setup_ui()
```

**Features:**
- **Message Bubbles**: User and AI messages with timestamps
- **Syntax Highlighting**: Code blocks and file paths are highlighted
- **Tool Execution**: Visual feedback when AI uses tools
- **File Attachments**: Drag-and-drop file sharing
- **Export Options**: Save conversations in multiple formats
- **Search**: Find specific messages in conversation history

### AI Settings

Comprehensive configuration through the Settings dialog:

**Model Configuration:**
- Provider selection (OpenAI, Anthropic, Azure)
- Model selection per provider
- API key management

**Conversation Settings:**
- Max conversation history (default: 50)
- Temperature setting (0.0-2.0, default: 0.7)
- Max tokens per response (default: 1000)

**Features:**
- Enable/disable tool execution
- Auto-suggestions for user input
- Conversation memory persistence
- Connection testing

## User Interface

### Main Window Layout

The main window is organized into logical sections:

```
┌─────────────────────────────────────┐
│ Menu Bar (File, View, AI, Help)     │
├─────────────────────────────────────┤
│ Project Input                       │
│ [Project Number/Search Term]        │
├─────────────────────────────────────┤
│ Navigation Mode                     │
│ ○ Folder Mode  ○ Document Mode      │
├─────────────────────────────────────┤
│ Mode-Specific Options               │
│ [Folder Selection / Doc Filters]    │
├─────────────────────────────────────┤
│ Options                            │
│ ☐ Debug Mode  ☐ Training Data      │
├─────────────────────────────────────┤
│ AI Assistant Toolbar               │
│ [Enable AI] [AI Chat] [Status]     │
├─────────────────────────────────────┤
│ Status Area                        │
│ Status: Ready...                   │
│ [Progress Bar]                     │
├─────────────────────────────────────┤
│ Action Buttons                     │
│ [Open Folder] [Find Docs] [Choose] │
└─────────────────────────────────────┘
```

### AI Chat Window

The AI chat opens in a separate resizable window:

```
┌─────────────────────────────────────┐
│ AI Assistant                        │
├─────────────────────────────────────┤
│ ┌─────────────────────────────────┐ │
│ │ Chat Messages Area              │ │
│ │                                 │ │
│ │ User: Find project 17741        │ │
│ │ AI: Found project 17741...      │ │
│ │                                 │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ [📎] [Input Message...] [Send]     │
├─────────────────────────────────────┤
│ [Clear] [Export] [Settings]        │
└─────────────────────────────────────┘
```

### Keyboard Shortcuts

- **Ctrl+Alt+Q**: Show/hide main window (global hotkey)
- **Enter**: Execute current action
- **Escape**: Close current dialog
- **Ctrl+,**: Open settings
- **F1**: Show help
- **Ctrl+Shift+A**: Toggle AI assistant
- **Ctrl+Shift+C**: Open AI chat

## Installation & Setup

### Prerequisites

**Required:**
- Python 3.8 or higher
- Windows 10/11, macOS 10.14+, or Linux

**Core Dependencies:**
```
tkinter (usually included with Python)
pathlib
json
threading
queue
logging
```

**Optional Dependencies:**
```
keyboard      # Global hotkey support
pystray      # System tray integration
litellm      # AI functionality
pillow       # Enhanced image support
```

### Installation Steps

1. **Clone or download the project**
2. **Install optional dependencies:**
   ```bash
   pip install keyboard pystray litellm pillow
   ```
3. **Run the application:**
   ```bash
   python quicknav/gui_launcher.py
   ```

### First-Time Setup

1. **Configure Project Paths**: Add your OneDrive or project root directories
2. **Set up AI (Optional)**: Add API keys for AI providers in Settings > AI
3. **Customize Theme**: Choose your preferred theme in Settings > General
4. **Configure Hotkeys**: Set up global shortcuts in Settings > General

## Usage Guide

### Basic Navigation

1. **Enter Project Information**:
   - Type a 5-digit project number (e.g., "17741")
   - Or enter a search term (e.g., "Test Project")

2. **Choose Navigation Mode**:
   - **Folder Mode**: Navigate to project subfolders
   - **Document Mode**: Search for specific documents

3. **Execute Action**:
   - Click "Open Folder" or "Find Documents"
   - Or press Enter to execute

### Document Search

1. **Select Document Mode**
2. **Choose Document Type**: CAD, Office, Images, etc.
3. **Apply Filters** (optional):
   - Room filter (e.g., "Meeting Room")
   - CO (Change Order) filter
   - Version preference
4. **Execute Search**

### AI Assistant Usage

1. **Enable AI**: Click "Enable AI" in toolbar or AI menu
2. **Open Chat**: Click "AI Chat" button or use Ctrl+Shift+C
3. **Ask Questions**:
   - "Find project 17741"
   - "What documents are in the System Designs folder?"
   - "Show me recent CAD files for project 17742"
4. **Review Responses**: AI will use tools to gather information

### Settings Configuration

Access settings via:
- **Menu**: File > Settings
- **Keyboard**: Ctrl+,

**General Tab:**
- Custom project root paths
- UI preferences (always on top, auto-hide)
- Global hotkey configuration
- Theme selection

**AI Tab:**
- Enable/disable AI features
- API key configuration
- Model selection
- Conversation settings
- Test connection

**Advanced Tab:**
- Performance options (caching)
- Debug mode
- Training data generation
- Auto-backup settings

## Configuration

### Settings File Location

Settings are stored in JSON format:

- **Windows**: `%APPDATA%\QuickNav\settings.json`
- **macOS**: `~/Library/Application Support/QuickNav/settings.json`
- **Linux**: `~/.config/QuickNav/settings.json`

### Example Settings Structure

```json
{
  "ui": {
    "theme": "light",
    "always_on_top": false,
    "auto_hide_delay": 1500,
    "window_geometry": "480x650"
  },
  "navigation": {
    "custom_roots": [
      "C:\\Users\\User\\OneDrive - Company\\Project Files"
    ],
    "default_mode": "folder",
    "remember_last_selections": true
  },
  "hotkeys": {
    "show_hide": "ctrl+alt+q"
  },
  "ai": {
    "enabled": true,
    "default_model": "gpt-3.5-turbo",
    "api_keys": {
      "openai": "sk-...",
      "anthropic": "sk-ant-..."
    },
    "max_conversation_history": 50,
    "temperature": 0.7,
    "max_tokens": 1000,
    "enable_tool_execution": true,
    "enable_auto_suggestions": true,
    "enable_conversation_memory": true
  },
  "advanced": {
    "cache_enabled": true,
    "cache_timeout_minutes": 5,
    "debug_mode": false,
    "training_data_enabled": false,
    "auto_backup_settings": true
  }
}
```

### Environment Variables

AI functionality supports environment variables for API keys:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI
export AZURE_API_KEY="your-azure-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
```

## Development

### Code Structure

The codebase follows clean architecture principles:

```python
# Main application entry point
class ProjectQuickNavGUI:
    def __init__(self):
        self.settings = SettingsManager()      # Configuration
        self.theme = ThemeManager()            # UI theming
        self.controller = GuiController()      # Business logic
        self.ai_client = AIClient()           # AI integration

# Business logic layer
class GuiController:
    def navigate_to_project(self, project_input, folder, debug_mode):
        # Handles project navigation logic
        pass

# Data persistence layer
class SettingsManager:
    def get(self, key, default=None):
        # Configuration access
        pass
```

### Adding New Features

1. **UI Components**: Add to `gui_widgets.py`
2. **Business Logic**: Extend `GuiController`
3. **Settings**: Add to `SettingsManager` schema
4. **AI Tools**: Register new tools in `AIClient`
5. **Tests**: Add tests to `test_ai_integration.py`

### Testing

Run the comprehensive test suite:

```bash
python quicknav/test_ai_integration.py
```

Tests cover:
- Module imports and dependencies
- Settings integration
- AI client functionality
- GUI component creation
- Tool registration

### Debugging

Enable debug mode for detailed logging:

1. **GUI**: Check "Show Debug Output" in Options
2. **Settings**: Set `advanced.debug_mode = true`
3. **Command Line**:
   ```bash
   python quicknav/gui_launcher.py --debug
   ```

Debug information includes:
- Backend process output
- Search result details
- AI conversation logs
- Performance metrics

### Contributing

When contributing to the GUI:

1. **Follow Architecture**: Maintain separation of concerns
2. **Add Tests**: Include tests for new functionality
3. **Update Documentation**: Document new features
4. **Theme Support**: Ensure new UI works with all themes
5. **Cross-Platform**: Test on different operating systems

## Troubleshooting

### Common Issues

**Q: Global hotkey not working**
A: Check if another application is using Ctrl+Alt+Q. Configure a different hotkey in Settings.

**Q: AI features disabled**
A: Install LiteLLM: `pip install litellm` and configure API keys in Settings > AI.

**Q: Project not found**
A: Verify project root paths in Settings > General > Custom Project Roots.

**Q: Performance issues**
A: Enable caching in Settings > Advanced and increase cache timeout.

**Q: UI appears blurry on high-DPI displays**
A: The application includes DPI awareness. Restart the application if issues persist.

### Log Files

Application logs are stored in:
- **Windows**: `%TEMP%\quicknav.log`
- **macOS/Linux**: `/tmp/quicknav.log`

Enable detailed logging with debug mode for troubleshooting.

## Recent Fixes & Improvements (v2.1)

### Critical Bug Fixes

1. **MessageBubble Initialization Error** - Fixed `_tkinter.TclError: unknown color name` errors
   - **Issue**: Theme color dictionaries were being passed directly to Tkinter as color strings
   - **Fix**: Enhanced `_get_theme_color` method with proper color extraction and validation
   - **Impact**: AI chat functionality now works reliably without crashes

2. **Theme Color Handling** - Resolved nested color structure navigation issues
   - **Issue**: `_configure_tk_defaults` method incorrectly treated color properties as states
   - **Fix**: Implemented proper color navigation: `element -> state -> property`
   - **Impact**: All themes now render correctly with proper colors

3. **Windows Hotkey Registration** - Enhanced error reporting and validation
   - **Issue**: Failed hotkey registration was incorrectly logged as successful
   - **Fix**: Added return value checking and accurate logging
   - **Impact**: Users now receive clear feedback on hotkey registration status

### UI/UX Enhancements

- Enhanced typography system with proper font hierarchy
- Improved color schemes for better contrast and readability
- Modernized layout with better spacing and visual hierarchy
- Added comprehensive error handling throughout the application
- Stabilized AI chat widget rendering and theming

### Technical Improvements

- Added defensive programming patterns for theme color handling
- Enhanced error logging and user feedback
- Improved code reliability and maintainability
- Added comprehensive test coverage for UI improvements

---

*This documentation covers the complete Tkinter GUI implementation. For backend functionality and MCP server integration, see the main project README.*