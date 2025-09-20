# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

Project QuickNav is a **multi-component navigation utility** for 5-digit project codes with four main components:

### Core Components

1. **Python Backend** (`src/find_project_path.py`, `src/doc_navigator.py`)
   - Primary resolver for 5-digit project codes to filesystem paths
   - Advanced document search and classification system
   - Handles both exact project number matching and fuzzy search by project name
   - Supports OneDrive for Business integration with fallback to test environment
   - Protocol: Returns standardized output (`SUCCESS:`, `ERROR:`, `SELECT:`, `SEARCH:`)

2. **Tkinter GUI Application** (`quicknav/gui.py`) **[NEW]**
   - Cross-platform Python GUI with modern interface
   - Enhanced document search with filtering and preview
   - AI-powered project navigation assistance
   - Theme support (Light, Dark, High Contrast)
   - Global hotkey support across Windows, macOS, and Linux
   - **Replaces and enhances the AutoHotkey implementation**

3. **AutoHotkey GUI Frontend** (`src/lld_navigator.ahk`) **[LEGACY]**
   - Windows-native GUI for user interaction
   - Integrates with Python backend via subprocess calls
   - Provides project subfolder selection (System Designs, Sales Handover, etc.)
   - Global hotkey: `Ctrl+Alt+Q` to show/hide interface
   - **Note**: Being phased out in favor of the Tkinter GUI

4. **MCP Server** (`mcp_server/`)
   - Model Context Protocol server for AI/automation integration
   - Exposes project navigation functionality via MCP tools and resources
   - Enables AI agents to locate and work with project directories

### NEW: AI Integration

The Tkinter GUI includes comprehensive AI integration:

- **AI Chat Assistant**: Natural language project navigation
- **Tool Functions**: AI can directly search projects and documents
- **Multi-Provider Support**: OpenAI, Anthropic, Azure, and local models via LiteLLM
- **Conversation Memory**: Persistent context across sessions
- **Settings Integration**: Full configuration UI for AI features

### Project Structure Hierarchy

The system expects this OneDrive for Business directory structure:
```
OneDrive - Pro AV Solutions/
├── Project Files/          # Container for all projects
│   ├── 10000 - 10999/     # Range folders (thousands)
│   │   ├── 10123 - Project A/
│   │   │   ├── 1. Sales Handover/
│   │   │   ├── 2. BOM & Orders/
│   │   │   ├── 4. System Designs/
│   │   │   └── 6. Customer Handover Documents/
│   │   └── 10456 - Project B/
│   └── 17000 - 17999/
│       ├── 17741 - Test Project/
│       └── 17742 - Another Project/
```

## Development Commands

### Testing
```bash
# Run all Python tests with coverage
python -m pytest
coverage run -m pytest
coverage xml

# Run specific test
python -m pytest tests/test_find_project_path.py -v

# Test the backend directly
python src/find_project_path.py 17741
python src/find_project_path.py "Test Project"
```

### Tkinter GUI Testing
```bash
# Run GUI application
python quicknav/gui_launcher.py

# Run AI integration tests
python quicknav/test_ai_integration.py

# Run LiteLLM functionality tests (requires LiteLLM)
python quicknav/test_litellm_functionality.py

# Run interactive GUI test
python test_gui.py
```

### AutoHotkey Integration Testing (Legacy)
```bash
# Run AHK integration tests (requires AutoHotkey v2 on Windows)
autohotkey tests/ahk/run_all_tests.ahk
```

### Code Quality
```bash
# Linting
ruff check mcp_server/ quicknav/

# Type checking
mypy mcp_server/ quicknav/
```

### Package Development
```bash
# Install in development mode
pip install -e .

# Install MCP server dependencies
pip install -r mcp_server/requirements.txt

# Install GUI dependencies (optional)
pip install keyboard pystray        # For hotkeys and system tray
pip install litellm                 # For AI functionality

# Run applications
python quicknav/gui_launcher.py     # Tkinter GUI
python -m mcp_server               # MCP server

# Test CLI entry points
quicknav 17741
quicknav-mcp
```

### Build Verification
```bash
# Basic import check
python -c "import mcp_server; import quicknav"
```

## Key Implementation Details

### Backend Protocol (find_project_path.py)
- **Input**: 5-digit project number OR search term
- **Output Formats**:
  - `SUCCESS:[path]` - Single match found
  - `ERROR:[message]` - No matches or validation error
  - `SELECT:[path1|path2|...]` - Multiple exact matches for project number
  - `SEARCH:[path1|path2|...]` - Multiple matches for search term

### Training Data Generation
The backend supports `--training-data` flag to generate JSON training data for AI analysis:
```bash
python src/find_project_path.py 17741 --training-data
# Generates training_data_17741.json with document catalog
```

### MCP Integration Points
- **Tool**: `navigate_project` - Resolve project codes to paths
- **Resource**: `projectnav://folders` - Live project structure access
- **Server Entry**: `quicknav-mcp` or `python -m mcp_server`

### Tkinter GUI Features (NEW)
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **AI Assistant**: Chat interface with natural language project navigation
- **Enhanced Search**: Advanced document filtering and preview
- **Theming**: Light, Dark, and High Contrast themes
- **Global Hotkeys**: Configurable system-wide shortcuts
- **Settings Management**: Comprehensive configuration interface
- **Real-time Validation**: Input validation and autocomplete
- **Performance**: Async operations and intelligent caching
- **Accessibility**: DPI-aware scaling and keyboard navigation

### AutoHotkey GUI Features (Legacy)
- Project code or search term input
- Subfolder selection (System Designs, Sales Handover, BOM & Orders, etc.)
- Debug mode for troubleshooting Python backend
- Training data generation toggle
- Search results browser for multiple matches
- System tray integration

## Testing Strategy

- **Python Backend**: Comprehensive pytest suite with environment simulation
- **Tkinter GUI**: AI integration tests and functionality verification
- **AHK Integration**: AutoHotkey v2 test suite for GUI workflows (legacy)
- **CI/CD**: GitHub Actions for build, test, lint, and typecheck
- **MCP Server**: Functional testing via MCP protocol

## Dependencies

### Core
- **Python**: 3.8+ with `mcp[cli]`
- **Development**: `pytest`, `coverage`, `ruff`, `mypy`

### GUI (Optional)
- **tkinter**: Usually included with Python
- **keyboard**: Global hotkey support
- **pystray**: System tray integration
- **litellm**: AI functionality (OpenAI, Anthropic, Azure, local models)

### Legacy
- **AutoHotkey**: v1.1 or v2 (Windows only, being phased out)

## Configuration Notes

- OneDrive path auto-detection via `%UserProfile%\OneDrive - Pro AV Solutions`
- Test environment fallback when OneDrive unavailable
- Training data saved to `C:/Users/SamLyndon/Projects/Work/av-project-analysis-tools/training_data/`
- GUI settings stored in platform-appropriate locations (AppData/Library/config)
- Cross-platform: Backend/MCP/GUI work on any OS; AutoHotkey requires Windows

## NEW: Tkinter GUI Application

The new Tkinter GUI (`quicknav/`) provides a modern, cross-platform replacement for the AutoHotkey implementation with the following enhancements:

### Architecture
```
quicknav/
├── gui.py                  # Main application (View)
├── gui_controller.py       # Business logic (Controller)
├── gui_settings.py         # Settings & configuration (Model)
├── gui_theme.py           # Theme management
├── gui_widgets.py         # Custom UI components
├── gui_hotkey.py          # Global hotkey support
├── gui_launcher.py        # Application launcher
├── ai_client.py           # AI integration
├── ai_chat_widget.py      # AI chat interface
└── test_*.py              # Test suites
```

### Key Features

1. **AI Integration**: Full LiteLLM support for multiple AI providers
2. **Cross-Platform**: Windows, macOS, and Linux compatibility
3. **Enhanced UI**: Modern interface with theming and accessibility
4. **Advanced Search**: Document filtering, preview, and classification
5. **Settings**: Comprehensive configuration with backup/restore
6. **Performance**: Async operations, caching, and optimization

### Usage
```bash
# Launch GUI application
python quicknav/gui_launcher.py

# Enable AI features (optional)
pip install litellm
# Then configure API keys in Settings > AI

# Test functionality
python quicknav/test_ai_integration.py
python quicknav/test_litellm_functionality.py
```

### AI Assistant Capabilities

The integrated AI assistant can:
- **Search Projects**: "Find project 17741" or "Show me projects containing 'Conference Room'"
- **Locate Documents**: "Find CAD files in project 17742"
- **Analyze Structure**: "What folders are in the System Designs directory?"
- **Recent Access**: "Show me recently accessed projects"
- **Navigation Help**: "How do I navigate to the Sales Handover folder?"

### Detailed Documentation

For complete documentation of the Tkinter GUI and AI integration, see:
- `docs/GUI_Documentation.md` - Comprehensive GUI documentation
- `quicknav/test_ai_integration.py` - AI functionality examples
- `quicknav/test_litellm_functionality.py` - LiteLLM integration examples