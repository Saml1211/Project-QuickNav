# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

Project QuickNav is a **multi-component navigation utility** for 5-digit project codes with three main components:

### Core Components

1. **Python Backend** (`src/find_project_path.py`)
   - Primary resolver for 5-digit project codes to filesystem paths
   - Handles both exact project number matching and fuzzy search by project name
   - Supports OneDrive for Business integration with fallback to test environment
   - Protocol: Returns standardized output (`SUCCESS:`, `ERROR:`, `SELECT:`, `SEARCH:`)

2. **AutoHotkey GUI Frontend** (`src/lld_navigator.ahk`)
   - Windows-native GUI for user interaction
   - Integrates with Python backend via subprocess calls
   - Provides project subfolder selection (System Designs, Sales Handover, etc.)
   - Global hotkey: `Ctrl+Alt+Q` to show/hide interface

3. **MCP Server** (`mcp_server/`)
   - Model Context Protocol server for AI/automation integration
   - Exposes project navigation functionality via MCP tools and resources
   - Enables AI agents to locate and work with project directories

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

### AutoHotkey Integration Testing
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

# Run MCP server
python -m mcp_server

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

### AutoHotkey GUI Features
- Project code or search term input
- Subfolder selection (System Designs, Sales Handover, BOM & Orders, etc.)
- Debug mode for troubleshooting Python backend
- Training data generation toggle
- Search results browser for multiple matches
- System tray integration

## Testing Strategy

- **Python Backend**: Comprehensive pytest suite with environment simulation
- **AHK Integration**: AutoHotkey v2 test suite for GUI workflows
- **CI/CD**: GitHub Actions for build, test, lint, and typecheck
- **MCP Server**: Functional testing via MCP protocol

## Dependencies

- **Python**: 3.8+ with `mcp[cli]`
- **AutoHotkey**: v1.1 or v2 (Windows GUI only)
- **Development**: `pytest`, `coverage`, `ruff`, `mypy`

## Configuration Notes

- OneDrive path auto-detection via `%UserProfile%\OneDrive - Pro AV Solutions`
- Test environment fallback when OneDrive unavailable
- Training data saved to `C:/Users/SamLyndon/Projects/Work/av-project-analysis-tools/training_data/`
- Cross-platform: Backend/MCP work on any OS; GUI requires Windows