# System Patterns – Project QuickNav

## Architecture Overview

Project QuickNav utilizes a three-tier architecture, each layer with a distinct responsibility:

1. **Python Backend (`find_project_path.py`):**  
   Central logic for locating project directories based on a 5-digit code.
   **NEW**: Enhanced with document discovery and training data generation capabilities.

2. **AutoHotkey (AHK) Frontend (`lld_navigator.ahk`):**  
   Provides the user interface and handles local user interaction.
   **NEW**: Includes training data generation toggle for optional AI/ML workflow integration.

3. **MCP Server (`mcp_server/`):**  
   Exposes backend navigation functions to AI agents or external systems through the Model Context Protocol.

## Key Technical Decisions

- **Stdout-based IPC (Inter-Process Communication):**  
  The Python backend communicates results via standard output, enabling robust and platform-neutral IPC. This choice maximizes compatibility and simplifies integration with both the AHK frontend and MCP server.
  **NEW**: Extended to include training data generation status messages.

- **Loose Coupling:**  
  Each component interacts only via clearly defined interfaces (CLI/stdout or MCP tools), minimizing dependencies and simplifying maintenance.

- **Explicit Separation of Concerns:**  
  - The backend is focused purely on directory resolution and document discovery.
  - The frontend is concerned only with UI/UX and user actions.
  - The MCP server bridges AI and automation workflows to the backend logic.

- **Optional Training Data Generation:**
  - Training data generation is opt-in via GUI toggle, maintaining core workflow simplicity.
  - Intelligent file organization prevents overwrites and enables tracking.

## Component Relationships

### Data Flow

1. **User-initiated Flow:**
   - User enters the project code into the AHK interface.
   - User optionally selects "Generate Training Data" checkbox.
   - AHK script invokes the Python backend on-demand, passing the job number and optional `--training-data` flag.
   - Backend returns resolved path via stdout.
   - **NEW**: If training data flag is set, backend also discovers documents and generates training JSON.
   - AHK presents the result for user action and optionally displays training data confirmation.

2. **AI-initiated Flow:**
   - AI agent connects via MCP to the server.
   - MCP server invokes Python backend on-demand with the proper job number.
   - Path is returned to the AI agent, enabling automated operations.

### Training Data System Architecture

**Document Discovery Flow:**
```
Project Path → discover_documents() → Document List → JSON Generation → training_data/training_data_[project].json
```

**File Organization Pattern:**
- Single projects: `training_data_17741.json`
- Search results: `training_data_search_YYYYMMDD_HHMMSS.json`
- Multiple exact matches: `training_data_17741_17742.json`

### Diagram (Textual)

`User ⇄ [AHK Frontend] ⇄ [Python Backend + Document Discovery]`
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  ↑ ↓ (training_data/*.json)  
`AI / Automation ⇄ [MCP Server] ⇄ [Python Backend + Document Discovery]`

## Training Data Patterns

- **Recursive Document Discovery**: Uses `os.walk()` for comprehensive file system traversal
- **File Type Filtering**: Supports .pdf, .docx, .doc, .rtf for maximum document coverage
- **Smart Filename Generation**: Prevents overwrites while enabling easy identification
- **JSON Schema Consistency**: Standardized structure for AI/ML training pipeline integration

## Summary

Project QuickNav's design ensures modularity, reliability, and extensibility. Each layer is independently testable and replaceable while maintaining a clear contract of interaction. The addition of training data capabilities enhances AI/ML workflow integration without compromising the core navigation functionality.