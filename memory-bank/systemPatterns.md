# System Patterns – Project QuickNav

## Architecture Overview

Project QuickNav utilizes a three-tier architecture, each layer with a distinct responsibility:

1. **Python Backend (`find_project_path.py`):**  
   Central logic for locating project directories based on a 5-digit code.

2. **AutoHotkey (AHK) Frontend (`lld_navigator.ahk`):**  
   Provides the user interface and handles local user interaction.

3. **MCP Server (`mcp_server/`):**  
   Exposes backend navigation functions to AI agents or external systems through the Model Context Protocol.

## Key Technical Decisions

- **Stdout-based IPC (Inter-Process Communication):**  
  The Python backend communicates results via standard output, enabling robust and platform-neutral IPC. This choice maximizes compatibility and simplifies integration with both the AHK frontend and MCP server.

- **Loose Coupling:**  
  Each component interacts only via clearly defined interfaces (CLI/stdout or MCP tools), minimizing dependencies and simplifying maintenance.

- **Explicit Separation of Concerns:**  
  - The backend is focused purely on directory resolution.
  - The frontend is concerned only with UI/UX and user actions.
  - The MCP server bridges AI and automation workflows to the backend logic.

## Component Relationships

### Data Flow

1. **User-initiated Flow:**
   - User enters the project code into the AHK interface.
   - AHK script invokes the Python backend, passing the code.
   - Backend returns resolved path via stdout.
   - AHK presents the result for user action.

2. **AI-initiated Flow:**
   - AI agent connects via MCP to the server.
   - MCP server invokes Python backend as needed.
   - Path is returned to the AI agent, enabling automated operations.

### Diagram (Textual)

`User ⇄ [AHK Frontend] ⇄ [Python Backend]`
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  ↑  
`AI / Automation ⇄ [MCP Server] ⇄ [Python Backend]`

## Summary

Project QuickNav's design ensures modularity, reliability, and extensibility. Each layer is independently testable and replaceable while maintaining a clear contract of interaction.