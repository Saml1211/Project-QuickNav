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
   - AHK script invokes the Python backend on-demand, passing the job number as an argument.
   - Backend returns resolved path via stdout.
   - AHK presents the result for user action.

2. **AI-initiated Flow:**
   - AI agent connects via MCP to the server.
   - MCP server invokes Python backend on-demand with the proper job number.
   - Path is returned to the AI agent, enabling automated operations.

### Diagram (Textual)

`User ⇄ [AHK Frontend] ⇄ [Python Backend]`
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  ↑  
`AI / Automation ⇄ [MCP Server] ⇄ [Python Backend]`

## Integration Test System Pattern

- **Test Organization:** All AHK-based integration tests are located in `tests/ahk/`, with each scenario as a discrete script following the `test_*.ahk` naming convention for easy discovery and automation.

- **Test Runner:** `run_all_tests.ahk` automatically locates and executes all non-utility test scripts in the directory, aggregates results, and returns an exit code and summary suitable for CI pipelines.

- **Shared Utilities:** Common assertions and helpers are centralized in `test_utils.ahk`. This encourages DRY patterns and makes test extension straightforward.

- **Maintainability Pattern:** Test scripts are modular and isolated, allowing contributors to add new edge cases or scenarios by copying existing tests and extending helpers as needed. The runner and utility structure minimize boilerplate and maximize discoverability.

- **Recommended Practice:** Refactor shared logic into `test_utils.ahk` as coverage grows. All new integration tests should exit with code 0 on pass, nonzero on failure, and be added to `tests/ahk/` for automatic inclusion in the suite.

## Summary

Project QuickNav's design ensures modularity, reliability, and extensibility. Each layer is independently testable and replaceable while maintaining a clear contract of interaction.