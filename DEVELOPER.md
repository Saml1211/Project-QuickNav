# Developer Guide – Project QuickNav

## Architecture Overview

Project QuickNav is structured as a three-tier modular system:

1. **Python Backend (`find_project_path.py`)**
   - Central logic for locating project directories by 5-digit code.
   - CLI/IPC interface for requests from both GUI and MCP server.

2. **AutoHotkey Frontend (`lld_navigator.ahk`)**
   - User interface for job code input and subfolder selection.
   - Handles user interaction, invokes backend, and launches folders.

3. **MCP Server (`mcp_server/`)**
   - Exposes backend navigation logic as Model Context Protocol (MCP) tools/resources.
   - Enables AI, automation scripts, and remote agents to resolve and access project folders.

![Architecture Diagram (text)](see memory-bank/systemPatterns.md)

```
User ⇄ [AHK Frontend] ⇄ [Python Backend]
                ↑
AI / Automation ⇄ [MCP Server] ⇄ [Python Backend]
```

---

## Component Details

### Backend (`find_project_path.py`)
- Implements folder search using Windows environment assumptions (OneDrive, "Project Folders").
- Exits with status-prefixed lines (e.g. `SUCCESS:`, `ERROR:`, `SELECT:`).
- Designed for robust command-line use and IPC.

### AutoHotkey Frontend (`lld_navigator.ahk`)
- Minimal GUI, single process, always-on-top window.
- Translates user interaction into backend calls.
- Handles error/success/multi-path selection via popup dialogs.

### MCP Server (`mcp_server/`)
- Wraps backend as an MCP tool (`navigate_project`).
- Exposes folder listing as a resource (`projectnav://folders`).
- Entry points: `server.py` (core logic), `tools.py` (tool schema), `resources.py` (resources), `test_server.py` (basic tests).

---

## How to Extend the Project

- **Add Backends/Root Path Logic:**  
  Edit `find_project_path.py` to support new root directories, naming patterns, or cross-platform logic.

- **Enhance GUI:**  
  Extend `lld_navigator.ahk` to add new subfolder options or integrate with other Windows utilities.

- **Add MCP Tools/Resources:**  
  Implement additional tools in `mcp_server/tools.py` or resources in `mcp_server/resources.py`. Decorate with `@mcp.tool()` or `@mcp.resource()`.

- **Cross-Platform Support:**  
  Replace the AHK frontend with a portable GUI (e.g., Tkinter, Electron), and update path resolution logic for non-Windows platforms.

---

## MCP Server Integration Guide

- **Dependencies:**  
  Install Python 3.8+, pip packages from `mcp_server/requirements.txt`, and ensure MCP SDK is present.

- **Tool Specification:**  
  - `navigate_project(project_number: str) -> dict`
    - Input: 5-digit string.
    - Output:  
      - `{ "status": "success", "path": <folder> }`  
      - `{ "status": "select", "paths": [<folder1>, ...] }`  
      - `{ "status": "error", "message": <msg> }`

- **Resource Specification:**  
  - `projectnav://folders`: Lists top-level project workspace entries.

- **Running the Server:**  
  ```sh
  python -m mcp_server
  ```

- **Testing:**  
  ```sh
  python -m mcp_server.test_server
  ```

---

## Testing Procedures

- **Unit/Integration Tests:**  
  See `mcp_server/test_server.py` for basic examples. Expand as needed for:
  - Edge case project numbers
  - Multi-directory matches
  - Error conditions (missing directories, bad input)

- **Manual Testing:**  
  - Use the AHK GUI to validate user workflows.
  - Run backend standalone:  
    ```sh
    python find_project_path.py 12345
    ```
    Observe stdout for correct status.

- **MCP Protocol Testing:**  
  - Use any MCP-compliant client, or script requests to the MCP server tool/resource endpoints.

---

## Conventions & Best Practices

- Keep all inter-component communication via stdout/CLI/MCP tools.
- Document any changes to CLI protocols or tool/resource schemas.
- All code should be internally documented. See code for function/class docstrings and script-level headers.
- For new OS/GUI support, maintain strict separation of backend logic and presentation/UI.

---

## Future Enhancements (Suggestions)

- Cross-platform frontend (e.g., multiplatform GUI)
- "Favorites" or search history
- Deeper IDE/editor integrations
- Advanced automation workflows

---

## AHK Linting and Best Practices

### AutoHotkey Linting

- **Include Directives:** 
  - Always use simple relative paths for `#Include` directives:
  ```ahk
  ; Good - direct relative path
  #Include lld_navigator_controller.ahk
  
  ; Good - explicit relative path for test scripts
  #Include ..\..\lld_navigator.ahk
  
  ; Avoid - may cause linter path resolution issues
  #Include %A_ScriptDir%\some_file.ahk
  ```
  
- **Test Utilities:**
  - Test scripts should use the shared utilities in `tests/ahk/test_utils.ahk`:
  ```ahk
  #Include test_utils.ahk
  ; Reset test state at the beginning of each test
  reset_test_state()
  ```
  - Never define duplicate test functions like `test_fail()` in individual test scripts. Use the ones provided by `test_utils.ahk`.

- **Linter Configuration:**
  - If using VS Code with the AutoHotkey extension:
    1. Set workspace as the project root, not the `tests/ahk/` directory.
    2. Configure the linter to correctly find include files by adding a `.vscode/settings.json` with:
    ```json
    {
      "AutoHotkey2.Include.Directories": [
        "${workspaceFolder}"
      ]
    }
    ```

- **Common Issues:**
  - The linter may sometimes incorrectly report errors for include paths, especially when:
    - The workspace root is not properly set.
    - Environment-dependent paths like `%A_ScriptDir%` are used in includes.
    - Multiple functions with the same name exist across included files.
  - For test files, ensure all utility functions are imported from `test_utils.ahk` and not redefined locally.

### AHK Code Organization

- Each AHK script should have a clear separation of concerns:
  - `lld_navigator.ahk`: GUI and presentation only; delegates business logic to controller.
  - `lld_navigator_controller.ahk`: Core business/backend logic, persistence, validation.
  - Test scripts: Individual tests with clear scope and purpose; shared logic in test utilities.

---