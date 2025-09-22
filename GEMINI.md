# Gemini Project Analysis: QuickNav

## Project Overview

Project QuickNav is a hybrid utility that combines a Python backend with an AutoHotkey (AHK) frontend to provide rapid navigation to project directories. The system is designed around a 5-digit project code convention.

**Core Components:**

1.  **Python Backend (`quicknav` package):** An installable Python package that provides a command-line interface (`quicknav`) for finding project and document paths. It can be installed via `pip` from `setup.py`.
2.  **AutoHotkey Frontend (`lld_navigator.ahk`):** A GUI wrapper that calls the `quicknav` CLI to provide a user-friendly search interface. It is configured to be launched with the `Ctrl+Alt+Q` hotkey.
3.  **MCP Server (`mcp_server`):** A Python-based server using the Model Context Protocol (`FastMCP`). It exposes the project navigation functionality as tools (e.g., `navigate_project`) for AI agents and automation.

**Architecture:**

The project demonstrates a clear separation of concerns:
- The Python backend handles all the file system search and resolution logic.
- The AHK script acts purely as a GUI front-end, delegating all logic to the Python CLI.
- The MCP server provides a dedicated, structured interface for programmatic access by AI agents.

## Building and Running

### Python Backend (CLI)

The Python package can be installed locally using `pip`.

```sh
# Install the package in editable mode
pip install -e .

# After installation, the following commands are available:
quicknav --version
quicknav project 12345
quicknav doc 12345 --type lld
```

### AutoHotkey GUI

1.  Ensure the `quicknav` Python package is installed and its script is in the system's PATH.
2.  Run the `lld_navigator.ahk` script using an AutoHotkey v2 interpreter.

### MCP Server

1.  Install the server's dependencies:
    ```sh
    pip install -r mcp_server/requirements.txt
    ```
2.  Run the server module:
    ```sh
    python -m mcp_server
    ```

### Testing

The project includes an integration test suite for the AutoHotkey GUI.

-   **Test Runner:** `tests\ahk\run_all_tests.ahk`
-   **Command:**
    ```sh
    autohotkey tests\ahk\run_all_tests.ahk
    ```

## Development Conventions

-   **Modularity:** The project is broken down into distinct components (Python backend, AHK frontend, MCP server) that interact through clearly defined interfaces (CLI arguments, MCP tool calls).
-   **Configuration:** The primary project root directory can be configured via the `QUICKNAV_PROJECT_ROOT` environment variable.
-   **Error Handling:** The Python CLI and MCP tools provide structured error messages (e.g., `ERROR:`, `SUCCESS:`), which the AHK script and other clients parse to determine the outcome of an operation.
-   **Testing:** The presence of a dedicated AHK test suite in `tests/ahk/` indicates a convention of writing automated tests for the GUI components. Tests are script-based and return exit codes to signal pass/fail, making them suitable for CI.
-   **Documentation:** The project is well-documented with a `README.md`, `INSTALL.md`, and other explanatory markdown files.
