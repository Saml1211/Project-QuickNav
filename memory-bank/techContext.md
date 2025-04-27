# Tech Context – Project QuickNav

## Technologies Used

- **Python 3.x:** Handles backend logic for project folder resolution.
- **AutoHotkey (AHK):** Provides the Windows desktop GUI and handles user input and actions.
- **MCP SDK:** Used for implementing the Model Context Protocol server, enabling AI and automation integration.

## Development Setup Requirements

- **Operating System:** Windows 10 or later (due to AHK frontend; backend and MCP may run on other platforms but full functionality requires Windows).
- **Python:** Version 3.8 or higher recommended.
- **AutoHotkey:** v1.1 or v2 (script written for compatibility with the installed version).
- **MCP SDK:** Installed via pip, as specified in `mcp_server/requirements.txt`.

## Technical Constraints

- **Windows-only:** AHK and full integration only supported on Windows. Other components may be portable but are untested on non-Windows platforms.
- **Path Handling:** All path operations and outputs are Windows-style. Care is taken in the Python backend and AHK frontend to ensure compatibility.
- **IPC (Inter-Process Communication):** All component communication uses stdout/stdin or explicit CLI calls. No sockets or shared memory used.

## Dependencies

- Python packages listed in `mcp_server/requirements.txt` (e.g., for MCP integration)
- AutoHotkey (installed globally)
- No external database or persistent storage required for operation

## Environmental Requirements

- Access to the root directory where all project folders are stored
- User permissions sufficient to launch AHK scripts and execute Python scripts

## Notes

- Ensure all tools are added to the system PATH as needed.
- For AI integration, ensure the MCP server is running and accessible.
- Future cross-platform support may require replacing the frontend layer.