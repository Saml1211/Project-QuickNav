# Tech Context – Project QuickNav

## Technologies Used

- **Python 3.x:** Handles backend logic for project folder resolution.
- **AutoHotkey v2:** Provides the Windows desktop GUI and handles user input and actions.
- **MCP SDK:** Used for implementing the Model Context Protocol server, enabling AI and automation integration.
- **AHK2EXE Compiler:** Compiles AutoHotkey scripts into standalone executable files for distribution.

## Development Setup Requirements

- **Operating System:** Windows 10 or later (due to AHK frontend; backend and MCP may run on other platforms but full functionality requires Windows).
- **Python:** Version 3.8 or higher recommended.
- **AutoHotkey v2:** Required for GUI functionality and EXE compilation (script written for v2 compatibility).
- **AHK2EXE Compiler:** Required for building standalone executables (typically included with AutoHotkey v2 installation).
- **MCP SDK:** Installed via pip, as specified in `mcp_server/requirements.txt`.

## Build System

- **Build Script:** `scripts/build_exe.ps1` - PowerShell script for automated EXE compilation
- **Compilation Target:** `src/lld_navigator.ahk` → `dist/quicknav-{version}-win64.exe`
- **Version Management:** Automatic version reading from `VERSION.txt`
- **Distribution Ready:** Creates standalone EXE requiring no additional installations
- **CRITICAL: Path Resolution:** Compiled EXE includes intelligent path handling to locate Python scripts in `../src/` relative to distribution directory

## Technical Constraints

- **Windows-only:** AHK and full integration only supported on Windows. Other components may be portable but are untested on non-Windows platforms.
- **AutoHotkey v2 Dependency:** Build system specifically requires AutoHotkey v2 with compiler support.
- **Path Handling:** All path operations and outputs are Windows-style. Care is taken in the Python backend and AHK frontend to ensure compatibility.
- **Distribution Architecture:** Compiled EXE must be placed in `/dist` directory to maintain correct relative path relationship with Python scripts in `/src`
- **IPC (Inter-Process Communication):** All component communication uses stdout/stdin or explicit CLI calls. No sockets or shared memory used.

## Dependencies

- Python packages listed in `mcp_server/requirements.txt` (e.g., for MCP integration)
- AutoHotkey v2 (installed globally with compiler support)
- No external database or persistent storage required for operation

## Environmental Requirements

- Access to the root directory where all project folders are stored
- User permissions sufficient to launch AHK scripts and execute Python scripts
- **Build Environment:** AutoHotkey v2 installation with AHK2EXE compiler for creating distribution builds
- **Distribution Environment:** No special requirements - compiled EXE includes all necessary path resolution logic

## Notes

- Ensure all tools are added to the system PATH as needed.
- For AI integration, ensure the MCP server is running and accessible.
- **Build System:** Run `scripts/build_exe.ps1` to create standalone executable for distribution.
- **Distribution Deployment:** Place compiled EXE in `/dist` directory to maintain proper relative path to Python scripts in `/src`
- **Path Resolution:** Compiled EXE automatically handles locating Python backend scripts without requiring environment configuration
- Future cross-platform support may require replacing the frontend layer.