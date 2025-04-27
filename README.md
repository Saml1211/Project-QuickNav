# Project QuickNav

QuickNav is a project directory navigation utility using a 5-digit project code system, with both CLI and GUI components.

**For installation and release instructions, see [INSTALL.md](./INSTALL.md) and [RELEASE.md](./RELEASE.md).**

---

## Overview

Project QuickNav is a utility designed to streamline navigation, access, and context sharing for large-scale project directories. It is purpose-built for environments where projects are organized by 5-digit codes, enabling rapid, reliable folder access for both human users and AI agents. With a simple interface and automation-ready backend, QuickNav minimizes search time, reduces workflow interruptions, and facilitates seamless integration into developer and automation pipelines.

---

## Features

- **Instant project folder location** via 5-digit code
- **User-friendly Windows GUI** with subfolder shortcuts (AutoHotkey)
- **Backend Python resolver** for robust directory search
- **AI/automation integration** through MCP (Model Context Protocol) server
- **Persistent context sharing** for project locations
- **Reliable, modular design** with clear separation of concerns

---

## System Requirements

- **Operating System:** Windows 10 or newer (full functionality; backend/MCP may run on other OS with limited features)
- **Python:** Version 3.8 or higher
- **AutoHotkey:** v1.1 or v2 (frontend script is compatible with both)
- **MCP SDK:** Installed via pip (see `mcp_server/requirements.txt`)
- **User permissions:** Must be able to run Python scripts and AHK executables
- **OneDrive for Business:** Used as root directory container (customizable in code)
- **Access:** To the base directory where all project folders are stored

---

## Installation

### 1. Clone the Repository

```sh
git clone <repo-url>
cd Project-QuickNav
```

### 2. Install the Python Backend

- Ensure Python 3.8+ is available.
- No special dependencies required for `find_project_path.py`.

### 3. Set Up the AutoHotkey Frontend

- Install [AutoHotkey](https://www.autohotkey.com/) (v1.1 or v2).
- Double-click `lld_navigator.ahk` to run, or create a desktop shortcut for rapid launch.

### 4. (Optional) Set Up the MCP Server for AI/Automation

- Navigate to `mcp_server/`
- Install dependencies:

```sh
pip install -r requirements.txt
```

- Start the MCP server:

```sh
python -m mcp_server
```

---

## Usage Guide

### Human Workflow

1. Run `lld_navigator.ahk`.
2. Enter a 5-digit project code.
3. Select the desired subfolder (e.g., System Designs, Sales Handover).
4. Click **Open**.
5. The folder opens in Windows Explorer. Errors or multiple matches are clearly reported for selection.

### Keyboard Shortcuts

- **Ctrl+Alt+Q:** Open or focus the Project QuickNav window globally, from anywhere in Windows. This shortcut will bring the QuickNav GUI to the front if already open, or launch it if not running.

### AI/Automation Workflow

- Integrate with the MCP server using the standard Model Context Protocol.
- Use the `navigate_project` tool to resolve project folders by code.
- Access live project structure via the `projectnav://folders` resource.

#### Example: MCP Tool Usage

Send a `navigate_project` request with a 5-digit string. The result will indicate:
- `status: "success"`: Folder path found.
- `status: "select"`: Multiple candidate paths, included as a list.
- `status: "error"`: Resolution failure or bad input.

---

## Troubleshooting

| Problem                                 | Solution                                                                                   |
|------------------------------------------|--------------------------------------------------------------------------------------------|
| *Invalid 5-digit code error*             | Ensure you enter exactly five digits – no letters or spaces.                               |
| *OneDrive/Project Folders not found*     | Check your OneDrive sync and directory structure. Update logic if root path differs.        |
| *Subfolder not found*                    | Confirm the selected subfolder exists inside the resolved project directory.                |
| *Python/AutoHotkey not recognized*       | Add both to your system PATH. Restart your terminal or PC if necessary.                    |
| *MCP server integration fails*           | Verify MCP server is running, and all dependencies in `mcp_server/requirements.txt` are met.|
| *Multiple project folders listed*        | Select the correct path in the dialog. If ambiguity persists, check your folder naming.     |
| *Cross-platform issues*                  | Only the backend/MCP server are portable; full GUI requires Windows/AutoHotkey.             |
| *Permission denied*                      | Run scripts as a user with sufficient access to the project root and subfolders.            |

---

## Contribution & Feedback

This project is actively stabilized and tested. Issues, feedback, and contributions are welcome – especially regarding cross-platform improvements, packaging, and advanced integrations.

---

## License

[Specify license here if applicable.]

---