# Project QuickNav

## Project Overview

"Project QuickNav" is a comprehensive project navigation and intelligence platform. It started as a utility for navigating project folders using a 5-digit code system and has evolved to include:

*   **An intelligent GUI:** Built with Python's Tkinter library, providing a user-friendly interface for project navigation.
*   **Machine Learning-powered features:** A recommendation engine suggests relevant projects based on user behavior, and an analytics dashboard provides insights into project usage.
*   **AI-powered search:** A natural language interface allows users to search for projects and documents using conversational queries.
*   **MCP Server:** A backend server that exposes AI and automation capabilities as a service.
*   **AutoHotkey Integration:** Some legacy features and hotkeys are implemented using AutoHotkey.

The project is well-structured, with separate directories for the main application (`quicknav`), the MCP server (`mcp_server`), documentation (`docs`), and tests (`tests`).

## Building and Running

### Dependencies

*   **Core application:** `pip install -e .`
*   **Machine Learning:** `pip install scikit-learn pandas numpy matplotlib watchdog`
*   **AI Features:** `pip install litellm keyboard pystray`
*   **MCP Server:** `pip install -r mcp_server/requirements.txt`

### Running the application

*   **Main GUI:**
    ```bash
    python quicknav/gui_launcher.py
    ```
*   **MCP Server:**
    ```bash
    python -m mcp_server
    ```

### Testing and Linting

*   **Run all tests:**
    ```bash
    pytest
    ```
*   **Linting:**
    ```bash
    ruff check .
    ```
*   **Type-checking:**
    ```bash
    mypy mcp_server/ quicknav/
    ```

## Development Conventions

The project follows a standard set of development conventions, as outlined in `CONTRIBUTING.md`:

*   **Branching:** The project uses a `main`/`dev` branching model. All new development should be done in `feature/*` or `bugfix/*` branches off of `dev`.
*   **Code Style:** Python code should adhere to the PEP8 style guide, enforced by the `ruff` linter.
*   **Testing:** All new features and bug fixes must include corresponding tests.
*   **Pull Requests:** Pull requests should be made against the `dev` branch and require at least one approving review and passing CI checks before being merged.
