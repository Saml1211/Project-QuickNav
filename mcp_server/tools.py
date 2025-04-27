# Tool definition module for Project QuickNav MCP server

import subprocess
import sys
from mcp_server.server import mcp, error_handler
from mcp_server import user_data

@mcp.tool()
@error_handler
def navigate_project(project_number: str) -> dict:
    """
    Find the folder path(s) for a given 5-digit project number using QuickNav logic.
    Tracks usage in user history.
    Arguments:
      project_number: The 5-digit project number (string).
    Returns:
      dict with:
        - status: One of "success", "select", or "error"
        - path: The resolved path (if success)
        - paths: List of candidate paths (if select)
        - message: Error or info message
    """
    # Validate input
    if not isinstance(project_number, str) or not project_number.isdigit() or len(project_number) != 5:
        result = {"status": "error", "message": "project_number must be a 5-digit string"}
        user_data.add_history_entry({
            "action": "navigate_project",
            "project_code": project_number,
            "status": "error",
            "message": result["message"]
        })
        return result

    try:
        # Call the script as a subprocess
        res = subprocess.run(
            [sys.executable, "find_project_path.py", project_number],
            capture_output=True, text=True, check=True
        )
        output = res.stdout.strip()
    except subprocess.CalledProcessError as e:
        result = {"status": "error", "message": f"QuickNav script failed: {e.stderr.strip() or e}"}
        user_data.add_history_entry({
            "action": "navigate_project",
            "project_code": project_number,
            "status": "error",
            "message": result["message"]
        })
        return result

    if output.startswith("SUCCESS:"):
        result = {"status": "success", "path": output[len("SUCCESS:"):].strip()}
    elif output.startswith("SELECT:"):
        parts = output[len("SELECT:"):].strip().split("|")
        result = {"status": "select", "paths": [p.strip() for p in parts]}
    elif output.startswith("ERROR:"):
        result = {"status": "error", "message": output[len("ERROR:"):].strip()}
    else:
        result = {"status": "error", "message": "Unrecognized output from QuickNav script"}

    # Record attempt and outcome in user history
    entry = {
        "action": "navigate_project",
        "project_code": project_number,
        "status": result["status"]
    }
    if "path" in result:
        entry["path"] = result["path"]
    if "paths" in result:
        entry["paths"] = result["paths"]
    if "message" in result:
        entry["message"] = result["message"]
    user_data.add_history_entry(entry)
    return result
@mcp.tool()
@error_handler
def list_projects() -> dict:
    """
    Return a list of all available 5-digit project codes by calling the 'project://list' resource.
    """
    from mcp_server.resources import list_project_codes
    return list_project_codes()
# === User Preferences and History Tools ===
from mcp_server import user_data

@mcp.tool()
@error_handler
def get_user_preferences() -> dict:
    """
    Retrieve all user preferences as a dict.
    """
    prefs = user_data.get_preferences()
    return {"status": "success", "preferences": prefs}

@mcp.tool()
@error_handler
def set_user_preferences(preferences: dict) -> dict:
    """
    Set (replace) all user preferences from a dict.
    """
    if not isinstance(preferences, dict):
        return {"status": "error", "message": "Preferences must be a dict"}
    user_data.set_preferences(preferences)
    return {"status": "success"}

@mcp.tool()
@error_handler
def clear_user_preferences() -> dict:
    """
    Clear all user preferences.
    """
    user_data.clear_preferences()
    return {"status": "success"}

@mcp.tool()
@error_handler
def get_user_history() -> dict:
    """
    Retrieve usage history as a list.
    """
    hist = user_data.get_history()
    return {"status": "success", "history": hist}

@mcp.tool()
@error_handler
def add_user_history_entry(entry: dict) -> dict:
    """
    Add an entry to user history.
    """
    if not isinstance(entry, dict):
        return {"status": "error", "message": "Entry must be a dict"}
    user_data.add_history_entry(entry)
    return {"status": "success"}

@mcp.tool()
@error_handler
def clear_user_history() -> dict:
    """
    Clear all usage history.
    """
    user_data.clear_history()
    return {"status": "success"}

@mcp.tool()
@error_handler
def recommend_projects(max_projects: int = 5) -> dict:
    """
    Recommend most relevant project codes (by usage).
    """
    recs = user_data.recommend_projects_from_history(max_projects)
    return {"status": "success", "recommended_projects": recs}

@mcp.tool()
@error_handler
def get_quicknav_usage_diagnostics() -> dict:
    """
    Provide usage and error statistics for diagnostics.
    """
    hist = user_data.get_history()
    total = len(hist)
    # Example: count errors
    errors = [h for h in hist if "status" in h and h["status"] == "error"]
    recent = hist[:10]
    return {
        "status": "success",
        "total_history_entries": total,
        "recent_entries": recent,
        "error_count": len(errors)
    }