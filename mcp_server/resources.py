# Resource definition module for Project QuickNav MCP server

import os
import re
from mcp_server.server import mcp, error_handler

@mcp.resource("projectnav://folders")
@error_handler
def list_project_folders() -> dict:
    """
    Get a list of top-level folders and files in the current Project QuickNav workspace.
    Returns:
      dict with a 'entries' key, listing names and types for each entry.
    """
    root = os.path.abspath(os.getcwd())
    entries = []
    for name in os.listdir(root):
        fullpath = os.path.join(root, name)
        entries.append({
            "name": name,
            "type": "dir" if os.path.isdir(fullpath) else "file"
        })
    return {"entries": entries}


@mcp.resource("project://list")
@error_handler
def list_project_codes() -> dict:
    """
    Returns a list of all 5-digit project codes available in the user's Pro AV OneDrive Project Folders.
    """
    # Import here for possible mocking
    import os

    # Step 1: Find OneDrive path
    user_profile = os.environ.get("UserProfile")
    if not user_profile:
        raise RuntimeError("UserProfile environment variable not found")
    onedrive_path = os.path.join(user_profile, "OneDrive - Pro AV Solutions")
    if not os.path.isdir(onedrive_path):
        raise RuntimeError("OneDrive folder not found: " + onedrive_path)
    pf_path = os.path.join(onedrive_path, "Project Folders")
    if not os.path.isdir(pf_path):
        raise RuntimeError("Project Folders not found: " + pf_path)

    code_set = set()
    range_folders = [os.path.join(pf_path, d) for d in os.listdir(pf_path)
                     if os.path.isdir(os.path.join(pf_path, d)) and re.fullmatch(r"\d{5,} - \d{5,}", d)]
    for range_path in range_folders:
        for entry in os.listdir(range_path):
            match = re.match(r"^(\d{5}) - .+", entry)
            full_entry = os.path.join(range_path, entry)
            if match and os.path.isdir(full_entry):
                code_set.add(match.group(1))

    codes = sorted(code_set)
    return {"project_codes": codes}