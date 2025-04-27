"""
User data management for preferences and history (Project QuickNav MCP server).

Stores user preferences and usage history in JSON format under user home dir.
Encapsulates all read/write/clear and update operations.
"""

import os
import json
from typing import Any, Dict, List, Optional

USER_DATA_FILENAME = ".quicknav_userdata.json"

def get_user_data_path() -> str:
    """Returns the full path to the JSON file used for user data."""
    home = os.path.expanduser("~")
    return os.path.join(home, USER_DATA_FILENAME)

def read_user_data() -> Dict[str, Any]:
    """Reads user data JSON from disk or returns default dict if missing/corrupt."""
    path = get_user_data_path()
    if not os.path.exists(path):
        return {"preferences": {}, "history": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid user data format")
        if "preferences" not in data:
            data["preferences"] = {}
        if "history" not in data:
            data["history"] = []
        return data
    except Exception:
        return {"preferences": {}, "history": []}

def write_user_data(data: Dict[str, Any]) -> None:
    """Writes the full user data dict to disk."""
    path = get_user_data_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_preferences() -> Dict[str, Any]:
    return read_user_data().get("preferences", {})

def set_preferences(prefs: Dict[str, Any]) -> None:
    d = read_user_data()
    d["preferences"] = prefs
    write_user_data(d)

def clear_preferences() -> None:
    d = read_user_data()
    d["preferences"] = {}
    write_user_data(d)

def get_history() -> List[Dict[str, Any]]:
    return read_user_data().get("history", [])

def add_history_entry(entry: Dict[str, Any]) -> None:
    d = read_user_data()
    hist = d.get("history", [])
    hist.insert(0, entry)
    # Optionally enforce a max length for history
    d["history"] = hist[:100]
    write_user_data(d)

def clear_history() -> None:
    d = read_user_data()
    d["history"] = []
    write_user_data(d)

def recommend_projects_from_history(max_projects: int = 5) -> List[str]:
    """
    Recommend most frequently or recently used project codes from history.
    Returns a list of unique project codes (most recent/frequent first).
    """
    hist = get_history()
    code_counts = {}
    code_order = []
    for entry in hist:
        code = entry.get("project_code")
        if not code:
            continue
        if code not in code_counts:
            code_counts[code] = 0
            code_order.append(code)
        code_counts[code] += 1
    # Sort by frequency, breaking ties by recency
    sorted_codes = sorted(code_order, key=lambda c: (-code_counts[c], code_order.index(c)))
    return sorted_codes[:max_projects]