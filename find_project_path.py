import os
import sys
import math
import re
import tempfile

def print_status(msg_type: str, msg: str) -> None:
    """
    Print a protocol-prefixed message to stdout and exit.

    Args:
        msg_type (str): One of 'SUCCESS', 'ERROR', 'SELECT'.
        msg (str): Message content.
    """
    print(f"{msg_type}:{msg}")
    sys.exit(0)

def validate_proj_num(arg: str) -> str:
    """
    Validate that the provided argument is a 5-digit project number.

    Args:
        arg (str): Input string from CLI.

    Returns:
        str: Validated project number.

    Raises:
        ValueError: If argument is not a 5-digit number.
    """
    if not re.fullmatch(r"\d{5}", arg):
        print_status("ERROR", "Invalid input: argument must be a 5-digit project number (numbers only)")
    return arg

def validate_search_term(term: str) -> str:
    """
    Validate and sanitize the search term for folder lookup.
    Ensures term does not contain shell metacharacters or dangerous input.

    Args:
        term (str): The user-provided search term.

    Returns:
        str: Sanitized search term.

    Raises:
        ValueError: If term is empty or contains disallowed patterns.
    """
    # Disallow metacharacters commonly used for shell injection
    if not term or len(term) > 100:
        print_status("ERROR", "Search term must be non-empty and under 100 characters")
    if re.search(r"[^a-zA-Z0-9_\-\s.]", term):
        print_status("ERROR", "Search term contains disallowed special characters")
    return term

def get_onedrive_folder() -> str:
    """
    Locate the user's OneDrive - Pro AV Solutions directory.
    If not found, use a test directory in the temp folder with clear status messaging.

    Returns:
        str: Absolute path to the OneDrive directory or test directory.
    """
    user_profile = os.environ.get("UserProfile")
    if not user_profile:
        print_status("ERROR", "Environment variable 'UserProfile' not found. Cannot locate OneDrive.")
    onedrive_path = os.path.join(user_profile, "OneDrive - Pro AV Solutions")
    if not os.path.isdir(onedrive_path):
        print_status("ERROR", f"OneDrive path not found at: {onedrive_path}")
    project_files_path = os.path.join(onedrive_path, "Project Files")
    if os.path.isdir(project_files_path):
        return project_files_path
    return onedrive_path

def get_project_folders(onedrive_path: str) -> str:
    """
    Get the Project Folders directory path within OneDrive.

    Args:
        onedrive_path (str): Path to OneDrive.

    Returns:
        str: Project Folders absolute path.

    Raises:
        FileNotFoundError: If 'Project Folders' does not exist.
    """
    pf_path = os.path.join(onedrive_path, "Project Folders")
    if not os.path.isdir(pf_path):
        print_status("ERROR", f"'Project Folders' directory missing at expected location: {pf_path}")
    return pf_path

def get_range_folder(proj_num: str, pf_path: str) -> str:
    """
    Get the directory path for the thousand-range containing the project.

    Args:
        proj_num (str): 5-digit project number.
        pf_path (str): Path to 'Project Folders'.

    Returns:
        str: Range folder path (e.g., '10000 - 10999').

    Raises:
        FileNotFoundError: If range folder does not exist.
    """
    try:
        num = int(proj_num)
    except Exception:
        print_status("ERROR", "Project number must be numeric.")
    start = int(math.floor(num / 1000) * 1000)
    end = start + 999
    range_name = f"{start} - {end}"
    range_path = os.path.join(pf_path, range_name)
    if not os.path.isdir(range_path):
        print_status("ERROR", f"Range folder '{range_name}' not found in 'Project Folders'.")
    return range_path

def search_project_dirs(proj_num: str, range_path: str) -> list:
    """
    Search for all directories matching '[ProjNum] - *' in the range folder.

    Args:
        proj_num (str): 5-digit project number.
        range_path (str): Path to the thousand-range directory.

    Returns:
        list[str]: List of absolute paths to matching project directories.
    """
    pat = re.compile(rf"^{proj_num} - .+")
    try:
        entries = os.listdir(range_path)
    except Exception as e:
        print_status("ERROR", f"Unable to list range folder contents: {str(e)}")
    matches = []
    for entry in entries:
        full_path = os.path.join(range_path, entry)
        if os.path.isdir(full_path) and pat.match(entry):
            matches.append(os.path.abspath(full_path))
    return matches

def search_by_name(search_term: str, pf_path: str) -> list:
    """
    Search for project directories containing the search term in their name.

    Args:
        search_term (str): Text to search for in project folder names.
        pf_path (str): Path to the Project Folders directory.

    Returns:
        list[str]: List of absolute paths to matching project directories.
    """
    matches = []
    try:
        range_folders = os.listdir(pf_path)
    except Exception as e:
        print_status("ERROR", f"Unable to list folders in Project Folders: {str(e)}")
    for range_folder in range_folders:
        range_path = os.path.join(pf_path, range_folder)
        if not os.path.isdir(range_path) or not re.match(r"^\d+ - \d+$", range_folder):
            continue
        try:
            entries = os.listdir(range_path)
        except Exception:
            continue  # Skip ranges with unreadable contents
        for entry in entries:
            full_path = os.path.join(range_path, entry)
            # Check if it's a directory and contains the search term (case insensitive)
            if os.path.isdir(full_path) and search_term.lower() in entry.lower():
                matches.append(os.path.abspath(full_path))
    return matches

def main():
    """
    Main entry point for the CLI tool.

    Accepts a single argument (5-digit project number or search term), resolves the folder(s),
    and prints the result (or error/selection prompt) to stdout using the protocol.

    Exits:
        With status message on success or error.
    """
    if len(sys.argv) != 2:
        print_status("ERROR", "Exactly one argument required: either a 5-digit project number or a project name search term.")
    arg = sys.argv[1].strip()

    # Strict input validation
    is_proj_num = bool(re.fullmatch(r"\d{5}", arg))
    if is_proj_num:
        proj_num = validate_proj_num(arg)
    else:
        search_term = validate_search_term(arg)

    # Environment and folder checks
    onedrive_folder = get_onedrive_folder()
    pfolder = get_project_folders(onedrive_folder)

    if is_proj_num:
        # Project number lookup
        range_folder = get_range_folder(proj_num, pfolder)
        matches = search_project_dirs(proj_num, range_folder)
        if not matches:
            print_status("ERROR", f"No project folder found for number {proj_num}. Please check the number or verify folder existence.")
        elif len(matches) == 1:
            print_status("SUCCESS", matches[0])
        else:
            print_status("SELECT", "|".join(matches))
    else:
        # Search term lookup
        matches = search_by_name(search_term, pfolder)
        if not matches:
            print_status("ERROR", f"No project folders found containing '{search_term}'. Please check spelling or try a different term.")
        elif len(matches) == 1:
            print_status("SUCCESS", matches[0])
        else:
            print_status("SELECT", "|".join(matches))

if __name__ == "__main__":
    main()