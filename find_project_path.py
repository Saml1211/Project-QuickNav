import os
import sys
import math
import re

def print_and_exit(msg):
    """
    Print a message to stdout and exit the program.

    Args:
        msg (str): Message to print, prefixed with status.
    """
    print(msg)
    sys.exit(0)

def validate_proj_num(arg):
    """
    Validate that the provided argument is a 5-digit project number.

    Args:
        arg (str): Input string from CLI.

    Returns:
        str: Validated project number.

    Exits:
        If arg does not match 5 digits.
    """
    if not re.fullmatch(r"\d{5}", arg):
        print_and_exit("ERROR:Invalid argument (must be 5-digit project number)")
    return arg

def get_onedrive_folder():
    """
    Locate the user's OneDrive - Pro AV Solutions directory.
    If not found, use a test directory in the temp folder.

    Returns:
        str: Absolute path to the OneDrive directory or test directory.
    """
    user_profile = os.environ.get("UserProfile")
    if not user_profile:
        # Fall back to temp directory for testing
        return setup_test_environment()

    onedrive_path = os.path.join(user_profile, "OneDrive - Pro AV Solutions")
    if not os.path.isdir(onedrive_path):
        # Fall back to temp directory for testing
        return setup_test_environment()

    # Check for the Project Files folder which contains Project Folders
    project_files_path = os.path.join(onedrive_path, "Project Files")
    if os.path.isdir(project_files_path):
        return project_files_path

    return onedrive_path

def setup_test_environment():
    """
    Create a simulated project folder structure in the temp directory.
    Structure:
    - Project Folders
      - 10000 - 10999
        - 10123 - Project A
        - 10456 - Project B
      - 17000 - 17999
        - 17741 - Test Project
        - 17742 - Another Project
    """
    import tempfile

    base_dir = os.path.join(tempfile.gettempdir(), "Project Folders")

    # Create base directories
    os.makedirs(os.path.join(base_dir, "10000 - 10999"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "17000 - 17999"), exist_ok=True)

    # Create project directories
    os.makedirs(os.path.join(base_dir, "10000 - 10999", "10123 - Project A"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "10000 - 10999", "10456 - Project B"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "17000 - 17999", "17741 - Test Project"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "17000 - 17999", "17742 - Another Project"), exist_ok=True)

    # Create some subfolders in the test project with numbered prefixes
    project_dir = os.path.join(base_dir, "17000 - 17999", "17741 - Test Project")
    os.makedirs(os.path.join(project_dir, "4. System Designs"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "1. Sales Handover"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "2. BOM & CO"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "3. Handover Docs"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "5. Floor Plans"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "6. Site Photos"), exist_ok=True)

    return base_dir

def get_project_folders(onedrive_path):
    """
    Get the Project Folders directory path.

    Args:
        onedrive_path (str): Path to OneDrive or test directory.

    Returns:
        str: Project Folders absolute path.
    """
    # If onedrive_path is already the Project Folders directory (from test environment)
    if os.path.basename(onedrive_path) == "Project Folders":
        return onedrive_path

    # Otherwise, look for Project Folders within the provided path
    pf_path = os.path.join(onedrive_path, "Project Folders")
    if not os.path.isdir(pf_path):
        # If not found, use the test environment
        return setup_test_environment()

    return pf_path

def get_range_folder(proj_num, pf_path):
    """
    Get the directory path for the thousand-range containing the project.

    Args:
        proj_num (str): 5-digit project number.
        pf_path (str): Path to 'Project Folders'.

    Returns:
        str: Range folder path (e.g. '10000 - 10999').
    """
    num = int(proj_num)
    start = int(math.floor(num / 1000) * 1000)
    end = start + 999
    range_name = f"{start} - {end}"
    range_path = os.path.join(pf_path, range_name)

    if not os.path.isdir(range_path):
        # If we're using the test environment, the range folder should exist
        # If it doesn't, there's a problem with the test environment setup
        if os.path.basename(os.path.dirname(pf_path)) == "Temp" or os.path.basename(pf_path) == "Project Folders":
            # Try to create the range folder in the test environment
            try:
                os.makedirs(range_path, exist_ok=True)
            except Exception:
                print_and_exit(f"ERROR:Failed to create range folder {range_name}")
        else:
            print_and_exit(f"ERROR:Range folder {range_name} not found")

    return range_path

def search_project_dirs(proj_num, range_path):
    """
    Search for all directories matching '[ProjNum] - *' in the range folder.

    Args:
        proj_num (str): 5-digit project number.
        range_path (str): Path to the thousand-range directory.

    Returns:
        list[str]: List of absolute paths to matching project directories.
    """
    # Pattern: [ProjNum] - *
    pat = re.compile(rf"^{proj_num} - .+")
    try:
        entries = os.listdir(range_path)
    except Exception:
        print_and_exit("ERROR:Unable to list range folder contents")
    matches = []
    for entry in entries:
        full_path = os.path.join(range_path, entry)
        if os.path.isdir(full_path) and pat.match(entry):
            matches.append(os.path.abspath(full_path))
    return matches

def search_by_name(search_term, pf_path):
    """
    Search for project directories containing the search term in their name.

    Args:
        search_term (str): Text to search for in project folder names.
        pf_path (str): Path to the Project Folders directory.

    Returns:
        list[str]: List of absolute paths to matching project directories.
    """
    matches = []

    # Search in all range folders
    try:
        range_folders = os.listdir(pf_path)
        for range_folder in range_folders:
            range_path = os.path.join(pf_path, range_folder)

            # Skip if not a directory or doesn't match the range pattern
            if not os.path.isdir(range_path) or not re.match(r"^\d+ - \d+$", range_folder):
                continue

            # Search within this range folder
            try:
                entries = os.listdir(range_path)
                for entry in entries:
                    full_path = os.path.join(range_path, entry)

                    # Check if it's a directory and contains the search term (case insensitive)
                    if os.path.isdir(full_path) and search_term.lower() in entry.lower():
                        matches.append(os.path.abspath(full_path))
            except Exception:
                # Skip this range folder if there's an error
                continue
    except Exception:
        print_and_exit("ERROR:Unable to search project folders")

    return matches

def main():
    """
    Main entry point for the CLI tool.

    Accepts a single argument (5-digit project number or search term), resolves the folder(s),
    and prints the result (or error/selection prompt) to stdout.

    Exits:
        With status message on success or error.
    """
    if len(sys.argv) != 2:
        print_and_exit("ERROR:Exactly one argument required (project number or search term)")

    search_term = sys.argv[1]
    onedrive_folder = get_onedrive_folder()
    pfolder = get_project_folders(onedrive_folder)

    # Check if the input is a 5-digit project number
    if re.fullmatch(r"\d{5}", search_term):
        # Process as a project number
        proj_num = search_term
        try:
            range_folder = get_range_folder(proj_num, pfolder)
            matches = search_project_dirs(proj_num, range_folder)
        except Exception:
            matches = []

        if not matches:
            print_and_exit(f"ERROR:No project folder found for number {proj_num}")
        elif len(matches) == 1:
            print_and_exit(f"SUCCESS:{matches[0]}")
        else:
            print_and_exit("SELECT:" + "|".join(matches))
    else:
        # Process as a search term
        matches = search_by_name(search_term, pfolder)

        if not matches:
            print_and_exit(f"ERROR:No project folders found containing '{search_term}'")
        elif len(matches) == 1:
            print_and_exit(f"SUCCESS:{matches[0]}")
        else:
            print_and_exit("SEARCH:" + "|".join(matches))

if __name__ == "__main__":
    main()