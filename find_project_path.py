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

    Returns:
        str: Absolute path to the OneDrive directory.

    Exits:
        If UserProfile env var or OneDrive folder is missing.
    """
    user_profile = os.environ.get("UserProfile")
    if not user_profile:
        print_and_exit("ERROR:UserProfile environment variable not found")
    onedrive_path = os.path.join(user_profile, "OneDrive - Pro AV Solutions")
    if not os.path.isdir(onedrive_path):
        print_and_exit("ERROR:OneDrive folder not found")
    return onedrive_path

def get_project_folders(onedrive_path):
    """
    Get the Project Folders directory path.

    Args:
        onedrive_path (str): Path to OneDrive.

    Returns:
        str: Project Folders absolute path.

    Exits:
        If the directory is missing.
    """
    pf_path = os.path.join(onedrive_path, "Project Folders")
    if not os.path.isdir(pf_path):
        print_and_exit("ERROR:Project Folders not found")
    return pf_path

def get_range_folder(proj_num, pf_path):
    """
    Get the directory path for the thousand-range containing the project.

    Args:
        proj_num (str): 5-digit project number.
        pf_path (str): Path to 'Project Folders'.

    Returns:
        str: Range folder path (e.g. '10000 - 10999').

    Exits:
        If the range folder is missing.
    """
    num = int(proj_num)
    start = int(math.floor(num / 1000) * 1000)
    end = start + 999
    range_name = f"{start} - {end}"
    range_path = os.path.join(pf_path, range_name)
    if not os.path.isdir(range_path):
        print_and_exit("ERROR:Range folder not found")
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

def main():
    """
    Main entry point for the CLI tool.

    Accepts a single argument (5-digit project number), resolves the folder, and prints the result
    (or error/selection prompt) to stdout.

    Exits:
        With status message on success or error.
    """
    if len(sys.argv) != 2:
        print_and_exit("ERROR:Exactly one argument required (5-digit project number)")
    proj_num = validate_proj_num(sys.argv[1])
    onedrive_folder = get_onedrive_folder()
    pfolder = get_project_folders(onedrive_folder)
    range_folder = get_range_folder(proj_num, pfolder)
    matches = search_project_dirs(proj_num, range_folder)
    if not matches:
        print_and_exit("ERROR:No project folder found for that number")
    elif len(matches) == 1:
        print_and_exit(f"SUCCESS:{matches[0]}")
    else:
        print_and_exit("SELECT:" + "|".join(matches))

if __name__ == "__main__":
    main()