"""
Test script that simulates the behavior of find_project_path.py without requiring
the actual OneDrive folder structure. It creates a simulated project folder structure
in the temp directory and searches for projects there.
"""

import sys
import os
import re
import tempfile

# Create a simulated project folder structure in the temp directory
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

def find_project(proj_num, base_dir):
    """
    Find project directories matching the pattern '[ProjNum] - *'
    """
    # Determine the range folder
    num = int(proj_num)
    start = int((num // 1000) * 1000)
    end = start + 999
    range_name = f"{start} - {end}"
    range_path = os.path.join(base_dir, range_name)

    if not os.path.isdir(range_path):
        return []

    # Find matching projects
    pat = re.compile(rf"^{proj_num} - .+")
    matches = []

    try:
        entries = os.listdir(range_path)
        for entry in entries:
            full_path = os.path.join(range_path, entry)
            if os.path.isdir(full_path) and pat.match(entry):
                matches.append(os.path.abspath(full_path))
    except Exception:
        return []

    return matches

def main():
    """
    Test function that simulates the behavior of find_project_path.py
    """
    # Check if we have the right number of arguments
    if len(sys.argv) != 2:
        print("ERROR:Exactly one argument required (5-digit project number)")
        sys.exit(0)

    # Check if the argument is a 5-digit number
    job_number = sys.argv[1]
    if not job_number.isdigit() or len(job_number) != 5:
        print("ERROR:Invalid argument (must be 5-digit project number)")
        sys.exit(0)

    # Setup the test environment
    base_dir = setup_test_environment()

    # Find matching projects
    matches = find_project(job_number, base_dir)

    if not matches:
        print(f"ERROR:No project folder found for {job_number}")
    elif len(matches) == 1:
        print(f"SUCCESS:{matches[0]}")
    else:
        print("SELECT:" + "|".join(matches))

if __name__ == "__main__":
    main()