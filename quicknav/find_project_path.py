import os
import sys
import math
import re
import json
from datetime import datetime

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

def discover_documents(project_path):
    """
    Recursively find all PDF, Word, and RTF documents in a project folder.

    Args:
        project_path (str): Path to the project directory to search.

    Returns:
        list[str]: List of full file paths to found documents (.pdf, .docx, .doc, .rtf).
    """
    document_extensions = {'.pdf', '.docx', '.doc', '.rtf'}
    document_paths = []
    
    try:
        for root, dirs, files in os.walk(project_path):
            for file in files:
                # Get file extension in lowercase for case-insensitive comparison
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in document_extensions:
                    full_path = os.path.join(root, file)
                    document_paths.append(os.path.abspath(full_path))
    except Exception:
        # If there's an error accessing the directory, return empty list
        pass
    
    return document_paths

def training_script(base_path, output_file='training_data.json'):
    """
    Create a comprehensive catalog of all documents across all projects.

    Args:
        base_path (str): Path to the Project Folders directory or OneDrive path.
        output_file (str): Name of the JSON file to save results to.

    Returns:
        list[dict]: Training data list with project and document information.
    """
    training_data = []
    
    # Get the Project Folders path
    if os.path.basename(base_path) == "Project Folders":
        pf_path = base_path
    else:
        pf_path = get_project_folders(base_path)
    
    try:
        # Iterate through all range folders (e.g., "10000 - 10999")
        range_folders = os.listdir(pf_path)
        for range_folder in range_folders:
            range_path = os.path.join(pf_path, range_folder)
            
            # Skip if not a directory or doesn't match the range pattern
            if not os.path.isdir(range_path) or not re.match(r"^\d+ - \d+$", range_folder):
                continue
            
            try:
                # Get all project folders within this range
                project_entries = os.listdir(range_path)
                for project_entry in project_entries:
                    project_path = os.path.join(range_path, project_entry)
                    
                    # Skip if not a directory
                    if not os.path.isdir(project_path):
                        continue
                    
                    # Find all documents in this project folder
                    documents = discover_documents(project_path)
                    
                    # Add each document to the training data
                    for doc_path in documents:
                        document_name = os.path.basename(doc_path)
                        training_entry = {
                            "project_folder": project_entry,
                            "document_path": doc_path,
                            "document_name": document_name,
                            "extracted_info": {}
                        }
                        training_data.append(training_entry)
                        
            except Exception:
                # Skip this range folder if there's an error
                continue
                
    except Exception:
        print_and_exit("ERROR:Unable to access project folders for training data collection")
    
    # Save training data to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
    except Exception:
        print_and_exit(f"ERROR:Unable to save training data to {output_file}")
    
    return training_data

def get_training_data_filename(project_paths, is_search=False):
    """
    Generate a unique filename for training data based on project path(s).
    
    Args:
        project_paths (str or list): Single project path or list of project paths.
        is_search (bool): Whether this is from a search operation with multiple results.
    
    Returns:
        str: Full path to the training data JSON file.
    """
    # Create training_data directory if it doesn't exist
    training_dir = os.environ.get('QUICKNAV_TRAINING_DIR', 
                                 os.path.join(os.path.expanduser('~'), '.quicknav', 'training_data'))
    os.makedirs(training_dir, exist_ok=True)
    
    if isinstance(project_paths, str):
        # Single project
        project_name = os.path.basename(project_paths)
        # Extract project number if it matches the pattern
        match = re.match(r"^(\d{5}) - ", project_name)
        if match:
            project_num = match.group(1)
            filename = f"training_data_{project_num}.json"
        else:
            # Fallback to sanitized project name
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', project_name)
            filename = f"training_data_{safe_name}.json"
    else:
        # Multiple projects
        if is_search:
            # For search results, use timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_search_{timestamp}.json"
        else:
            # For multiple exact matches, combine project numbers
            project_nums = []
            for path in project_paths:
                project_name = os.path.basename(path)
                match = re.match(r"^(\d{5}) - ", project_name)
                if match:
                    project_nums.append(match.group(1))
            
            if project_nums:
                filename = f"training_data_{'_'.join(project_nums)}.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_data_multiple_{timestamp}.json"
    
    return os.path.join(training_dir, filename)

def main():
    """
    Main entry point for the CLI tool.

    Accepts a single argument (5-digit project number or search term), resolves the folder(s),
    and prints the result (or error/selection prompt) to stdout.
    Optionally accepts --training-data flag to generate training data after successful operation.

    Exits:
        With status message on success or error.
    """
    # Check for training data flag
    generate_training_data = False
    args = sys.argv[1:]
    
    if "--training-data" in args:
        generate_training_data = True
        args.remove("--training-data")
    
    if len(args) != 1:
        print_and_exit("ERROR:Exactly one argument required (project number or search term)")

    search_term = args[0]
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
            result_message = f"SUCCESS:{matches[0]}"
            print(result_message)
            # Generate training data if requested and operation was successful
            if generate_training_data:
                try:
                    # Generate training data only for this specific project
                    training_data = []
                    documents = discover_documents(matches[0])
                    project_name = os.path.basename(matches[0])
                    
                    for doc_path in documents:
                        document_name = os.path.basename(doc_path)
                        training_entry = {
                            "project_folder": project_name,
                            "document_path": doc_path,
                            "document_name": document_name,
                            "extracted_info": {}
                        }
                        training_data.append(training_entry)
                    
                    # Save training data to JSON file
                    filename = get_training_data_filename(matches[0])
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(training_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"TRAINING:Generated training data for project {project_name} with {len(training_data)} documents (saved to {filename})")
                except Exception as e:
                    print(f"TRAINING_ERROR:Failed to generate training data: {str(e)}")
            sys.exit(0)
        else:
            print_and_exit("SELECT:" + "|".join(matches))
    else:
        # Process as a search term
        matches = search_by_name(search_term, pfolder)

        if not matches:
            print_and_exit(f"ERROR:No project folders found containing '{search_term}'")
        elif len(matches) == 1:
            result_message = f"SUCCESS:{matches[0]}"
            print(result_message)
            # Generate training data if requested and operation was successful
            if generate_training_data:
                try:
                    # Generate training data only for this specific project
                    training_data = []
                    documents = discover_documents(matches[0])
                    project_name = os.path.basename(matches[0])
                    
                    for doc_path in documents:
                        document_name = os.path.basename(doc_path)
                        training_entry = {
                            "project_folder": project_name,
                            "document_path": doc_path,
                            "document_name": document_name,
                            "extracted_info": {}
                        }
                        training_data.append(training_entry)
                    
                    # Save training data to JSON file
                    filename = get_training_data_filename(matches[0])
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(training_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"TRAINING:Generated training data for project {project_name} with {len(training_data)} documents (saved to {filename})")
                except Exception as e:
                    print(f"TRAINING_ERROR:Failed to generate training data: {str(e)}")
            sys.exit(0)
        else:
            result_message = "SEARCH:" + "|".join(matches)
            print(result_message)
            # Generate training data if requested and operation was successful
            if generate_training_data:
                try:
                    # Generate training data for all found projects from search
                    training_data = []
                    for project_path in matches:
                        documents = discover_documents(project_path)
                        project_name = os.path.basename(project_path)
                        
                        for doc_path in documents:
                            document_name = os.path.basename(doc_path)
                            training_entry = {
                                "project_folder": project_name,
                                "document_path": doc_path,
                                "document_name": document_name,
                                "extracted_info": {}
                            }
                            training_data.append(training_entry)
                    
                    # Save training data to JSON file
                    filename = get_training_data_filename(matches, True)
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(training_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"TRAINING:Generated training data for {len(matches)} projects with {len(training_data)} documents (saved to {filename})")
                except Exception as e:
                    print(f"TRAINING_ERROR:Failed to generate training data: {str(e)}")
            sys.exit(0)

if __name__ == "__main__":
    main()