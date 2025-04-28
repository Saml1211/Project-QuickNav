"""
Simple test script to verify the find_project_path.py script functionality.
This creates a dummy response that mimics a successful project path lookup.
"""

import sys
import os

def main():
    """
    Test function that mimics the behavior of find_project_path.py
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
    
    # For testing purposes, always return a SUCCESS response with a dummy path
    print(f"SUCCESS:C:\\Projects\\{job_number}\\Main Project")
    
    # Optional: Uncomment to test the SELECT response
    # print(f"SELECT:C:\\Projects\\{job_number}\\Path1|C:\\Projects\\{job_number}\\Path2")
    
    # Optional: Uncomment to test an ERROR response
    # print("ERROR:No project folder found for that number")

if __name__ == "__main__":
    main()