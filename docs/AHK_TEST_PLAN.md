# Project QuickNav - AutoHotkey Frontend Test Plan

This document outlines manual test cases for the `lld_navigator.ahk` script.

## 1. UI Interaction Tests

### Test Case 1.1: Launch GUI
- **Steps:**
    1. Double-click `lld_navigator.ahk` or run it via the AutoHotkey executable.
- **Expected Result:** The QuickNav GUI window appears, displaying an input field labeled "Enter Project Number:" and an "Open" button. The window should have focus.

### Test Case 1.2: Enter Project Number
- **Steps:**
    1. Launch the GUI.
    2. Click into the input field.
    3. Type a 5-digit number (e.g., "12345").
- **Expected Result:** The typed number "12345" appears correctly in the input field.

### Test Case 1.3: Click "Open" Button
- **Steps:**
    1. Launch the GUI.
    2. Enter a valid 5-digit project number (e.g., "12345") into the input field.
    3. Click the "Open" button with the mouse.
- **Expected Result:** The script attempts to execute `find_project_path.py` with "12345" as an argument. The GUI might briefly show a processing state. (Actual folder opening depends on subsequent tests).

### Test Case 1.4: Use Enter Key in Input Field
- **Steps:**
    1. Launch the GUI.
    2. Enter a valid 5-digit project number (e.g., "12345") into the input field.
    3. Press the Enter key while the input field has focus.
- **Expected Result:** Same outcome as clicking the "Open" button (Test Case 1.3). The script attempts to execute `find_project_path.py`.

### Test Case 1.5: Close GUI (X Button)
- **Steps:**
    1. Launch the GUI.
    2. Click the standard window close button ('X') in the title bar.
- **Expected Result:** The GUI window closes, and the `lld_navigator.ahk` script terminates without errors.

### Test Case 1.6: Close GUI (Escape Key)
- **Steps:**
    1. Launch the GUI.
    2. Press the Escape (Esc) key.
- **Expected Result:** The GUI window closes, and the `lld_navigator.ahk` script terminates without errors.

## 2. Input Validation Tests (Handled by AHK)

### Test Case 2.1: Invalid Length (Too Short)
- **Steps:**
    1. Launch the GUI.
    2. Enter "123" into the input field.
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears with an error like "Invalid input: Must be a 5-digit project number." No Python script execution is attempted.

### Test Case 2.2: Invalid Length (Too Long)
- **Steps:**
    1. Launch the GUI.
    2. Enter "123456" into the input field.
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears with an error like "Invalid input: Must be a 5-digit project number." No Python script execution is attempted.

### Test Case 2.3: Invalid Characters (Non-Digits)
- **Steps:**
    1. Launch the GUI.
    2. Enter "abcde" into the input field.
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears with an error like "Invalid input: Must be a 5-digit project number." No Python script execution is attempted.

### Test Case 2.4: Empty Input
- **Steps:**
    1. Launch the GUI.
    2. Leave the input field empty.
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears with an error like "Invalid input: Must be a 5-digit project number." No Python script execution is attempted.

## 3. Subprocess Handling Tests (Python Script Interaction)

*(These tests require the ability to control or mock the output of `find_project_path.py`)*

### Test Case 3.1: Python Script Success (Single Path)
- **Setup:** Configure the environment/mock `find_project_path.py` to return `SUCCESS:/path/to/your/project/12345 - Project Alpha` when run with argument "12345".
- **Steps:**
    1. Launch the GUI.
    2. Enter "12345".
    3. Click "Open" or press Enter.
- **Expected Result:** Windows Explorer opens directly to the directory `/path/to/your/project/12345 - Project Alpha`. The AHK GUI closes automatically.

### Test Case 3.2: Python Script Select (Multiple Paths)
- **Setup:** Configure the environment/mock `find_project_path.py` to return `SELECT:/path/a/54321 - Proj|/path/b/54321 - Proj Alt` when run with argument "54321".
- **Steps:**
    1. Launch the GUI.
    2. Enter "54321".
    3. Click "Open" or press Enter.
- **Expected Result:** A second AHK GUI window or menu appears, listing the two paths: "/path/a/54321 - Proj" and "/path/b/54321 - Proj Alt". Clicking one of the options opens Explorer to that specific path and closes both AHK windows. Clicking cancel/close on the selection window closes both windows without opening Explorer.

### Test Case 3.3: Python Script Error (e.g., Project Not Found)
- **Setup:** Configure the environment/mock `find_project_path.py` to return `ERROR:No project folder found for that number` when run with argument "99999".
- **Steps:**
    1. Launch the GUI.
    2. Enter "99999".
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears displaying the error message: "No project folder found for that number". The main GUI remains open.

### Test Case 3.4: Python Script Error (e.g., Range Folder Not Found)
- **Setup:** Configure the environment/mock `find_project_path.py` to return `ERROR:Range folder not found` when run with argument "88888".
- **Steps:**
    1. Launch the GUI.
    2. Enter "88888".
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears displaying the error message: "Range folder not found". The main GUI remains open.

### Test Case 3.5: Python Script Unexpected Output
- **Setup:** Configure the environment/mock `find_project_path.py` to return unexpected output (e.g., just "Some random text") when run with argument "77777".
- **Steps:**
    1. Launch the GUI.
    2. Enter "77777".
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears indicating an "Unexpected output from backend script" or similar error, possibly showing the raw output received. The main GUI remains open.

## 4. Error Scenarios (Environment/Setup)

### Test Case 4.1: `find_project_path.py` Not Found/Accessible
- **Setup:** Rename or move `find_project_path.py` so it's not in the same directory as `lld_navigator.ahk` or accessible via the path used in the script.
- **Steps:**
    1. Launch the GUI.
    2. Enter a valid project number (e.g., "12345").
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears indicating the backend script (`find_project_path.py`) could not be found or executed.

### Test Case 4.2: Python Executable Not Found/Accessible
- **Setup:** Modify the `lld_navigator.ahk` script to point to an invalid Python executable path, or ensure the default `py.exe` (if used) is not available.
- **Steps:**
    1. Launch the GUI.
    2. Enter a valid project number (e.g., "12345").
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears indicating that the Python executable could not be run.

### Test Case 4.3: OneDrive / Project Folders Not Found (Handled by Python)
- **Note:** These environment issues within the Python script's logic should be caught by the Python script itself. The AHK script's role is to display the error message returned by Python.
- **Setup:** Configure the user's environment such that `find_project_path.py` would fail and return `ERROR:OneDrive folder not found` or `ERROR:Project Folders not found` or `ERROR:UserProfile environment variable not found`.
- **Steps:**
    1. Launch the GUI.
    2. Enter a valid project number (e.g., "12345").
    3. Click "Open" or press Enter.
- **Expected Result:** An AHK message box appears displaying the specific error message received from the Python script's output (e.g., "ERROR:OneDrive folder not found"). The main GUI remains open.