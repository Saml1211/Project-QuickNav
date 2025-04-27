# Project QuickNav - Testing Guide

This document provides instructions for running tests for the Project QuickNav utility and guidelines for contributing new tests.

## Test Suites Overview

Project QuickNav uses a combination of automated and manual tests:

1.  **Python Backend Unit Tests (`test_find_project_path.py`):** These tests use Python's `unittest` framework to verify the logic of individual functions within `find_project_path.py` in isolation, mocking external dependencies like filesystem access and environment variables.
2.  **MCP Server Integration Tests (`mcp_server/test_server.py`):** These tests use `unittest` to verify the MCP server's tools and resources. They specifically test the integration between the MCP tool (`navigate_project`) and the `find_project_path.py` script by mocking the `subprocess` call, and test the `list_project_folders` resource by mocking `os` functions.
3.  **AutoHotkey Frontend Manual Tests (`AHK_TEST_PLAN.md`):** Due to the nature of GUI testing in AutoHotkey, these tests are designed to be performed manually. They cover UI interactions, input validation, and how the AHK script handles various outputs and errors from the Python backend.

## Running Automated Tests (Python)

The Python unit and integration tests can be run using the `unittest` discovery feature from the project's root directory.

**Prerequisites:**
*   Python 3 installed.
*   Dependencies for the MCP server installed (if applicable, although tests currently mock dependencies). Check `mcp_server/requirements.txt`.

**Command:**

Navigate to the root directory of the `Project-QuickNav` project in your terminal and run:

```bash
python -m unittest discover -v
```

*   `-m unittest`: Runs the `unittest` module.
*   `discover`: Automatically finds tests within the current directory and subdirectories (it will find `test_find_project_path.py` and `mcp_server/test_server.py`).
*   `-v`: Enables verbose output, showing results for each individual test case.

**Expected Output:**
You should see output indicating the number of tests run and whether they passed or failed. For example:

```
test_get_onedrive_folder_no_userprofile (test_find_project_path.TestFindProjectPath) ... ok
test_get_onedrive_folder_not_found (test_find_project_path.TestFindProjectPath) ... ok
# ... many more lines ...
test_list_project_folders_mixed (mcp_server.test_server.TestMCPServerIntegration) ... ok
test_navigate_project_invalid_input_length (mcp_server.test_server.TestMCPServerIntegration) ... ok
# ... etc ...

----------------------------------------------------------------------
Ran XX tests in Y.YYYs

OK
```
*(Where XX is the total number of tests and Y.YYY is the time taken)*

## Running Manual Tests (AutoHotkey)

Refer to the detailed steps and expected results outlined in `AHK_TEST_PLAN.md`.

**Prerequisites:**
*   AutoHotkey installed.
*   A configured environment that allows `find_project_path.py` to run (or mocks/modifications to simulate its different outputs as described in the test plan).

**Procedure:**
Follow the steps for each test case in `AHK_TEST_PLAN.md` and verify that the actual results match the expected results.

## Test Coverage Overview

*   **`find_project_path.py`:** All functions (`validate_proj_num`, `get_onedrive_folder`, `get_project_folders`, `get_range_folder`, `search_project_dirs`, `main`) are covered by unit tests in `test_find_project_path.py`. Tests cover valid inputs, invalid inputs, error conditions (e.g., folders not found, OS errors), and edge cases. Mocking is used extensively to isolate the script from the actual filesystem and environment.
*   **`mcp_server/`:**
    *   `tools.py` (`navigate_project`): Tested in `mcp_server/test_server.py`. Covers input validation and simulates all possible output scenarios from the backend script (`SUCCESS`, `SELECT`, `ERROR`, unexpected) and subprocess errors.
    *   `resources.py` (`list_project_folders`): Tested in `mcp_server/test_server.py`. Covers listing files/directories in various scenarios (mixed content, empty directory) and OS errors during listing. Mocking is used to simulate filesystem interactions.
*   **`lld_navigator.ahk`:** Covered by the manual test plan in `AHK_TEST_PLAN.md`, including UI elements, input validation, handling of backend script outputs, and environment error conditions.

## Adding New Tests

### Python Tests (`unittest`)
1.  **Identify the need:** Add tests when adding new functions, modifying existing logic, or fixing bugs.
2.  **Locate the file:** Add unit tests for `find_project_path.py` to `test_find_project_path.py`. Add integration tests for MCP components to `mcp_server/test_server.py`.
3.  **Write the test:**
    *   Create a new method within the appropriate `unittest.TestCase` subclass (e.g., `TestFindProjectPath` or `TestMCPServerIntegration`).
    *   Method names should start with `test_`.
    *   Use `unittest.mock.patch` to mock dependencies (filesystem, subprocesses, other functions).
    *   Use `self.assertEqual()`, `self.assertTrue()`, `self.assertRaises()`, etc., to make assertions about the expected behavior.
    *   Ensure mocks are checked (e.g., `mock_function.assert_called_once_with(...)`).
4.  **Run tests:** Run `python -m unittest discover -v` to ensure your new test passes and doesn't break existing ones.

### AutoHotkey Manual Tests
1.  **Identify the need:** Add tests for new UI features, changes in interaction logic, or new error handling cases.
2.  **Edit the file:** Open `AHK_TEST_PLAN.md`.
3.  **Add the test case:**
    *   Add a new section or subsection following the existing structure.
    *   Clearly define the **Setup** (if any specific environment is needed), **Steps** to perform, and the **Expected Result**.