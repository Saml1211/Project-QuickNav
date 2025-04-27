# Custom Orchestrator Rules: AHK/Python Desktop Utilities (like Project QuickNav)

When tasked with developing desktop utilities involving an AutoHotkey (AHK) frontend and a Python backend communicating via stdout:

1.  **Prioritize Backend First:** Delegate the implementation of the Python backend script (`find_project_path.py` or similar) as the first major subtask. This script often contains core logic needed by the frontend.
    *   **Delegate To:** Typically the "Code" mode (`code`) is appropriate for Python implementation.
    *   **Context Provided:** Ensure the `new_task` message for the Python subtask includes the detailed specifications for its logic, inputs (command-line args), required `stdout` output formats (`SUCCESS:`, `ERROR:`, `SELECT:`), standard library constraints, and the target OS (Windows) path handling requirements.

2.  **Frontend Implementation:** Once the Python backend subtask is completed and its results confirmed, delegate the implementation of the AHK frontend script (`lld_navigator.ahk` or similar).
    *   **Delegate To:** The "Code" mode (`code`) is likely suitable, assuming it has AHK proficiency. If a specialized AHK mode exists, consider that.
    *   **Context Provided:** Ensure the `new_task` message for the AHK subtask includes the detailed GUI specifications, the *exact* interaction protocol with the Python script (how to call it, how to parse `stdout`), logic for handling Python's different responses (including presenting `SELECT:` options), final path construction, subfolder verification logic, and error handling requirements. Explicitly mention the dependency on the completed Python script.

3.  **IPC Protocol is Key:** Emphasize the `stdout`-based communication protocol in the instructions for *both* subtasks. The Python task needs to know *what* to print, and the AHK task needs to know *how* to parse it.

4.  **Windows Environment:** Remind subtasks that the target environment is Windows, and path handling (e.g., using `os.path` correctly in Python, handling backslashes in AHK) is critical.

5.  **Modularity:** Reinforce the separation of concerns: Python finds the *main* path, AHK handles *all* UI, final path construction, and subfolder checks. Subtask instructions should reflect this strict separation.

6.  **Tool Usage:** Encourage subtasks (especially the "Code" mode) to use appropriate tools like `read_file` to understand existing code (if applicable later), `write_to_file` for generating the initial scripts, and `apply_diff` or `insert_content` for potential future modifications.

7.  **Completion Signal:** Ensure instructions for both subtasks mandate the use of `attempt_completion` with a clear summary of the implemented functionality upon successful completion.