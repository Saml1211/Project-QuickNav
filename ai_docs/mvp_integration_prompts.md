# Prompts for MVP Integration Plan Execution

This document provides a set of effective, copy-paste-ready prompts for directing an AI assistant to execute each of the eight phases of the integration plan.

Each prompt clearly defines the scope, lists the specific actions required, provides the exact commit messages, and requests confirmation upon completion.

---

### **Prompt for PHASE 0: Preservation**

Execute **PHASE 0: Preservation** from our integration plan.

1.  Ensure you have a local copy or remote pointing to the `Project-QuickNav-MVP` repository.
2.  Check out its main branch.
3.  Create the tag `v1.0-mvp` at its current `HEAD`.
4.  Push the new tag to the remote origin to lock in the MVP's history.

Confirm once the tag is successfully created and pushed.

---

### **Prompt for PHASE 1: Integration Branch**

Proceed with **PHASE 1: Integration Branch**.

1.  Check out the `main` branch of the `Project-QuickNav` repository and pull the latest changes.
2.  Create a new branch named `integrate-mvp-refactor` from `main`.
3.  Push the new branch to origin and set it to track.

Confirm once the `integrate-mvp-refactor` branch exists on the remote.

---

### **Prompt for PHASE 2: Directory-Swap Import**

Begin **PHASE 2: Directory-Swap Import**.

1.  On the `integrate-mvp-refactor` branch, delete the existing root-level `quicknav/` directory.
2.  From your local `Project-QuickNav-MVP` checkout, copy the following into the root of the current project:
    *   `quicknav/` (the new package)
    *   `mcp_server/`
    *   `src/`
    *   `quicknav-1.0.0-win64.spec`
    *   The contents of `.github/workflows/` (overwrite if conflicts).
3.  Stage all changes.
4.  Commit everything with the exact message: `feat: import MVP refactor (directory-swap)`

Confirm once the commit is created.

---

### **Prompt for PHASE 3: Asset Re-integration from Main**

Execute **PHASE 3: Asset Re-integration from Main**.

1.  Use `git checkout main -- <path>` to restore the following assets from the `main` branch:
    *   The entire `tests/ahk/` directory.
    *   The debug helper scripts: `debug_helper.ahk` and `debug_quicknav.ahk`.
2.  Modify `setup.py`: ensure `packages=find_packages()` is set correctly and add `mcp[cli]` to the `install_requires` list.
3.  Stage all changes.
4.  Commit with the exact message: `chore: re-add tests, debug tools, update setup.py`

Confirm once the commit is created.

---

### **Prompt for PHASE 4: Import-Path & Code Fix-ups**

Proceed with **PHASE 4: Import-Path & Code Fix-ups**.

1.  Create a virtual environment if needed and run `pip install -e .` to install the project in editable mode.
2.  Execute the Python unit tests (`src/test_find_project_path.py` and `mcp_server/test_server.py`).
3.  Run the AHK test suite using `tests/ahk/run_all_tests.ahk`.
4.  Analyze any failures and apply necessary patches to fix import errors or incorrect paths in the tests and scripts.
5.  Once all tests pass, commit the fixes with the exact message: `fix: align tests to new package versions`

Report the test results and confirm when the commit is ready.

---

### **Prompt for PHASE 5: Build System Consolidation**

Begin **PHASE 5: Build System Consolidation**.

1.  Validate that the `quicknav-1.0.0-win64.spec` file and `scripts/build_exe.ps1` are correctly configured for the merged file structure.
2.  Attempt to build the project using the PyInstaller spec file.
3.  Apply any necessary fixes to the spec file or build scripts.
4.  Once the build succeeds, commit the changes with the exact message: `build: PyInstaller spec validated for merged tree`

Confirm once the build process works and the commit is created.

---

### **Prompt for PHASE 6: Continuous Integration**

Execute **PHASE 6: Continuous Integration**.

1.  Review the merged `.github/workflows/` files.
2.  Ensure the CI pipeline correctly:
    *   Installs dependencies (Python and AutoHotkey).
    *   Runs all Python and AHK tests.
    *   Builds the PyInstaller executable.
    *   Uploads the executable as a workflow artifact.
3.  Commit any adjustments with the exact message: `ci: unify workflows, run tests & build exe`

Confirm once the unified CI workflow file is complete and committed.

---

### **Prompt for PHASE 7 & 8: Finalization & PR**

We are now at the final stage: **PHASE 7 (PR & Sanity Check)** and **PHASE 8 (Post-Merge Cleanup)**.

1.  Push all your commits on the `integrate-mvp-refactor` branch to the remote.
2.  Generate a detailed summary for a pull request. The PR will merge `integrate-mvp-refactor` into `main`. The summary should mention the key changes, how the MVP history is preserved via the `v1.0-mvp` tag, and link to our integration plan rule.
3.  Finally, list the recommended cleanup tasks and follow-up issues to be created after the merge is complete, as outlined in Phase 8.

Provide the PR summary and the list of follow-up tasks. 