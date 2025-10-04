Execute a comprehensive refactoring and bug-fixing pass on the Project QuickNav Tkinter application, which currently uses a modern, modular architecture with separated concerns including GUIState for centralized state management, LayoutManager for DPI scaling and responsive layout, EventCoordinator for event handling, and specialized section components for individual UI areas.

### **Priority 1: Critical Functional Bugs**

**1.1. Fix Fatal `AttributeError` on Document Search**

* **Problem:** The application crashes when the "Find Document" button is clicked due to a missing `search_documents` method in the `GuiController` class, despite being called in `gui.py` at line 505. The existing `navigate_to_document` method handles document navigation but doesn't provide the expected `search_documents` interface.
* **Reproduction:**
    1. Launch the application.
    2. Enter a project number (e.g., `20797`).
    3. Select the "Find Documents" radio button.
    4. Choose any "Document Type" from the dropdown.
    5. Click the "Find Document" button.
* **Action:** Implement the missing `search_documents` method in the `GuiController` class that accepts project ID, document type, and filter parameters, executes the search logic using the existing `navigate_to_document` method, and returns results compatible with the current UI expectations. Ensure the method signature matches the calling code in `gui.py`.

**1.2. Resolve `TypeError` in Settings Menu and Remove Redundancy**

* **Problem:** The application crashes when trying to open the settings window due to incorrect argument count passed to `SettingsDialog._init_()`. The SettingsDialog class expects specific parameters that aren't being provided correctly. Additionally, the "Settings" option appears redundantly in both File and Help menus.
* **Reproduction:**
    1. Launch the application.
    2. Click `File > Settings` OR `Help > Settings`. Both actions trigger the crash.
* **Action:**
    1. Correct the instantiation of the `SettingsDialog` class in `gui.py` to pass the correct number and type of arguments required by its `__init__` method, ensuring proper parent window, settings manager, and theme manager parameters.
    2. Remove the duplicate "Settings" entry from the `Help` menu, keeping it only under the `File` menu following standard GUI conventions.

---

### **Priority 2: Major UI/UX and Usability Flaws**

**2.1. Redesign Core Interaction Model and State Management**

* **Problem:** The UI state management is confusing with multiple conflicting action buttons enabled simultaneously, and the UI doesn't reset after action completion. The current `ActionButtonsSection` shows different buttons for folder vs document modes but doesn't provide a unified interaction model.
* **Action:**
    1. **Consolidate Action Buttons:** Remove the separate "Open Folder" and "Find Document" buttons from `ActionButtonsSection`. Replace them with a single, context-aware button (e.g., labeled "Go" or "Navigate") that dynamically updates its text and command based on the selected radio button ("Open Project Folder" or "Find Documents").
    2. **Enforce Input Validation:** The consolidated action button must be `disabled` by default and only become `normal` when a valid project number is entered in the "Project Input" field, integrating with the existing validation system in `EventCoordinator`.
    3. **Reset UI After Actions:** After a successful action, implement UI reset logic that clears the "Project Input" field, resets dropdowns to default values, and updates the status bar with success confirmation.

**2.2. Provide Comprehensive User Feedback**

* **Problem:** The application provides minimal feedback on operations. When a folder is successfully opened, the UI remains static, leaving users uncertain about action success. The "Recent" projects feature exists but isn't properly populated or persisted.
* **Action:**
    1. **Implement a Status Bar:** Enhance the existing `StatusSection` to display persistent, helpful messages: "Ready", "Please enter a project number", "Navigating to project '20797'...", "Successfully opened folder.", "No documents found for the selected criteria." Include progress indicators during background operations.
    2. **Activate and Persist Recent Projects:** Implement logic in `SettingsManager` to track successfully accessed projects using the existing `add_recent_project` method. On success, add the project number to the top of the "Recent" list with proper metadata including access timestamps and project paths.

**2.3. Restructure Layout for Ergonomics and Clarity**

* **Problem:** The layout is cluttered with controls not grouped by function, forcing users' eyes to jump around. Inconsistent spacing exists between elements, and the grid layout doesn't properly expand horizontally.
* **Action:**
    1. **Use `ttk.Labelframe` for Grouping:** Reorganize the entire layout in `gui.py` into logical `Labelframe` sections:
        * `Project`: Contains the "Project Input" field and the "Recent" projects list from `ProjectInputSection`.
        * `Navigation Mode`: Contains the "Open Project Folder" and "Find Documents" radio buttons from `NavigationModeSection`.
        * `Actions & Filters`: A central frame that contains the contextual controls (subfolder list from `FolderModeSection` OR document filters from `DocumentModeSection`) and the new consolidated action button.
        * `Utilities`: Contains the "Options" and "AI Assistant" sections from `OptionsSection` and `AIAssistantSection`.
    2. **Improve Spacing:** Apply consistent `padx` and `pady` values (e.g., `padx=10, pady=5`) to all frames and widgets using the existing `LayoutManager.get_consistent_padding()` and `get_consistent_spacing()` methods.
    3. **Enhance Window Resizing:** Configure the layout's grid weights in `gui.py` so that central frames expand horizontally, utilizing the existing `LayoutManager` for responsive behavior and preventing controls from bunching up in narrow columns.

---

### **Priority 3: Code Quality, Polish, and Best Practices**

**3.1. Implement Robust, User-Friendly Error Handling**

* **Problem:** All exceptions currently crash the application and expose raw tracebacks in the console, despite having logging infrastructure in place.
* **Action:** Wrap all controller calls within the view in `try...except` blocks. On any exception, use `tkinter.messagebox.showerror` to display user-friendly error dialogs (e.g., "Operation Failed. Check logs for details.") and use the existing `logging` module to write full tracebacks to `app.log` with appropriate log levels.

**3.2. Refactor Monolithic GUI Class toward MVC**

* **Problem:** While the application already has a good modular structure, the `gui.py` file contains business logic mixed with UI concerns, making it harder to maintain than the already well-separated controller and model components.
* **Action:** Further refactor toward a strict Model-View-Controller (MVC) pattern:
  * **View (`gui.py`):** Restrict this file to widget creation, layout management using `LayoutManager`, and binding commands to the controller. Remove any remaining business logic and delegate all operations to the controller.
  * **Controller (`gui_controller.py`):** This class already handles event logic well and acts as an intermediary. Enhance it with the missing `search_documents` method and improve error handling.
  * **Model (`gui_state.py` and `gui_settings.py`):** These already handle data operations effectively. Enhance `gui_settings.py` to better manage recent projects and add project categorization features.

**3.3. Standardize Tooltips and Affordances**

* **Problem:** While some elements have tooltips via the existing tooltip system, others don't, creating inconsistent user experience. The purpose of checkboxes like "Booms" or "CD" in the options section isn't immediately clear.
* **Action:** Implement tooltips for every interactive element using the existing `_add_tooltip` method in `gui.py`. This includes all buttons, checkboxes, dropdown menus, and input fields to clarify their purpose, especially for domain-specific options like "Booms" (likely room filters) and "CD" (likely change document filters).

**3.4. Enhance Background Task Management**

* **Problem:** While the application already uses threading for background operations, the task management could be more robust with better error recovery and user feedback.
* **Action:** Improve the existing background task processing in `gui.py` by adding timeout handling, retry mechanisms for failed operations, and better progress reporting through the status bar. Ensure all background operations properly update the UI state and handle cancellation gracefully.

**3.5. Implement Comprehensive Input Validation**

* **Problem:** While basic validation exists in `EventCoordinator`, edge cases and user input errors aren't handled comprehensively.
* **Action:** Enhance the existing validation system to handle edge cases like empty inputs, invalid project numbers, malformed search terms, and provide immediate feedback through the status bar. Integrate validation feedback with the consolidated action button state management.

**3.6. Add Keyboard Navigation and Accessibility**

* **Problem:** While the application has some keyboard shortcuts, comprehensive keyboard navigation and accessibility features are missing.
* **Action:** Enhance the existing `EventCoordinator` to provide comprehensive keyboard navigation, ensuring all interactive elements are accessible via keyboard, and add proper focus management and tab order throughout the interface.

**3.7. Implement Data Persistence and Recovery**

* **Problem:** While settings are persisted, operational data like recent projects and user preferences aren't robustly saved and restored.
* **Action:** Enhance the existing `SettingsManager` to implement robust data persistence for recent projects, user preferences, and application state. Add backup and recovery mechanisms for critical data.

**3.8. Add Performance Monitoring and Optimization**

* **Problem:** The application lacks performance monitoring and optimization features despite having caching mechanisms.
* **Action:** Implement performance monitoring using the existing logging infrastructure to track operation times, cache hit rates, and resource usage. Add optimization features like lazy loading for heavy components and memory management for large result sets.

**3.9. Enhance Error Reporting and Debugging**

* **Problem:** While logging exists, error reporting and debugging capabilities are limited for end users and developers.
* **Action:** Enhance the existing logging system to provide detailed error reports, debugging information, and user-friendly error messages. Add diagnostic features that can be enabled through the settings interface.

**3.10. Implement Comprehensive Testing Framework**

* **Problem:** While some test files exist, comprehensive testing coverage is missing for the refactored components.
* **Action:** Create a comprehensive testing framework that covers all major components, including unit tests for the controller, integration tests for the UI components, and end-to-end tests for complete workflows. Ensure tests cover both success and failure scenarios for all major features.