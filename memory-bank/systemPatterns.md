1 | # System Patterns – Project QuickNav
 2 | 
 3 | ## Architecture Overview
 4 | 
 5 | Project QuickNav utilizes a three-tier architecture, each layer with a distinct responsibility:
 6 | 
 7 | 1. **Python Backend (`find_project_path.py`):**  
 8 |    Central logic for locating project directories based on a 5-digit code.
 9 | 
10 | 2. **AutoHotkey (AHK) Frontend (`lld_navigator.ahk`):**  
11 |    Provides the user interface and handles local user interaction.
12 | 
13 | 3. **MCP Server (`mcp_server/`):**  
14 |    Exposes backend navigation functions to AI agents or external systems through the Model Context Protocol.
15 | 
16 | ## Key Technical Decisions
17 | 
18 | - **Stdout-based IPC (Inter-Process Communication):**  
19 |   The Python backend communicates results via standard output, enabling robust and platform-neutral IPC. This choice maximizes compatibility and simplifies integration with both the AHK frontend and MCP server.
20 | 
21 | - **Loose Coupling:**  
22 |   Each component interacts only via clearly defined interfaces (CLI/stdout or MCP tools), minimizing dependencies and simplifying maintenance.
23 | 
24 | - **Explicit Separation of Concerns:**  
25 |   - The backend is focused purely on directory resolution.
26 |   - The frontend is concerned only with UI/UX and user actions.
27 |   - The MCP server bridges AI and automation workflows to the backend logic.
28 | 
29 | ## Component Relationships
30 | 
31 | ### Data Flow
32 | 
33 | 1. **User-initiated Flow:**
34 |    - User enters the project code into the AHK interface.
35 |    - AHK script invokes the Python backend on-demand, passing the job number as an argument.
36 |    - Backend returns resolved path via stdout.
37 |    - AHK presents the result for user action.
38 | 
39 | 2. **AI-initiated Flow:**
40 |    - AI agent connects via MCP to the server.
41 |    - MCP server invokes Python backend on-demand with the proper job number.
42 |    - Path is returned to the AI agent, enabling automated operations.
43 | 
44 | ### Diagram (Textual)
45 | 
46 | `User ⇄ [AHK Frontend] ⇄ [Python Backend]`
47 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  ↑  
48 | `AI / Automation ⇄ [MCP Server] ⇄ [Python Backend]`
49 | 
50 | ## Integration Test System Pattern
51 | 
52 | - **Test Organization:** All AHK-based integration tests are located in `tests/ahk/`, with each scenario as a discrete script following the `test_*.ahk` naming convention for easy discovery and automation.
53 | 
54 | - **Test Runner:** `run_all_tests.ahk` automatically locates and executes all non-utility test scripts in the directory, aggregates results, and returns an exit code and summary suitable for CI pipelines.
55 | 
56 | - **Shared Utilities:** Common assertions and helpers are centralized in `test_utils.ahk`. This encourages DRY patterns and makes test extension straightforward.
57 | 
58 | - **Maintainability Pattern:** Test scripts are modular and isolated, allowing contributors to add new edge cases or scenarios by copying existing tests and extending helpers as needed. The runner and utility structure minimize boilerplate and maximize discoverability.
59 | 
60 | - **Recommended Practice:** Refactor shared logic into `test_utils.ahk` as coverage grows. All new integration tests should exit with code 0 on pass, nonzero on failure, and be added to `tests/ahk/` for automatic inclusion in the suite.
61 | 
62 | ## AHK Scripting Best Practices
63 | 
64 | ### Robust Object/Map Variable Handling
65 | 
66 | When working with AHK variables that are expected to be objects or maps (especially those returned by functions or parsed from external data), it's crucial to validate their structure before accessing members. This prevents runtime errors if the variable is not an object or lacks expected keys/properties.
67 | 
68 | **Key Practices:**
69 | 1.  **Type Check:** Verify the variable is an object using `IsObject(varName)`.
70 | 2.  **Property/Key Existence:** Before accessing a property like `varName.propertyName` or `varName[keyName]`, check if it exists using `varName.Has("propertyName")` for objects or `varName.Has(keyName)` for Maps.
71 | 3.  **Graceful Failure:** If validation fails (e.g., not an object, or missing key):
72 |     - Log a detailed error, preferably with context (function name, input values).
73 |     - Provide a safe fallback behavior or display a user-friendly error message.
74 |     - Avoid letting the script crash.
75 | 
76 | **Example (from `lld_navigator_controller.ahk`):**
77 | The `v` variable, resulting from `ValidateAndNormalizeInputs`, is handled as follows:
78 | 
79 | ```ahk
80 | ; v is expected to be a Map like {valid: true/false, errorMsg: "...", ...}
81 | v := ValidateAndNormalizeInputs(jobNumber, selectedFolder)
82 | 
83 | if (!IsObject(v) || !v.Has("valid")) {
84 |     SafeShowInlineHint("Input validation process failed unexpectedly.", "error")
85 |     LogError("ValidateAndNormalizeInputs did not return a valid object. ...", "FunctionName", "CRITICAL")
86 |     SafeResetGUI()
87 |     Return
88 | }
89 | 
90 | if (!v.valid) {
91 |     local errorToShow := v.Has("errorMsg") && v.errorMsg != "" ? v.errorMsg : "Default error message."
92 |     SafeShowInlineHint(errorToShow, "error")
93 |     ; ... further error handling ...
94 |     Return
95 | }
96 | ; Proceed with using v.properties
97 | ```
98 | This pattern significantly improves script stability.
99 | 
100| ## Summary
101| 
102| Project QuickNav's design ensures modularity, reliability, and extensibility. Each layer is independently testable and replaceable while maintaining a clear contract of interaction.