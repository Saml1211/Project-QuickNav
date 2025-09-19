/*
Project QuickNav – Enhanced AutoHotkey Frontend with Document Navigation

Purpose:
    Provides a comprehensive GUI for entering a 5-digit project code and navigating to
    specific document types with version/revision selection, high-DPI scaling support,
    and advanced filtering options.

Requirements:
    - AutoHotkey v2
    - find_project_path.py and doc_navigator.py in the quicknav package
    - Python 3.x installed and in PATH

Features:
    - Document type selection (LLD, HLD, Change Orders, Floor Plans, etc.)
    - Version/revision filtering (Latest, As-Built, Initial, All)
    - Room and Change Order number filtering
    - High-DPI scaling support for 4K displays
    - Thumbnail previews for photos and images
    - Custom root path configuration

Usage:
    - Run this script or press Ctrl+Alt+Q to show/focus the GUI
    - Enter a valid 5-digit job number or search term
    - Select document type and filters
    - Click "Find Documents" to search or "Open Folder" for basic navigation

Keyboard Shortcuts:
    - Ctrl+Alt+Q: Show or focus Project QuickNav window
    - Enter: Execute search/navigation
    - Escape: Close/hide window
*/

#Requires AutoHotkey v2.0
try DllCall("User32\SetProcessDpiAwarenessContext", "ptr", -4)  ; DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
#SingleInstance Force

; Global variables
global mainGui, settingsGui
global currentProject := ""
global dpiScale := A_ScreenDPI / 96.0
global configPath := A_AppData . "\QuickNav\settings.json"

; Document types configuration
docTypes := Map(
    "lld", "Low-Level Design (LLD)",
    "hld", "High-Level Design (HLD)",
    "change_order", "Change Orders",
    "sales_po", "Sales & PO Reports",
    "floor_plans", "Floor Plans",
    "scope", "Scope Documents",
    "qa_itp", "QA/ITP Reports",
    "swms", "SWMS",
    "supplier_quotes", "Supplier Quotes",
    "photos", "Site Photos"
)

; Version filter options
versionFilters := ["Auto (Latest/Best)", "Latest Version", "As-Built Only", "Initial Version", "All Versions"]

; Initialize on startup
InitializeApp()

; Hotkey assignment
^!q:: ToggleGui()

; Main functions
InitializeApp() {
    CreateMainGui()
    LoadSettings()
    AddSystemTrayMenu()

    ; Show GUI on startup
    ShowMainGui()
}

CreateMainGui() {
    global mainGui, dpiScale

    ; Calculate scaled dimensions
    baseWidth := 400
    baseHeight := 550
    guiWidth := Round(baseWidth * dpiScale)
    guiHeight := Round(baseHeight * dpiScale)

    ; Create main GUI with DPI scaling and resizable window
    mainGui := Gui("+AlwaysOnTop -DPIScale +Resize", "Project QuickNav - Enhanced")
    
    ; Set scaled font size for better readability on high-DPI displays
    mainGui.SetFont("s" . Round(10 * dpiScale))
    
    ; Minimum window size adapted for DPI
    mainGui.MinSize := Round(400 * dpiScale) . "x" . Round(500 * dpiScale)
    
    ; Calculate scaled positions and sizes
    margin := Round(20 * dpiScale)
    rowHeight := Round(25 * dpiScale)
    currentY := margin

    ; Project input section
    projectLabel := mainGui.Add("Text", Format("x{} y{} w{}", margin, currentY, Round(130 * dpiScale)), "Project Number:")
    projectEdit := mainGui.Add("Edit", Format("x{} y{} w{} h{} vProjectInput", margin + Round(135 * dpiScale), currentY, Round(250 * dpiScale), Round(26 * dpiScale)))
    projectEdit.ToolTip := "Enter a 5-digit project number (e.g., 17741) or search term"
    currentY += Round(45 * dpiScale)

    ; Navigation mode selection
    navGroup := mainGui.Add("GroupBox", Format("x{} y{} w{} h{}", margin, currentY, Round(370 * dpiScale), Round(120 * dpiScale)), "Navigation Mode")
    navGroupY := currentY + Round(20 * dpiScale)
    folderModeRadio := mainGui.Add("Radio", Format("x{} y{} w{} h{} vFolderMode Checked", margin + Round(10 * dpiScale), navGroupY, Round(180 * dpiScale), Round(24 * dpiScale)), "Open Project Folder")
    docModeRadio := mainGui.Add("Radio", Format("x{} y{} w{} h{} vDocMode", margin + Round(10 * dpiScale), navGroupY + Round(30 * dpiScale), Round(180 * dpiScale), Round(24 * dpiScale)), "Find Documents")
    
    ; Settings button (moved below radios to avoid overlap)
    settingsBtn := mainGui.Add("Button", Format("x{} y{} w{} h{} vSettingsBtn", margin + Round(245 * dpiScale), navGroupY + Round(60 * dpiScale), Round(110 * dpiScale), Round(30 * dpiScale)), "Settings")
    currentY += Round(130 * dpiScale)

    ; Folder selection (for folder mode)
    folderGroup := mainGui.Add("GroupBox", Format("x{} y{} w{} h{} vFolderGroup", margin, currentY, Round(370 * dpiScale), Round(200 * dpiScale)), "Select Subfolder")
    folderGroupY := currentY + Round(20 * dpiScale)
    
    ; Folder radio buttons - optimized batch creation with better spacing
    folderRadios := []
    folderNames := [
        "4. System Designs",
        "1. Sales Handover",
        "2. BOM & Orders",
        "6. Customer Handover Documents",
        "5. Floor Plans",
        "6. Site Photos"
    ]
    
    ; Pre-calculate common values for performance
    radioX := margin + Round(15 * dpiScale)
    radioWidth := Round(350 * dpiScale)
    radioHeight := Round(30 * dpiScale)  ; vertical rhythm
    
    loop folderNames.Length {
        radioY := folderGroupY + Round((A_Index - 1) * radioHeight)
        radio := mainGui.Add("Radio", Format("x{} y{} w{} h{} vFolder{} {}",
            radioX, radioY, radioWidth, Round(24 * dpiScale),
            A_Index, A_Index == 1 ? "Checked" : ""))
        radio.Text := folderNames[A_Index]
        folderRadios.Push(radio)
    }
    currentY += Round(210 * dpiScale)

    ; Document type selection (for document mode)
    docGroup := mainGui.Add("GroupBox", Format("x{} y{} w{} h{} vDocGroup Hidden", margin, currentY - Round(200 * dpiScale), Round(370 * dpiScale), Round(200 * dpiScale)), "Document Type & Filters")
    docGroupY := currentY - Round(180 * dpiScale)
    
    ; Document type dropdown
    docTypeLabel := mainGui.Add("Text", Format("x{} y{}", margin + Round(10 * dpiScale), docGroupY), "Document Type:")
    docTypeOptions := []
    for key, value in docTypes
        docTypeOptions.Push(value)
    docTypeCombo := mainGui.Add("ComboBox", Format("x{} y{} w{} vDocType Choose1", margin + Round(100 * dpiScale), docGroupY, Round(280 * dpiScale)), docTypeOptions)
    docTypeCombo.ToolTip := "Select the type of document to search for"
    docGroupY += Round(30 * dpiScale)
    
    ; Version filter
    versionLabel := mainGui.Add("Text", Format("x{} y{}", margin + Round(10 * dpiScale), docGroupY), "Version Filter:")
    versionCombo := mainGui.Add("ComboBox", Format("x{} y{} w{} vVersionFilter Choose1", margin + Round(100 * dpiScale), docGroupY, Round(200 * dpiScale)), versionFilters)
    versionCombo.ToolTip := "Choose how to filter document versions"
    docGroupY += Round(30 * dpiScale)
    
    ; Room filter
    roomLabel := mainGui.Add("Text", Format("x{} y{}", margin + Round(10 * dpiScale), docGroupY), "Room:")
    roomEdit := mainGui.Add("Edit", Format("x{} y{} w{} h{} vRoomFilter Number", margin + Round(50 * dpiScale), docGroupY, Round(60 * dpiScale), Round(20 * dpiScale)))
    roomEdit.ToolTip := "Enter room number to filter documents (optional)"
    
    ; CO filter
    coLabel := mainGui.Add("Text", Format("x{} y{}", margin + Round(125 * dpiScale), docGroupY), "CO:")
    coEdit := mainGui.Add("Edit", Format("x{} y{} w{} h{} vCoFilter Number", margin + Round(155 * dpiScale), docGroupY, Round(60 * dpiScale), Round(20 * dpiScale)))
    coEdit.ToolTip := "Enter change order number to filter documents (optional)"
    
    ; Include archive checkbox
    archiveCheck := mainGui.Add("Checkbox", Format("x{} y{} w{} vIncludeArchive", margin + Round(230 * dpiScale), docGroupY, Round(140 * dpiScale)), "Include Archive")
    archiveCheck.ToolTip := "Include documents from archive folders"
    currentY += Round(20 * dpiScale)

    ; Options section
    optionsGroup := mainGui.Add("GroupBox", Format("x{} y{} w{} h{}", margin, currentY, Round(370 * dpiScale), Round(60 * dpiScale)), "Options")
    optionsY := currentY + Round(20 * dpiScale)
    debugCheck := mainGui.Add("Checkbox", Format("x{} y{} w{} vDebugMode", margin + Round(10 * dpiScale), optionsY, Round(170 * dpiScale)), "Show Debug Output")
    trainingCheck := mainGui.Add("Checkbox", Format("x{} y{} w{} vGenerateTraining", margin + Round(190 * dpiScale), optionsY, Round(170 * dpiScale)), "Generate Training Data")
    currentY += Round(70 * dpiScale)

    ; Status and progress
    statusText := mainGui.Add("Text", Format("x{} y{} w{} h{} vStatusText", margin, currentY, Round(370 * dpiScale), Round(60 * dpiScale)), "Ready - Select navigation mode and enter project number")
    currentY += Round(55 * dpiScale)

    ; Action buttons
    btnWidth := Round(120 * dpiScale)
    btnHeight := Round(35 * dpiScale)
    btnSpacing := Round(20 * dpiScale)

    ; Calculate button positions to center them
    totalBtnWidth := (btnWidth * 2) + btnSpacing
    btnStartX := margin + Round((370 * dpiScale - totalBtnWidth) / 2)

    openBtn := mainGui.Add("Button", Format("x{} y{} w{} h{} Default vOpenBtn", btnStartX, currentY, btnWidth, btnHeight), "Open Folder")
    findBtn := mainGui.Add("Button", Format("x{} y{} w{} h{} vFindBtn Hidden", btnStartX, currentY, btnWidth, btnHeight), "Find Documents")
    chooseBtn := mainGui.Add("Button", Format("x{} y{} w{} h{} vChooseBtn Hidden", btnStartX + btnWidth + btnSpacing, currentY, btnWidth, btnHeight), "Choose From List")

    ; Set up event handlers
    folderModeRadio.OnEvent("Click", (*) => SwitchToFolderMode())
    docModeRadio.OnEvent("Click", (*) => SwitchToDocumentMode())
    openBtn.OnEvent("Click", (*) => ExecuteProjectNavigation())
    findBtn.OnEvent("Click", (*) => ExecuteDocumentNavigation())
    chooseBtn.OnEvent("Click", (*) => ExecuteDocumentNavigation(true))
    settingsBtn.OnEvent("Click", (*) => ShowSettingsDialog())
    projectEdit.OnEvent("Change", (*) => ValidateInputs())

    ; GUI event handlers
    mainGui.OnEvent("Close", (*) => mainGui.Hide())
    mainGui.OnEvent("Escape", (*) => mainGui.Hide())
    mainGui.OnEvent("Size", MainGuiSize)
    
    ; Store references to controls
    mainGui.projectLabel := projectLabel
    mainGui.projectEdit := projectEdit
    mainGui.navGroup := navGroup
    mainGui.folderModeRadio := folderModeRadio
    mainGui.docModeRadio := docModeRadio
    mainGui.settingsBtn := settingsBtn
    mainGui.folderGroup := folderGroup
    mainGui.docGroup := docGroup
    mainGui.optionsGroup := optionsGroup
    mainGui.statusText := statusText
    mainGui.openBtn := openBtn
    mainGui.findBtn := findBtn
    mainGui.chooseBtn := chooseBtn
    mainGui.folderRadios := folderRadios
    mainGui.docTypeLabel := docTypeLabel
    mainGui.docTypeCombo := docTypeCombo
    mainGui.versionLabel := versionLabel
    mainGui.versionCombo := versionCombo
    mainGui.roomLabel := roomLabel
    mainGui.roomEdit := roomEdit
    mainGui.coLabel := coLabel
    mainGui.coEdit := coEdit
    mainGui.archiveCheck := archiveCheck
    mainGui.debugCheck := debugCheck
    mainGui.trainingCheck := trainingCheck
    ; ---- Panel control lists and initial visibility ----
    mainGui.folderCtrls := mainGui.folderRadios.Clone()  ; radios for folder mode

    mainGui.docCtrls := [
        mainGui.docTypeLabel,  mainGui.docTypeCombo,
        mainGui.versionLabel,  mainGui.versionCombo,
        mainGui.roomLabel,     mainGui.roomEdit,
        mainGui.coLabel,       mainGui.coEdit,
        mainGui.archiveCheck
    ]

    ; Start in Folder mode: hide document controls so they don't overlap
    for ctrl in mainGui.docCtrls
        ctrl.Visible := false
}

SwitchToFolderMode() {
    global mainGui
    
    ; Show folder selection, hide document options
    mainGui.folderGroup.Visible := true
    mainGui.docGroup.Visible := false
    mainGui.openBtn.Visible := true
    mainGui.findBtn.Visible := false
    mainGui.chooseBtn.Visible := false
        ; Toggle panel controls
    for c in mainGui.folderCtrls
        c.Visible := true
    for c in mainGui.docCtrls
        c.Visible := false

    ; Reapply layout after mode switch
    try {
        mainGui.GetPos(, , &w, &h)
        MainGuiSize(mainGui, 0, w, h)
    } catch {
    }
    
    UpdateStatus("Folder mode - Select a project subfolder to open")
}

SwitchToDocumentMode() {
    global mainGui
    
    ; Show document options, hide folder selection
    mainGui.folderGroup.Visible := false
    mainGui.docGroup.Visible := true
    mainGui.openBtn.Visible := false
    mainGui.findBtn.Visible := true
    mainGui.chooseBtn.Visible := true
        ; Toggle panel controls
    for c in mainGui.folderCtrls
        c.Visible := false
    for c in mainGui.docCtrls
        c.Visible := true

    ; Reapply layout after mode switch
    try {
        mainGui.GetPos(, , &w, &h)
        MainGuiSize(mainGui, 0, w, h)
    } catch {
    }
    
    UpdateStatus("Document mode - Find specific documents by type and filters")
}

ValidateInputs() {
    global mainGui

    projectInput := Trim(mainGui.projectEdit.Text)

    ; Validate project input
    if (projectInput = "") {
        UpdateStatus("Enter a 5-digit project number or search term")
        mainGui.projectEdit.ToolTip := "This field cannot be empty"
        return false
    }

    ; Check for invalid characters (filesystem unsafe)
    if (RegExMatch(projectInput, "[<>:|?*]")) {
        UpdateStatus("Invalid characters in project input")
        mainGui.projectEdit.ToolTip := "Project input contains invalid filesystem characters"
        return false
    }

    ; Validate length
    if (StrLen(projectInput) > 100) {
        UpdateStatus("Project input too long (max 100 characters)")
        mainGui.projectEdit.ToolTip := "Project input cannot exceed 100 characters"
        return false
    }

    ; Clear any previous error tooltips
    mainGui.projectEdit.ToolTip := "Enter a 5-digit project number (e.g., 17741) or search term"

    if (RegExMatch(projectInput, "^\d{5}$")) {
        UpdateStatus("Ready - Project number: " . projectInput)
    } else {
        UpdateStatus("Ready - Search term: " . projectInput)
    }

    return true
}

ExecuteProjectNavigation() {
    global mainGui, currentProject

    try {
        ; Get form data
        formData := mainGui.Submit(false)

        if (formData.ProjectInput = "") {
            ShowError("Please enter a project number or search term")
            return
        }

        UpdateStatus("Searching for project...")
        currentProject := formData.ProjectInput

        ; Get selected folder
        selectedFolder := ""
        loop mainGui.folderRadios.Length {
            if (formData.HasOwnProp("Folder" . A_Index) && formData.GetProp("Folder" . A_Index)) {
                selectedFolder := mainGui.folderRadios[A_Index].Text
                break
            }
        }

        ; Execute Python backend for project navigation
        result := ExecutePythonScript("project", formData.ProjectInput, "", formData)
        ProcessProjectResult(result, selectedFolder, formData)

    } catch as e {
        ShowError("Error during project navigation: " . e.Message)
        UpdateStatus("Error occurred")
    }
}

ExecuteDocumentNavigation(chooseMode := false) {
    global mainGui, currentProject, docTypes

    try {
        ; Get form data
        formData := mainGui.Submit(false)

        if (formData.ProjectInput = "") {
            ShowError("Please enter a project number or search term")
            return
        }

        UpdateStatus("Searching for documents...")
        currentProject := formData.ProjectInput

        ; Get selected document type
        docTypeText := formData.DocType
        docTypeKey := ""
        for key, value in docTypes {
            if (value = docTypeText) {
                docTypeKey := key
                break
            }
        }

        if (docTypeKey = "") {
            ShowError("Please select a document type")
            return
        }

        ; Build command arguments
        args := []
        args.Push("--type", docTypeKey)

        ; Add selection mode
        if (chooseMode) {
            args.Push("--choose")
        } else {
            versionFilter := formData.VersionFilter
            if (InStr(versionFilter, "Latest")) {
                args.Push("--latest")
            }
        }

        ; Add filters
        if (formData.RoomFilter != "") {
            args.Push("--room", formData.RoomFilter)
        }

        if (formData.CoFilter != "") {
            args.Push("--co", formData.CoFilter)
        }

        if (formData.IncludeArchive) {
            args.Push("--include-archive")
        }

        ; Execute Python backend for document navigation
        result := ExecutePythonScript("doc", formData.ProjectInput, args, formData)
        ProcessDocumentResult(result, formData)

    } catch as e {
        ShowError("Error during document navigation: " . e.Message)
        UpdateStatus("Error occurred")
    }
}

ExecutePythonScript(command, projectInput, extraArgs, formData) {
    try {
        ; Validate inputs
        if (command = "") {
            throw Error("Command cannot be empty")
        }
        if (projectInput = "") {
            throw Error("Project input cannot be empty")
        }

        ; Build command with proper escaping (invoke the CLI module explicitly)
        cmd := "python -m quicknav.cli " . command . " " . Chr(34) . StrReplace(projectInput, Chr(34), Chr(34) . Chr(34)) . Chr(34)

        ; Add extra arguments with validation
        if (IsObject(extraArgs)) {
            for arg in extraArgs {
                if (arg != "") {
                    cmd .= " " . arg
                }
            }
        } else if (extraArgs != "") {
            cmd .= " " . extraArgs
        }

        ; Add training data flag if enabled
        if (formData.HasOwnProp("GenerateTraining") && formData.GenerateTraining) {
            cmd .= " --training-data"
        }

        ; Create unique temp file to avoid conflicts
        tempFile := A_Temp . "\quicknav_output_" . A_TickCount . ".txt"

        ; Build full command with error redirection
        fullCmd := A_ComSpec . " /C " . Chr(34) . cmd . Chr(34) . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"

        ; Execute command with timeout protection
        try {
            RunWait(fullCmd, "", "Hide", &exitCode)
        } catch as runError {
            throw Error("Failed to execute Python command: " . runError.Message)
        }

        ; Check if command execution was successful, else try legacy fallback for 'project'
        if (exitCode != 0) {
            if (command = "project") {
                ; Legacy fallback: call find_project_path.py directly if quicknav module is unavailable
                scriptPath := "find_project_path.py"
                if (!FileExist(scriptPath)) {
                    scriptPath := "test_find_project_path.py"
                    if (!FileExist(scriptPath)) {
                        scriptPath := "..\src\find_project_path.py"
                        if (!FileExist(scriptPath)) {
                            scriptPath := "..\src\test_find_project_path.py"
                        }
                    }
                }
                if (FileExist(scriptPath)) {
                    legacyCmd := A_ComSpec . " /C " . Chr(34) . "python " . Chr(34) . scriptPath . Chr(34) . " " . Chr(34) . StrReplace(projectInput, Chr(34), Chr(34) . Chr(34)) . Chr(34)
                    if (formData.HasOwnProp("GenerateTraining") && formData.GenerateTraining) {
                        legacyCmd .= " --training-data"
                    }
                    legacyCmd .= " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"
                    try {
                        RunWait(legacyCmd, "", "Hide", &exitCode)
                    } catch {
                        ; ignore and fall through to error
                    }
                }
            }
            if (exitCode != 0) {
                throw Error("Python command failed with exit code: " . exitCode)
            }
        }

        ; Check if output file was created
        if (!FileExist(tempFile)) {
            throw Error("Python command did not produce output file")
        }

        ; Read output with size limit to prevent memory issues
        try {
            output := FileRead(tempFile, "UTF-8")
        } catch as readError {
            throw Error("Failed to read Python output: " . readError.Message)
        }

        ; Clean up temp file
        try {
            FileDelete(tempFile)
        } catch as cleanupError {
            ; Log but don't fail on cleanup error
            if (formData.HasOwnProp("DebugMode") && formData.DebugMode) {
                OutputDebug("Warning: Failed to clean up temp file: " . cleanupError.Message)
            }
        }

        ; Validate output
        output := Trim(output)
        if (output = "") {
            throw Error("Python command produced no output")
        }

        ; Show debug output if enabled
        if (formData.HasOwnProp("DebugMode") && formData.DebugMode) {
            MsgBox("Command: " . cmd . "`n`nOutput:`n" . output, "Debug Output", 64)
        }

        return output

    } catch as e {
        ; Clean up temp file if it exists
        if (FileExist(tempFile)) {
            try {
                FileDelete(tempFile)
            } catch {
                ; Ignore cleanup errors in exception handler
            }
        }
        throw e
    }
}

ProcessProjectResult(result, selectedFolder, formData) {
    global mainGui

    lines := StrSplit(result, "`n")
    statusLine := Trim(lines[1])

    if (InStr(statusLine, "ERROR:") == 1) {
        ShowError(SubStr(statusLine, 7))
        UpdateStatus("Project not found")
        return
    }

    if (InStr(statusLine, "SELECT:") == 1) {
        ; Multiple exact matches
        ShowProjectSelectionDialog(SubStr(statusLine, 8), selectedFolder)
        return
    }

    if (InStr(statusLine, "SEARCH:") == 1) {
        ; Search results
        ShowProjectSearchDialog(SubStr(statusLine, 8), selectedFolder)
        return
    }

    if (InStr(statusLine, "SUCCESS:") == 1) {
        ; Single match found
        projectPath := Trim(SubStr(statusLine, 9))
        OpenProjectFolder(projectPath, selectedFolder)
        return
    }

    ShowError("Unexpected response from backend")
}

ProcessDocumentResult(result, formData) {
    global mainGui

    lines := StrSplit(result, "`n")
    statusLine := Trim(lines[1])

    if (InStr(statusLine, "ERROR:") == 1) {
        ShowError(SubStr(statusLine, 7))
        UpdateStatus("Documents not found")
        return
    }

    if (InStr(statusLine, "SELECT:") == 1) {
        ; Multiple documents found
        ShowDocumentSelectionDialog(SubStr(statusLine, 8))
        return
    }

    if (InStr(statusLine, "SUCCESS:") == 1) {
        ; Single document found
        documentPath := Trim(SubStr(statusLine, 9))
        OpenDocument(documentPath)
        return
    }

    ShowError("Unexpected response from backend")
}

OpenProjectFolder(projectPath, selectedFolder) {
    global mainGui

    try {
        ; Validate inputs
        if (projectPath = "") {
            RaiseError("Project path cannot be empty")
        }
        if (selectedFolder = "") {
            RaiseError("Selected folder cannot be empty")
        }

        ; Construct full path with proper path handling
        ; Maintain legacy compatibility:
        ; - "Floor Plans" and "Site Photos" live under "\1. Sales Handover\"
        ; - Handle labels with numeric prefixes like "5. Floor Plans" / "6. Site Photos"
        if (InStr(selectedFolder, "Floor Plans")) {
            fullPath := projectPath . "\1. Sales Handover\Floor Plans"
        } else if (InStr(selectedFolder, "Site Photos")) {
            fullPath := projectPath . "\1. Sales Handover\Site Photos"
        } else {
            fullPath := projectPath . "\" . selectedFolder
        }

        ; Normalize path separators
        fullPath := StrReplace(fullPath, "/", "\")

        ; Check if path exists
        if (!FileExist(fullPath)) {
            RaiseError("Subfolder '" . selectedFolder . "' not found at: " . fullPath)
        }

        ; Check if it's actually a directory
        if (!DirExist(fullPath)) {
            RaiseError("Path exists but is not a directory: " . fullPath)
        }

        ; Attempt to open with Explorer
        try {
            Run("explorer.exe " . Chr(34) . fullPath . Chr(34))
        } catch as runError {
            ; Fallback to alternative method
            try {
                Run("explorer.exe /select," . Chr(34) . fullPath . Chr(34))
            } catch as fallbackError {
                RaiseError("Failed to open folder with both methods: " . runError.Message . " | " . fallbackError.Message)
            }
        }

        UpdateStatus("Opened folder: " . selectedFolder)

        ; Auto-hide after success
        SetTimer(() => mainGui.Hide(), -1500)

    } catch as e {
        ShowError("Failed to open project folder: " . e.Message)
        UpdateStatus("Error opening folder")
    }
}

OpenDocument(documentPath) {
    global mainGui

    if (!FileExist(documentPath)) {
        ShowError("Document not found: " . documentPath)
        return
    }

    try {
        Run(Chr(34) . documentPath . Chr(34))
        UpdateStatus("Opened document")

        ; Auto-hide after success
        SetTimer(() => mainGui.Hide(), -1500)

    } catch as e {
        ShowError("Failed to open document: " . e.Message)
    }
}

ShowProjectSelectionDialog(pathsString, selectedFolder) {
    global mainGui, dpiScale

    try {
        ; Parse paths
        arrPaths := StrSplit(pathsString, "|")

        ; Create selection GUI with DPI scaling
        selGui := Gui("+AlwaysOnTop -DPIScale", "Select Project Folder")
        selGui.SetFont("s" . Round(10 * dpiScale))

        ; Calculate scaled dimensions
        margin := Round(15 * dpiScale)
        currentY := margin
        guiWidth := Round(450 * dpiScale)
        guiHeight := Round(150 * dpiScale)

        ; Instructions text
        selGui.Add("Text", Format("x{} y{} w{}", margin, currentY, Round(420 * dpiScale)),
            "Multiple project folders found:`nSelect the correct path:")
        currentY += Round(40 * dpiScale)

        ; Create dropdown options
        choices := ""
        loop arrPaths.Length {
            choices .= A_Index . ": " . arrPaths[A_Index] . "|"
        }
        choices := SubStr(choices, 1, -1)

        ; Dropdown list
        selGui.Add("DropDownList", Format("x{} y{} w{} vSelChoice AltSubmit Choose1",
            margin, currentY, Round(420 * dpiScale)), choices)
        currentY += Round(40 * dpiScale)

        ; Buttons
        btnWidth := Round(80 * dpiScale)
        btnHeight := Round(30 * dpiScale)
        btnSpacing := Round(20 * dpiScale)
        btnY := currentY

        okBtn := selGui.Add("Button", Format("x{} y{} w{} h{} Default",
            margin + Round(420 * dpiScale) - btnWidth - btnSpacing - btnWidth, btnY, btnWidth, btnHeight), "OK")
        cancelBtn := selGui.Add("Button", Format("x{} y{} w{} h{}",
            margin + Round(420 * dpiScale) - btnWidth, btnY, btnWidth, btnHeight), "Cancel")

        ; Event handlers
        okBtn.OnEvent("Click", (*) => ProcessSelection(selGui, arrPaths, selectedFolder))
        cancelBtn.OnEvent("Click", (*) => selGui.Destroy())
        selGui.OnEvent("Close", (*) => selGui.Destroy())
        selGui.OnEvent("Escape", (*) => selGui.Destroy())

        ; Show modal dialog
        selGui.Show(Format("w{} h{}", guiWidth, guiHeight))

    } catch as e {
        ShowError("Error creating project selection dialog: " . e.Message)
    }
}

ProcessSelection(selGui, arrPaths, selectedFolder) {
    try {
        ; Get selection
        formData := selGui.Submit()
        selGui.Destroy()

        if (!formData.SelChoice) {
            UpdateStatus("Selection cancelled")
            MsgBox("Selection cancelled.", "Cancelled", 48)
            return
        }

        ; Extract chosen path
        chosenIdx := formData.SelChoice
        projectPath := arrPaths[chosenIdx]

        ; Open the selected project folder
        OpenProjectFolder(projectPath, selectedFolder)

    } catch as e {
        ShowError("Error processing selection: " . e.Message)
    }
}

ShowProjectSearchDialog(pathsString, selectedFolder) {
    global mainGui, dpiScale

    try {
        ; Parse paths
        arrPaths := StrSplit(pathsString, "|")

        ; Create search results GUI with DPI scaling
        searchGui := Gui("+Resize -DPIScale", "Search Results")
        searchGui.SetFont("s" . Round(10 * dpiScale))

        ; Calculate scaled dimensions
        margin := Round(15 * dpiScale)
        currentY := margin
        baseWidth := 650
        baseHeight := 420
        guiWidth := Round(baseWidth * dpiScale)
        guiHeight := Round(baseHeight * dpiScale)

        ; Instructions text
        searchGui.Add("Text", Format("x{} y{} w{}",
            margin, currentY, Round(620 * dpiScale)),
            "Found " . arrPaths.Length . " project folders matching your search:")
        currentY += Round(30 * dpiScale)

        ; Create ListView
        lvWidth := Round(620 * dpiScale)
        lvHeight := Round(300 * dpiScale)
        LV := searchGui.Add("ListView", Format("x{} y{} w{} h{} vSearchList -Multi",
            margin, currentY, lvWidth, lvHeight),
            ["Project Number", "Project Name", "Full Path"])

        ; Set column widths with scaling
        LV.ModifyCol(1, Round(120 * dpiScale))  ; Project Number
        LV.ModifyCol(2, Round(200 * dpiScale))  ; Project Name
        LV.ModifyCol(3, Round(300 * dpiScale))  ; Full Path

        ; Populate ListView
        loop arrPaths.Length {
            path := arrPaths[A_Index]
            SplitPath(path, &fileName, &dirPath)

            ; Extract project number and name using regex
            if (RegExMatch(fileName, "^(\d{5}) - (.+)$", &match)) {
                projNum := match[1]
                projName := match[2]
            } else {
                projNum := "N/A"
                projName := fileName
            }

            LV.Add("", projNum, projName, path)
        }

        currentY += lvHeight + Round(20 * dpiScale)

        ; Buttons
        btnWidth := Round(80 * dpiScale)
        btnHeight := Round(30 * dpiScale)
        btnSpacing := Round(20 * dpiScale)
        btnY := currentY

        openBtn := searchGui.Add("Button", Format("x{} y{} w{} h{} Default vBtnOpen",
            margin + lvWidth - btnWidth - btnSpacing - btnWidth, btnY, btnWidth, btnHeight), "Open")
        cancelBtn := searchGui.Add("Button", Format("x{} y{} w{} h{} vBtnCancel",
            margin + lvWidth - btnWidth, btnY, btnWidth, btnHeight), "Cancel")

        ; Event handlers
        LV.OnEvent("DoubleClick", (*) => ProcessSearchSelection(searchGui, LV, arrPaths, selectedFolder))
        openBtn.OnEvent("Click", (*) => ProcessSearchSelection(searchGui, LV, arrPaths, selectedFolder))
        cancelBtn.OnEvent("Click", (*) => searchGui.Destroy())
        searchGui.OnEvent("Close", (*) => searchGui.Destroy())
        searchGui.OnEvent("Escape", (*) => searchGui.Destroy())

        ; Handle window resize
        searchGui.OnEvent("Size", SearchGuiSize)

        ; Show GUI
        searchGui.Show(Format("w{} h{}", guiWidth, guiHeight))

    } catch as e {
        ShowError("Error creating search results dialog: " . e.Message)
    }
}

SearchGuiSize(thisGui, MinMax, Width, Height) {
    if (MinMax = -1)  ; Minimized
        return

    try {
        ; Resize ListView to fit new window size
        margin := Round(15 * dpiScale)
        thisGui["SearchList"].Move(, , Width - (margin * 2), Height - Round(120 * dpiScale))

        ; Reposition buttons
        btnWidth := Round(80 * dpiScale)
        btnHeight := Round(30 * dpiScale)
        btnSpacing := Round(20 * dpiScale)
        btnY := Height - Round(50 * dpiScale)

        thisGui["BtnOpen"].Move(Width - margin - btnWidth - btnSpacing - btnWidth, btnY)
        thisGui["BtnCancel"].Move(Width - margin - btnWidth, btnY)

    } catch as e {
        ; Ignore resize errors
    }
}

ProcessSearchSelection(searchGui, LV, arrPaths, selectedFolder) {
    try {
        ; Get selected row
        selectedRow := LV.GetNext()
        if (selectedRow = 0) {
            UpdateStatus("No project selected")
            MsgBox("Please select a project from the list.", "No Selection", 48)
            return
        }

        ; Get the full path from the third column
        selectedPath := LV.GetText(selectedRow, 3)
        searchGui.Destroy()

        ; Open the selected project folder
        OpenProjectFolder(selectedPath, selectedFolder)

    } catch as e {
        ShowError("Error processing search selection: " . e.Message)
    }
}

ShowDocumentSelectionDialog(pathsString) {
    global mainGui, dpiScale

    try {
        ; Parse document paths
        arrPaths := StrSplit(pathsString, "|")

        ; Create document selection GUI with DPI scaling
        docGui := Gui("+Resize -DPIScale", "Select Document")
        docGui.SetFont("s" . Round(10 * dpiScale))

        ; Calculate scaled dimensions
        margin := Round(15 * dpiScale)
        currentY := margin
        baseWidth := 800
        baseHeight := 500
        guiWidth := Round(baseWidth * dpiScale)
        guiHeight := Round(baseHeight * dpiScale)

        ; Instructions text
        docGui.Add("Text", Format("x{} y{} w{}",
            margin, currentY, Round(770 * dpiScale)),
            "Found " . arrPaths.Length . " documents. Select one to open:")
        currentY += Round(30 * dpiScale)

        ; Create ListView with document columns
        lvWidth := Round(770 * dpiScale)
        lvHeight := Round(350 * dpiScale)
        LV := docGui.Add("ListView", Format("x{} y{} w{} h{} vDocList -Multi",
            margin, currentY, lvWidth, lvHeight),
            ["Document Name", "Version", "Type", "Path"])

        ; Set column widths with scaling
        LV.ModifyCol(1, Round(200 * dpiScale))  ; Document Name
        LV.ModifyCol(2, Round(100 * dpiScale))  ; Version
        LV.ModifyCol(3, Round(100 * dpiScale))  ; Type
        LV.ModifyCol(4, Round(370 * dpiScale))  ; Path

        ; Populate ListView with document metadata
        loop arrPaths.Length {
            path := arrPaths[A_Index]
            SplitPath(path, &fileName, &dirPath)

            ; Parse document metadata from filename
            docInfo := ParseDocumentInfo(fileName, path)
            LV.Add("", docInfo.name, docInfo.version, docInfo.type, path)
        }

        currentY += lvHeight + Round(20 * dpiScale)

        ; Buttons
        btnWidth := Round(80 * dpiScale)
        btnHeight := Round(30 * dpiScale)
        btnSpacing := Round(20 * dpiScale)
        btnY := currentY

        openBtn := docGui.Add("Button", Format("x{} y{} w{} h{} Default vBtnOpen",
            margin + lvWidth - btnWidth - btnSpacing - btnWidth, btnY, btnWidth, btnHeight), "Open")
        cancelBtn := docGui.Add("Button", Format("x{} y{} w{} h{} vBtnCancel",
            margin + lvWidth - btnWidth, btnY, btnWidth, btnHeight), "Cancel")

        ; Event handlers
        LV.OnEvent("DoubleClick", (*) => ProcessDocumentSelection(docGui, LV, arrPaths))
        openBtn.OnEvent("Click", (*) => ProcessDocumentSelection(docGui, LV, arrPaths))
        cancelBtn.OnEvent("Click", (*) => docGui.Destroy())
        docGui.OnEvent("Close", (*) => docGui.Destroy())
        docGui.OnEvent("Escape", (*) => docGui.Destroy())

        ; Handle window resize
        docGui.OnEvent("Size", DocGuiSize)

        ; Show GUI
        docGui.Show(Format("w{} h{}", guiWidth, guiHeight))

    } catch as e {
        ShowError("Error creating document selection dialog: " . e.Message)
    }
}

ParseDocumentInfo(fileName, fullPath) {
    ; Parse document metadata from filename and path
    docInfo := Map(
        "name", fileName,
        "version", "N/A",
        "type", "Unknown"
    )

    try {
        ; Extract file extension to determine type
        if (InStr(fileName, ".vsdx") || InStr(fileName, ".vsd"))
            docInfo["type"] := "Visio"
        else if (InStr(fileName, ".pdf"))
            docInfo["type"] := "PDF"
        else if (InStr(fileName, ".docx") || InStr(fileName, ".doc"))
            docInfo["type"] := "Word"
        else if (InStr(fileName, ".jpg") || InStr(fileName, ".png") || InStr(fileName, ".jpeg"))
            docInfo["type"] := "Image"

        ; Extract version information using common patterns
        if (RegExMatch(fileName, "REV\s*(\d+(?:\.\d+)?)", &match))
            docInfo["version"] := "REV " . match[1]
        else if (RegExMatch(fileName, "\((\d+(?:\.\d+)?)\)", &match))
            docInfo["version"] := "(" . match[1] . ")"
        else if (RegExMatch(fileName, "(\d+(?:\.\d+)?)", &match))
            docInfo["version"] := match[1]

        ; Clean up document name by removing version info for display
        displayName := RegExReplace(fileName, "\s*REV\s*\d+(?:\.\d+)?", "")
        displayName := RegExReplace(displayName, "\s*\(\d+(?:\.\d+)?\)", "")
        if (displayName != "")
            docInfo["name"] := displayName

    } catch as e {
        ; Keep defaults if parsing fails
    }

    return docInfo
}

DocGuiSize(thisGui, MinMax, Width, Height) {
    if (MinMax = -1)  ; Minimized
        return

    try {
        ; Resize ListView to fit new window size
        margin := Round(15 * dpiScale)
        thisGui["DocList"].Move(, , Width - (margin * 2), Height - Round(120 * dpiScale))

        ; Reposition buttons
        btnWidth := Round(80 * dpiScale)
        btnHeight := Round(30 * dpiScale)
        btnSpacing := Round(20 * dpiScale)
        btnY := Height - Round(50 * dpiScale)

        thisGui["BtnOpen"].Move(Width - margin - btnWidth - btnSpacing - btnWidth, btnY)
        thisGui["BtnCancel"].Move(Width - margin - btnWidth, btnY)

    } catch as e {
        ; Ignore resize errors
    }
}

ProcessDocumentSelection(docGui, LV, arrPaths) {
    try {
        ; Get selected row
        selectedRow := LV.GetNext()
        if (selectedRow = 0) {
            UpdateStatus("No document selected")
            MsgBox("Please select a document from the list.", "No Selection", 48)
            return
        }

        ; Get the full path from the fourth column
        selectedPath := LV.GetText(selectedRow, 4)
        docGui.Destroy()

        ; Open the selected document
        OpenDocument(selectedPath)

    } catch as e {
        ShowError("Error processing document selection: " . e.Message)
    }
}

ShowSettingsDialog() {
    global mainGui, dpiScale, configPath

    try {
        ; Load current settings
        settings := LoadSettings()

        ; Create settings GUI with DPI scaling
        settingsGui := Gui("+AlwaysOnTop -DPIScale", "Settings")
        settingsGui.SetFont("s" . Round(10 * dpiScale))

        ; Calculate scaled dimensions
        margin := Round(15 * dpiScale)
        currentY := margin
        baseWidth := 500
        baseHeight := 400
        guiWidth := Round(baseWidth * dpiScale)
        guiHeight := Round(baseHeight * dpiScale)

        ; Custom Roots section
        settingsGui.Add("GroupBox", Format("x{} y{} w{} h{}",
            margin, currentY, Round(470 * dpiScale), Round(300 * dpiScale)), "Custom Root Directories")
        currentY += Round(25 * dpiScale)

        ; ListBox for root paths
        lbWidth := Round(440 * dpiScale)
        lbHeight := Round(200 * dpiScale)
        rootList := settingsGui.Add("ListBox", Format("x{} y{} w{} h{} vRootList",
            margin + Round(10 * dpiScale), currentY, lbWidth, lbHeight))

        ; Populate with current roots
        if (settings.Has("custom_roots")) {
            for root in settings["custom_roots"] {
                rootList.Add([root])
            }
        }

        currentY += lbHeight + Round(15 * dpiScale)

        ; Buttons for managing roots
        btnWidth := Round(80 * dpiScale)
        btnHeight := Round(25 * dpiScale)
        btnSpacing := Round(10 * dpiScale)
        btnY := currentY

        addBtn := settingsGui.Add("Button", Format("x{} y{} w{} h{} vAddBtn",
            margin + Round(10 * dpiScale), btnY, btnWidth, btnHeight), "Add")
        editBtn := settingsGui.Add("Button", Format("x{} y{} w{} h{} vEditBtn",
            margin + Round(10 * dpiScale) + btnWidth + btnSpacing, btnY, btnWidth, btnHeight), "Edit")
        removeBtn := settingsGui.Add("Button", Format("x{} y{} w{} h{} vRemoveBtn",
            margin + Round(10 * dpiScale) + (btnWidth + btnSpacing) * 2, btnY, btnWidth, btnHeight), "Remove")

        currentY += btnHeight + Round(20 * dpiScale)

        ; Save/Cancel buttons
        saveBtn := settingsGui.Add("Button", Format("x{} y{} w{} h{} Default vSaveBtn",
            margin + Round(470 * dpiScale) - btnWidth - btnSpacing - btnWidth, currentY, btnWidth, btnHeight), "Save")
        cancelBtn := settingsGui.Add("Button", Format("x{} y{} w{} h{} vCancelBtn",
            margin + Round(470 * dpiScale) - btnWidth, currentY, btnWidth, btnHeight), "Cancel")

        ; Event handlers
        addBtn.OnEvent("Click", (*) => AddRootPath(settingsGui, rootList))
        editBtn.OnEvent("Click", (*) => EditRootPath(settingsGui, rootList))
        removeBtn.OnEvent("Click", (*) => RemoveRootPath(settingsGui, rootList))
        saveBtn.OnEvent("Click", (*) => SaveSettingsDialog(settingsGui, rootList))
        cancelBtn.OnEvent("Click", (*) => settingsGui.Destroy())
        settingsGui.OnEvent("Close", (*) => settingsGui.Destroy())
        settingsGui.OnEvent("Escape", (*) => settingsGui.Destroy())

        ; Show GUI
        settingsGui.Show(Format("w{} h{}", guiWidth, guiHeight))

    } catch as e {
        ShowError("Error creating settings dialog: " . e.Message)
    }
}

AddRootPath(settingsGui, rootList) {
    try {
        ; Show input dialog for new root path
        newPath := InputBox("Enter the full path to a project root directory:", "Add Root Path", "w400")

        if (newPath.Result = "OK" && newPath.Value != "") {
            ; Validate the path
            if (!DirExist(newPath.Value)) {
                result := MsgBox("The specified path does not exist. Add it anyway?", "Path Not Found", 4)
                if (result = "No")
                    return
            }

            ; Add to list
            rootList.Add([newPath.Value])
        }

    } catch as e {
        ShowError("Error adding root path: " . e.Message)
    }
}

EditRootPath(settingsGui, rootList) {
    try {
        ; Get selected item
        selectedIndex := rootList.Value
        if (selectedIndex = 0) {
            MsgBox("Please select a root path to edit.", "No Selection", 48)
            return
        }

        ; Get current path
        currentPath := rootList.Text

        ; Show input dialog for editing
        editPath := InputBox("Edit the root path:", "Edit Root Path", "w400", currentPath)

        if (editPath.Result = "OK" && editPath.Value != "") {
            ; Validate the path
            if (!DirExist(editPath.Value)) {
                result := MsgBox("The specified path does not exist. Update it anyway?", "Path Not Found", 4)
                if (result = "No")
                    return
            }

            ; Update the list
            rootList.Delete(selectedIndex)
            rootList.Insert(selectedIndex, editPath.Value)
        }

    } catch as e {
        ShowError("Error editing root path: " . e.Message)
    }
}

RemoveRootPath(settingsGui, rootList) {
    try {
        ; Get selected item
        selectedIndex := rootList.Value
        if (selectedIndex = 0) {
            MsgBox("Please select a root path to remove.", "No Selection", 48)
            return
        }

        ; Confirm deletion
        result := MsgBox("Remove the selected root path?", "Confirm Removal", 4)
        if (result = "Yes") {
            rootList.Delete(selectedIndex)
        }

    } catch as e {
        ShowError("Error removing root path: " . e.Message)
    }
}

SaveSettingsDialog(settingsGui, rootList) {
    try {
        ; Collect all root paths from the list
        rootPaths := []
        loop rootList.GetCount() {
            rootPaths.Push(rootList.GetText(A_Index))
        }

        ; Create settings object
        newSettings := Map(
            "custom_roots", rootPaths,
            "version", "1.0"
        )

        ; Save to JSON
        SaveSettingsToFile(newSettings)
        settingsGui.Destroy()

        UpdateStatus("Settings saved successfully")

    } catch as e {
        ShowError("Error saving settings: " . e.Message)
    }
}

UpdateStatus(message) {
    global mainGui
    mainGui.statusText.Text := message
}

ShowError(message) {
    global mainGui
    MsgBox(message, "Error", 16)
    UpdateStatus("Error: " . message)
}

LoadSettings() {
    ; Load settings from JSON file
    global configPath

    try {
        if (FileExist(configPath)) {
            ; Read JSON file content
            jsonContent := FileRead(configPath)

            ; Parse JSON manually (simple implementation)
            loadedSettings := ParseSimpleJson(jsonContent)
            return loadedSettings
        }
    } catch as e {
        ; Use defaults if loading fails
    }

    ; Return default settings
    return Map(
        "custom_roots", [],
        "version", "1.0"
    )
}

ParseSimpleJson(jsonString) {
    ; Simple JSON parser for basic structures
    settings := Map()

    try {
        ; Remove whitespace
        jsonString := RegExReplace(jsonString, "\s+", "")

        ; Extract custom_roots array
        if (RegExMatch(jsonString, '"custom_roots":\[([^\]]*)\]', &match)) {
            rootsString := match[1]
            if (rootsString != "") {
                ; Split by comma and clean up quotes
                rootArray := []
                roots := StrSplit(rootsString, '","')
                for root in roots {
                    root := Trim(root, '"')
                    if (root != "")
                        rootArray.Push(root)
                }
                settings["custom_roots"] := rootArray
            } else {
                settings["custom_roots"] := []
            }
        } else {
            settings["custom_roots"] := []
        }

        ; Extract version
        if (RegExMatch(jsonString, '"version":"([^"]*)"', &match)) {
            settings["version"] := match[1]
        } else {
            settings["version"] := "1.0"
        }

    } catch as e {
        ; Return empty settings on parse error
        settings := Map("custom_roots", [], "version", "1.0")
    }

    return settings
}

SaveSettingsToFile(settings) {
    ; Save settings to JSON file
    global configPath

    try {
        ; Ensure directory exists
        configDir := ""
        SplitPath(configPath, , &configDir)
        if (!FileExist(configDir)) {
            DirCreate(configDir)
        }

        ; Generate JSON string
        jsonString := GenerateSimpleJson(settings)

        ; Write to file
        FileDelete(configPath)  ; Delete if exists
        FileAppend(jsonString, configPath, "UTF-8")

    } catch as e {
        ShowError("Failed to save settings: " . e.Message)
    }
}

GenerateSimpleJson(settings) {
    ; Generate simple JSON string
    json := "{`n"

    ; Add custom_roots array
    json .= '  "custom_roots": ['
    if (settings.Has("custom_roots") && settings["custom_roots"].Length > 0) {
        for i, root in settings["custom_roots"] {
            json .= '"' . EscapeJsonString(root) . '"'
            if (i < settings["custom_roots"].Length)
                json .= ', '
        }
    }
    json .= '],`n'

    ; Add version
    version := settings.Has("version") ? settings["version"] : "1.0"
    json .= '  "version": "' . EscapeJsonString(version) . '"`n'

    json .= "}"
    return json
}

EscapeJsonString(str) {
    ; Simple JSON string escaping
    str := StrReplace(str, "\", "\\")
    str := StrReplace(str, '"', '\"')
    str := StrReplace(str, "`n", "\n")
    str := StrReplace(str, "`r", "\r")
    str := StrReplace(str, "`t", "\t")
    return str
}

ShowMainGui() {
    global mainGui

    ; Center the GUI on screen
    mainGui.Show()
}

ToggleGui() {
    global mainGui

    if (WinExist("ahk_id " . mainGui.Hwnd)) {
        mainGui.Hide()
    } else {
        ShowMainGui()
    }
}

AddSystemTrayMenu() {
    ; Clear default menu and add custom items
    A_TrayMenu.Delete()
    
    A_TrayMenu.Add("Show/Hide QuickNav", (*) => ToggleGui())
    A_TrayMenu.Add("Settings", (*) => ShowSettingsDialog())
    A_TrayMenu.Add()
    A_TrayMenu.Add("Exit", (*) => ExitApp())
    
    A_TrayMenu.Default := "Show/Hide QuickNav"
    TraySetIcon("shell32.dll", 44)
}

; Main GUI resize handler: dynamic layout, spacing, and bounds checks
MainGuiSize(thisGui, MinMax, Width, Height) {
    global mainGui, dpiScale
    if (MinMax = -1)
        return
    try {
        ; Fallback if event did not pass client size
        if (!Width || !Height) {
            thisGui.GetPos(, , &Width, &Height)
        }
        margin := Round(20 * dpiScale)
        pad := Round(10 * dpiScale)
        labelWidth := Round(130 * dpiScale)
        editMinW := Round(250 * dpiScale)
        btnWidth := Round(120 * dpiScale)
        btnHeight := Round(35 * dpiScale)
        btnSpacing := Round(20 * dpiScale)
        usableW := Max(Round(320 * dpiScale), Width - (margin * 2))
        
        ; Project label/edit
        if IsObject(mainGui.projectLabel) {
            mainGui.projectLabel.Move(margin)
        }
        if IsObject(mainGui.projectEdit) {
            mainGui.projectEdit.Move(margin + labelWidth + pad, , Max(editMinW, usableW - labelWidth - pad))
        }
        
        ; Stretch nav group
        if IsObject(mainGui.navGroup) {
            mainGui.navGroup.Move(margin, , usableW)
            mainGui.navGroup.GetPos(&ngX,&ngY,&ngW,&ngH)
            ; Radios spacing
            if IsObject(mainGui.folderModeRadio)
                mainGui.folderModeRadio.Move(ngX + pad + Round(10 * dpiScale), ngY + Round(20 * dpiScale), Round(220 * dpiScale), Round(24 * dpiScale))
            if IsObject(mainGui.docModeRadio)
                mainGui.docModeRadio.Move(ngX + pad + Round(10 * dpiScale), ngY + Round(50 * dpiScale), Round(220 * dpiScale), Round(24 * dpiScale))
            ; Settings bottom-right
            if IsObject(mainGui.settingsBtn)
                mainGui.settingsBtn.Move(ngX + ngW - Round(110 * dpiScale) - pad, ngY + ngH - btnHeight - pad, Round(110 * dpiScale), btnHeight)
        } else {
            ngY := margin + Round(40 * dpiScale), ngH := Round(120 * dpiScale)
            ngX := margin, ngW := usableW
        }
        
        ; Compute lower panels
        optionsHeight := Round(60 * dpiScale)
        statusHeight := Round(60 * dpiScale) ; increased for multi-line status text
        buttonsY := Height - margin - btnHeight
        statusY := buttonsY - pad - statusHeight
        optionsY := statusY - pad - optionsHeight

        optionsTopMin := ngY + ngH + pad
        if (optionsY < optionsTopMin)
            optionsY := optionsTopMin

        ; clamp status and buttons to avoid overlap on small heights
        if (statusY < optionsY + pad)
            statusY := optionsY + pad
        if (buttonsY < statusY + pad + btnHeight)
            buttonsY := statusY + pad + btnHeight

        midY := ngY + ngH + pad
        midH := Max(Round(120 * dpiScale), optionsY - midY - pad)
        
        ; Stretch groups
        if IsObject(mainGui.folderGroup)
            mainGui.folderGroup.Move(margin, midY, usableW, midH)
        if IsObject(mainGui.docGroup)
            mainGui.docGroup.Move(margin, midY, usableW, midH)
        
        ; Reflow folder radios if visible
        if (IsObject(mainGui.folderGroup) && mainGui.folderGroup.Visible) {
            mainGui.folderGroup.GetPos(&fgX,&fgY,&fgW,&fgH)
            innerPadX := Round(15 * dpiScale)
            innerW := Max(Round(200 * dpiScale), fgW - innerPadX * 2)
            columns := (innerW > Round(600 * dpiScale)) ? 2 : 1
            colW := innerW // columns
            radioCtrlH := Round(24 * dpiScale)
            rowH := Round(30 * dpiScale)
            topY := fgY + Round(20 * dpiScale)
            loop mainGui.folderRadios.Length {
                idx := A_Index
                col := Mod(idx - 1, columns)
                row := Floor((idx - 1) / columns)
                x := fgX + innerPadX + col * colW
                y := topY + row * rowH
                ; Clamp to avoid negative/overflow positions on small windows
                x := Max(margin, x)
                y := Max(midY + Round(5 * dpiScale), y)
                w := Max(Round(160 * dpiScale), colW - pad)
                mainGui.folderRadios[idx].Move(x, y, w, radioCtrlH)
            }
        }
        
        ; Stretch doc controls if visible
        if (IsObject(mainGui.docGroup) && mainGui.docGroup.Visible) {
            mainGui.docGroup.GetPos(&dgX,&dgY,&dgW,&dgH)
            fieldX := dgX + Round(110 * dpiScale)
            minX := dgX + Round(10 * dpiScale)
            minY := dgY + Round(10 * dpiScale)
            fieldX := Max(fieldX, minX)
            availW := Max(Round(120 * dpiScale), dgW - (fieldX - dgX) - pad)
            fieldW := Min(Round(200 * dpiScale), availW) ; cap width to keep fields aligned and readable

            ; Labels
            if IsObject(mainGui.docTypeLabel)
                mainGui.docTypeLabel.Move(dgX + Round(10 * dpiScale), dgY + Round(20 * dpiScale))
            if IsObject(mainGui.versionLabel)
                mainGui.versionLabel.Move(dgX + Round(10 * dpiScale), dgY + Round(50 * dpiScale))
            if IsObject(mainGui.roomLabel)
                mainGui.roomLabel.Move(dgX + Round(10 * dpiScale), dgY + Round(80 * dpiScale))
            if IsObject(mainGui.coLabel)
                mainGui.coLabel.Move(dgX + Round(125 * dpiScale), dgY + Round(80 * dpiScale))

            ; Fields (use same width for alignment)
            if IsObject(mainGui.docTypeCombo)
                mainGui.docTypeCombo.Move(fieldX, Max(minY, dgY + Round(20 * dpiScale)), fieldW)
            if IsObject(mainGui.versionCombo)
                mainGui.versionCombo.Move(fieldX, Max(minY, dgY + Round(50 * dpiScale)), fieldW)
            if IsObject(mainGui.roomEdit)
                mainGui.roomEdit.Move(fieldX, Max(minY, dgY + Round(80 * dpiScale)))
            if IsObject(mainGui.coEdit)
                mainGui.coEdit.Move(fieldX + Round(105 * dpiScale), Max(minY, dgY + Round(80 * dpiScale)))
            if IsObject(mainGui.archiveCheck) {
                xCheck := Max(minX, dgX + dgW - Round(150 * dpiScale) - pad)
                mainGui.archiveCheck.Move(xCheck, Max(minY, dgY + Round(80 * dpiScale)))
            }
        }
        
        ; Options group, status, and buttons
        if IsObject(mainGui.optionsGroup) {
            mainGui.optionsGroup.Move(margin, optionsY, usableW, optionsHeight)
            ; Reposition checkboxes inside options group responsively
            mainGui.optionsGroup.GetPos(&ogX,&ogY,&ogW,&ogH)
            ogPad := Round(10 * dpiScale)
            chkH := Round(24 * dpiScale)
            leftX := ogX + ogPad
            rightX := ogX + Round(190 * dpiScale)
            sameLine := (rightX + Round(160 * dpiScale) <= ogX + ogW - pad)
            if IsObject(mainGui.debugCheck)
                mainGui.debugCheck.Move(leftX, ogY + Round(20 * dpiScale))
            if IsObject(mainGui.trainingCheck) {
                if (sameLine)
                    mainGui.trainingCheck.Move(rightX, ogY + Round(20 * dpiScale))
                else
                    mainGui.trainingCheck.Move(leftX, ogY + Round(20 * dpiScale) + chkH + Round(6 * dpiScale))
            }
        }
        if IsObject(mainGui.statusText)
            mainGui.statusText.Move(margin, statusY, usableW, statusHeight)
        
        totalBtnW := (mainGui.openBtn.Visible ? btnWidth : (btnWidth * 2 + btnSpacing))
        btnStartX := margin + Floor((usableW - totalBtnW) / 2)
        if (mainGui.openBtn.Visible) {
            mainGui.openBtn.Move(btnStartX, buttonsY, btnWidth, btnHeight)
        } else {
            mainGui.findBtn.Move(btnStartX, buttonsY, btnWidth, btnHeight)
            mainGui.chooseBtn.Move(btnStartX + btnWidth + btnSpacing, buttonsY, btnWidth, btnHeight)
        }
    } catch {
        ; Ignore layout errors to prevent flicker during live resize
    }
}

; Helper to raise errors in a linter-friendly way
RaiseError(message) {
    ; AHK v2: wrap throw Error(...) to avoid some linters mis-parsing inline throw
    throw Error(message)
}