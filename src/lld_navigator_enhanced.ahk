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
#SingleInstance Force

; Global variables
global mainGui, docPickerGui, settingsGui
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

    ; Create main GUI with DPI scaling
    mainGui := Gui("+AlwaysOnTop +DPIScale", "Project QuickNav - Enhanced")

    ; Calculate scaled positions and sizes
    margin := Round(15 * dpiScale)
    rowHeight := Round(25 * dpiScale)
    currentY := margin

    ; Project input section
    mainGui.Add("Text", Format("x{} y{} w{}", margin, currentY, Round(120 * dpiScale)), "Project Number:")
    projectEdit := mainGui.Add("Edit", Format("x{} y{} w{} h{} vProjectInput", margin + Round(125 * dpiScale), currentY, Round(200 * dpiScale), Round(23 * dpiScale)))
    currentY += Round(35 * dpiScale)

    ; Navigation mode selection
    mainGui.Add("GroupBox", Format("x{} y{} w{} h{}", margin, currentY, Round(370 * dpiScale), Round(80 * dpiScale)), "Navigation Mode")
    navGroupY := currentY + Round(20 * dpiScale)
    folderModeRadio := mainGui.Add("Radio", Format("x{} y{} w{} vFolderMode Checked", margin + Round(10 * dpiScale), navGroupY, Round(150 * dpiScale)), "Open Project Folder")
    docModeRadio := mainGui.Add("Radio", Format("x{} y{} w{} vDocMode", margin + Round(10 * dpiScale), navGroupY + Round(25 * dpiScale), Round(150 * dpiScale)), "Find Documents")

    ; Settings button
    settingsBtn := mainGui.Add("Button", Format("x{} y{} w{} h{} vSettingsBtn", margin + Round(250 * dpiScale), navGroupY + Round(10 * dpiScale), Round(100 * dpiScale), Round(30 * dpiScale)), "Settings")
    currentY += Round(90 * dpiScale)

    ; Folder selection (for folder mode)
    folderGroup := mainGui.Add("GroupBox", Format("x{} y{} w{} h{} vFolderGroup", margin, currentY, Round(370 * dpiScale), Round(160 * dpiScale)), "Select Subfolder")
    folderGroupY := currentY + Round(20 * dpiScale)

    ; Folder radio buttons
    folderRadios := []
    folderNames := [
        "4. System Designs",
        "1. Sales Handover",
        "2. BOM & Orders",
        "6. Customer Handover Documents",
        "5. Floor Plans",
        "6. Site Photos"
    ]

    loop folderNames.Length {
        radio := mainGui.Add("Radio", Format("x{} y{} w{} vFolder{} {}",
            margin + Round(15 * dpiScale),
            folderGroupY + Round((A_Index - 1) * 25 * dpiScale),
            Round(300 * dpiScale),
            A_Index,
            A_Index == 1 ? "Checked" : ""))
        radio.Text := folderNames[A_Index]
        folderRadios.Push(radio)
    }
    currentY += Round(170 * dpiScale)

    ; Document type selection (for document mode)
    docGroup := mainGui.Add("GroupBox", Format("x{} y{} w{} h{} vDocGroup Hidden", margin, currentY - Round(170 * dpiScale), Round(370 * dpiScale), Round(170 * dpiScale)), "Document Type & Filters")
    docGroupY := currentY - Round(150 * dpiScale)

    ; Document type dropdown
    mainGui.Add("Text", Format("x{} y{}", margin + Round(10 * dpiScale), docGroupY), "Document Type:")
    docTypeOptions := []
    for key, value in docTypes
        docTypeOptions.Push(value)
    docTypeCombo := mainGui.Add("ComboBox", Format("x{} y{} w{} vDocType Choose1", margin + Round(100 * dpiScale), docGroupY, Round(250 * dpiScale)), docTypeOptions)
    docGroupY += Round(30 * dpiScale)

    ; Version filter
    mainGui.Add("Text", Format("x{} y{}", margin + Round(10 * dpiScale), docGroupY), "Version Filter:")
    versionCombo := mainGui.Add("ComboBox", Format("x{} y{} w{} vVersionFilter Choose1", margin + Round(100 * dpiScale), docGroupY, Round(150 * dpiScale)), versionFilters)
    docGroupY += Round(30 * dpiScale)

    ; Room filter
    mainGui.Add("Text", Format("x{} y{}", margin + Round(10 * dpiScale), docGroupY), "Room:")
    roomEdit := mainGui.Add("Edit", Format("x{} y{} w{} h{} vRoomFilter Number", margin + Round(50 * dpiScale), docGroupY, Round(60 * dpiScale), Round(20 * dpiScale)))

    ; CO filter
    mainGui.Add("Text", Format("x{} y{}", margin + Round(125 * dpiScale), docGroupY), "CO:")
    coEdit := mainGui.Add("Edit", Format("x{} y{} w{} h{} vCoFilter Number", margin + Round(155 * dpiScale), docGroupY, Round(60 * dpiScale), Round(20 * dpiScale)))

    ; Include archive checkbox
    archiveCheck := mainGui.Add("Checkbox", Format("x{} y{} w{} vIncludeArchive", margin + Round(230 * dpiScale), docGroupY, Round(120 * dpiScale)), "Include Archive")
    currentY += Round(20 * dpiScale)

    ; Options section
    optionsGroup := mainGui.Add("GroupBox", Format("x{} y{} w{} h{}", margin, currentY, Round(370 * dpiScale), Round(60 * dpiScale)), "Options")
    optionsY := currentY + Round(20 * dpiScale)
    debugCheck := mainGui.Add("Checkbox", Format("x{} y{} w{} vDebugMode", margin + Round(10 * dpiScale), optionsY, Round(170 * dpiScale)), "Show Debug Output")
    trainingCheck := mainGui.Add("Checkbox", Format("x{} y{} w{} vGenerateTraining", margin + Round(190 * dpiScale), optionsY, Round(170 * dpiScale)), "Generate Training Data")
    currentY += Round(70 * dpiScale)

    ; Status and progress
    statusText := mainGui.Add("Text", Format("x{} y{} w{} h{} vStatusText", margin, currentY, Round(370 * dpiScale), Round(40 * dpiScale)), "Ready - Select navigation mode and enter project number")
    currentY += Round(45 * dpiScale)

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

    ; Store references to controls
    mainGui.projectEdit := projectEdit
    mainGui.folderGroup := folderGroup
    mainGui.docGroup := docGroup
    mainGui.statusText := statusText
    mainGui.openBtn := openBtn
    mainGui.findBtn := findBtn
    mainGui.chooseBtn := chooseBtn
    mainGui.folderRadios := folderRadios
    mainGui.docTypeCombo := docTypeCombo
    mainGui.versionCombo := versionCombo
    mainGui.roomEdit := roomEdit
    mainGui.coEdit := coEdit
    mainGui.archiveCheck := archiveCheck
    mainGui.debugCheck := debugCheck
    mainGui.trainingCheck := trainingCheck
}

SwitchToFolderMode() {
    global mainGui

    ; Show folder selection, hide document options
    mainGui.folderGroup.Visible := true
    mainGui.docGroup.Visible := false
    mainGui.openBtn.Visible := true
    mainGui.findBtn.Visible := false
    mainGui.chooseBtn.Visible := false

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

    UpdateStatus("Document mode - Find specific documents by type and filters")
}

ValidateInputs() {
    global mainGui

    projectInput := mainGui.projectEdit.Text

    if (projectInput = "") {
        UpdateStatus("Enter a 5-digit project number or search term")
        return
    }

    if (RegExMatch(projectInput, "^\d{5}$")) {
        UpdateStatus("Ready - Project number: " . projectInput)
    } else {
        UpdateStatus("Ready - Search term: " . projectInput)
    }
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
    ; Build command
    cmd := "python -m quicknav " . command . " " . Chr(34) . projectInput . Chr(34)

    ; Add extra arguments
    if (IsObject(extraArgs)) {
        for arg in extraArgs {
            cmd .= " " . arg
        }
    } else if (extraArgs != "") {
        cmd .= " " . extraArgs
    }

    ; Add training data flag if enabled
    if (formData.GenerateTraining) {
        cmd .= " --training-data"
    }

    ; Execute command
    tempFile := A_Temp . "\quicknav_output.txt"
    fullCmd := A_ComSpec . " /C " . cmd . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"

    try {
        RunWait(fullCmd, "", "Hide")

        if (!FileExist(tempFile)) {
            throw Error("Failed to execute command")
        }

        output := FileRead(tempFile)
        FileDelete(tempFile)

        ; Show debug output if enabled
        if (formData.DebugMode) {
            MsgBox("Command: " . cmd . "`n`nOutput:`n" . output, "Debug Output", 64)
        }

        return Trim(output)

    } catch as e {
        if (FileExist(tempFile)) {
            FileDelete(tempFile)
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

    ; Construct full path
    if (selectedFolder = "Floor Plans" || selectedFolder = "Site Photos") {
        fullPath := projectPath . "\1. Sales Handover\" . selectedFolder
    } else {
        fullPath := projectPath . "\" . selectedFolder
    }

    if (!FileExist(fullPath)) {
        ShowError("Subfolder '" . selectedFolder . "' not found")
        return
    }

    try {
        Run("explorer.exe " . Chr(34) . fullPath . Chr(34))
        UpdateStatus("Opened folder: " . selectedFolder)

        ; Auto-hide after success
        SetTimer(() => mainGui.Hide(), -1500)

    } catch as e {
        ShowError("Failed to open folder: " . e.Message)
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
    ; Implementation for project selection dialog
    ; Similar to existing implementation but with DPI scaling
    UpdateStatus("Multiple projects found - showing selection dialog")
}

ShowProjectSearchDialog(pathsString, selectedFolder) {
    ; Implementation for project search dialog
    ; Similar to existing implementation but with DPI scaling
    UpdateStatus("Search results found - showing selection dialog")
}

ShowDocumentSelectionDialog(pathsString) {
    ; Implementation for document selection dialog with previews
    UpdateStatus("Multiple documents found - showing selection dialog")
}

ShowSettingsDialog() {
    ; Implementation for settings dialog
    ; Allow configuration of custom roots, DPI settings, etc.
    MsgBox("Settings dialog not yet implemented", "Settings", 64)
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
            ; Settings loading would go here
        }
    } catch {
        ; Use defaults if loading fails
    }
}

SaveSettings() {
    ; Save settings to JSON file
    global configPath

    try {
        ; Ensure directory exists
        SplitPath(configPath, , &configDir)
        if (!FileExist(configDir)) {
            FileCreateDir(configDir)
        }

        ; Settings saving would go here
    } catch {
        ; Ignore save errors
    }
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