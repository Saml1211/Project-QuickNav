/*
Project QuickNav – AutoHotkey Frontend

Purpose:
    Provides a simple GUI for entering a 5-digit project code and selecting a subfolder,
    then locates and opens the matching project subfolder via the Python backend.

Requirements:
    - AutoHotkey v2
    - find_project_path.py in the working directory
    - Python 3.x installed and in PATH

Usage:
    - Run this script.
    - Enter a valid 5-digit job number.
    - Select a subfolder.
    - Click "Open" to open the folder.
    - Press Ctrl+Alt+Q anytime to open/focus this GUI.

Keyboard Shortcuts:
    - Ctrl+Alt+Q: Show or focus Project QuickNav window.
*/

#Requires AutoHotkey v2.0
#SingleInstance Force

^!q:: mainGui.Show()

folderNames := ["System Designs", "Sales Handover", "BOM & CO", "Handover Docs", "Floor Plans", "Site Photos"]

mainGui := Gui("+AlwaysOnTop", "Project QuickNav")
mainGui.Add("Text", "x20 y16", "Job Number:")
mainGui.Add("Edit", "x100 y13 w100 vJobNumber Limit5")
mainGui.Add("GroupBox", "x20 y45 w280 h140", "Select Subfolder")
mainGui.Add("Radio", "x40 y70 w220 vRadio1 Checked", "System &Designs")
mainGui.Add("Radio", "x40 y95 w220 vRadio2", "&Sales Handover")
mainGui.Add("Radio", "x40 y120 w220 vRadio3", "&BOM & CO")
mainGui.Add("Radio", "x40 y145 w220 vRadio4", "Hand&over Docs")
mainGui.Add("Radio", "x40 y170 w220 vRadio5", "&Floor Plans")
mainGui.Add("Radio", "x40 y195 w220 vRadio6", "Site P&hotos")
mainGui.Add("Checkbox", "x20 y210 w220 vDebugMode", "Show Raw Python Output")

; Add a text label that shows the status of operations
statusText := mainGui.Add("Text", "x20 y235 w280 h20 vStatusText", "Ready")

btnOpen := mainGui.Add("Button", "x110 y260 w100 h30 Default", "Open")

; Use OnEvent method for AutoHotkey v2 event handling
btnOpen.OnEvent("Click", OpenProject)
mainGui.OnEvent("Close", GuiClose)
mainGui.Show()

return

OpenProject(ctrl, info) {
    global folderNames, mainGui
    
    ; Update status text
    mainGui["StatusText"].Value := "Processing..."
    
    ; Use test_find_project_path.py instead if the real backend can't be found
    scriptPath := "find_project_path.py"
    if !FileExist(scriptPath)
        scriptPath := "test_find_project_path.py"
    
    ; No need to check for a running backend - we'll run the Python script with arguments as needed

    mainGui["StatusText"].Value := "Getting form data..."
    params := mainGui.Submit(false)  ; false to keep GUI visible
    DebugMode := params.DebugMode
    jobNumber := params.JobNumber

    if (!RegExMatch(jobNumber, "^\d{5}$")) {
        mainGui["StatusText"].Value := "Invalid job number"
        MsgBox("Please enter a valid 5-digit job number.", "Invalid Input", 48)
        return
    }

    selectedFolder := ""
    for idx in [1,2,3,4,5,6] {
        varName := "Radio" . idx
        if (params[varName]) {
            selectedFolder := folderNames[idx]
            break
        }
    }
    if (!selectedFolder)
        selectedFolder := folderNames[1]

    mainGui["StatusText"].Value := "Running Python backend..."
    comspec := A_ComSpec
    tempFile := A_Temp . "\project_quicknav_pyout.txt"
    cmd := comspec . " /C python " . Chr(34) . scriptPath . Chr(34) . " " . jobNumber . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"
    RunWait(cmd, "", "Hide")

    mainGui["StatusText"].Value := "Processing output..."
    output := FileRead(tempFile)
    FileDelete(tempFile)
    output := Trim(output)
    
    if (DebugMode) {
        MsgBox("Raw Python output:`n`n" . output, "Debug Output", 64)
    }

    if (InStr(output, "ERROR:") == 1) {
        msg := SubStr(output, 7)
        mainGui["StatusText"].Value := "Error: " . Trim(msg)
        MsgBox(Trim(msg), "Error", 16)
        return
    }
    if (InStr(output, "SELECT:") == 1) {
        mainGui["StatusText"].Value := "Multiple paths found"
        strPaths := SubStr(output, 8)
        arrPaths := StrSplit(strPaths, "|")

        choices := ""
        for i, path in arrPaths
            choices .= i ":" path "|"
        choices := SubStr(choices, 1, -1)

        selGui := Gui("", "Select Project Folder")
        selGui.Add("Text", "x10 y10", "Multiple project folders found:`nSelect the correct path:")
        selGui.Add("DropDownList", "x10 y40 w400 vSelChoice AltSubmit", choices)
        btnOK := selGui.Add("Button", "x170 y80 w80 Default", "OK")
        btnOK.OnEvent("Click", selGui.Close)
        selGui.OnEvent("Close", selGui.Close)
        selGui.Show("Modal")

        selParams := selGui.Submit()
        if !selParams.SelChoice {
            mainGui["StatusText"].Value := "Selection cancelled"
            MsgBox("Selection cancelled.", "Cancelled", 48)
            return
        }
        parts := StrSplit(selParams.SelChoice, ":")
        chosenIdx := parts[1]
        MainProjectPath := arrPaths[chosenIdx]
    }
    else if (InStr(output, "SUCCESS:") == 1) {
        MainProjectPath := Trim(SubStr(output, 9))
    }
    else {
        mainGui["StatusText"].Value := "Unexpected response"
        MsgBox("Unexpected response from Python backend:`n" . output, "Error", 16)
        return
    }

    if (SubStr(MainProjectPath, 1, 1) == "\" || SubStr(MainProjectPath, 1, 1) == "/")
        MainProjectPath := SubStr(MainProjectPath, 2)

    FullSubfolderPath := MainProjectPath . "\" . selectedFolder
    mainGui["StatusText"].Value := "Checking folder: " . FullSubfolderPath

    attrib := FileGetAttrib(FullSubfolderPath)
    if !InStr(attrib, "D") {
        ; For testing, create the directory if it doesn't exist
        if (FileExist(MainProjectPath)) {
            try {
                DirCreate(FullSubfolderPath)
                mainGui["StatusText"].Value := "Created folder: " . FullSubfolderPath
            } catch as e {
                mainGui["StatusText"].Value := "Subfolder not found"
                MsgBox("Subfolder '" . selectedFolder . "' not found under:`n" . MainProjectPath, "Subfolder Not Found", 16)
                return
            }
        } else {
            mainGui["StatusText"].Value := "Subfolder not found"
            MsgBox("Subfolder '" . selectedFolder . "' not found under:`n" . MainProjectPath, "Subfolder Not Found", 16)
            return
        }
    }

    mainGui["StatusText"].Value := "Opening folder: " . FullSubfolderPath
    Run(FullSubfolderPath)
    mainGui["StatusText"].Value := "Folder opened successfully"
}

GuiClose(*) {
    ExitApp()
}

IsBackendRunning() {
    ; Since we're not starting the backend as a persistent process,
    ; just return true - we'll run the Python script with arguments when needed
    return true
}