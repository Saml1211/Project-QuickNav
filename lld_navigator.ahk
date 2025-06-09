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

; Function to toggle the GUI (show if hidden, hide if visible)
ToggleGui() {
    global mainGui

    ; Check if the GUI exists and is visible
    if WinExist("ahk_id " . mainGui.Hwnd) {
        ; If the GUI is visible, hide it
        mainGui.Hide()
    } else {
        ; If the GUI is not visible, show it
        mainGui.Show()
    }
}

; Assign Ctrl+Alt+Q to toggle the GUI
^!q:: ToggleGui()

folderNames := [
    "4. System Designs",
    "1. Sales Handover",
    "2. BOM & Orders",
    "6. Customer Handover Documents",
    "Floor Plans",
    "Site Photos"
]

mainGui := Gui("+AlwaysOnTop", "Project QuickNav")
mainGui.Add("Text", "x20 y16", "Job Number or Search:")
mainGui.Add("Edit", "x150 y13 w150 vJobNumber")
mainGui.Add("GroupBox", "x20 y40 w280 h185", "Select Subfolder")
mainGui.Add("Radio", "x40 y65 w220 vRadio1 Checked", "System Designs")
mainGui.Add("Radio", "x40 y90 w220 vRadio2", "Sales Handover")
mainGui.Add("Radio", "x40 y115 w220 vRadio3", "BOM && Orders")
mainGui.Add("Radio", "x40 y140 w220 vRadio4", "Customer Handover Documents")
mainGui.Add("Radio", "x40 y165 w220 vRadio5", "Floor Plans")
mainGui.Add("Radio", "x40 y190 w220 vRadio6", "Site Photos")
mainGui.Add("Checkbox", "x20 y235 w220 vDebugMode", "Show Raw Python Output")
mainGui.Add("Checkbox", "x20 y255 w220 vGenerateTrainingData", "Generate Training Data")
mainGui.Add("Progress", "x20 y275 w280 h1 Disabled -Theme")

; Add a text label that shows the status of operations
statusText := mainGui.Add("Text", "x20 y280 w280 h60 vStatusText", "Ready")

btnOpen := mainGui.Add("Button", "x110 y315 w100 h30 Default", "Open")

; Use OnEvent method for AutoHotkey v2 event handling
btnOpen.OnEvent("Click", OpenProject)
mainGui.OnEvent("Close", GuiClose)

; Initialize the system tray menu
AddSystemTrayMenu()

mainGui.Show()

return

; Function to reset the GUI to its initial state
ResetGUI() {
    global mainGui
    mainGui["StatusText"].Value := "Ready"
    ; Re-enable the Open button if it was disabled
    btnOpen.Enabled := true
}

OpenProject(ctrl, info) {
    global folderNames, mainGui, btnOpen

    ; Disable the Open button to prevent multiple clicks
    btnOpen.Enabled := false

    ; Update status text
    mainGui["StatusText"].Value := "Processing..."

    ; Use test_find_project_path.py instead if the real backend can't be found
    scriptPath := "find_project_path.py"
    if !FileExist(scriptPath)
        scriptPath := "test_find_project_path.py"

    ; Get form data
    params := mainGui.Submit(false)  ; false to keep GUI visible
    DebugMode := params.DebugMode
    GenerateTrainingData := params.GenerateTrainingData
    jobNumber := params.JobNumber

    ; Check if the input is empty
    if (jobNumber = "") {
        mainGui["StatusText"].Value := "Empty input"
        MsgBox("Please enter a job number or search term.", "Empty Input", 48)
        ResetGUI()
        return
    }

    ; If it's a 5-digit number, treat it as a job number
    ; Otherwise, treat it as a search term
    isJobNumber := RegExMatch(jobNumber, "^\d{5}$")

    ; Display what we're doing
    if (isJobNumber) {
        mainGui["StatusText"].Value := "Searching for job number: " . jobNumber
    } else {
        mainGui["StatusText"].Value := "Searching for projects containing: " . jobNumber
    }

    ; Get selected folder
    selectedFolder := ""
    for idx in [1, 2, 3, 4, 5, 6] {
        varName := "Radio" . idx
        ; Use HasOwnProp() method for key existence check and GetProp() for value check
        if (params.HasOwnProp(varName) && params.%varName%) {
            selectedFolder := folderNames[idx]
            break
        }
    }
    if (!selectedFolder)
        selectedFolder := folderNames[1]

    ; Run the Python script to find the project folder
    mainGui["StatusText"].Value := "Searching for project folder..."

    ; Prepare the command to run the Python script
    comspec := A_ComSpec
    tempFile := A_Temp . "\project_quicknav_pyout.txt"
    
    ; Build the command with optional training data flag
    trainingFlag := GenerateTrainingData ? " --training-data" : ""
    cmd := comspec . " /C python " . Chr(34) . scriptPath . Chr(34) . " " . jobNumber . trainingFlag . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"

    try {
        ; Run the Python script and wait for it to complete
        RunWait(cmd, "", "Hide")

        ; Check if the output file exists
        if !FileExist(tempFile) {
            mainGui["StatusText"].Value := "Error: Python execution failed"
            MsgBox("Failed to execute Python script. Make sure Python is installed and in your PATH.", "Python Error", 16)
            ResetGUI()
            return
        }

        ; Read the output from the Python script
        output := FileRead(tempFile)
        FileDelete(tempFile)
        output := Trim(output)

        ; Show debug output if enabled
        if (DebugMode) {
            MsgBox("Raw Python output:`n`n" . output, "Debug Output", 64)
        }

        ; Check for training data messages and handle them
        if (InStr(output, "TRAINING:") > 0) {
            ; Extract training message
            lines := StrSplit(output, "`n")
            for i, line in lines {
                if (InStr(line, "TRAINING:") == 1) {
                    trainingMsg := SubStr(line, 10)
                    MsgBox("Training data generated successfully:`n" . trainingMsg, "Training Data", 64)
                    break
                }
            }
        }
        else if (InStr(output, "TRAINING_ERROR:") > 0) {
            ; Extract training error message
            lines := StrSplit(output, "`n")
            for i, line in lines {
                if (InStr(line, "TRAINING_ERROR:") == 1) {
                    trainingError := SubStr(line, 16)
                    MsgBox("Training data generation failed:`n" . trainingError, "Training Error", 16)
                    break
                }
            }
        }

        ; Process the output from the Python script
        if (InStr(output, "ERROR:") == 1) {
            ; Handle error response
            msg := SubStr(output, 7)
            mainGui["StatusText"].Value := "Error: " . Trim(msg)
            MsgBox(Trim(msg), "Error", 16)
            ResetGUI()
            return
        }
        else if (InStr(output, "SELECT:") == 1) {
            ; Handle multiple exact matches
            mainGui["StatusText"].Value := "Multiple paths found"
            strPaths := SubStr(output, 8)
            arrPaths := StrSplit(strPaths, "|")

            ; Create a selection dialog
            choices := ""
            for i, path in arrPaths
                choices .= i ":" path "|"
            choices := SubStr(choices, 1, -1)

            ; Create global variables for the selection GUI
            global selGui, selResult

            ; Create a function to handle the OK button click
            OkHandler(thisCtrl, *) {
                selResult := selGui.Submit()
                selGui.Destroy()
            }

            ; Create a function to handle the Cancel/Close action
            CloseHandler(thisGui, *) {
                selResult := {SelChoice: 0}
                selGui.Destroy()
            }

            ; Initialize the selection result
            selResult := {SelChoice: 0}

            ; Create the selection GUI
            selGui := Gui("", "Select Project Folder")
            selGui.Add("Text", "x10 y10", "Multiple project folders found:`nSelect the correct path:")
            selGui.Add("DropDownList", "x10 y40 w400 vSelChoice AltSubmit Choose1", choices)
            btnOK := selGui.Add("Button", "x170 y80 w80 Default", "OK")
            btnOK.OnEvent("Click", OkHandler)
            selGui.OnEvent("Close", CloseHandler)

            ; Show the GUI and wait for it to close
            selGui.Show("Modal")

            ; Process the selection result
            if !selResult.SelChoice {
                mainGui["StatusText"].Value := "Selection cancelled"
                MsgBox("Selection cancelled.", "Cancelled", 48)
                ResetGUI()
                return
            }

            parts := StrSplit(selResult.SelChoice, ":")
            chosenIdx := parts[1]
            MainProjectPath := arrPaths[chosenIdx]
        }
        else if (InStr(output, "SEARCH:") == 1) {
            ; Handle search results (multiple matches from name search)
            mainGui["StatusText"].Value := "Search results found"
            strPaths := SubStr(output, 8)
            arrPaths := StrSplit(strPaths, "|")

            ; Create global variables for the search results GUI
            global searchGui, searchResult

            ; Function to handle window resize
            SearchGuiSize(thisGui, MinMax, Width, Height) {
                if (MinMax = -1)  ; If window is minimized, do nothing
                    return

                ; Resize the ListView to fit the window
                thisGui["SearchList"].Move(,, Width - 20, Height - 80)

                ; Reposition the buttons
                thisGui["BtnOpen"].Move(Width - 170, Height - 40)
                thisGui["BtnCancel"].Move(Width - 80, Height - 40)
            }

            ; Create a function to handle the OK button click
            SearchOkHandler(thisCtrl, *) {
                global searchGui, searchResult
                ; Get the selected row from the ListView
                LV := searchGui["SearchList"]
                selectedRow := LV.GetNext()
                if (selectedRow > 0) {
                    ; Get the full path from the third column
                    searchResult := LV.GetText(selectedRow, 3)
                }
                searchGui.Destroy()
            }

            ; Create a function to handle the Cancel/Close action
            SearchCloseHandler(thisGui, *) {
                global searchResult, searchGui
                searchResult := ""
                searchGui.Destroy()
            }

            ; Initialize the search result
            searchResult := ""

            ; Create the search results GUI
            searchGui := Gui("+Resize", "Search Results")
            searchGui.Add("Text", "x10 y10", "Found " . arrPaths.Length . " project folders matching your search:")

            ; Add a ListView to display the search results
            LV := searchGui.Add("ListView", "x10 y30 w600 h300 vSearchList -Multi", ["Project Number", "Project Name", "Full Path"])

            ; Set column widths
            LV.ModifyCol(1, 100)  ; Project Number
            LV.ModifyCol(2, 250)  ; Project Name
            LV.ModifyCol(3, 250)  ; Full Path

            ; Populate the ListView with search results
            for i, path in arrPaths {
                ; Extract project number and name from the path
                SplitPath(path, &fileName, &dirPath)

                ; Try to extract project number and name using regex
                if (RegExMatch(fileName, "^(\d{5}) - (.+)$", &match)) {
                    projNum := match[1]
                    projName := match[2]
                } else {
                    projNum := "N/A"
                    projName := fileName
                }

                ; Add the row to the ListView
                LV.Add("", projNum, projName, path)
            }

            ; Add buttons
            btnOK := searchGui.Add("Button", "x450 y340 w80 Default vBtnOpen", "Open")
            btnCancel := searchGui.Add("Button", "x540 y340 w80 vBtnCancel", "Cancel")

            ; Set up event handlers
            LV.OnEvent("DoubleClick", SearchOkHandler)
            btnOK.OnEvent("Click", SearchOkHandler)
            btnCancel.OnEvent("Click", SearchCloseHandler)
            searchGui.OnEvent("Close", SearchCloseHandler)

            ; Handle window resize
            searchGui.OnEvent("Size", SearchGuiSize)

            ; Show the GUI and wait for it to close
            searchGui.Show("w620 h380")
            WinWaitClose("ahk_id " . searchGui.Hwnd)

            ; Process the selection result
            if (searchResult = "") {
                mainGui["StatusText"].Value := "Selection cancelled"
                MsgBox("No project selected.", "Cancelled", 48)
                ResetGUI()
                return
            }

            MainProjectPath := searchResult
        }
        else if (InStr(output, "SUCCESS:") == 1) {
            ; Handle success response
            MainProjectPath := Trim(SubStr(output, 9))
        }
        else {
            ; Handle unexpected response
            mainGui["StatusText"].Value := "Unexpected response"
            MsgBox("Unexpected response from Python backend:`n" . output, "Error", 16)
            ResetGUI()
            return
        }

        ; Clean up the path - remove any leading/trailing quotes or spaces
        MainProjectPath := Trim(MainProjectPath, " `t`n`r`"")

        ; Remove leading slash if present
        if (SubStr(MainProjectPath, 1, 1) == "\" || SubStr(MainProjectPath, 1, 1) == "/")
            MainProjectPath := SubStr(MainProjectPath, 2)

        ; Construct the full subfolder path
        ; For "Floor Plans" and "Site Photos", make them subfolders of "1. Sales Handover"
        if (selectedFolder = "Floor Plans" || selectedFolder = "Site Photos") {
            FullSubfolderPath := MainProjectPath . "\1. Sales Handover\" . selectedFolder
        } else {
            FullSubfolderPath := MainProjectPath . "\" . selectedFolder
        }

        ; Debug the path if enabled
        if (DebugMode) {
            MsgBox("Main Project Path: " . MainProjectPath . "`nSelected Folder: " . selectedFolder . "`nFull Path: " . FullSubfolderPath, "Path Debug", 64)
        }
        mainGui["StatusText"].Value := "Checking folder: " . FullSubfolderPath

        ; Check if the subfolder exists
        if !FileExist(FullSubfolderPath) {
            mainGui["StatusText"].Value := "Subfolder not found"
            MsgBox("Subfolder '" . selectedFolder . "' not found under:`n" . MainProjectPath, "Subfolder Not Found", 16)
            ResetGUI()
            return
        }

        ; Open the folder in Windows Explorer
        mainGui["StatusText"].Value := "Opening folder: " . FullSubfolderPath

        ; Make sure the path is properly formatted for Windows Explorer
        ; Escape any special characters and ensure proper quotes
        explorerPath := "explorer.exe " . Chr(34) . FullSubfolderPath . Chr(34)

        try {
            ; Try using Run with explorer.exe explicitly
            RunWait(explorerPath)
            mainGui["StatusText"].Value := "Folder opened successfully"

            ; If debug mode is enabled, show the command that was executed
            if (DebugMode) {
                MsgBox("Command executed: " . explorerPath, "Debug Info", 64)
            }
        } catch as e {
            ; If that fails, try an alternative method
            try {
                ; Try using ShellRun to open the folder
                Run("explorer.exe /select," . Chr(34) . FullSubfolderPath . Chr(34))
                mainGui["StatusText"].Value := "Folder opened successfully (alternative method)"
            } catch as e2 {
                ; If both methods fail, show an error
                mainGui["StatusText"].Value := "Error opening folder: " . e2.Message
                MsgBox("Failed to open folder: " . e2.Message . "`n`nPath: " . FullSubfolderPath, "Error", 16)
                ResetGUI()
                return
            }
        }

        ; Wait a moment before resetting the GUI to show the success message
        SetTimer(() => ResetGUI(), -2000)  ; Reset GUI after 2 seconds
    }
    catch as e {
        ; Handle any exceptions
        mainGui["StatusText"].Value := "Error: " . e.Message
        MsgBox("An error occurred: " . e.Message, "Error", 16)
        ResetGUI()
    }
}

; Handle the GUI close event (X button)
GuiClose(*) {
    ; Hide the GUI instead of exiting the app
    mainGui.Hide()
    return
}

; Add a system tray menu to allow exiting the application
; This gives users a way to completely exit the app if needed
AddSystemTrayMenu() {
    global

    ; Create a tray menu
    A_TrayMenu.Delete() ; Clear default menu

    ; Create menu callback functions with correct parameter signatures
    ToggleGuiMenu(ItemName, ItemPos, MyMenu) {
        ToggleGui()
    }

    ExitAppMenu(ItemName, ItemPos, MyMenu) {
        ExitApp()
    }

    ; Add menu items with proper callback functions
    A_TrayMenu.Add("Show/Hide QuickNav", ToggleGuiMenu)
    A_TrayMenu.Add() ; Add a separator
    A_TrayMenu.Add("Exit", ExitAppMenu)

    ; Set the default item using the correct method
    try {
        ; Try to set the default item
        A_TrayMenu.Default := "Show/Hide QuickNav"
    } catch {
        ; If that fails, just continue without setting a default
    }

    ; Set a custom tray icon (optional)
    TraySetIcon("shell32.dll", 44) ; Use a folder icon from shell32.dll
}

; Function to exit the application
ExitApp(*) {
    ExitApp()
}

; IsBackendRunning function definition
IsBackendRunning() {
    ; Since we're not starting the backend as a persistent process,
    ; just return true - we'll run the Python script with arguments when needed
    return true
}