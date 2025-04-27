/*
Project QuickNav – AutoHotkey Frontend

Purpose:
    Provides a simple GUI for entering a 5-digit project code and selecting a subfolder,
    then locates and opens the matching project subfolder via the Python backend.

Requirements:
    - AutoHotkey v1.1 or v2
    - find_project_path.py in the working directory
    - Python 3.x installed and in PATH

Usage:
    1. Run this script (double-click or via shortcut).
    2. Enter a valid 5-digit job number.
    3. Select the desired subfolder type.
    4. Click 'Open' to locate and open the folder in Explorer.
    5. Press Ctrl+Alt+Q globally to open or focus the QuickNav GUI window at any time.

Integration:
    Communicates with the Python backend via stdout capture and parses results per protocol.

Keyboard Shortcuts:
    - Ctrl+Alt+Q: Opens or focuses the Project QuickNav window globally (does not launch a second instance).
*/

#SingleInstance Force

; --- Global hotkey: Ctrl+Alt+Q opens or focuses the QuickNav window ---
^!q::
    ; This routine runs when Ctrl+Alt+Q is pressed globally.
    ; If the QuickNav window is already open, bring it to the foreground and restore if minimized.
    ; Otherwise, show the GUI window.
    WinTitle := "Project QuickNav"
    if WinExist(WinTitle)
    {
        ; If window found, activate and restore if needed
        WinActivate
        WinRestore
    }
    else
    {
        ; Otherwise, show the GUI window
        Gui, Show, w320 h270, %WinTitle%
    }
return

; --- GUI Definition ---
Gui, +AlwaysOnTop
Gui, Add, Text, x20 y16, Job Number:
Gui, Add, Edit, x100 y13 w100 vJobNumber Limit5

Gui, Add, GroupBox, x20 y45 w280 h140, Select Subfolder
Gui, Add, Radio, x40 y70 w220 vSelectedFolder Checked, System &Designs
Gui, Add, Radio, x40 y95 w220, &Sales Handover
Gui, Add, Radio, x40 y120 w220, &BOM & CO
Gui, Add, Radio, x40 y145 w220, Hand&over Docs
Gui, Add, Radio, x40 y170 w220, &Floor Plans
Gui, Add, Radio, x40 y195 w220, Site P&hotos

Gui, Add, Button, x110 y225 w100 h30 Default gOpenProject, &Open

Gui, Show, w320 h270, Project QuickNav
Return

; --- Main logic for handling 'Open' button click ---
OpenProject:
    Gui, Submit, NoHide

    JobNumber := JobNumber
    ; Validate: must be 5 digits only
    if (!RegExMatch(JobNumber, "^\d{5}$")) {
        MsgBox, 48, Invalid Input, Please enter a valid 5-digit job number.
        Return
    }

    ; --- Map selected radio to subfolder name ---
    folderNames := ["System Designs", "Sales Handover", "BOM & CO", "Handover Docs", "Floor Plans", "Site Photos"]
    SelectedFolder := ""
    Loop, 6
    {
        GuiControlGet, checked, , % "Button" . (A_Index + 1) ; Button2..Button7 (radios)
        if (checked)
        {
            SelectedFolder := folderNames[A_Index - 1]
            Break
        }
    }
    ; Fallback if not found (shouldn't happen)
    if (SelectedFolder = "")
        SelectedFolder := folderNames[1]

    ; --- Prepare call to Python backend (find_project_path.py) ---
    pythonExe := "python"
    scriptPath := "find_project_path.py"
    cmd := pythonExe . " """ . scriptPath . """ " . JobNumber

    ; --- Run Python script, capture stdout via temp file ---
    tempFile := A_Temp . "\project_quicknav_pyout.txt"
    RunWait, %ComSpec% /C %cmd% > "%tempFile%", , Hide

    FileRead, output, %tempFile%
    FileDelete, %tempFile%

    output := Trim(output)

    ; --- Interpret protocol from backend: ERROR, SELECT (multiple), or SUCCESS ---
    if (SubStr(output, 1, 6) = "ERROR:") {
        Msg := SubStr(output, 7)
        MsgBox, 16, Error, % Trim(Msg)
        Return
    } else if (SubStr(output, 1, 7) = "SELECT:") {
        paths := SubStr(output, 8)
        stringsplit, pathArr, paths, |
        choiceList := ""
        Loop, %pathArr0%
        {
            idx := A_Index
            choiceList .= idx . ": " . pathArr%A_Index% . "|"
        }
        ; Remove final |
        choiceList := SubStr(choiceList, 1, StrLen(choiceList)-1)
        ; --- Show selection dialog for multiple folders found ---
        Gui, 2:New
        Gui, 2:Add, Text, x10 y10, Multiple project folders found:`nSelect the correct path:
        Gui, 2:Add, DropDownList, x10 y40 w400 vChosenPath AltSubmit, %choiceList%
        Gui, 2:Add, Button, x170 y80 w80 gChoosePathOk Default, &OK
        Gui, 2:Show, w430 h120, Select Project Folder
        ; Modal
        WinWaitClose, Select Project Folder
        if (!ChosenPath) {
            MsgBox, 48, Cancelled, Selection cancelled.
            Return
        }
        chosenIdx := RegExReplace(ChosenPath, "^(\d+):.*", "$1")
        MainProjectPath := pathArr%chosenIdx%
        if (MainProjectPath = "") {
            MsgBox, 16, Error, No selection made.
            Return
        }
    } else if (SubStr(output, 1, 8) = "SUCCESS:") {
        MainProjectPath := Trim(SubStr(output, 9))
    } else {
        MsgBox, 16, Error, Unexpected response from Python backend:`n%output%
        Return
    }

    ; --- Combine project path with selected subfolder ---
    if (SubStr(MainProjectPath, 0) = "\" || SubStr(MainProjectPath, 0) = "/")
        MainProjectPath := SubStr(MainProjectPath, 1, StrLen(MainProjectPath)-1)
    FullSubfolderPath := MainProjectPath . "\" . SelectedFolder

    ; --- Ensure subfolder exists before opening ---
    if !(FileExist(FullSubfolderPath) && InStr(FileExist(FullSubfolderPath), "D")) {
        MsgBox, 16, Subfolder Not Found, Subfolder '[%SelectedFolder%]' not found under:`n%MainProjectPath%
        Return
    }

    Run, %FullSubfolderPath%
    ExitApp
Return

; --- Dialog event handlers for selection dialog and GUI close ---

2ButtonOK:
ChoosePathOk:
    Gui, 2:Submit
    Gui, 2:Destroy
Return

GuiClose:
    ExitApp
Return

2GuiClose:
    Gui, 2:Destroy
Return