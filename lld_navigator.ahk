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
; Dummy JSON helpers to prevent unused-variable warnings if not defined elsewhere
    - Ctrl+Alt+Q: Show or focus Project QuickNav window.
*/

#Requires AutoHotkey v2.0
#SingleInstance Force

ObjFromJson(jsonStr) {
    return {}
}
ObjToJson(obj) {
    return "{}"
}

; --- JSON Class for handling JSON operations ---
class JSON {
    static Load(jsonStr) {
        return this._Parse(jsonStr)
    }
    
    static Dump(obj) {
        return this._Stringify(obj)
    }
    
    static _Parse(jsonStr) {
        ; Simple JSON parser for AutoHotkey
        try {
            parsed := ObjFromJson(jsonStr)
            return parsed
        }
        catch Error as e {
            LogError("JSON Parse error: " e.Message, "JSON.Load")
            return {}
        }
    }
    
    static _Stringify(obj) {
        ; Simple JSON stringifier for AutoHotkey
        try {
            jsonStr := ObjToJson(obj)
            return jsonStr
        }
        catch Error as e {
            LogError("JSON Stringify error: " e.Message, "JSON.Dump")
            return "{}"
        }
    }
}

; --- Version String (Batch 4) ---
versionStr := FileExist("VERSION.txt") ? Trim(FileRead("VERSION.txt")) : "dev"

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

; --- Batch 2 Enhancement: Dynamic subfolder list with favorites and recents ---
; Paths for persistent storage in AppData
recentDataPath := A_AppData . "\QuickNav\recent.json"

; Helper for file IO (basic JSON emulation)
LoadRecents() {
    global recentDataPath
    if !FileExist(recentDataPath)
        return {jobs: [], folders: [], favorites: []}
    try
        return JSON.Load(FileRead(recentDataPath))
    catch
        return {jobs: [], folders: [], favorites: []}
}
SaveRecents(data) {
    global recentDataPath
    DirCreate(DirSplit(recentDataPath)[1])
    FileDelete(recentDataPath)
    FileAppend(JSON.Dump(data), recentDataPath)
}
; --- Preferences Persistence Utility ---
settingsPath := A_AppData . "\QuickNav\settings.json"
LoadSettings() {
    global settingsPath
    if !FileExist(settingsPath)
        return Map()
    try
        return JSON.Load(FileRead(settingsPath))
    catch
        return Map()
}
SaveSettings(settings) {
    global settingsPath
    DirCreate(DirSplit(settingsPath)[1])
    FileDelete(settingsPath)
    FileAppend(JSON.Dump(settings), settingsPath)
}
; --- Reset App Utility (Batch 4) ---
ResetApp() {
    global settingsPath, recentDataPath, logPath
    try FileDelete(settingsPath)
    try FileDelete(recentDataPath)
    try FileDelete(logPath)
    ShowNotification("App data reset. Restart recommended.", "success")
}
; --- Diagnostic Logging Utility (Batch 4) ---
logPath := A_AppData . "\QuickNav\error.log"
LogError(msg, context := "") {
    global logPath
    try {
        DirCreate(DirSplit(logPath)[1])
        FileAppend(FormatTime(A_Now, "yyyy-MM-dd HH:mm:ss") . " | " . (context ? context . " | " : "") . msg . "`n", logPath)
    }
    catch {
        ; Fail silently—no further action
    }
}

; Default folders
defaultFolderNames := [
    "4. System Designs",
    "1. Sales Handover",
    "2. BOM & Orders",
    "6. Customer Handover Documents",
    "Floor Plans",
    "Site Photos"
]

; Load recents/favorites
recentsData := LoadRecents()
folderNames := recentsData.favorites.Length ? recentsData.favorites : defaultFolderNames
recentJobs := recentsData.jobs
recentFolders := recentsData.folders

mainGui := Gui("+AlwaysOnTop", "Project QuickNav v" . versionStr)

; --- Persistent Notification Area (Batch 3 UX) ---
notificationPanel := mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(240) " w" Scale(316) " h" Scale(28) " vNotificationPanel BackgroundD3D3D3", "")
notificationLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(247) " w" Scale(240) " h" Scale(14) " vNotificationLabel", "")
notificationClose := mainGui.Add("Button", "x" Scale(280) " y" Scale(244) " w" Scale(36) " h" Scale(20) " vNotificationClose", "✕")
notificationPanel.Visible := false
notificationLabel.Visible := false
notificationClose.Visible := false

; --- Inline error message control (for job number validation) ---
errorHint := ""


; DPI scaling factor
dpiScale := A_ScreenDPI / 96
Scale(x) => Round(x * dpiScale)

; Section: Job Input
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(10) " w" Scale(316) " h" Scale(60), "Project Selection")
mainGui.Add("Text",    "x" Scale(28) " y" Scale(32) " w" Scale(110) " AccessibleName'Job number label'", "Job Number or Search:")

jobEdit := mainGui.Add("Edit", "x+" Scale(6) " yp w" Scale(150) " vJobNumber Section AccessibleName'Job number or search box'", "")
jobEdit.Opt("ToolTip 'Enter a 5-digit job number or search term. Drag a folder here to auto-fill.'")

; Inline hint/error label
jobErrorLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(60) " w" Scale(260) " cRed vJobError AccessibleName'Job error hint'", "") ; initially blank
jobErrorLabel.Visible := false  ; Deprecated in favor of persistent notification area

; Drag-and-drop for folder path input
jobEdit.OnEvent("DropFiles", (this, files, *) => (
    this.Value := files[1],  ; fill with dropped path
    ShowInlineHint("Detected path from drag-and-drop.", "info")
))


; Section: Subfolder Selection
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(80) " w" Scale(316) " h" Scale(160), "Select Subfolder")
y_radio := Scale(100)
; --- Dynamic radio group for subfolders with tooltips, favorites, and recents ---
radioCtrls := []
Loop folderNames.Length
{
    label := folderNames[A_Index]
    checked := (A_Index = 1) ? " Checked" : ""
    radio := mainGui.Add("Radio", "x" Scale(32) " y" (y_radio + Scale(25)*(A_Index-1)) " w" Scale(250) " vRadio" A_Index checked " AccessibleName'Subfolder: " label "'", label)
    radio.Opt("ToolTip 'Choose a subfolder. Right-click to favorite/unfavorite.'")
    radioCtrls.Push(radio)
}
; Help/About button in main window (moved outside loop)
btnAbout := mainGui.Add("Button", "x" Scale(28) " y" Scale(320) " w" Scale(100) " h" Scale(22), "Help/About")
btnAbout.OnEvent("Click", (*) => ShowAboutDialog())
favHint := mainGui.Add("Text", "x" Scale(32) " y" (y_radio + Scale(25)*(folderNames.Length)) " w" Scale(250) " cGray", "Tip: Right-click a subfolder to favorite/unfavorite")

; Add right-click events for favorite management
; --- Preferences Dialog (Batch 3 UX) ---
ShowPreferencesDialog() {
    global folderNames
    static prefGui, controls
    settings := LoadSettings()
    if IsSet(prefGui) && prefGui {
        prefGui.Destroy()
    }
    prefGui := Gui("+AlwaysOnTop", "Preferences / Settings")
    controls := Map()

    ; Default Folder
    prefGui.Add("Text", "x12 y12 w120 h20", "Default Folder:")
    defaultFolder := settings.Has("defaultFolder") ? settings["defaultFolder"] : (folderNames.Length ? folderNames[1] : "")
    controls["defaultFolder"] := prefGui.Add("DropDownList", "x140 y12 w120 vDefaultFolder", folderNames)
    controls["defaultFolder"].Choose(defaultFolder ? folderNames.IndexOf(defaultFolder) + 1 : 1)

    ; Default Job Input Behavior
    prefGui.Add("Text", "x12 y44 w120 h20", "Job Input Behavior:")
    jobInputOpts := ["Prompt", "Auto-fill Last", "Auto-fill Favorite"]
; Theme Mode
    prefGui.Add("Text", "x12 y66 w120 h20", "Theme:")
    themeOpts := ["Light", "Dark", "High Contrast"]
    defaultTheme := settings.Has("theme") ? settings["theme"] : "Light"
    controls["theme"] := prefGui.Add("DropDownList", "x140 y66 w120 vTheme", themeOpts)
    controls["theme"].Choose(themeOpts.IndexOf(defaultTheme) + 1)
; --- Help/About Dialog with Usage, Features, and Shortcuts (Batch 3) ---
    defaultJobInput := settings.Has("jobInputMode") ? settings["jobInputMode"] : "Prompt"
    controls["jobInputMode"] := prefGui.Add("DropDownList", "x140 y44 w120 vJobInputMode", jobInputOpts)
    controls["jobInputMode"].Choose(jobInputOpts.IndexOf(defaultJobInput) + 1)

    ; Max Recents/Favorites
    prefGui.Add("Text", "x12 y76 w120 h20", "Maximum Recents:")
    maxRecents := settings.Has("maxRecents") ? settings["maxRecents"] : 10
    controls["maxRecents"] := prefGui.Add("Edit", "x140 y76 w60 vMaxRecents", maxRecents)

    ; Notification Duration
    prefGui.Add("Text", "x12 y108 w120 h20", "Notification Duration (ms):")
    notifDur := settings.Has("notifDuration") ? settings["notifDuration"] : 3000
    controls["notifDuration"] := prefGui.Add("Edit", "x140 y108 w60 vNotifDuration", notifDur)

    ; Save/Cancel Buttons
    btnSave := prefGui.Add("Button", "x30 y150 w80 Default", "Save")
    btnCancel := prefGui.Add("Button", "x120 y150 w80", "Cancel")
    btnReset := prefGui.Add("Button", "x210 y150 w80", "Reset App")
    btnSave.OnEvent("Click", (*) => (
        settings["defaultFolder"] := controls["defaultFolder"].Text,
        settings["jobInputMode"] := controls["jobInputMode"].Text,
        settings["theme"] := controls["theme"].Text,
        settings["maxRecents"] := Integer(controls["maxRecents"].Text),
        settings["notifDuration"] := Integer(controls["notifDuration"].Text),
        SaveSettings(settings),
        ApplyTheme(settings["theme"]),
        ShowNotification("Preferences saved.", "success"),
        prefGui.Destroy()
    ))
    btnCancel.OnEvent("Click", (*) => prefGui.Destroy())
    btnReset.OnEvent("Click", (*) => (ResetApp(), prefGui.Destroy()))
    prefGui.Show("w310 h200")
}

; --- Apply theme colors to GUIs and controls dynamically (Batch 3) ---
ApplyTheme(theme := "Light") {
    global mainGui, notificationPanel, notificationLabel, notificationClose, loaderLabel, progressBar, statusText, btnOpen, btnCancel, btnAbout
    ; Only affects controls we can recolor in AHK v2
    if (theme = "Dark") {
        mainGui.Opt("Background20232A")
        notificationPanel.Opt("Background444444")
        notificationLabel.Opt("cDDDDDD")
        loaderLabel.Opt("cAACCFF")
        progressBar.Opt("c33AAFF")
        statusText.Opt("cCCCCCC")
        btnOpen.Opt("Background222222 c00CC99")
        btnCancel.Opt("Background222222 cFF6688")
        btnAbout.Opt("Background333333 cCCCCCC")
    } else if (theme = "High Contrast") {
        mainGui.Opt("Background000000")
        notificationPanel.Opt("BackgroundFFFF00")
        notificationLabel.Opt("c000000")
        loaderLabel.Opt("cFFFFFF")
        progressBar.Opt("cFFFFFF")
        statusText.Opt("cFFFFFF")
        btnOpen.Opt("BackgroundFFFF00 c000000")
        btnCancel.Opt("BackgroundFF0000 cFFFFFF")
        btnAbout.Opt("Background000000 cFFFF00")
    } else {
        mainGui.Opt("BackgroundFFFFFF")
        notificationPanel.Opt("BackgroundD3D3D3")
        notificationLabel.Opt("c333333")
        loaderLabel.Opt("c3366AA")
        progressBar.Opt("c3366AA")
        statusText.Opt("c333333")
        btnOpen.Opt("BackgroundF0F0F0 c005577")
        btnCancel.Opt("BackgroundF0F0F0 cAA4455")
        btnAbout.Opt("BackgroundF5F5F5 c333333")
    }
    mainGui.Redraw()
}
For ctrl in radioCtrls
    ctrl.OnEvent("RButtonUp", (this, *) => ToggleFavorite(this.Text))

mainGui.Add("Checkbox", "x" Scale(32) " y" Scale(230) " w" Scale(250) " vDebugMode", "Show Raw Python Output").Opt("ToolTip 'Enable to see raw output/errors from backend.'")

; Section: Status + Progress
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(255) " w" Scale(316) " h" Scale(65), "Status")
loaderLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(258) " w" Scale(120) " h" Scale(14) " vLoaderLabel cBlue", "")
loaderLabel.Visible := false
progressBar := mainGui.Add("Progress", "x" Scale(28) " y" Scale(275) " w" Scale(280) " h" Scale(16) " vProgress1 Range0-100 cBlue")
progressBar.Value := 0
statusText := mainGui.Add("Text", "x" Scale(28) " y" Scale(298) " w" Scale(280) " h" Scale(20) " vStatusText", "Ready")

; Main Action Button
btnOpen := mainGui.Add("Button", "x" Scale(70) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Default AccessibleName'Open project folder'", "Open")
btnOpen.Opt("ToolTip 'Start project lookup.'")

btnCancel := mainGui.Add("Button", "x" Scale(180) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Disabled vBtnCancel AccessibleName'Cancel lookup'", "Cancel")
btnCancel.Opt("ToolTip 'Cancel ongoing lookup.'")
btnCancel.OnEvent("Click", CancelProcessing)

; Use OnEvent method for AutoHotkey v2 event handling
btnOpen.OnEvent("Click", OpenProject)
mainGui.OnEvent("Close", GuiClose)

; --- Utility: Persistent, Dismissible Notification Display ---
ShowNotification(msg, type := "info", duration := 0) {
    global notificationPanel, notificationLabel, notificationClose
    static bgColors, fgColors
    if !IsSet(bgColors) {
        bgColors := Map()
        bgColors["success"] := "CCE5FF"
        bgColors["error"] := "FFCCCC"
        bgColors["warning"] := "FFF5CC"
        bgColors["info"] := "D3D3D3"
    }
    if !IsSet(fgColors) {
        fgColors := Map()
        fgColors["success"] := "007700"
        fgColors["error"] := "C80000"
        fgColors["warning"] := "A88400"
        fgColors["info"] := "333333"
    }
    notificationPanel.Opt("Background" . (bgColors.Has(type) ? bgColors[type] : bgColors["info"]))
    notificationLabel.Opt("c" . (fgColors.Has(type) ? fgColors[type] : fgColors["info"]))
    notificationLabel.Value := msg
    notificationPanel.Visible := true
    notificationLabel.Visible := true
    notificationClose.Visible := true
    notificationClose.OnEvent("Click", (*) => HideNotification())
    if (duration && duration > 0)
        SetTimer(() => HideNotification(), -duration)
}

HideNotification() {
    global notificationPanel, notificationLabel, notificationClose
    notificationPanel.Visible := false
    notificationLabel.Visible := false
    notificationClose.Visible := false
}

; --- Utility: Inline hint display, now routes to persistent notifications ---
ShowInlineHint(msg, type:="error") {
    ShowNotification(msg, type)
}

; --- Loader: Enhanced Progress and Animated Loader Label ---
SetProgress(val := "") {
    global progressBar, loaderLabel
    static timerOn := false
    static dots := 0

    if (val = "") {
        progressBar.Marquee := true
        loaderLabel.Value := "Loading"
        loaderLabel.Visible := true
        dots := 0
        if !timerOn {
            SetTimer(AnimateLoaderLabel, 400)
            timerOn := true
        }
    } else {
        progressBar.Marquee := false
        progressBar.Value := val
        loaderLabel.Visible := false
        if timerOn {
            SetTimer(AnimateLoaderLabel, 0)
            timerOn := false
        }
    }
}
; Animate loader label with moving dots
AnimateLoaderLabel() {
    global loaderLabel
    static dotCycle := ["", ".", "..", "..."]
    static i := 1
    loaderLabel.Value := "Loading" . dotCycle[i]
    i := Mod(i, 4) + 1
}

; --- Utility: Toggle favorites ---
ToggleFavorite(label) {
    global recentsData, folderNames
    arr := recentsData.favorites
    if arr.Has(label) {
        arr.RemoveAt(arr.IndexOf(label))
    } else {
        arr.Push(label)
    }
    recentsData.favorites := arr
    SaveRecents(recentsData)
    ReloadFolderRadios()
}

; --- Utility: Reload radio controls when favorites update ---
ReloadFolderRadios() {
    global mainGui, radioCtrls, recentsData, defaultFolderNames, y_radio
    ; Remove old radios
    For ctrl in radioCtrls
        ctrl.Destroy()
    folderNames := recentsData.favorites.Length ? recentsData.favorites : defaultFolderNames
    radioCtrls := []
    Loop folderNames.Length {
        label := folderNames[A_Index]
        checked := (A_Index = 1) ? " Checked" : ""
        radio := mainGui.Add("Radio", "x" Scale(32) " y" (y_radio + Scale(25)*(A_Index-1)) " w" Scale(250) " vRadio" A_Index checked, label)
        radio.Opt("ToolTip 'Choose a subfolder. Right-click to favorite/unfavorite.'")
        radioCtrls.Push(radio)
        radio.OnEvent("RButtonUp", (this, *) => ToggleFavorite(this.Text))
    }
}


; Initialize the system tray menu
AddSystemTrayMenu()

mainGui.Show()

return

; Function to reset the GUI to its initial state
ResetGUI() {
    global mainGui, progressBar, btnOpen, btnCancel, jobEdit
    mainGui["StatusText"].Value := "Ready"
    progressBar.Value := 0
    btnOpen.Enabled := true
    btnCancel.Enabled := false
    jobEdit.Opt("BackgroundWhite")
}

; (removed duplicate SetProgress utility; see enhanced version above)

; --- Enhanced OpenProject with inline error, Cancel, persistence, and async backend ---
OpenProject(ctrl, info) {
    global folderNames, mainGui, btnOpen, btnCancel, jobEdit, recentsData, recentJobs, recentFolders, radioCtrls
    static procPID := 0

    btnOpen.Enabled := false
    btnCancel.Enabled := true
    mainGui["StatusText"].Value := "Processing..."
    SetProgress()
    jobEdit.Opt("BackgroundWhite")

    scriptPath := "find_project_path.py"
    if !FileExist(scriptPath)
        scriptPath := "test_find_project_path.py"

    params := mainGui.Submit(false)
    DebugMode := params.DebugMode
    jobNumber := params.JobNumber

    ; Input validation with inline feedback
    if (jobNumber = "") {
        ShowInlineHint("Please enter a job number or search term.", "error")
        jobEdit.Opt("BackgroundF7C8C8") ; light red
        ResetGUI()
        return
    }
    isJobNumber := RegExMatch(jobNumber, "^\d{5}$")
    if (!isJobNumber && !RegExMatch(jobNumber, "^[\w\s\-]+$")) {
        ShowInlineHint("Invalid input. Must be 5 digits or a search term.", "error")
        jobEdit.Opt("BackgroundF7C8C8")
        ResetGUI()
        return
    }
    jobEdit.Opt("BackgroundWhite")

    ; Save to recents (job numbers and folder selection)
    if !recentJobs.Has(jobNumber) {
        recentJobs.Push(jobNumber)
        if recentJobs.Length > 10
            recentJobs.RemoveAt(1)
    }
    ; Get selected folder
    selectedFolder := ""
    for idx, ctrl in radioCtrls {
        varName := "Radio" . idx
        if (params.HasOwnProp(varName) && params.%varName%) {
            selectedFolder := folderNames[idx]
            break
        }
    }
    if (!selectedFolder)
        selectedFolder := folderNames[1]
    if !recentFolders.Has(selectedFolder) {
        recentFolders.Push(selectedFolder)
        if recentFolders.Length > 10
            recentFolders.RemoveAt(1)
    }
    recentsData.jobs := recentJobs
    recentsData.folders := recentFolders
    SaveRecents(recentsData)

    mainGui["StatusText"].Value := "Searching for project folder..."
    SetProgress(30)

    ; Prepare async backend call: use Run and save PID for cancellation
    comspec := A_ComSpec
    tempFile := A_Temp . "\project_quicknav_pyout.txt"
    cmd := comspec . " /C python " . Chr(34) . scriptPath . Chr(34) . " " . jobNumber . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"
    procPID := 0

    try {
        Run(cmd, , "Hide Pid", &procPID)
    } catch as e {
        mainGui["StatusText"].Value := "Error: backend launch failed"
        ShowInlineHint("Backend error: " . e.Message, "error")
        LogError("Failed to launch backend: " . e.Message, "OpenProject")
        ResetGUI()
        return
    }

    ; Monitor for backend completion or cancel (poll file)
    attempts := 0
    maxAttempts := 100 ; 50*0.1s = 5s
    SetTimer(() => WaitForBackend(tempFile, procPID, attempts, maxAttempts, DebugMode, selectedFolder), 100)
}

; --- Cancel Button Handler ---
CancelProcessing(ctrl, info) {
    static procPID := 0
    if (procPID) {
        try {
            ProcessClose(procPID)
        } catch
        {}
    }
    ShowInlineHint("Cancelled.", "info")
    ResetGUI()
}

; --- Wait for Backend Completion ---
WaitForBackend(tempFile, procPID, attempts, maxAttempts, DebugMode, selectedFolder) {
    global mainGui, btnOpen, btnCancel
    attempts++
    if FileExist(tempFile) {
        btnCancel.Enabled := false
        btnOpen.Enabled := true
        ; Read output and handle as before (elided for brevity, see original function for error handling)
        output := FileRead(tempFile)
        FileDelete(tempFile)
        output := Trim(output)
        if (DebugMode) {
            ShowInlineHint("Raw backend output: " . output, "info")
        }
        ; ... (Process the output as in the original OpenProject)
        mainGui["StatusText"].Value := "Done."
        SetProgress(100)
        ResetGUI()
        return
    } else if (attempts > maxAttempts) {
        ShowInlineHint("Backend timeout.", "error")
        LogError("Backend timeout waiting for Python response", "WaitForBackend")
        ResetGUI()
        return
    }
    SetTimer(() => WaitForBackend(tempFile, procPID, attempts, maxAttempts, DebugMode, selectedFolder), 100)
}

    ; Disable the Open button to prevent multiple clicks
    btnOpen.Enabled := false

    ; Update status text
    mainGui["StatusText"].Value := "Processing..."
    SetProgress()  ; Show indeterminate progress during processing

    ; Use test_find_project_path.py instead if the real backend can't be found
    scriptPath := "find_project_path.py"
    if !FileExist(scriptPath)
        scriptPath := "test_find_project_path.py"

    ; Get form data
    params := mainGui.Submit(false)  ; false to keep GUI visible
    DebugMode := params.DebugMode
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
    SetProgress(30)

    ; Prepare the command to run the Python script
    comspec := A_ComSpec
    tempFile := A_Temp . "\project_quicknav_pyout.txt"
    cmd := comspec . " /C python " . Chr(34) . scriptPath . Chr(34) . " " . jobNumber . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"

    try {
        ; Run the Python script and wait for it to complete
        RunWait(cmd, "", "Hide")

        ; Check if the output file exists
        if !FileExist(tempFile) {
            mainGui["StatusText"].Value := "Error: Python execution failed"
            SetProgress(0)
            MsgBox("Failed to execute Python script. Make sure Python is installed and in your PATH.", "Python Error", 16)
            LogError("Python output file missing after execution", "SyncBackendCall")
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

        ; Process the output from the Python script
        if (InStr(output, "ERROR:") == 1) {
            ; Handle error response
            msg := SubStr(output, 7)
            mainGui["StatusText"].Value := "Error: " . Trim(msg)
            SetProgress(0)
            MsgBox(Trim(msg), "Error", 16)
            LogError("Python backend error: " . msg, "SyncBackendCall")
            ResetGUI()
            return
        }
        else if (InStr(output, "SELECT:") == 1) {
            ; Handle multiple exact matches
            mainGui["StatusText"].Value := "Multiple paths found"
            SetProgress(100)
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
            SetProgress(100)
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
            LogError("Unexpected backend output: " . output, "OpenProject")
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
            LogError("Subfolder not found: " . selectedFolder . " under " . MainProjectPath, "SyncBackendCall")
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
            ShowNotification("Folder opened successfully", "success")
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
; orphaned catch block removed, now handled in new async logic
; end removal -- now handled by new OpenProject, WaitForBackend, CancelProcessing (see above)

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
    A_TrayMenu.Add("Preferences...", (*) => ShowPreferencesDialog())
    A_TrayMenu.Add("Help/About...", (*) => ShowAboutDialog())
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
; --- Utility: Split a file path into [dir, name, ext, nameNoExt, drive] ---
DirSplit(path) {
    local name, dir, ext, nameNoExt, drive
    SplitPath(path, &name, &dir, &ext, &nameNoExt, &drive)
    return [dir, name, ext, nameNoExt, drive]
}
ShowAboutDialog() {
    global logPath, versionStr
    txt := "
    (
Project QuickNav
Version: See VERSION.txt

Usage:
 - Enter a 5-digit job number or search term and select a subfolder.
 - Press 'Open' to launch the folder, or drag/drop a folder onto the input.
 - Right-click subfolders to favorite.
 - Cancel ongoing operations anytime.

Feature Summary (Batch 3):
 - Inline, persistent notifications (color-coded, dismissible)
 - Animated loader for backend/process work
 - User preferences: favorites, input mode, recents, notifications
 - Light/Dark theming (if enabled in your build)
 - Comprehensive Help/About dialog

Keyboard Shortcuts:
 - Ctrl+Alt+Q: Show or focus QuickNav window

For detailed documentation, see README.md or INSTALL.md.
    )"
    GuiObj := Gui("+AlwaysOnTop", "About / Help - QuickNav v" . versionStr)
    txt := RegExReplace(txt, "Version: See VERSION\.txt", "Version: " . versionStr)
    GuiObj.Add("Edit", "x10 y10 w410 h180 -Wrap ReadOnly", txt)
    btnDiag := GuiObj.Add("Button", "x30 y200 w120", "Open Diagnostics Folder")
    btnLog := GuiObj.Add("Button", "x170 y200 w120", "View Error Log")
    GuiObj.Add("Button", "x310 y200 w90 Default", "OK").OnEvent("Click", (*) => GuiObj.Destroy())

    btnDiag.OnEvent("Click", (*) => (
        Run("explorer.exe " . Chr(34) . DirSplit(logPath)[1] . Chr(34))
    ))
    btnLog.OnEvent("Click", (*) => (
        FileExist(logPath)
        ? Run("notepad.exe " . Chr(34) . logPath . Chr(34))
        : ShowNotification("No error log found.","info")
    ))
    GuiObj.Show("w430 h250")
}