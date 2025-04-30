/*
Project QuickNav – AutoHotkey Frontend (GUI Script)

This script provides the GUI for Project QuickNav and delegates all business/controller logic
(validation, persistence, backend invocation, state transitions) to lld_navigator_controller.ahk.

- All controller/business routines are located in lld_navigator_controller.ahk and #Included below.
- The interface between GUI and controller is via Controller_* functions (see controller script for details).
- NO business logic is present here; only GUI setup, event hookup, and UI utility routines.

See lld_navigator_controller.ahk for controller implementation and documentation.
*/

#Requires AutoHotkey v2.0
#SingleInstance Force

#Include c:\Users\SamLyndon\Projects\Personal\Project-QuickNav\lld_navigator_controller.ahk

; --- DPI scaling factor and scaling function ---
dpiScale := A_ScreenDPI / 96
Scale(x) => Round(x * dpiScale)

; --- Version String ---
versionStr := FileExist("VERSION.txt") ? Trim(FileRead("VERSION.txt")) : "dev"

; --- Paths for persistence ---
recentDataPath := A_AppData . "\QuickNav\recent.json"
settingsPath := A_AppData . "\QuickNav\settings.json"
logPath := A_AppData . "\QuickNav\error.log"

; --- Default folders ---
defaultFolderNames := [
    "4. System Designs",
    "1. Sales Handover",
    "2. BOM & Orders",
    "6. Customer Handover Documents",
    "Floor Plans",
    "Site Photos"
]

; --- Load recents/favorites and settings ---
recentsData := LoadRecents()
folderNames := []
if (recentsData.favorites.Length) {
    for idx, fname in recentsData.favorites {
        v := ValidateAndNormalizeInputs("00000", fname)
        if v.valid
            folderNames.Push(v.normalizedFolder)
    }
}
if (folderNames.Length == 0) {
    for idx, fname in defaultFolderNames
        folderNames.Push(fname)
}
recentJobs := []
for idx, job in recentsData.jobs {
    v := ValidateAndNormalizeInputs(job, folderNames[1])
    if v.valid
        recentJobs.Push(v.normalizedJob)
}
recentFolders := []
for idx, f in recentsData.folders {
    v := ValidateAndNormalizeInputs("00000", f)
    if v.valid
        recentFolders.Push(v.normalizedFolder)
}

; --- Instantiate main GUI window ---
mainGui := Gui("+AlwaysOnTop", "Project QuickNav v" . versionStr)

; --- Persistent Notification Area ---
notificationPanel := mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(240) " w" Scale(316) " h" Scale(28) " vNotificationPanel BackgroundD3D3D3", "")
notificationLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(247) " w" Scale(240) " h" Scale(14) " vNotificationLabel", "")
notificationClose := mainGui.Add("Button", "x" Scale(280) " y" Scale(244) " w" Scale(36) " h" Scale(20) " vNotificationClose", "✕")
notificationPanel.Visible := false
notificationLabel.Visible := false
notificationClose.Visible := false

; --- Job Input Section ---
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(10) " w" Scale(316) " h" Scale(60), "Project Selection")
jobLabel := mainGui.Add("Text",    "x" . Scale(28) . " y" . Scale(32) . " w" . Scale(110), "Job Number or Search:")
jobEdit := mainGui.Add("Edit", "x+" . Scale(6) . " yp w" . Scale(150) . " vJobNumber Section", "")
jobEdit.ToolTip := "Enter a 5-digit job number or search term. Drag a folder here to auto-fill."
jobErrorLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(60) " w" Scale(260) " cRed vJobError", "")
jobErrorLabel.Visible := false

; --- Drag-and-drop for job input ---
OnMessage(0x233, WM_DROPFILES)
WM_DROPFILES(wParam, lParam, msg, hwnd) {
    global jobEdit, folderNames
    files := DllCall("shell32\DragQueryFile", "ptr", wParam, "uint", 0xFFFFFFFF, "ptr", 0, "uint", 0)
    if (files > 0) {
        path := Buffer(1024)
        DllCall("shell32\DragQueryFile", "ptr", wParam, "uint", 0, "ptr", path, "uint", 1024)
        droppedPath := Trim(StrGet(path))
        v := ValidateAndNormalizeInputs(droppedPath, folderNames[1])
        if (!v.valid) {
            MsgBox(v.errorMsg, "Invalid Input", 48)
        } else {
            jobEdit.Value := v.normalizedJob
            ShowInlineHint("Detected path from drag-and-drop.", "info")
        }
    }
    DllCall("shell32\DragFinish", "ptr", wParam)
}

; --- Subfolder Selection (Radio group) ---
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(80) " w" Scale(316) " h" Scale(160), "Select Subfolder")
y_radio := Scale(100)
radioCtrls := []
Loop folderNames.Length {
    label := folderNames[A_Index]
    checked := (A_Index = 1) ? " Checked" : ""
    radio := mainGui.Add("Radio", "x" Scale(32) " y" (y_radio + Scale(25)*(A_Index-1)) " w" Scale(250) " vRadio" A_Index checked, label)
    radio.ToolTip := "Choose a subfolder. Right-click to favorite/unfavorite."
    radioCtrls.Push(radio)
}
btnAbout := mainGui.Add("Button", "x" Scale(28) " y" Scale(320) " w" Scale(100) " h" Scale(22), "Help/About")
btnAbout.OnEvent("Click", (*) => ShowAboutDialog())
favHint := mainGui.Add("Text", "x" Scale(32) " y" (y_radio + Scale(25)*(folderNames.Length)) " w" Scale(250) " cGray", "Tip: Right-click a subfolder to favorite/unfavorite")

; --- Right-click favorites context menu ---
global currentRadioCtrl := 0
global radioContextMenu := Menu()
radioContextMenu.Add("Toggle Favorite", (*) => Controller_ToggleFavorite(currentRadioCtrl.Text))
For ctrl in radioCtrls {
    ctrl.OnEvent("ContextMenu", (ctrlObj,*) => (currentRadioCtrl := ctrlObj, radioContextMenu.Show()))
}

; --- Preferences Dialog, Theming, and UX helpers ---
ShowPreferencesDialog() {
    global folderNames
    static prefGui, controls
    settings := LoadSettings()
    if IsSet(prefGui) && prefGui {
        prefGui.Destroy()
    }
    prefGui := Gui("+AlwaysOnTop", "Preferences / Settings")
    controls := Map()
    prefGui.Add("Text", "x12 y12 w120 h20", "Default Folder:")
    defaultFolder := settings.Has("defaultFolder") ? settings["defaultFolder"] : (folderNames.Length ? folderNames[1] : "")
    controls["defaultFolder"] := prefGui.Add("DropDownList", "x140 y12 w120 vDefaultFolder", folderNames)
    controls["defaultFolder"].Choose(defaultFolder ? folderNames.IndexOf(defaultFolder) + 1 : 1)
    prefGui.Add("Text", "x12 y44 w120 h20", "Job Input Behavior:")
    jobInputOpts := ["Prompt", "Auto-fill Last", "Auto-fill Favorite"]
    prefGui.Add("Text", "x12 y66 w120 h20", "Theme:")
    themeOpts := ["Light", "Dark", "High Contrast"]
    defaultTheme := settings.Has("theme") ? settings["theme"] : "Light"
    controls["theme"] := prefGui.Add("DropDownList", "x140 y66 w120 vTheme", themeOpts)
    controls["theme"].Choose(themeOpts.IndexOf(defaultTheme) + 1)
    defaultJobInput := settings.Has("jobInputMode") ? settings["jobInputMode"] : "Prompt"
    controls["jobInputMode"] := prefGui.Add("DropDownList", "x140 y44 w120 vJobInputMode", jobInputOpts)
    controls["jobInputMode"].Choose(jobInputOpts.IndexOf(defaultJobInput) + 1)
    prefGui.Add("Text", "x12 y76 w120 h20", "Maximum Recents:")
    maxRecents := settings.Has("maxRecents") ? settings["maxRecents"] : 10
    controls["maxRecents"] := prefGui.Add("Edit", "x140 y76 w60 vMaxRecents", maxRecents)
    prefGui.Add("Text", "x12 y108 w120 h20", "Notification Duration (ms):")
    notifDur := settings.Has("notifDuration") ? settings["notifDuration"] : 3000
    controls["notifDuration"] := prefGui.Add("Edit", "x140 y108 w60 vNotifDuration", notifDur)
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
ApplyTheme(theme := "Light") {
    global mainGui, notificationPanel, notificationLabel, notificationClose, loaderLabel, progressBar, statusText, btnOpen, btnCancel, btnAbout
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

; --- Status + Progress ---
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(255) " w" Scale(316) " h" Scale(65), "Status")
loaderLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(258) " w" Scale(120) " h" Scale(14) " vLoaderLabel cBlue", "")
loaderLabel.Visible := false
progressBar := mainGui.Add("Progress", "x" Scale(28) " y" Scale(275) " w" Scale(280) " h" Scale(16) " vProgress1 Range0-100 cBlue")
progressBar.Value := 0
statusText := mainGui.Add("Text", "x" Scale(28) " y" Scale(298) " w" Scale(280) " h" Scale(20) " vStatusText", "Ready")

; --- Main Action Buttons ---
btnOpen := mainGui.Add("Button", "x" Scale(70) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Default", "Open")
btnOpen.ToolTip := "Start project lookup."
btnOpen.OnEvent("Click", OpenProject)
btnCancel := mainGui.Add("Button", "x" Scale(180) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Disabled vBtnCancel", "Cancel")
btnCancel.ToolTip := "Cancel ongoing lookup."
btnCancel.OnEvent("Click", (*) => Controller_CancelProcessing())

; --- Debug checkbox ---
debugCheckbox := mainGui.Add("Checkbox", "x" Scale(32) " y" Scale(230) " w" Scale(250) " vDebugMode", "Show Raw Python Output")
debugCheckbox.ToolTip := "Enable to see raw output/errors from backend."

; --- Loader/progress animation helpers ---
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
AnimateLoaderLabel() {
    global loaderLabel
    static dotCycle := ["", ".", "..", "..."]
    static i := 1
    loaderLabel.Value := "Loading" . dotCycle[i]
    i := Mod(i, 4) + 1
}

; --- Notification and inline hint utilities ---
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
ShowInlineHint(msg, type:="error") {
    ShowNotification(msg, type)
}

; --- GUI event handlers: delegate to controller ---
OpenProject(ctrl, info) {
    global folderNames, mainGui, radioCtrls
    params := mainGui.Submit(false)
    DebugMode := params.DebugMode
    jobNumber := params.JobNumber
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
    Controller_OpenProject(jobNumber, selectedFolder, DebugMode)
}

; --- Main GUI close event: hide instead of exit ---
GuiClose(*) {
    mainGui.Hide()
    return
}

; --- System Tray and About/Help ---
AddSystemTrayMenu() {
    global
    A_TrayMenu.Delete()
    ToggleGuiMenu(ItemName, ItemPos, MyMenu) {
        ToggleGui()
    }
    ExitAppMenu(ItemName, ItemPos, MyMenu) {
        ExitApp()
    }
    A_TrayMenu.Add("Show/Hide QuickNav", ToggleGuiMenu)
    A_TrayMenu.Add("Preferences...", (*) => ShowPreferencesDialog())
    A_TrayMenu.Add("Help/About...", (*) => ShowAboutDialog())
    A_TrayMenu.Add()
    A_TrayMenu.Add("Exit", ExitAppMenu)
    try {
        A_TrayMenu.Default := "Show/Hide QuickNav"
    } catch {
        dummy := 1 ; ignore failure
    }
    TraySetIcon("shell32.dll", 44)
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
    btnDiag.OnEvent("Click", (*) => (Run("explorer.exe " . Chr(34) . DirSplit(logPath)[1] . Chr(34))))
    btnLog.OnEvent("Click", (*) => (
        FileExist(logPath)
        ? Run("notepad.exe " . Chr(34) . logPath . Chr(34))
        : ShowNotification("No error log found.","info")
    ))
    GuiObj.Show("w430 h250")
}

; --- Other app/utility handlers ---
ToggleGui() {
    global mainGui
    if WinExist("ahk_id " . mainGui.Hwnd) {
        mainGui.Hide()
    } else {
        mainGui.Show()
    }
}
^!q:: ToggleGui()

ResetGUI() {
    global mainGui, progressBar, btnOpen, btnCancel, jobEdit
    mainGui["StatusText"].Value := "Ready"
    progressBar.Value := 0
    btnOpen.Enabled := true
    btnCancel.Enabled := false
    jobEdit.Opt("BackgroundWhite")
}

; --- App entrypoint: initialize tray, show GUI ---
AddSystemTrayMenu()
mainGui.OnEvent("Close", GuiClose)
mainGui.Show()
return