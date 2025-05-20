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

; Enable debug mode for development
global DEBUG_MODE := true

; Define global for dark mode support (updated after settings load)
global A_IsDarkMode := false
global settings ; To store loaded settings
global projectSelectionGB ; For GuiSize

; Set up error handling
; Error handler function
QuickNav_ErrorHandler(err) {
    LogError("Unhandled error: " . err.Message . " at line " . err.Line, "ErrorHandler")
    MsgBox("An error occurred:`n" . err.Message . "`n`nLine: " . err.Line . "`n`nSee error log for details.", "QuickNav Error", 16)
    return true  ; Continue running the script
}

; Set up error handling
OnError(QuickNav_ErrorHandler)

; Include the controller script - use a direct relative path
#Include lld_navigator_controller.ahk ; Controller script must be in the same directory

; Load settings early to apply theme and A_IsDarkMode
settings := Controller_LoadSettings() ; Assuming this function exists in the controller
if (IsObject(settings) && settings.Has("theme") && settings.theme == "Dark") {
    A_IsDarkMode := true
} else {
    A_IsDarkMode := false
}


; --- DPI scaling factor and scaling functions ---
dpiScale := A_ScreenDPI / 96

; Enhanced scaling function with different scaling modes
Scale(x, mode := "default") {
    global dpiScale

    ; Different scaling modes for different types of UI elements
    if (mode = "text")
        return Round(x * Min(dpiScale, 1.5))  ; Cap text scaling to avoid overly large text
    else if (mode = "spacing")
        return Round(x * Max(dpiScale, 1.0))  ; Ensure minimum spacing
    else if (mode = "icon")
        return Round(x * Min(dpiScale, 2.0))  ; Scale icons but cap at 2x
    else if (mode = "font")
        return Round(x * Sqrt(dpiScale))      ; More gradual font scaling
    else
        return Round(x * dpiScale)            ; Default scaling
}

; --- Version String ---
versionStr := FileExist("VERSION.txt") ? Trim(FileRead("VERSION.txt")) : "dev"

; --- Paths for persistence ---
recentDataPath := A_AppData . "\QuickNav\recent.json"
settingsPath := A_AppData . "\QuickNav\settings.json" ; Used by controller
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
if (recentsData.Has("favorites") && (TypeOf(recentsData.favorites) = "Array") && recentsData.favorites.Length) {
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
if (recentsData.Has("jobs") && (TypeOf(recentsData.jobs) = "Array")) {
    for idx, job in recentsData.jobs {
        v := ValidateAndNormalizeInputs(job, folderNames.Length > 0 ? folderNames[1] : "") ; Assumes folderNames is not empty
        if v.valid
            recentJobs.Push(v.normalizedJob)
    }
}
recentFolders := []
if (recentsData.Has("folders") && (TypeOf(recentsData.folders) = "Array")) {
    for idx, f in recentsData.folders {
        v := ValidateAndNormalizeInputs("00000", f)
        if v.valid
            recentFolders.Push(v.normalizedFolder)
    }
}

; --- Instantiate main GUI window with resizing support ---
mainGui := Gui("+AlwaysOnTop +Resize +MinSize" . Scale(340) . "x" . Scale(380), "Project QuickNav v" . versionStr)

; --- Handle window resize events ---
mainGui.OnEvent("Size", GuiSize)

; --- Persistent Notification Area (positioned dynamically) ---
notificationPanel := mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(0) " w" Scale(316) " h" Scale(28) " vNotificationPanel BackgroundD3D3D3", "")
notificationLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(0) " w" Scale(240) " h" Scale(14) " vNotificationLabel", "")
notificationClose := mainGui.Add("Button", "x" Scale(280) " y" Scale(0) " w" Scale(36) " h" Scale(20) " vNotificationClose", "✕")
notificationPanel.Visible := false
notificationLabel.Visible := false
notificationClose.Visible := false

; --- Job Input Section ---
projectSelectionGB := mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(10) " w" Scale(316) " h" Scale(60), "Project Selection")
jobLabel := mainGui.Add("Text",    "x" . Scale(28) . " y" . Scale(32) . " w" . Scale(110), "Job Number or Search:")
jobEdit := mainGui.Add("Edit", "x+" . Scale(6) . " yp w" . Scale(150) . " vJobNumber Section", "")
jobEdit.ToolTip := "Enter a 5-digit job number or search term. Drag a folder here to auto-fill."
jobErrorLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(60) " w" Scale(260) " cRed vJobError", "")
jobErrorLabel.Visible := false

; --- Drag-and-drop for job input with enhanced error handling ---
OnMessage(0x233, WM_DROPFILES)
WM_DROPFILES(wParam, lParam, msg, hwnd) {
    global jobEdit, folderNames, jobErrorLabel, folderListView, radioValues

    try {
        files := DllCall("shell32\DragQueryFile", "ptr", wParam, "uint", 0xFFFFFFFF, "ptr", 0, "uint", 0)
        if (files <= 0) {
            ShowInlineHint("No valid files were dropped.", "warning")
            return
        }

        pathBuf := Buffer(1024)
        if (!DllCall("shell32\DragQueryFile", "ptr", wParam, "uint", 0, "ptr", pathBuf, "uint", pathBuf.Size)) {
            ShowInlineHint("Could not retrieve the dropped path.", "error")
            LogError("DragQueryFile failed to retrieve path", "WM_DROPFILES")
            return
        }
        droppedPath := Trim(StrGet(pathBuf))

        if (!FileExist(droppedPath)) {
            ShowInlineHint("The dropped path does not exist: " . droppedPath, "error")
            return
        }
        if (!DirExist(droppedPath)) {
            ShowInlineHint("Please drop a folder, not a file.", "warning")
            return
        }

        v := ValidateAndNormalizeInputs(droppedPath, folderNames.Length > 0 ? folderNames[1] : "")
        if (!v.valid) {
            jobErrorLabel.Value := v.errorMsg
            jobErrorLabel.Visible := true
            jobEdit.Opt("BackgroundFFDDDD")
            ShowInlineHint("Invalid path format. See error below.", "error")
            SetTimer(() => (jobErrorLabel.Visible := false, ApplyThemeToControl(jobEdit, settings.theme)), -5000)
        } else {
            jobErrorLabel.Visible := false
            ApplyThemeToControl(jobEdit, settings.theme)
            jobEdit.Value := v.normalizedJob
            ShowInlineHint("Detected job " . v.normalizedJob . " from path.", "success")

            if (v.normalizedFolder && folderNames.Length > 0) {
                folderIndex := folderNames.IndexOf(v.normalizedFolder)
                if (folderIndex > 0) {
                    folderListView.Modify(folderIndex, "Select Focus")
                    folderListView.EnsureVisible(folderIndex)
                    radioValues[folderIndex] := true
                    FlashListViewItem(folderIndex)
                }
            }
        }
    } catch as e {
        LogError("Drag and drop error: " . e.Message, "WM_DROPFILES")
        ShowInlineHint("An error occurred processing the dropped item.", "error")
    } finally {
        if (wParam)
            DllCall("shell32\DragFinish", "ptr", wParam)
    }
}

; --- Subfolder Selection ---
folderGroupBox := mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(80) " w" Scale(316) " h" Scale(160), "Select Subfolder")
folderListView := mainGui.Add("ListView", "x" Scale(20) " y" Scale(100) " w" Scale(300) " h" Scale(130) " -Hdr -Multi -ReadOnly -LV0x10000 AltSubmit Grid Background" . (A_IsDarkMode ? "222222" : "FFFFFF") . " vFolderListView", ["Folder"])
folderListView.OnEvent("Click", FolderListViewClick)
folderListView.OnEvent("ItemFocus", FolderListViewFocus)
OnMessage(0x200, WM_MOUSEMOVE)

radioValues := []
Loop folderNames.Length {
    folderListView.Add("", folderNames[A_Index])
    radioValues.Push(false)
}
if (folderNames.Length > 0) {
    folderListView.Modify(1, "Select Focus")
    radioValues[1] := true
}

btnAbout := mainGui.Add("Button", "x" Scale(28) " y" Scale(320) " w" Scale(100) " h" Scale(22), "Help/About")
btnAbout.OnEvent("Click", (*) => ShowAboutDialog())
favHint := mainGui.Add("Text", "x" Scale(32) " y" Scale(240) " w" Scale(250) " cGray", "Tip: Right-click a subfolder to favorite/unfavorite")

FolderListViewClick(ctrl, rowNum) {
    global radioValues, folderListView
    Loop radioValues.Length
        radioValues[A_Index] := false
    if (rowNum > 0 && rowNum <= radioValues.Length) {
        radioValues[rowNum] := true
        folderListView.Modify(rowNum, "Select Focus")
        folderListView.EnsureVisible(rowNum)
        FlashListViewItem(rowNum)
    }
}

FolderListViewFocus(ctrl, rowNum) {
    global radioValues
    Loop radioValues.Length
        radioValues[A_Index] := false
    if (rowNum > 0 && rowNum <= radioValues.Length) {
        radioValues[rowNum] := true
    }
}

WM_MOUSEMOVE(wParam, lParam, msg, hwnd) {
    global folderListView, radioValues
    static lastHoverRow := 0
    hoverRow := 0

    if (hwnd = folderListView.Hwnd) {
        lvhti := Buffer(24)
        lvhti.Fill(0)
        NumPut("Int", lParam & 0xFFFF, lvhti, 0)
        NumPut("Int", (lParam >> 16) & 0xFFFF, lvhti, 4)
        SendMessage(0x1012, 0, lvhti.Ptr, folderListView.Hwnd) ; LVM_SUBITEMHITTEST
        itemIndex := NumGet(lvhti, 12, "Int")

        if IsNumber(itemIndex) && itemIndex >= 0 {
            hoverRow := itemIndex + 1
        } else {
            hoverRow := 0
        }

        if (hoverRow != lastHoverRow) {
            if (lastHoverRow > 0 && folderListView.GetCount() >= lastHoverRow && !(radioValues.Has(lastHoverRow) && radioValues[lastHoverRow])) {
                folderListView.Modify(lastHoverRow, "-Select")
            }
            if (hoverRow > 0 && folderListView.GetCount() >= hoverRow && !(radioValues.Has(hoverRow) && radioValues[hoverRow])) {
                folderListView.Modify(hoverRow, "+Select -Focus")
            }
            lastHoverRow := hoverRow
        }
    }
}

FlashListViewItem(rowNum) {
    return ; Simplified
}

global currentFolderIndex := 0
global folderContextMenu := Menu()
folderContextMenu.Add("Toggle Favorite", (*) => Controller_ToggleFavorite(folderNames[currentFolderIndex]))

FolderListViewContextMenu(ctrl, rowNum) {
    global currentFolderIndex, folderContextMenu, folderNames
    if (rowNum > 0 && rowNum <= folderNames.Length) {
        currentFolderIndex := rowNum
        folderContextMenu.Show()
    }
}
folderListView.OnEvent("ContextMenu", FolderListViewContextMenu)


; --- Preferences Dialog ---
global prefControlsMap := Map() ; To store controls for preferences dialog

ShowPreferencesDialog() {
    global folderNames, settings, prefControlsMap ; Use global settings and prefControlsMap
    
    currentSettings := Type(settings) = "Map" ? settings : Controller_LoadSettings()

    static prefGui ; Make prefGui static to allow Destroy if already open
    if IsObject(prefGui) && prefGui.Hwnd {
        prefGui.Destroy()
    }
    prefGui := Gui("+AlwaysOnTop", "Preferences / Settings")
    prefControlsMap.Clear() ; Clear previous controls

    yPos := Scale(12)
    yStep := Scale(32)

    prefGui.Add("Text", "x" . Scale(12) . " y" . yPos . " w" . Scale(120) . " h" . Scale(20), "Default Folder:")
    defaultFolderVal := currentSettings.Has("defaultFolder") ? currentSettings["defaultFolder"] : (folderNames.Length ? folderNames[1] : "")
    prefControlsMap["defaultFolder"] := prefGui.Add("DropDownList", "x" . Scale(140) . " y" . yPos . " w" . Scale(150) . " vDefaultFolder", folderNames)
    if (folderNames.Length > 0)
        prefControlsMap["defaultFolder"].Choose(defaultFolderVal && folderNames.HasValue(defaultFolderVal) ? folderNames.IndexOf(defaultFolderVal) : 1)
    else
        prefControlsMap["defaultFolder"].Enabled := false

    yPos += yStep
    prefGui.Add("Text", "x" . Scale(12) . " y" . yPos . " w" . Scale(120) . " h" . Scale(20), "Job Input Behavior:")
    jobInputOpts := ["Prompt", "Auto-fill Last", "Auto-fill Favorite"]
    defaultJobInputVal := currentSettings.Has("jobInputMode") ? currentSettings["jobInputMode"] : "Prompt"
    prefControlsMap["jobInputMode"] := prefGui.Add("DropDownList", "x" . Scale(140) . " y" . yPos . " w" . Scale(150) . " vJobInputMode", jobInputOpts)
    prefControlsMap["jobInputMode"].Choose(jobInputOpts.HasValue(defaultJobInputVal) ? jobInputOpts.IndexOf(defaultJobInputVal) : 1)

    yPos += yStep
    prefGui.Add("Text", "x" . Scale(12) . " y" . yPos . " w" . Scale(120) . " h" . Scale(20), "Theme:")
    themeOpts := ["Light", "Dark", "High Contrast"]
    defaultThemeVal := currentSettings.Has("theme") ? currentSettings["theme"] : "Light"
    prefControlsMap["theme"] := prefGui.Add("DropDownList", "x" . Scale(140) . " y" . yPos . " w" . Scale(150) . " vTheme", themeOpts)
    prefControlsMap["theme"].Choose(themeOpts.HasValue(defaultThemeVal) ? themeOpts.IndexOf(defaultThemeVal) : 1)

    yPos += yStep
    prefGui.Add("Text", "x" . Scale(12) . " y" . yPos . " w" . Scale(120) . " h" . Scale(20), "Maximum Recents:")
    maxRecentsVal := currentSettings.Has("maxRecents") ? currentSettings["maxRecents"] : 10
    prefControlsMap["maxRecents"] := prefGui.Add("Edit", "x" . Scale(140) . " y" . yPos . " w" . Scale(60) . " vMaxRecents", maxRecentsVal)

    yPos += yStep
    prefGui.Add("Text", "x" . Scale(12) . " y" . yPos . " w" . Scale(120) . " h" . Scale(20), "Notification Duration (ms):")
    notifDurVal := currentSettings.Has("notifDuration") ? currentSettings["notifDuration"] : 3000
    prefControlsMap["notifDuration"] := prefGui.Add("Edit", "x" . Scale(140) . " y" . yPos . " w" . Scale(60) . " vNotifDuration", notifDurVal)

    yPos += yStep + Scale(10)
    btnSave := prefGui.Add("Button", "x" . Scale(30) . " y" . yPos . " w" . Scale(80) . " Default", "Save")
    btnCancel := prefGui.Add("Button", "x" . Scale(120) . " y" . yPos . " w" . Scale(80), "Cancel")
    btnReset := prefGui.Add("Button", "x" . Scale(210) . " y" . yPos . " w" . Scale(80), "Reset App")

    btnSave.OnEvent("Click", HandleSavePreferences.Bind(prefGui)) ; Pass prefGui to destroy it
    btnCancel.OnEvent("Click", (*) => prefGui.Destroy())
    btnReset.OnEvent("Click", (*) => (ResetApp(), prefGui.Destroy()))

    ApplyThemeToPrefsDialog(prefGui, currentSettings.Has("theme") ? currentSettings["theme"] : "Light")
    prefGui.Show("w" . Scale(310) . " h" . Scale(yPos + 40))
}

HandleSavePreferences(prefGuiBound, *) { ; prefGuiBound is the GUI object passed by Bind
    global settings, A_IsDarkMode, prefControlsMap ; Ensure access to global settings and prefControlsMap
    
    newSettings := Map()
    newSettings["defaultFolder"] := prefControlsMap["defaultFolder"].Text
    newSettings["jobInputMode"] := prefControlsMap["jobInputMode"].Text
    newSettings["theme"] := prefControlsMap["theme"].Text
    newSettings["maxRecents"] := Integer(prefControlsMap["maxRecents"].Text)
    newSettings["notifDuration"] := Integer(prefControlsMap["notifDuration"].Text)
    
    SaveSettings(newSettings)
    settings := newSettings ; Update global settings cache
    A_IsDarkMode := (settings.Has("theme") && settings.theme == "Dark")
    ApplyTheme(settings["theme"])
    ShowNotification("Preferences saved.", "success")
    prefGuiBound.Destroy()
}


ApplyThemeToPrefsDialog(guiObj, theme) {
    bg := "", fgText := "", fgButton := "", bgButton := "", fgInput := "", bgInput := ""
    if (theme = "Dark") {
        bg := "20232A", fgText := "CCCCCC", fgButton := "CCCCCC", bgButton := "333333", fgInput := "EEEEEE", bgInput := "333333"
    } else if (theme = "High Contrast") {
        bg := "000000", fgText := "FFFFFF", fgButton := "000000", bgButton := "FFFF00", fgInput := "000000", bgInput := "FFFFFF"
    } else {
        bg := "FFFFFF", fgText := "333333", fgButton := "333333", bgButton := "F0F0F0", fgInput := "000000", bgInput := "White"
    }
    guiObj.Opt("Background" . bg)
    for ctrl in guiObj {
        if InStr(ctrl.Type, "Text") || InStr(ctrl.Type, "GroupBox_Label")
            ctrl.Opt("c" . fgText)
        else if InStr(ctrl.Type, "Button")
            ctrl.Opt("Background" . bgButton . " c" . fgButton)
        else if InStr(ctrl.Type, "DropDownList") || InStr(ctrl.Type, "Edit")
            ctrl.Opt("Background" . bgInput . " c" . fgInput)
    }
}

ApplyTheme(theme := "Light") {
    global mainGui, notificationPanel, notificationLabel, notificationClose, loaderLabel, progressBar, statusText
    global btnOpen, btnCancel, btnAbout, jobLabel, jobEdit, jobErrorLabel, favHint, debugCheckbox, folderListView

    bg := "", panelBg := "", panelFg := "", loaderFg := "", progressBg := "", progressFg := "", statusFg := ""
    btnOpenBg := "", btnOpenFg := "", btnCancelBg := "", btnCancelFg := "", btnAboutBg := "", btnAboutFg := ""
    notifCloseBg := "", notifCloseFg := "", jobLabelFg := "", jobEditBg := "", jobEditFg := "", jobErrorFg := ""
    favHintFg := "", debugCbFg := "", lvBg := "", lvFg := "", gbLabelFg := ""

    if (theme = "Dark") {
        bg := "20232A", panelBg := "444444", panelFg := "DDDDDD", loaderFg := "AACCFF", progressBg := "20232A", progressFg := "33AAFF", statusFg := "CCCCCC"
        btnOpenBg := "222222", btnOpenFg := "00CC99", btnCancelBg := "222222", btnCancelFg := "FF6688", btnAboutBg := "333333", btnAboutFg := "CCCCCC"
        notifCloseBg := "333333", notifCloseFg := "DDDDDD", jobLabelFg := "CCCCCC", jobEditBg := "333333", jobEditFg := "EEEEEE", jobErrorFg := "FF6666"
        favHintFg := "AAAAAA", debugCbFg := "CCCCCC", lvBg := "222222", lvFg := "CCCCCC", gbLabelFg := "AAAAAA"
    } else if (theme = "High Contrast") {
        bg := "000000", panelBg := "FFFF00", panelFg := "000000", loaderFg := "FFFFFF", progressBg := "000000", progressFg := "FFFFFF", statusFg := "FFFFFF"
        btnOpenBg := "FFFF00", btnOpenFg := "000000", btnCancelBg := "FF0000", btnCancelFg := "FFFFFF", btnAboutBg := "000000", btnAboutFg := "FFFF00"
        notifCloseBg := "FFFF00", notifCloseFg := "000000", jobLabelFg := "FFFFFF", jobEditBg := "FFFFFF", jobEditFg := "000000", jobErrorFg := "FF0000"
        favHintFg := "FFFF00", debugCbFg := "FFFFFF", lvBg := "000000", lvFg := "FFFFFF", gbLabelFg := "FFFFFF"
    } else {
        bg := "FFFFFF", panelBg := "D3D3D3", panelFg := "333333", loaderFg := "3366AA", progressBg := "FFFFFF", progressFg := "3366AA", statusFg := "333333"
        btnOpenBg := "F0F0F0", btnOpenFg := "005577", btnCancelBg := "F0F0F0", btnCancelFg := "AA4455", btnAboutBg := "F5F5F5", btnAboutFg := "333333"
        notifCloseBg := "F0F0F0", notifCloseFg := "333333", jobLabelFg := "333333", jobEditBg := "White", jobEditFg := "000000", jobErrorFg := "DD0000"
        favHintFg := "Gray", debugCbFg := "333333", lvBg := "FFFFFF", lvFg := "333333", gbLabelFg := "333333"
    }
    mainGui.Opt("Background" . bg)
    notificationPanel.Opt("Background" . panelBg), notificationLabel.Opt("c" . panelFg)
    loaderLabel.Opt("c" . loaderFg), progressBar.Opt("Background" . progressBg . " c" . progressFg), statusText.Opt("c" . statusFg)
    btnOpen.Opt("Background" . btnOpenBg . " c" . btnOpenFg), btnCancel.Opt("Background" . btnCancelBg . " c" . btnCancelFg), btnAbout.Opt("Background" . btnAboutBg . " c" . btnAboutFg)
    notificationClose.Opt("Background" . notifCloseBg . " c" . notifCloseFg)
    jobLabel.Opt("c" . jobLabelFg), jobEdit.Opt("Background" . jobEditBg . " c" . jobEditFg), jobErrorLabel.Opt("c" . jobErrorFg)
    favHint.Opt("c" . favHintFg), debugCheckbox.Opt("c" . debugCbFg)
    folderListView.Opt("Background" . lvBg), folderListView.SetTextColor(lvFg)
    for ctrl in mainGui {
        if InStr(ctrl.Type, "GroupBox")
            ctrl.Opt("c" . gbLabelFg)
    }
    mainGui.Redraw()
}

ApplyThemeToControl(ctrlObj, theme) { ; Helper to apply theme to a single control, e.g., jobEdit
    jobEditBg := "", jobEditFg := ""
    if (theme = "Dark") {
        jobEditBg := "333333", jobEditFg := "EEEEEE"
    } else if (theme = "High Contrast") {
        jobEditBg := "FFFFFF", jobEditFg := "000000"
    } else {
        jobEditBg := "White", jobEditFg := "000000"
    }
    if (ctrlObj == jobEdit) {
        ctrlObj.Opt("Background" . jobEditBg . " c" . jobEditFg)
    }
}

; --- Status + Progress ---
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(255) " w" Scale(316) " h" Scale(65), "Status")
loaderLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(258) " w" Scale(120) " h" Scale(14) " vLoaderLabel cBlue", "")
loaderLabel.Visible := false
progressBar := mainGui.Add("Progress", "x" Scale(28) " y" Scale(275) " w" Scale(280) " h" Scale(16) " vProgress1 Range0-100 cBlue +Smooth")
progressBar.Value := 0
statusText := mainGui.Add("Text", "x" Scale(28) " y" Scale(298) " w" Scale(280) " h" Scale(20) " vStatusText", "Ready")

WinSetExStyle("+0x02000000", mainGui.Hwnd)

btnOpen := mainGui.Add("Button", "x" Scale(70) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Default", "Open")
btnOpen.ToolTip := "Start project lookup."
btnOpen.OnEvent("Click", OpenProject)
btnCancel := mainGui.Add("Button", "x" Scale(180) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Disabled vBtnCancel", "Cancel")
btnCancel.ToolTip := "Cancel ongoing lookup."
btnCancel.OnEvent("Click", (*) => Controller_CancelProcessing())

Hotkey "Enter", OpenHotkey, "On"
Hotkey "Esc", CancelHotkey, "On"
Hotkey "F1", HelpHotkey, "On"
Hotkey "^p", PrefsHotkey, "On"

OpenHotkey(*) {
    global mainGui, jobEdit, folderListView, btnOpen
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    if (jobEdit.Focused || folderListView.Focused || Gui.FocusedCtrl = btnOpen)
        btnOpen.Click()
}

CancelHotkey(*) {
    global mainGui, btnCancel
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    if (btnCancel.Enabled)
        btnCancel.Click()
    else if (WinExist("ahk_class #32770") && (WinGetTitle() ~= "Preferences" || WinGetTitle() ~= "About / Help")) { ; Check for dialogs
        Send "{Esc}" ; Send Esc to close dialog if Cancel button is not active
    }
}

HelpHotkey(*) {
    global mainGui, btnAbout
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    ShowAboutDialog() ; Changed from btnAbout.Click() to directly call
}

PrefsHotkey(*) {
    global mainGui
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    ShowPreferencesDialog()
}

btnOpen.ToolTip := "Start project lookup. Shortcut: Enter"
btnCancel.ToolTip := "Cancel ongoing lookup. Shortcut: Esc"
btnAbout.ToolTip := "Show help and about information. Shortcut: F1"
jobEdit.ToolTip := "Enter a 5-digit job number or search term. Drag a folder here to auto-fill. Press Enter to start lookup."
folderListView.ToolTip := "Select a subfolder. Use arrow keys to navigate. Press Enter to open."

AnnounceStatus(msg) {
    global statusText
    statusText.Value := msg
}

debugCheckbox := mainGui.Add("Checkbox", "x" Scale(32) " y" Scale(230) " w" Scale(250) " vDebugMode", "Show Raw Python Output")
debugCheckbox.ToolTip := "Enable to see raw output/errors from backend."

UpdateNotificationPanelPosition() {
    global notificationPanel, notificationLabel, notificationClose, debugCheckbox, folderGroupBox
    folderGroupBoxPos := GetControlPosition(folderGroupBox)
    notifY := folderGroupBoxPos.y + folderGroupBoxPos.h + Scale(10, "spacing")
    notificationPanel.Move(, notifY)
    notificationLabel.Move(, notifY + Scale(7))
    notificationClose.Move(, notifY + Scale(4))
    debugCheckbox.Move(, notifY - Scale(25))
}

GetControlPosition(ctrl) {
    x := 0, y := 0, w := 0, h := 0
    ctrl.GetPos(&x, &y, &w, &h)
    return Map("x", x, "y", y, "w", w, "h", h)
}

SetProgress(val := "") {
    global progressBar, loaderLabel
    static timerOn := false, dots := 0, lastVal := -1
    Critical true
    if (val != "" && lastVal != -1 && Abs(val - lastVal) < 5 && val != 0 && val != 100) {
        Critical false
        return
    }
    lastVal := val
    if (val = "") { 
        progressBar.Opt("+Smooth")
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
        progressBar.Opt("+Smooth")
        progressBar.Value := val
        loaderLabel.Visible := false
        if timerOn {
            SetTimer(AnimateLoaderLabel, 0)
            timerOn := false
        }
    }
    Critical false
}
AnimateLoaderLabel() {
    global loaderLabel
    static dotCycle := ["", ".", "..", "..."], i := 1
    loaderLabel.Value := "Loading" . dotCycle[i]
    i := Mod(i, dotCycle.Length) + 1
}

ShowNotification(msg, type := "info", duration := 0) {
    global notificationPanel, notificationLabel, notificationClose, settings
    static bgColors, fgColors, lastMsg := "", lastType := ""

    if !IsSet(bgColors) {
        bgColors := Map("success", "CCE5FF", "error", "FFCCCC", "warning", "FFF5CC", "info", "D3D3D3")
        fgColors := Map("success", "007700", "error", "C80000", "warning", "A88400", "info", "333333")
    }

    effectiveDuration := duration
    if (duration == 0 && IsObject(settings) && settings.Has("notifDuration")) {
        effectiveDuration := Integer(settings.notifDuration)
    }

    if (msg == lastMsg && type == lastType && notificationPanel.Visible) {
        if (effectiveDuration && effectiveDuration > 0) {
            SetTimer(HideNotification, 0) 
            SetTimer(HideNotification, -Abs(effectiveDuration)) 
        }
        return
    }
    lastMsg := msg
    lastType := type

    Critical true
    notificationPanel.Opt("Background" . (bgColors.Has(type) ? bgColors[type] : bgColors["info"]))
    notificationLabel.Opt("c" . (fgColors.Has(type) ? fgColors[type] : fgColors["info"]))
    notificationLabel.Value := msg

    if (!notificationPanel.Visible) {
        notificationPanel.Visible := true
        notificationLabel.Visible := true
        notificationClose.Visible := true
    }
    notificationClose.OnEvent("Click", HideNotification)

    if (effectiveDuration && effectiveDuration > 0) {
        SetTimer(HideNotification, 0) 
        SetTimer(HideNotification, -Abs(effectiveDuration)) 
    }
    Critical false
}
HideNotification(*) { 
    global notificationPanel, notificationLabel, notificationClose
    notificationPanel.Visible := false
    notificationLabel.Visible := false
    notificationClose.Visible := false
}
ShowInlineHint(msg, type:="error") {
    ShowNotification(msg, type, 5000) 
}

OpenProject(ctrlObj, info) {
    global folderNames, mainGui, radioValues, folderListView, settings

    params := mainGui.Submit(false)
    currentDebugMode := params.DebugMode
    currentJobNumber := Trim(params.JobNumber)

    selectedFolder := ""
    selectedRow := 0
    focusedRow := folderListView.GetNext(0, "Focused")

    if (focusedRow > 0 && focusedRow <= radioValues.Length) {
        selectedRow := focusedRow
    } else {
        Loop radioValues.Length {
            if (radioValues[A_Index]) {
                selectedRow := A_Index
                break
            }
        }
    }

    if (selectedRow > 0 && selectedRow <= folderNames.Length) {
        selectedFolder := folderNames[selectedRow]
    } else if (folderNames.Length > 0) {
        selectedFolder := folderNames[1] 
    } else {
        ShowNotification("Error: No subfolders configured.", "error")
        return
    }
    Controller_OpenProject(currentJobNumber, selectedFolder, currentDebugMode)
}

GuiClose(*) {
    ExitApp 
}

GuiSize(guiObj, MinMax, Width, Height) {
    global mainGui, notificationPanel, notificationLabel, notificationClose, progressBar, statusText
    global btnOpen, btnCancel, folderListView, folderGroupBox, jobEdit, jobLabel, projectSelectionGB

    if (MinMax = -1)
        return

    contentWidth := Width - Scale(24)

    for ctrl in mainGui {
        if InStr(ctrl.Type, "GroupBox") {
            ctrl.Move(, , contentWidth)
        }
    }
    
    ; Resize jobEdit based on its groupbox (projectSelectionGB)
    jobEditXRelToGB := jobLabel.X - projectSelectionGB.X + jobLabel.W + Scale(6)
    newJobEditW := projectSelectionGB.W - jobEditXRelToGB - Scale(16) ; Scale(16) as right padding in GB
    jobEdit.Move(jobLabel.X + jobLabel.W + Scale(6), , newJobEditW < Scale(50) ? Scale(50) : newJobEditW)


    lvXRelToGB := folderListView.X - folderGroupBox.X
    newLvW := folderGroupBox.W - (lvXRelToGB * 2) ; Assuming same padding left/right for LV in GB
    folderListView.Move(, , newLvW)

    notificationPanel.Move(, , contentWidth)
    notificationLabel.Move(, , contentWidth - Scale(76))
    notificationClose.Move(Width - Scale(12) - notificationClose.W)

    progressX := progressBar.X
    statusX := statusText.X
    progressW := contentWidth - (progressX - Scale(12)) * 2
    statusW := contentWidth - (statusX - Scale(12)) * 2
    progressBar.Move(, , progressW)
    statusText.Move(, , statusW)
    
    btnWidth := btnOpen.W
    centerX := Width / 2
    btnOpen.Move(centerX - btnWidth - Scale(5))
    btnCancel.Move(centerX + Scale(5))

    UpdateNotificationPanelPosition()
    mainGui.Redraw()
}

AddSystemTrayMenu() {
    A_TrayMenu.Delete() 
    ToggleGuiMenu(*) => ToggleGui()
    ExitAppMenu(*) => ExitApp()
    ShowPrefsMenu(*) => ShowPreferencesDialog()
    ShowAboutMenu(*) => ShowAboutDialog()

    A_TrayMenu.Add("Show/Hide QuickNav", ToggleGuiMenu)
    A_TrayMenu.Add("Preferences...", ShowPrefsMenu)
    A_TrayMenu.Add("Help/About...", ShowAboutMenu)
    A_TrayMenu.Add() 
    A_TrayMenu.Add("Exit", ExitAppMenu)
    try A_TrayMenu.Default := "Show/Hide QuickNav"
    catch { ; ignore error
    }
    TraySetIcon("shell32.dll", 44) 
}
ShowAboutDialog() {
    global logPath, versionStr, settings
    txt := "
    (LTrim Join`r`n
Project QuickNav
Version: " . versionStr . "

Usage:
 - Enter a 5-digit job number or search term and select a subfolder.
 - Press 'Open' to launch the folder, or drag/drop a folder onto the input.
 - Right-click subfolders to favorite.
 - Cancel ongoing operations anytime.

Feature Summary:
 - Inline, persistent notifications (color-coded, dismissible)
 - Animated loader for backend/process work
 - User preferences: favorites, input mode, recents, notifications
 - Light/Dark/High Contrast theming
 - Comprehensive Help/About dialog

Keyboard Shortcuts:
 - Ctrl+Alt+Q: Show or focus QuickNav window
 - Enter: Open Project (when input or list focused)
 - Esc: Cancel Operation / Close Dialogs
 - F1: Help/About
 - Ctrl+P: Preferences

For detailed documentation, see README.md or INSTALL.md.
    )"
    static aboutGuiObj ; Make static to allow Destroy if already open
    if IsObject(aboutGuiObj) && aboutGuiObj.Hwnd {
        aboutGuiObj.Destroy()
    }
    static aboutGuiObj ; Make static to allow Destroy if already open
    if IsObject(aboutGuiObj) && aboutGuiObj.Hwnd {
        aboutGuiObj.Destroy()
    }
    aboutGuiObj := Gui("+AlwaysOnTop", "About / Help - QuickNav v" . versionStr)


    editWidth := Scale(410)
    editHeight := Scale(220, "text") 
    buttonWidth := Scale(140)
    buttonHeight := Scale(30)
    smallButtonWidth := Scale(90)
    xPadding := Scale(10, "spacing")
    yPadding := Scale(10, "spacing")
    buttonY := editHeight + yPadding * 2

    dialogWidth := editWidth + (xPadding * 2)
    dialogHeight := buttonY + buttonHeight + yPadding

    aboutGuiObj.Add("Edit", "x" . xPadding . " y" . yPadding . " w" . editWidth . " h" . editHeight . " -Wrap ReadOnly", txt)

    btnCount := 3
    totalButtonWidth := buttonWidth * 2 + smallButtonWidth + xPadding * (btnCount -1)
    startX := (dialogWidth - totalButtonWidth) / 2

    btnDiag := aboutGuiObj.Add("Button", "x" . startX . " y" . buttonY . " w" . buttonWidth . " h" . buttonHeight, "Open Diagnostics Folder")
    btnLog := aboutGuiObj.Add("Button", "x+" . xPadding . " yp w" . buttonWidth . " h" . buttonHeight, "View Error Log")
    aboutGuiObj.Add("Button", "x+" . xPadding . " yp w" . smallButtonWidth . " h" . buttonHeight . " Default", "OK").OnEvent("Click", (*) => aboutGuiObj.Destroy())

    btnDiag.OnEvent("Click", (*) => Run('explorer.exe "' . DirSplit(logPath)[1] . '"'))
    btnLog.OnEvent("Click", (*) => (FileExist(logPath) ? Run('notepad.exe "' . logPath . '"') : ShowNotification("No error log found.","info")))
    
    ApplyThemeToPrefsDialog(aboutGuiObj, IsObject(settings) && settings.Has("theme") ? settings.theme : "Light") 
    aboutGuiObj.Show("w" . dialogWidth . " h" . dialogHeight)
}

ToggleGui() {
    global mainGui
    if (mainGui.Visible) {
        mainGui.Hide()
    } else {
        mainGui.Show()
        WinActivate("ahk_id " mainGui.Hwnd)
    }
}
^!q:: ToggleGui()

ResetGUI() {
    global mainGui, progressBar, btnOpen, btnCancel, jobEdit, statusText, jobErrorLabel, settings
    statusText.Value := "Ready"
    progressBar.Value := 0
    SetProgress(0) 
    btnOpen.Enabled := true
    btnCancel.Enabled := false
    ApplyThemeToControl(jobEdit, IsObject(settings) && settings.Has("theme") ? settings.theme : "Light")
    jobErrorLabel.Visible := false
    ApplyTheme(IsObject(settings) && settings.Has("theme") ? settings.theme : "Light") 
}

ReloadFolderRadios() {
    global folderNames, radioValues, recentsData, defaultFolderNames, mainGui, folderListView

    recentsData := LoadRecents() 
    folderNames := []
    if (IsObject(recentsData) && recentsData.Has("favorites") && (TypeOf(recentsData.favorites) = "Array") && recentsData.favorites.Length) {
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

    folderListView.Delete() 
    radioValues := []

    Loop folderNames.Length {
        folderListView.Add("", folderNames[A_Index])
        radioValues.Push(false)
    }

    if (folderNames.Length > 0) {
        folderListView.Modify(1, "Select Focus")
        radioValues[1] := true
    } else {
        ; Handle case where there are no folders (e.g. all favorites removed and no defaults)
        ShowNotification("No subfolders available. Please check favorites or configuration.", "warning")
    }

    UpdateNotificationPanelPosition()
    mainGui.Redraw()
    ShowNotification("Favorites updated.", "success", 2000)
}

AddSystemTrayMenu()
mainGui.OnEvent("Close", GuiClose)

if IsObject(settings) && settings.Has("theme") {
    ApplyTheme(settings.theme)
} else {
    ApplyTheme("Light") 
}

UpdateNotificationPanelPosition() 
mainGui.Show("w" . Scale(340) . " h" . Scale(380))
return