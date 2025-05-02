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

; Set up error handling
OnError QuickNav_ErrorHandler

; Error handler function
QuickNav_ErrorHandler(err) {
    LogError("Unhandled error: " . err.Message . " at line " . err.Line, "ErrorHandler")
    MsgBox("An error occurred:`n" . err.Message . "`n`nLine: " . err.Line . "`n`nSee error log for details.", "QuickNav Error", 16)
    return true  ; Continue running the script
}

#Include %A_ScriptDir%\lld_navigator_controller.ahk

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

; --- Instantiate main GUI window with resizing support ---
mainGui := Gui("+AlwaysOnTop +Resize +MinSize340x380", "Project QuickNav v" . versionStr)

; --- Handle window resize events ---
mainGui.OnEvent("Size", GuiSize)

; --- Persistent Notification Area (positioned dynamically) ---
; Create notification controls with placeholder positions - will be positioned later
notificationPanel := mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(0) " w" Scale(316) " h" Scale(28) " vNotificationPanel BackgroundD3D3D3", "")
notificationLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(0) " w" Scale(240) " h" Scale(14) " vNotificationLabel", "")
notificationClose := mainGui.Add("Button", "x" Scale(280) " y" Scale(0) " w" Scale(36) " h" Scale(20) " vNotificationClose", "✕")
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

; --- Drag-and-drop for job input with enhanced error handling ---
OnMessage(0x233, WM_DROPFILES)
WM_DROPFILES(wParam, lParam, msg, hwnd) {
    global jobEdit, folderNames, jobErrorLabel

    try {
        ; Check if we have any files dropped
        files := DllCall("shell32\DragQueryFile", "ptr", wParam, "uint", 0xFFFFFFFF, "ptr", 0, "uint", 0)
        if (files <= 0) {
            ShowInlineHint("No valid files were dropped.", "warning")
            return
        }

        ; Get the first dropped file/folder path
        path := Buffer(1024)
        if (!DllCall("shell32\DragQueryFile", "ptr", wParam, "uint", 0, "ptr", path, "uint", 1024)) {
            ShowInlineHint("Could not retrieve the dropped path.", "error")
            LogError("DragQueryFile failed to retrieve path", "WM_DROPFILES")
            return
        }

        ; Process the dropped path
        droppedPath := Trim(StrGet(path))

        ; Check if path exists
        if (!FileExist(droppedPath)) {
            ShowInlineHint("The dropped path does not exist: " . droppedPath, "error")
            return
        }

        ; Check if it's a directory
        if (!DirExist(droppedPath)) {
            ShowInlineHint("Please drop a folder, not a file.", "warning")
            return
        }

        ; Validate and normalize the input
        v := ValidateAndNormalizeInputs(droppedPath, folderNames[1])
        if (!v.valid) {
            ; Show error in the UI instead of a modal dialog
            jobErrorLabel.Value := v.errorMsg
            jobErrorLabel.Visible := true
            jobEdit.Opt("BackgroundFFDDDD")  ; Highlight with error color
            ShowInlineHint("Invalid path format. See error below.", "error")

            ; Clear error after a delay
            SetTimer(() => (
                jobErrorLabel.Visible := false,
                jobEdit.Opt("BackgroundWhite")
            ), -5000)
        } else {
            ; Clear any previous errors
            jobErrorLabel.Visible := false
            jobEdit.Opt("BackgroundWhite")

            ; Set the job number and show success message
            jobEdit.Value := v.normalizedJob
            ShowInlineHint("Detected job " . v.normalizedJob . " from path.", "success")

            ; If we have a valid folder match, select it in the ListView
            if (v.normalizedFolder && folderNames.Length > 0) {
                folderIndex := folderNames.IndexOf(v.normalizedFolder)
                if (folderIndex > 0) {
                    folderListView.Modify(folderIndex, "Select Focus Vis")
                    radioValues[folderIndex] := true
                    FlashListViewItem(folderIndex)
                }
            }
        }
    } catch as e {
        ; Log the error and show a user-friendly message
        LogError("Drag and drop error: " . e.Message, "WM_DROPFILES")
        ShowInlineHint("An error occurred processing the dropped item.", "error")
    } finally {
        ; Always clean up the drop handle
        if (IsSet(wParam) && wParam)
            DllCall("shell32\DragFinish", "ptr", wParam)
    }
}

; --- Subfolder Selection (Radio group with scrolling) ---
folderGroupBox := mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(80) " w" Scale(316) " h" Scale(160), "Select Subfolder")

; Create a ListView to act as a scrollable container with visual feedback
folderListView := mainGui.Add("ListView", "x" Scale(20) " y" Scale(100) " w" Scale(300) " h" Scale(130) " -Hdr -Multi -ReadOnly -LV0x10000 AltSubmit Grid Background" . (A_IsDarkMode ? "222222" : "FFFFFF"), ["Folder"])
folderListView.OnEvent("Click", FolderListViewClick)
folderListView.OnEvent("ItemFocus", FolderListViewFocus)

; Add hover effect for ListView
OnMessage(0x200, WM_MOUSEMOVE)  ; Track mouse movement for hover effects

; Set up radio buttons
y_radio := Scale(100)
radioCtrls := []
radioValues := []

; Add folders to the ListView
Loop folderNames.Length {
    folderListView.Add("", folderNames[A_Index])
    radioValues.Push(false)  ; Initialize all to false
}

; Set the first item as selected
folderListView.Modify(1, "Select Focus")
radioValues[1] := true

; Add Help/About button
btnAbout := mainGui.Add("Button", "x" Scale(28) " y" Scale(320) " w" Scale(100) " h" Scale(22), "Help/About")
btnAbout.OnEvent("Click", (*) => ShowAboutDialog())

; Add favorite hint
favHint := mainGui.Add("Text", "x" Scale(32) " y" Scale(240) " w" Scale(250) " cGray", "Tip: Right-click a subfolder to favorite/unfavorite")

; Handle ListView click events
FolderListViewClick(ctrl, rowNum) {
    global radioValues, folderListView

    ; Clear previous selection
    Loop radioValues.Length
        radioValues[A_Index] := false

    ; Set new selection
    if (rowNum > 0 && rowNum <= radioValues.Length)
        radioValues[rowNum] := true

    ; Update visual selection
    folderListView.Modify(rowNum, "Select Focus Vis")

    ; Provide visual feedback
    FlashListViewItem(rowNum)
}

; Handle ListView focus events
FolderListViewFocus(ctrl, rowNum) {
    global radioValues

    ; Clear previous selection
    Loop radioValues.Length
        radioValues[A_Index] := false

    ; Set new selection
    if (rowNum > 0 && rowNum <= radioValues.Length)
        radioValues[rowNum] := true
}

; Provide visual feedback when hovering over ListView items
WM_MOUSEMOVE(wParam, lParam, msg, hwnd) {
    global folderListView
    static lastHoverRow := 0

    ; Check if mouse is over the ListView
    if (hwnd = folderListView.Hwnd) {
        ; Get the item under the mouse
        VarSetStrCapacity(&lvhti, 24)
        NumPut("UInt", 24, lvhti, 0)
        NumPut("Int", lParam & 0xFFFF, lvhti, 8)
        NumPut("Int", (lParam >> 16) & 0xFFFF, lvhti, 12)

        SendMessage(0x1012, 0, &lvhti, folderListView.Hwnd)  ; LVM_HITTEST
        hoverRow := NumGet(lvhti, 16, "Int") + 1

        ; If hovering over a different row, update hover effect
        if (hoverRow != lastHoverRow && hoverRow > 0) {
            ; Reset previous hover row
            if (lastHoverRow > 0 && folderListView.GetNext(lastHoverRow - 1, "Selected") != lastHoverRow)
                folderListView.Modify(lastHoverRow, "-Select")

            ; Set hover effect on current row if not already selected
            if (folderListView.GetNext(hoverRow - 1, "Selected") != hoverRow)
                folderListView.Modify(hoverRow, "+Select -Focus")

            lastHoverRow := hoverRow
        }
    }
}

; Flash an item in the ListView for visual feedback
FlashListViewItem(rowNum) {
    global folderListView

    ; Save current colors
    originalBgColor := folderListView.GetBGColor()
    originalTextColor := folderListView.GetTextColor()

    ; Flash with highlight color
    folderListView.SetBGColor("3399FF")
    folderListView.SetTextColor("FFFFFF")
    folderListView.Redraw()

    ; Reset after a short delay
    SetTimer(() => (
        folderListView.SetBGColor(originalBgColor),
        folderListView.SetTextColor(originalTextColor),
        folderListView.Redraw()
    ), -150)
}

; --- Right-click favorites context menu ---
global currentFolderIndex := 0
global folderContextMenu := Menu()
folderContextMenu.Add("Toggle Favorite", (*) => Controller_ToggleFavorite(folderNames[currentFolderIndex]))

; Add context menu to ListView
folderListView.OnEvent("ContextMenu", FolderListViewContextMenu)

; Handle ListView context menu events
FolderListViewContextMenu(ctrl, rowNum) {
    global currentFolderIndex, folderContextMenu, folderNames

    ; Get the row under the mouse
    if (rowNum > 0 && rowNum <= folderNames.Length) {
        currentFolderIndex := rowNum
        folderContextMenu.Show()
    }
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

    ; Create controls with consistent spacing
    yPos := 12
    yStep := 32

    ; Default Folder
    prefGui.Add("Text", "x12 y" . yPos . " w120 h20", "Default Folder:")
    defaultFolder := settings.Has("defaultFolder") ? settings["defaultFolder"] : (folderNames.Length ? folderNames[1] : "")
    controls["defaultFolder"] := prefGui.Add("DropDownList", "x140 y" . yPos . " w150 vDefaultFolder", folderNames)
    controls["defaultFolder"].Choose(defaultFolder ? folderNames.IndexOf(defaultFolder) + 1 : 1)

    ; Job Input Behavior
    yPos += yStep
    prefGui.Add("Text", "x12 y" . yPos . " w120 h20", "Job Input Behavior:")
    jobInputOpts := ["Prompt", "Auto-fill Last", "Auto-fill Favorite"]
    defaultJobInput := settings.Has("jobInputMode") ? settings["jobInputMode"] : "Prompt"
    controls["jobInputMode"] := prefGui.Add("DropDownList", "x140 y" . yPos . " w150 vJobInputMode", jobInputOpts)
    controls["jobInputMode"].Choose(jobInputOpts.IndexOf(defaultJobInput) + 1)

    ; Theme
    yPos += yStep
    prefGui.Add("Text", "x12 y" . yPos . " w120 h20", "Theme:")
    themeOpts := ["Light", "Dark", "High Contrast"]
    defaultTheme := settings.Has("theme") ? settings["theme"] : "Light"
    controls["theme"] := prefGui.Add("DropDownList", "x140 y" . yPos . " w150 vTheme", themeOpts)
    controls["theme"].Choose(themeOpts.IndexOf(defaultTheme) + 1)

    ; Maximum Recents
    yPos += yStep
    prefGui.Add("Text", "x12 y" . yPos . " w120 h20", "Maximum Recents:")
    maxRecents := settings.Has("maxRecents") ? settings["maxRecents"] : 10
    controls["maxRecents"] := prefGui.Add("Edit", "x140 y" . yPos . " w60 vMaxRecents", maxRecents)

    ; Notification Duration
    yPos += yStep
    prefGui.Add("Text", "x12 y" . yPos . " w120 h20", "Notification Duration (ms):")
    notifDur := settings.Has("notifDuration") ? settings["notifDuration"] : 3000
    controls["notifDuration"] := prefGui.Add("Edit", "x140 y" . yPos . " w60 vNotifDuration", notifDur)

    ; Buttons
    yPos += yStep + 10
    btnSave := prefGui.Add("Button", "x30 y" . yPos . " w80 Default", "Save")
    btnCancel := prefGui.Add("Button", "x120 y" . yPos . " w80", "Cancel")
    btnReset := prefGui.Add("Button", "x210 y" . yPos . " w80", "Reset App")

    ; Button event handlers
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

    ; Apply current theme to preferences dialog
    ApplyThemeToPrefsDialog(prefGui, settings.Has("theme") ? settings["theme"] : "Light")

    ; Show dialog with appropriate size
    prefGui.Show("w310 h" . (yPos + 40))
}

; Helper function to apply theme to preferences dialog
ApplyThemeToPrefsDialog(prefGui, theme) {
    if (theme = "Dark") {
        prefGui.Opt("Background20232A")
        for ctrl in prefGui
            if InStr(ctrl.Type, "Text")
                ctrl.Opt("cCCCCCC")
            else if InStr(ctrl.Type, "Button")
                ctrl.Opt("Background333333 cCCCCCC")
    } else if (theme = "High Contrast") {
        prefGui.Opt("Background000000")
        for ctrl in prefGui
            if InStr(ctrl.Type, "Text")
                ctrl.Opt("cFFFFFF")
            else if InStr(ctrl.Type, "Button")
                ctrl.Opt("BackgroundFFFF00 c000000")
    } else {
        prefGui.Opt("BackgroundFFFFFF")
        for ctrl in prefGui
            if InStr(ctrl.Type, "Text")
                ctrl.Opt("c333333")
            else if InStr(ctrl.Type, "Button")
                ctrl.Opt("BackgroundF0F0F0 c333333")
    }
}
ApplyTheme(theme := "Light") {
    global mainGui, notificationPanel, notificationLabel, notificationClose, loaderLabel, progressBar, statusText
    global btnOpen, btnCancel, btnAbout, jobLabel, jobEdit, jobErrorLabel, radioCtrls, favHint, debugCheckbox

    if (theme = "Dark") {
        ; Main window and panels
        mainGui.Opt("Background20232A")
        notificationPanel.Opt("Background444444")
        notificationLabel.Opt("cDDDDDD")

        ; Status and progress indicators
        loaderLabel.Opt("cAACCFF")
        progressBar.Opt("c33AAFF")
        statusText.Opt("cCCCCCC")

        ; Buttons
        btnOpen.Opt("Background222222 c00CC99")
        btnCancel.Opt("Background222222 cFF6688")
        btnAbout.Opt("Background333333 cCCCCCC")
        notificationClose.Opt("Background333333 cDDDDDD")

        ; Input fields and labels
        jobLabel.Opt("cCCCCCC")
        jobEdit.Opt("Background333333 cEEEEEE")
        jobErrorLabel.Opt("cFF6666")

        ; Radio buttons and hints
        favHint.Opt("cAAAAAA")
        for ctrl in radioCtrls
            ctrl.Opt("cCCCCCC")

        ; Debug checkbox
        debugCheckbox.Opt("cCCCCCC")

        ; ListView styling
        folderListView.SetBGColor("222222")
        folderListView.SetTextColor("CCCCCC")

        ; GroupBoxes - find all GroupBox controls and style them
        for ctrl in mainGui
            if InStr(ctrl.Type, "GroupBox")
                ctrl.Opt("cAAAAAA")

    } else if (theme = "High Contrast") {
        ; Main window and panels
        mainGui.Opt("Background000000")
        notificationPanel.Opt("BackgroundFFFF00")
        notificationLabel.Opt("c000000")

        ; Status and progress indicators
        loaderLabel.Opt("cFFFFFF")
        progressBar.Opt("cFFFFFF")
        statusText.Opt("cFFFFFF")

        ; Buttons
        btnOpen.Opt("BackgroundFFFF00 c000000")
        btnCancel.Opt("BackgroundFF0000 cFFFFFF")
        btnAbout.Opt("Background000000 cFFFF00")
        notificationClose.Opt("BackgroundFFFF00 c000000")

        ; Input fields and labels
        jobLabel.Opt("cFFFFFF")
        jobEdit.Opt("BackgroundFFFFFF c000000")
        jobErrorLabel.Opt("cFF0000")

        ; Radio buttons and hints
        favHint.Opt("cFFFF00")
        for ctrl in radioCtrls
            ctrl.Opt("cFFFFFF")

        ; Debug checkbox
        debugCheckbox.Opt("cFFFFFF")

        ; ListView styling
        folderListView.SetBGColor("000000")
        folderListView.SetTextColor("FFFFFF")

        ; GroupBoxes - find all GroupBox controls and style them
        for ctrl in mainGui
            if InStr(ctrl.Type, "GroupBox")
                ctrl.Opt("cFFFFFF")

    } else {
        ; Main window and panels
        mainGui.Opt("BackgroundFFFFFF")
        notificationPanel.Opt("BackgroundD3D3D3")
        notificationLabel.Opt("c333333")

        ; Status and progress indicators
        loaderLabel.Opt("c3366AA")
        progressBar.Opt("c3366AA")
        statusText.Opt("c333333")

        ; Buttons
        btnOpen.Opt("BackgroundF0F0F0 c005577")
        btnCancel.Opt("BackgroundF0F0F0 cAA4455")
        btnAbout.Opt("BackgroundF5F5F5 c333333")
        notificationClose.Opt("BackgroundF0F0F0 c333333")

        ; Input fields and labels
        jobLabel.Opt("c333333")
        jobEdit.Opt("BackgroundWhite c000000")
        jobErrorLabel.Opt("cDD0000")

        ; Radio buttons and hints
        favHint.Opt("cGray")
        for ctrl in radioCtrls
            ctrl.Opt("c333333")

        ; Debug checkbox
        debugCheckbox.Opt("c333333")

        ; ListView styling
        folderListView.SetBGColor("FFFFFF")
        folderListView.SetTextColor("333333")

        ; GroupBoxes - find all GroupBox controls and style them
        for ctrl in mainGui
            if InStr(ctrl.Type, "GroupBox")
                ctrl.Opt("c333333")
    }

    mainGui.Redraw()
}

; --- Status + Progress ---
mainGui.Add("GroupBox", "x" Scale(12) " y" Scale(255) " w" Scale(316) " h" Scale(65), "Status")
loaderLabel := mainGui.Add("Text", "x" Scale(28) " y" Scale(258) " w" Scale(120) " h" Scale(14) " vLoaderLabel cBlue", "")
loaderLabel.Visible := false
progressBar := mainGui.Add("Progress", "x" Scale(28) " y" Scale(275) " w" Scale(280) " h" Scale(16) " vProgress1 Range0-100 cBlue +Smooth")
progressBar.Value := 0
statusText := mainGui.Add("Text", "x" Scale(28) " y" Scale(298) " w" Scale(280) " h" Scale(20) " vStatusText", "Ready")

; --- Reduce flickering with double-buffering ---
WinSetExStyle("+0x02000000", mainGui.Hwnd)  ; WS_EX_COMPOSITED style to reduce flicker

; --- Main Action Buttons ---
btnOpen := mainGui.Add("Button", "x" Scale(70) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Default", "Open")
btnOpen.ToolTip := "Start project lookup."
btnOpen.OnEvent("Click", OpenProject)
btnCancel := mainGui.Add("Button", "x" Scale(180) " y" Scale(330) " w" Scale(80) " h" Scale(32) " Disabled vBtnCancel", "Cancel")
btnCancel.ToolTip := "Cancel ongoing lookup."
btnCancel.OnEvent("Click", (*) => Controller_CancelProcessing())
; --- Accessibility: Keyboard Shortcuts and Screen Reader Announcements ---

; Hotkeys for main actions (active only when main window is active)
Hotkey("Enter", OpenHotkey, "On")
Hotkey("Esc", CancelHotkey, "On")
Hotkey("F1", HelpHotkey, "On")
Hotkey("^p", PrefsHotkey, "On")

OpenHotkey(*) {
    global mainGui, jobEdit, folderListView, btnOpen
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    if (jobEdit.Focused || folderListView.Focused)
        btnOpen.Click()
}

CancelHotkey(*) {
    global mainGui, btnCancel
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    if (btnCancel.Enabled)
        btnCancel.Click()
}

HelpHotkey(*) {
    global mainGui, btnAbout
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    btnAbout.Click()
}

PrefsHotkey(*) {
    global mainGui
    if !WinActive("ahk_id " mainGui.Hwnd)
        return
    ShowPreferencesDialog()
}

; Ensure all controls have tooltips for screen readers
btnOpen.ToolTip := "Start project lookup. Shortcut: Enter"
btnCancel.ToolTip := "Cancel ongoing lookup. Shortcut: Esc"
btnAbout.ToolTip := "Show help and about information. Shortcut: F1"
debugCheckbox.ToolTip := "Enable to see raw output/errors from backend."
jobEdit.ToolTip := "Enter a 5-digit job number or search term. Drag a folder here to auto-fill. Press Enter to start lookup."
folderListView.ToolTip := "Select a subfolder. Use arrow keys to navigate. Press Enter to open."

; Announce status changes for screen readers
AnnounceStatus(msg) {
    global statusText
    statusText.Value := msg
    SetTimer(() => statusText.Value := "", -2000)
}

; --- Debug checkbox ---
debugCheckbox := mainGui.Add("Checkbox", "x" Scale(32) " y" Scale(230) " w" Scale(250) " vDebugMode", "Show Raw Python Output")
debugCheckbox.ToolTip := "Enable to see raw output/errors from backend."

; --- Update notification panel position based on folder list ---
UpdateNotificationPanelPosition() {
    global notificationPanel, notificationLabel, notificationClose, debugCheckbox, folderGroupBox

    ; Get the position and size of the folder group box
    folderGroupBoxPos := GetControlPosition(folderGroupBox)

    ; Calculate position based on folder group box
    notifY := folderGroupBoxPos.y + folderGroupBoxPos.h + Scale(10, "spacing")

    ; Update positions
    notificationPanel.Move(, notifY)
    notificationLabel.Move(, notifY + Scale(7))
    notificationClose.Move(, notifY + Scale(4))

    ; Update debug checkbox position to be just above notification panel
    debugCheckbox.Move(, notifY - Scale(25))
}

; Helper function to get control position and size
GetControlPosition(ctrl) {
    x := y := w := h := 0
    ctrl.GetPos(&x, &y, &w, &h)
    pos := Map("x", x, "y", y, "w", w, "h", h)
    return pos
}

; --- Loader/progress animation helpers ---
SetProgress(val := "") {
    global progressBar, loaderLabel
    static timerOn := false
    static dots := 0
    static lastVal := -1

    ; Batch updates to reduce flickering
    Critical "On"

    ; Only update if value has changed significantly (at least 5%)
    if (val != "" && lastVal != -1 && Abs(val - lastVal) < 5 && val != 0 && val != 100) {
        Critical "Off"
        return
    }

    ; Update last value
    lastVal := val

    if (val = "") {
        ; Indeterminate progress
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
        ; Determinate progress
        progressBar.Marquee := false
        progressBar.Opt("+Smooth")
        progressBar.Value := val
        loaderLabel.Visible := false
        if timerOn {
            SetTimer(AnimateLoaderLabel, 0)
            timerOn := false
        }
    }

    ; End critical section
    Critical "Off"
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
    static bgColors, fgColors, lastMsg := "", lastType := ""
    static notificationTimer := 0

    ; Initialize color maps if needed
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

    ; Avoid redundant updates for the same message and type
    if (msg == lastMsg && type == lastType && notificationPanel.Visible) {
        ; Just reset the timer if there is one
        if (duration && duration > 0) {
            if (notificationTimer)
                SetTimer(notificationTimer, 0)
            notificationTimer := ObjBindMethod(this, "HideNotification")
            SetTimer(notificationTimer, -duration)
        }
        return
    }

    ; Store current message and type
    lastMsg := msg
    lastType := type

    ; Batch UI updates to reduce flickering
    Critical "On"

    ; Update notification appearance
    notificationPanel.Opt("Background" . (bgColors.Has(type) ? bgColors[type] : bgColors["info"]))
    notificationLabel.Opt("c" . (fgColors.Has(type) ? fgColors[type] : fgColors["info"]))
    notificationLabel.Value := msg

    ; Make notification visible
    if (!notificationPanel.Visible) {
        notificationPanel.Visible := true
        notificationLabel.Visible := true
        notificationClose.Visible := true
    }

    ; Set up close button
    notificationClose.OnEvent("Click", (*) => HideNotification())

    ; Set auto-hide timer if duration is specified
    if (duration && duration > 0) {
        if (notificationTimer)
            SetTimer(notificationTimer, 0)
        notificationTimer := ObjBindMethod(this, "HideNotification")
        SetTimer(notificationTimer, -duration)
    }

    Critical "Off"
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
    global folderNames, mainGui, radioValues, folderListView

    ; Get form values
    params := mainGui.Submit(false)
    DebugMode := params.DebugMode
    jobNumber := params.JobNumber

    ; Get selected folder from ListView
    selectedFolder := ""
    selectedRow := 0

    ; First try to get the focused row
    focusedRow := folderListView.GetNext(0, "Focused")
    if (focusedRow > 0) {
        selectedRow := focusedRow
    } else {
        ; Fall back to radioValues array
        Loop radioValues.Length {
            if (radioValues[A_Index]) {
                selectedRow := A_Index
                break
            }
        }
    }

    ; Get the folder name for the selected row
    if (selectedRow > 0 && selectedRow <= folderNames.Length) {
        selectedFolder := folderNames[selectedRow]
    } else {
        ; Default to first folder if nothing is selected
        selectedFolder := folderNames[1]
    }

    ; Call the controller function
    Controller_OpenProject(jobNumber, selectedFolder, DebugMode)
}

; --- Main GUI event handlers ---
GuiClose(*) {
    mainGui.Hide()
    return
}

; Handle window resize events
GuiSize(GuiObj, MinMax, Width, Height) {
    global mainGui, notificationPanel, notificationLabel, notificationClose, progressBar, statusText
    global btnOpen, btnCancel, btnAbout, radioCtrls, favHint, debugCheckbox

    ; Skip if minimized
    if (MinMax = -1)
        return

    ; Calculate new widths based on window size
    newWidth := Width - Scale(24)  ; Padding on both sides

    ; Resize GroupBoxes
    for ctrl in mainGui
        if InStr(ctrl.Type, "GroupBox")
            ctrl.Move(, , newWidth)

    ; Resize notification panel and its controls
    notificationPanel.Move(, , newWidth)
    notificationLabel.Move(, , newWidth - Scale(76))
    notificationClose.Move(Scale(newWidth - 36))

    ; Resize progress bar and status text
    progressBar.Move(, , newWidth - Scale(36))
    statusText.Move(, , newWidth - Scale(36))

    ; Resize radio buttons
    for ctrl in radioCtrls
        ctrl.Move(, , newWidth - Scale(66))

    ; Reposition buttons
    centerX := Width / 2
    btnOpen.Move(centerX - Scale(90))
    btnCancel.Move(centerX + Scale(10))

    ; Update notification panel position
    UpdateNotificationPanelPosition()
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

    ; Use proper scaling for all dimensions
    editWidth := Scale(410)
    editHeight := Scale(180, "text")
    buttonWidth := Scale(120)
    buttonHeight := Scale(30)
    smallButtonWidth := Scale(90)
    xPadding := Scale(10, "spacing")
    yPadding := Scale(10, "spacing")
    buttonY := editHeight + Scale(20, "spacing")

    ; Calculate dialog dimensions
    dialogWidth := editWidth + (xPadding * 2)
    dialogHeight := buttonY + buttonHeight + yPadding

    ; Add controls with scaled dimensions
    GuiObj.Add("Edit", "x" . xPadding . " y" . yPadding . " w" . editWidth . " h" . editHeight . " -Wrap ReadOnly", txt)

    ; Position buttons with proper spacing
    btn1X := Scale(30)
    btn2X := btn1X + buttonWidth + Scale(20, "spacing")
    btn3X := btn2X + buttonWidth + Scale(20, "spacing")

    btnDiag := GuiObj.Add("Button", "x" . btn1X . " y" . buttonY . " w" . buttonWidth . " h" . buttonHeight, "Open Diagnostics Folder")
    btnLog := GuiObj.Add("Button", "x" . btn2X . " y" . buttonY . " w" . buttonWidth . " h" . buttonHeight, "View Error Log")
    GuiObj.Add("Button", "x" . btn3X . " y" . buttonY . " w" . smallButtonWidth . " h" . buttonHeight . " Default", "OK").OnEvent("Click", (*) => GuiObj.Destroy())

    btnDiag.OnEvent("Click", (*) => (Run("explorer.exe " . Chr(34) . DirSplit(logPath)[1] . Chr(34))))
    btnLog.OnEvent("Click", (*) => (
        FileExist(logPath)
        ? Run("notepad.exe " . Chr(34) . logPath . Chr(34))
        : ShowNotification("No error log found.","info")
    ))

    ; Show dialog with scaled dimensions
    GuiObj.Show("w" . dialogWidth . " h" . dialogHeight)
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

; --- Reload folder list when favorites change ---
ReloadFolderRadios() {
    global folderNames, radioValues, recentsData, defaultFolderNames, mainGui, folderListView

    ; Rebuild folderNames from favorites
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

    ; Clear ListView and radioValues
    folderListView.Delete()
    radioValues := []

    ; Add folders to ListView
    Loop folderNames.Length {
        folderListView.Add("", folderNames[A_Index])
        radioValues.Push(false)
    }

    ; Select first item
    if (folderNames.Length > 0) {
        folderListView.Modify(1, "Select Focus")
        radioValues[1] := true
    }

    ; Update notification panel position and redraw GUI
    UpdateNotificationPanelPosition()
    mainGui.Redraw()

    ShowNotification("Favorites updated.", "success", 2000)
}

; --- App entrypoint: initialize tray, show GUI ---
AddSystemTrayMenu()
mainGui.OnEvent("Close", GuiClose)

; Position notification panel based on current radio buttons
UpdateNotificationPanelPosition()

; Show the main GUI
mainGui.Show("w" . Scale(340) . " h" . Scale(380))
return