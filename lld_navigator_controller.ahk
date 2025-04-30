/*
lld_navigator_controller.ahk
Controller and business logic for Project QuickNav AHK frontend.
Handles:
- Input validation and normalization
- Recents, favorites, and settings persistence
- Project backend invocation and result handling
- State transitions, logging, cancellation
- All non-GUI business logic

Interface: 
- Designed to be #Included by lld_navigator.ahk.
- All controller functions are prefixed with Controller_*
- UI script delegates: Controller_OpenProject, Controller_CancelProcessing, Controller_ToggleFavorite, Controller_ReloadFolderRadios, etc.
*/

; === Input Validation & Normalization ===
ValidateAndNormalizeInputs(jobInput, folderInput) {
    jobIn := Trim(jobInput, " `t`r`n")
    folderIn := Trim(folderInput, " `t`r`n")
    forbiddenChars := "|&;><`'" . Chr(34) . "/\\"
    Loop StrLen(jobIn) {
        ch := SubStr(jobIn, A_Index, 1)
        if InStr(forbiddenChars, ch) or (ch = "`r" or ch = "`n")
            return { valid: false, errorMsg: "Invalid or unsafe character '" . ch . "' detected in job/search input.", normalizedJob: jobIn, normalizedFolder: folderIn }
    }
    Loop StrLen(folderIn) {
        ch := SubStr(folderIn, A_Index, 1)
        if InStr(forbiddenChars, ch) or (ch = "`r" or ch = "`n")
            return { valid: false, errorMsg: "Invalid or unsafe character '" . ch . "' detected in folder name.", normalizedJob: jobIn, normalizedFolder: folderIn }
    }
    if (RegExMatch(jobIn, "^\d{5}$")) {
        ; Valid job number
    } else if (RegExMatch(jobIn, "^[A-Za-z0-9 _\-]+$")) {
        ; Valid search
    } else {
        return { valid: false, errorMsg: "Invalid job/search input: only 5 digits (job) or letters, digits, dash, underscore, space (search) allowed.", normalizedJob: jobIn, normalizedFolder: folderIn }
    }
    if (!RegExMatch(folderIn, "^[A-Za-z0-9 _\-&.]+$")) {
        return { valid: false, errorMsg: "Invalid folder: Only letters, digits, dash, underscore, ampersand, dot, space allowed.", normalizedJob: jobIn, normalizedFolder: folderIn }
    }
    return { valid: true, errorMsg: "", normalizedJob: jobIn, normalizedFolder: folderIn }
}

; === Persistence Helpers ===
JSON_Load_From_File(path) {
    if (!FileExist(path))
        return Map()
    try {
        cmd := "python " . A_ScriptDir . "\ahk_json_bridge.py load `"" . path . "`""
        shell := ComObject("WScript.Shell")
        exec := shell.Exec(cmd)
        stdout := ""
        while !exec.StdOut.AtEndOfStream
            stdout .= exec.StdOut.ReadAll()
        stdout := Trim(stdout)
        if (stdout = "")
            return Map()
        try
            return JSON.Parse(stdout)
        catch
            return Map()
    } catch as e {
        MsgBox("Error parsing JSON: " . e.Message)
        return Map()
    }
}
JSON_Dump_To_File(obj, path) {
    try {
        jsonStr := ""
        try
            jsonStr := JSON.Stringify(obj)
        catch as e {
            MsgBox("Could not encode JSON: " . e.Message)
            return
        }
        safeJson := StrReplace(jsonStr, "`"", "\`"")
        cmd := "python " . A_ScriptDir . "\ahk_json_bridge.py dump `"" . path . "`" `"" . safeJson . "`""
        shell := ComObject("WScript.Shell")
        exec := shell.Exec(cmd)
        while !exec.StdOut.AtEndOfStream
            exec.StdOut.ReadAll()
    } catch as e {
        MsgBox("Error saving JSON: " . e.Message)
    }
}

LoadRecents() {
    global recentDataPath
    rec := JSON_Load_From_File(recentDataPath)
    if (!rec.Has("jobs"))
        rec["jobs"] := []
    if (!rec.Has("folders"))
        rec["folders"] := []
    if (!rec.Has("favorites"))
        rec["favorites"] := []
    return rec
}
SaveRecents(data) {
    global recentDataPath
    DirCreate(DirSplit(recentDataPath)[1])
    JSON_Dump_To_File(data, recentDataPath)
}
LoadSettings() {
    global settingsPath
    s := JSON_Load_From_File(settingsPath)
    return s
}
SaveSettings(settings) {
    global settingsPath
    DirCreate(DirSplit(settingsPath)[1])
    JSON_Dump_To_File(settings, settingsPath)
}
ResetApp() {
    global settingsPath, recentDataPath, logPath
    try FileDelete(settingsPath)
    try FileDelete(recentDataPath)
    try FileDelete(logPath)
    ShowNotification("App data reset. Restart recommended.", "success")
}
LogError(msg, context := "") {
    global logPath
    try {
        DirCreate(DirSplit(logPath)[1])
        FileAppend(FormatTime(A_Now, "yyyy-MM-dd HH:mm:ss") . " | " . (context ? context . " | " : "") . msg . "`n", logPath)
    }
    catch {
        ; Fail silently
    }
}
DirSplit(path) {
    local name, dir, ext, nameNoExt, drive
    SplitPath(path, &name, &dir, &ext, &nameNoExt, &drive)
    return [dir, name, ext, nameNoExt, drive]
}

; === Controller Functions ===

Controller_OpenProject(jobNumber, selectedFolder, DebugMode := false) {
    global folderNames, mainGui, btnOpen, btnCancel, jobEdit, recentsData, recentJobs, recentFolders, procPID
    btnOpen.Enabled := false
    btnCancel.Enabled := true
    mainGui["StatusText"].Value := "Processing..."
    SetProgress()
    jobEdit.Opt("BackgroundWhite")

    scriptPath := "find_project_path.py"
    if !FileExist(scriptPath)
        scriptPath := "test_find_project_path.py"

    v := ValidateAndNormalizeInputs(jobNumber, selectedFolder)
    if (!v.valid) {
        ShowInlineHint(v.errorMsg, "error")
        jobEdit.Opt("BackgroundF7C8C8")
        ResetGUI()
        return
    }
    jobEdit.Opt("BackgroundWhite")

    if !recentJobs.Has(jobNumber) {
        recentJobs.Push(jobNumber)
        if recentJobs.Length > 10
            recentJobs.RemoveAt(1)
    }
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

    comspec := A_ComSpec
    tempFile := A_Temp . "\project_quicknav_pyout.txt"

    safeJob := StrReplace(jobNumber, "`"", "")
    safeJob := Trim(safeJob, " `t`n`r")
    cmd := comspec . " /C python " . Chr(34) . scriptPath . Chr(34) . " " . Chr(34) . safeJob . Chr(34) . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"
    procPID := 0

    try {
        Run(cmd, , "Hide Pid", &procPID)
    } catch as e {
        mainGui["StatusText"].Value := "Error: backend launch failed"
        ShowInlineHint("Backend error: " . e.Message, "error")
        LogError("Failed to launch backend: " . e.Message, "Controller_OpenProject")
        ResetGUI()
        return
    }

    attempts := 0
    maxAttempts := 100
    SetTimer(() => Controller_WaitForBackend(tempFile, procPID, attempts, maxAttempts, DebugMode, selectedFolder), 100)
}
Controller_WaitForBackend(tempFile, pid, attempts, maxAttempts, DebugMode, selectedFolder) {
    global mainGui, btnOpen, btnCancel, procPID
    procPID := pid
    attempts++
    if FileExist(tempFile) {
        btnCancel.Enabled := false
        btnOpen.Enabled := true

        output := ""
        fileReadSuccess := false
        fileDeleteSuccess := false
        retryCount := 0
        maxRetries := 5

        while (!fileReadSuccess && retryCount < maxRetries) {
            try {
                output := FileRead(tempFile)
                fileReadSuccess := true
            } catch as e {
                retryCount++
                if (retryCount >= maxRetries) {
                    ShowInlineHint("Error reading output file: " . e.Message, "error")
                    LogError("File read error: " . e.Message, "Controller_WaitForBackend")
                    ResetGUI()
                    return
                }
                Sleep(100)
            }
        }
        retryCount := 0
        while (!fileDeleteSuccess && retryCount < maxRetries) {
            try {
                FileDelete(tempFile)
                fileDeleteSuccess := true
            } catch as e {
                retryCount++
                if (retryCount >= maxRetries) {
                    LogError("Could not delete temp file: " . e.Message, "Controller_WaitForBackend")
                    break
                }
                Sleep(100)
            }
        }
        output := Trim(output)
        if (DebugMode) {
            ShowInlineHint("Raw backend output: " . output, "info")
        }
        if (InStr(output, "ERROR:") == 1)
            return Controller_HandleBackendError(output, selectedFolder)
        else if (InStr(output, "SELECT:") == 1)
            return Controller_HandleBackendSelect(output, selectedFolder, DebugMode)
        else if (InStr(output, "SEARCH:") == 1)
            return Controller_HandleBackendSearch(output, selectedFolder, DebugMode)
        else if (InStr(output, "SUCCESS:") == 1)
            return Controller_HandleBackendSuccess(output, selectedFolder, DebugMode)
        else {
            mainGui["StatusText"].Value := "Unexpected response"
            MsgBox("Unexpected response from Python backend:`n" . output, "Error", 16)
            ResetGUI()
            return
        }
    } else if (attempts > maxAttempts) {
        ShowInlineHint("Backend timeout.", "error")
        LogError("Backend timeout waiting for Python response", "Controller_WaitForBackend")
        ResetGUI()
        return
    }
    SetTimer(() => Controller_WaitForBackend(tempFile, procPID, attempts, maxAttempts, DebugMode, selectedFolder), 100)
}
Controller_HandleBackendError(output, selectedFolder) {
    global mainGui
    msg := SubStr(output, 7)
    mainGui["StatusText"].Value := "Error: " . Trim(msg)
    SetProgress(0)
    MsgBox(Trim(msg), "Error", 16)
    ResetGUI()
}
Controller_HandleBackendSelect(output, selectedFolder, DebugMode) {
    global mainGui
    mainGui["StatusText"].Value := "Multiple paths found"
    SetProgress(100)
    strPaths := SubStr(output, 8)
    arrPaths := StrSplit(strPaths, "|")
    global selGui, selResult

    OkHandler(thisCtrl, *) {
        selResult := selGui.Submit()
        selGui.Destroy()
    }
    CloseHandler(thisGui, *) {
        selResult := {SelChoice: 0}
        selGui.Destroy()
    }
    selResult := {SelChoice: 0}
    selGui := Gui("", "Select Project Folder")
    selGui.Add("Text", "x10 y10", "Multiple project folders found:`nSelect the correct path:")
    choices := ""
    for i, path in arrPaths
        choices .= i ":" path "|"
    choices := SubStr(choices, 1, -1)
    selGui.Add("DropDownList", "x10 y40 w400 vSelChoice AltSubmit Choose1", choices)
    btnOK := selGui.Add("Button", "x170 y80 w80 Default", "OK")
    btnOK.OnEvent("Click", OkHandler)
    selGui.OnEvent("Close", CloseHandler)
    selGui.Show("Modal")
    if !selResult.SelChoice {
        mainGui["StatusText"].Value := "Selection cancelled"
        MsgBox("Selection cancelled.", "Cancelled", 48)
        ResetGUI()
        return
    }
    parts := StrSplit(selResult.SelChoice, ":")
    chosenIdx := parts[1]
    MainProjectPath := arrPaths[chosenIdx]
    Controller_OpenSelectedFolder(MainProjectPath, selectedFolder, DebugMode)
}
Controller_HandleBackendSearch(output, selectedFolder, DebugMode) {
    global mainGui
    mainGui["StatusText"].Value := "Search results found"
    SetProgress(100)
    strPaths := SubStr(output, 8)
    arrPaths := StrSplit(strPaths, "|")
    global searchGui, searchResult

    SearchGuiSize(thisGui, MinMax, Width, Height) {
        if (MinMax = -1)
            return
        thisGui["SearchList"].Move(,, Width - 20, Height - 80)
        thisGui["BtnOpen"].Move(Width - 170, Height - 40)
        thisGui["BtnCancel"].Move(Width - 80, Height - 40)
    }
    SearchOkHandler(thisCtrl, *) {
        global searchGui, searchResult
        LV := searchGui["SearchList"]
        selectedRow := LV.GetNext()
        if (selectedRow > 0)
            searchResult := LV.GetText(selectedRow, 3)
        searchGui.Destroy()
    }
    SearchCloseHandler(thisGui, *) {
        global searchResult, searchGui
        searchResult := ""
        searchGui.Destroy()
    }
    searchResult := ""
    searchGui := Gui("+Resize", "Search Results")
    searchGui.Add("Text", "x10 y10", "Found " . arrPaths.Length . " project folders matching your search:")
    LV := searchGui.Add("ListView", "x10 y30 w600 h300 vSearchList -Multi", ["Project Number", "Project Name", "Full Path"])
    LV.ModifyCol(1, 100)
    LV.ModifyCol(2, 250)
    LV.ModifyCol(3, 250)
    for i, path in arrPaths {
        SplitPath(path, &fileName, &dirPath)
        if (RegExMatch(fileName, "^(\d{5}) - (.+)$", &match)) {
            projNum := match[1]
            projName := match[2]
        } else {
            projNum := "N/A"
            projName := fileName
        }
        LV.Add("", projNum, projName, path)
    }
    btnOK := searchGui.Add("Button", "x450 y340 w80 Default vBtnOpen", "Open")
    btnCancel := searchGui.Add("Button", "x540 y340 w80 vBtnCancel", "Cancel")
    LV.OnEvent("DoubleClick", SearchOkHandler)
    btnOK.OnEvent("Click", SearchOkHandler)
    btnCancel.OnEvent("Click", SearchCloseHandler)
    searchGui.OnEvent("Close", SearchCloseHandler)
    searchGui.OnEvent("Size", SearchGuiSize)
    searchGui.Show("w620 h380")
    WinWaitClose("ahk_id " . searchGui.Hwnd)
    if (searchResult = "") {
        mainGui["StatusText"].Value := "Selection cancelled"
        MsgBox("No project selected.", "Cancelled", 48)
        ResetGUI()
        return
    }
    MainProjectPath := searchResult
    Controller_OpenSelectedFolder(MainProjectPath, selectedFolder, DebugMode)
}
Controller_HandleBackendSuccess(output, selectedFolder, DebugMode) {
    MainProjectPath := Trim(SubStr(output, 9))
    Controller_OpenSelectedFolder(MainProjectPath, selectedFolder, DebugMode)
}
Controller_OpenSelectedFolder(MainProjectPath, selectedFolder, DebugMode) {
    global mainGui
    v := ValidateAndNormalizeInputs("", selectedFolder)
    if (!v.valid) {
        ShowInlineHint(v.errorMsg, "error")
        ResetGUI()
        return
    }
    MainProjectPath := Trim(MainProjectPath, " `t`n`r`"")

    if (SubStr(MainProjectPath, 1, 1) == "\" || SubStr(MainProjectPath, 1, 1) == "/")
        MainProjectPath := SubStr(MainProjectPath, 2)

    if (selectedFolder = "Floor Plans" || selectedFolder = "Site Photos") {
        FullSubfolderPath := MainProjectPath . "\1. Sales Handover\" . selectedFolder
    } else {
        FullSubfolderPath := MainProjectPath . "\" . selectedFolder
    }
    FullSubfolderPath := Trim(FullSubfolderPath, " `t`n`r`"")

    if (DebugMode)
        MsgBox("Main Project Path: " . MainProjectPath . "`nSelected Folder: " . selectedFolder . "`nFull Path: " . FullSubfolderPath, "Path Debug", 64)

    mainGui["StatusText"].Value := "Checking folder: " . FullSubfolderPath

    if !FileExist(FullSubfolderPath) {
        mainGui["StatusText"].Value := "Subfolder not found"
        MsgBox("Subfolder '" . selectedFolder . "' not found under:`n" . MainProjectPath, "Subfolder Not Found", 16)
        ResetGUI()
        return
    }
    mainGui["StatusText"].Value := "Opening folder: " . FullSubfolderPath

    explorerPath := "explorer.exe " . Chr(34) . FullSubfolderPath . Chr(34)

    try {
        Run(explorerPath)
        ShowNotification("Folder opened successfully", "success")
        mainGui["StatusText"].Value := "Folder opened successfully"
        if (DebugMode)
            MsgBox("Command executed: " . explorerPath, "Debug Info", 64)
    } catch as e {
        try {
            Run("explorer.exe /select," . Chr(34) . FullSubfolderPath . Chr(34))
            mainGui["StatusText"].Value := "Folder opened successfully (alternative method)"
        } catch as e2 {
            mainGui["StatusText"].Value := "Error opening folder: " . e2.Message
            MsgBox("Failed to open folder: " . e2.Message . "`n`nPath: " . FullSubfolderPath, "Error", 16)
            ResetGUI()
            return
        }
    }
    SetTimer(() => ResetGUI(), -2000)
}
Controller_CancelProcessing() {
    global procPID
    if (procPID && ProcessExist(procPID)) {
        try {
            ProcessClose(procPID)
            ShowInlineHint("Process cancelled.", "info")
        } catch as e {
            ShowInlineHint("Error cancelling process: " . e.Message, "warning")
            LogError("Error cancelling process: " . e.Message, "Controller_CancelProcessing")
        }
    } else {
        ShowInlineHint("No active process to cancel.", "info")
    }
    tempFile := A_Temp . "\project_quicknav_pyout.txt"
    if (FileExist(tempFile)) {
        try {
            FileDelete(tempFile)
        } catch {
            ; ignore errors
        }
    }
    ResetGUI()
}
Controller_ToggleFavorite(label) {
    global recentsData
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
Controller_ReloadFolderRadios() {
    ReloadFolderRadios()
}