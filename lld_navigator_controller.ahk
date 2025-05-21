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

; === Global Variables ===
; Process and path variables
procPID := 0
recentDataPath := A_AppData . "\QuickNav\recent.json"
settingsPath := A_AppData . "\QuickNav\settings.json"
logPath := A_AppData . "\QuickNav\error.log"
processStartTime := 0  ; Initialize globally
processTimeoutMs := 30000  ; Initialize globally

; Initialize data structures
recentJobs := []
recentFolders := []
recentsData := Map()
recentsData.jobs := recentJobs
recentsData.folders := recentFolders
recentsData.favorites := []

; External variables declared in the main script
; These are referenced here but defined in lld_navigator.ahk
global mainGui, btnOpen, btnCancel, jobEdit, folderNames

; External functions - these are defined in lld_navigator.ahk
; We don't declare them as global since they're functions
; The following functions are referenced in this file but defined in lld_navigator.ahk:
; - ShowNotification(message, type := "info") - Displays notifications with styling
; - SetProgress(percent := 0) - Updates progress indicator in GUI
; - ShowInlineHint(message, type := "info") - Shows inline hints/messages in UI
; - ResetGUI() - Resets GUI controls to initial state
; - ReloadFolderRadios() - Updates folder radio buttons with favorites status

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
        ; Read the file content
        fileContent := ""
        try {
            file := FileOpen(path, "r")
            if (file) {
                fileContent := file.Read()
            } else {
                LogError("Could not open file for reading: " . path, "JSON_Load_From_File", "WARNING")
                return Map()
            }
        } finally {
            if (IsSet(file) && file) {
                file.Close()
                file := ""
            }
        }

        ; Parse JSON content
        if (fileContent = "")
            return Map()

        try {
            ; Use native JSON parsing in AHK v2
            parsed := Jxon_Load(fileContent)
            return parsed
        } catch as e {
            LogError("JSON parsing error: " . e.Message, "JSON_Load_From_File", "WARNING")
            return Map()
        }
    } catch as e {
        LogError("Error loading JSON: " . e.Message, "JSON_Load_From_File", "ERROR")
        return Map()
    }
}

JSON_Dump_To_File(obj, path) {
    try {
        ; Create directory if it doesn't exist
        dir := DirSplit(path)[1]
        if (dir && !DirExist(dir))
            DirCreate(dir)

        ; Convert object to JSON string
        try {
            jsonStr := Jxon_Dump(obj, 4)  ; 4 spaces indentation for readability
        } catch as e {
            LogError("JSON encoding error: " . e.Message, "JSON_Dump_To_File", "ERROR")
            return false
        }

        ; Write to file
        try {
            file := FileOpen(path, "w")
            if (file) {
                file.Write(jsonStr)
                return true
            } else {
                LogError("Could not open file for writing: " . path, "JSON_Dump_To_File", "WARNING")
                return false
            }
        } finally {
            if (IsSet(file) && file) {
                file.Close()
                file := ""
            }
        }
    } catch as e {
        LogError("Error saving JSON: " . e.Message, "JSON_Dump_To_File", "ERROR")
        return false
    }
}

; === JSON Helper Functions ===
; Jxon_Load() and Jxon_Dump() - Native JSON functions for AHK v2
; Based on the JXON library by coco
Jxon_Load(src, args*) {
    static q := Chr(34)

    key := "", is_key := false
    stack := [tree := []]
    is_arr := Map(tree, 1)
    next := q . "{[01234567890-tfn"
    pos := 0

    while ((ch := SubStr(src, ++pos, 1)) != "") {
        if InStr(" `t`n`r", ch)
            continue
        if !InStr(next, ch, true) {
            testArr := StrSplit(SubStr(src, 1, pos), "`n")

            ln := testArr.Length
            col := pos - InStr(src, "`n",, -(StrLen(src)-pos+1))

            msg := Format("{}: line {} col {} (char {})"
                , (next == "")      ? ["Extra data", ch := SubStr(src, pos)][1]
                : (next == q)       ? "Expecting quotation mark at beginning of string"
                : (next == q . "}") ? "Expecting object key or closing bracket"
                : (next == ":")     ? "Expecting colon"
                : (next == q . ",") ? "Expecting comma or closing bracket"
                : (next == ",}")    ? "Expecting comma or closing bracket"
                : (next == ",]")    ? "Expecting comma or closing square bracket"
                : (next == "")      ? "Expecting object key, value or closing bracket"
                : "Expecting JSON value(string, number, [true, false, null], object or array)"
                , ln, col, pos)

            throw Error(msg, -1, ch)
        }

        obj := stack[stack.Length]
        is_array := is_arr.Has(obj)

        if (ch == "{") {
            val := Map()
            is_arr[val] := 0
            obj.Push(val)
            stack.Push(val)
            next := q . "}"
        } else if (ch == "[") {
            val := []
            is_arr[val] := 1
            obj.Push(val)
            stack.Push(val)
            next := q . "{[0123456789-tfn]"
        } else if (ch == "}") {
            stack.Pop()
            next := is_arr.Has(stack[stack.Length]) ? ",]" : ",}"
        } else if (ch == "]") {
            stack.Pop()
            next := is_arr.Has(stack[stack.Length]) ? ",]" : ",}"
        } else if (ch == ",") {
            is_key := false
            next := is_arr.Has(stack[stack.Length]) ? q . "{[0123456789-tfn" : q
        } else if (ch == ":") {
            is_key := true
            next := q . "{[0123456789-tfn"
        } else if (ch == q) { ; Quotation mark
            i := pos
            while (i := InStr(src, q,, i+1)) {
                val := StrReplace(SubStr(src, pos+1, i-pos-1), "\\", "\u005C")
                if (SubStr(val, -1) != "\")
                    break
            }
            if !i
                throw Error("Missing closing quotation mark", -1, src)

            pos := i
            val := StrReplace(val, "\/", "/")
            val := StrReplace(val, "\" . q, q)
            val := StrReplace(val, "\b", "`b")
            val := StrReplace(val, "\f", "`f")
            val := StrReplace(val, "\n", "`n")
            val := StrReplace(val, "\r", "`r")
            val := StrReplace(val, "\t", "`t")

            i := 0
            while (i := InStr(val, "\u",, i+1)) {
                if ((esc := Abs("0x" . SubStr(val, i+2, 4))) < 0x100)
                    val := StrReplace(val, SubStr(val, i, 6), Chr(esc))
            }

            if is_key {
                key := val
                next := ":"
            } else {
                obj.Push(val)
                next := is_arr.Has(obj) ? ",]" : ",}"
            }
        } else if (ch == "t" && SubStr(src, pos, 4) == "true") {
            obj.Push(true)
            pos += 3
            next := is_arr.Has(obj) ? ",]" : ",}"
        } else if (ch == "f" && SubStr(src, pos, 5) == "false") {
            obj.Push(false)
            pos += 4
            next := is_arr.Has(obj) ? ",]" : ",}"
        } else if (ch == "n" && SubStr(src, pos, 4) == "null") {
            obj.Push(Map())
            pos += 3
            next := is_arr.Has(obj) ? ",]" : ",}"
        } else if InStr("0123456789-", ch) {
            i := pos
            while InStr("0123456789.eE+-", SubStr(src, ++i, 1))
                continue
            val := SubStr(src, pos, i-pos)
            if (val ~= "^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$") {
                obj.Push(val + 0)
                pos := i - 1
            } else {
                throw Error("Invalid number format", -1, val)
            }
            next := is_arr.Has(obj) ? ",]" : ",}"
        } else {
            throw Error("Unexpected character", -1, ch)
        }
    }

    return tree[1]
}

Jxon_Dump(obj, indent:="", lvl:=1) {
    static q := Chr(34)

    ; Helper functions
    _indent(str, level) {
        result := ""
        Loop level
            result .= str
        return result
    }

    _join(obj, delim) {
        result := ""
        for val in obj
            result .= (result ? delim : "") . val
        return result
    }

    _escape(str) {
        str := StrReplace(str, "\", "\\")
        str := StrReplace(str, "/", "\/")
        str := StrReplace(str, q, "\" . q)
        str := StrReplace(str, "`b", "\b")
        str := StrReplace(str, "`f", "\f")
        str := StrReplace(str, "`n", "\n")
        str := StrReplace(str, "`r", "\r")
        str := StrReplace(str, "`t", "\t")
        return str
    }

    if IsObject(obj) {
        if (obj.HasOwnProp("Length")) { ; Array
            for i, val in obj {
                if IsObject(val)
                    obj[i] := Jxon_Dump(val, indent, lvl+1)
            }
            return "[" . (indent ? "`n" . _indent(indent, lvl) : "")
                . (indent ? _join(obj, "," . "`n" . _indent(indent, lvl)) : _join(obj, ","))
                . (indent ? "`n" . _indent(indent, lvl-1) : "") . "]"
        } else { ; Map
            str := ""
            for key, val in obj {
                if (key == "")
                    continue
                if IsObject(val)
                    val := Jxon_Dump(val, indent, lvl+1)
                key := q . _escape(key) . q
                str .= (str ? "," : "") . (indent ? "`n" . _indent(indent, lvl) : "") . key . ":" . (indent ? " " : "") . val
            }
            return "{" . str . (indent ? "`n" . _indent(indent, lvl-1) : "") . "}"
        }
    } else if (obj == "")
        return q . q
    else if (obj * 0 == 0 && obj != "" && !InStr(obj, ".") && !InStr(obj, "e") && !InStr(obj, "E"))
        return obj
    else
        return q . _escape(obj) . q
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

; === UI Helper Functions ===
; These are safe wrappers for the UI functions defined in lld_navigator.ahk
; They ensure the controller can be used standalone for testing

; Function to safely call ShowNotification if it exists
SafeShowNotification(message, type := "info") {
    try {
        ; Check if the function exists by trying to call it
        IsFunc("ShowNotification")
        ; If we get here, the function exists, so call it
        ShowNotification(message, type)
    } catch {
        ; Fallback if ShowNotification is not defined
        ToolTip("Notification: " message "`nType: " type)
        SetTimer () => ToolTip(), -3000
    }
}

; Function to safely call SetProgress if it exists
SafeSetProgress(percent := 0) {
    static lastPercent := -1
    
    try {
        ; Check if the function exists by trying to call it
        IsFunc("SetProgress")
        ; If we get here, the function exists, so call it
        SetProgress(percent)
    } catch {
        ; Fallback if SetProgress is not defined (only log if percent changed)
        if (percent != lastPercent) {
            LogError("Progress: " percent "%", "SafeSetProgress", "DEBUG")
            lastPercent := percent
        }
    }
}

; Function to safely call ShowInlineHint if it exists
SafeShowInlineHint(message, type := "info") {
    try {
        ; Check if the function exists by trying to call it
        IsFunc("ShowInlineHint")
        ; If we get here, the function exists, so call it
        ShowInlineHint(message, type)
    } catch {
        ; Fallback if ShowInlineHint is not defined
        ToolTip("Hint: " message "`nType: " type)
        SetTimer () => ToolTip(), -3000
    }
}

; Function to safely call ResetGUI if it exists
SafeResetGUI() {
    try {
        ; Check if the function exists by trying to call it
        IsFunc("ResetGUI")
        ; If we get here, the function exists, so call it
        ResetGUI()
    } catch {
        ; Fallback if ResetGUI is not defined
        global btnOpen, btnCancel
        LogError("ResetGUI called (fallback implementation)", "SafeResetGUI", "DEBUG")
        
        ; Basic reset functionality
        if (IsSet(btnOpen) && btnOpen)
            btnOpen.Enabled := true
        if (IsSet(btnCancel) && btnCancel)  
            btnCancel.Enabled := false
    }
}

; Function to safely call ReloadFolderRadios if it exists
SafeReloadFolderRadios() {
    try {
        ; Check if the function exists by trying to call it
        IsFunc("ReloadFolderRadios")
        ; If we get here, the function exists, so call it
        ReloadFolderRadios()
    } catch {
        ; Fallback if ReloadFolderRadios is not defined
        LogError("ReloadFolderRadios called (fallback - no implementation)", "SafeReloadFolderRadios", "DEBUG")
    }
}

; For backward compatibility, set up globals for these functions
ResetApp() {
    global settingsPath, recentDataPath, logPath
    try FileDelete(settingsPath)
    try FileDelete(recentDataPath)
    try FileDelete(logPath)
    SafeShowNotification("App data reset. Restart recommended.", "success")
}

LogError(msg, context := "", severity := "ERROR") {
    global logPath

    ; Create a more detailed log entry
    try {
        ; Ensure log directory exists
        DirCreate(DirSplit(logPath)[1])

        ; Get system information for the log
        sysInfo := GetSystemInfo()

        ; Format the log entry with timestamp, severity, context, and message
        logEntry := FormatTime(A_Now, "yyyy-MM-dd HH:mm:ss")
        logEntry .= " | " . severity
        logEntry .= " | " . (context ? context : "UNKNOWN")
        logEntry .= " | " . msg

        ; Add system info for ERROR severity
        if (severity = "ERROR") {
            logEntry .= "`n    System: " . sysInfo.os . ", AHK: " . A_AhkVersion
            logEntry .= "`n    Memory: " . sysInfo.memory . ", CPU: " . sysInfo.cpu . "%"

            ; Add stack trace if available
            try {
                ; Force an error to get stack trace
                local e := Error("Dummy error for stack trace")
                if (e.Stack) {
                    stackLines := StrSplit(e.Stack, "`n")
                    ; Skip the first line which is this function
                    if (stackLines.Length > 1) {
                        logEntry .= "`n    Stack:"
                        Loop Min(5, stackLines.Length - 1) {  ; Show up to 5 stack frames
                            logEntry .= "`n      " . stackLines[A_Index + 1]
                        }
                    }
                }
            } catch {
                ; Ignore stack trace errors
            }
        }

        ; Write to log file with proper handle management
        try {
            file := FileOpen(logPath, "a")
            if (file) {
                file.WriteLine(logEntry)
                file.WriteLine("---")  ; Separator between log entries
            }
        } finally {
            if IsSet(file) && file {
                file.Close()
                file := ""
            }
        }

        ; For critical errors, also create a separate crash log
        if (severity = "CRITICAL") {
            crashLogPath := A_AppData . "\QuickNav\crash_" . FormatTime(A_Now, "yyyyMMdd_HHmmss") . ".log"
            try {
                FileAppend(logEntry, crashLogPath)
            } catch {
                ; Fail silently for crash log
            }
        }

        return true
    } catch as e {
        ; If we can't log to file, try to show a message box for critical errors
        if (severity = "CRITICAL") {
            try {
                MsgBox("Critical error occurred but could not be logged: " . msg, "Logging Error", 16)
            } catch {
                ; Last resort - fail silently
            }
        }
        return false
    }
}

; Helper function to get system information for logs
GetSystemInfo() {
    info := Map()

    ; Get OS information
    info.os := A_OSVersion

    ; Get memory usage
    try {
        memoryStatus := Buffer(64, 0)
        NumPut("UInt", 64, memoryStatus, 0)
        DllCall("kernel32\GlobalMemoryStatusEx", "Ptr", memoryStatus)
        totalPhysMB := NumGet(memoryStatus, 8, "UInt64") / 1048576
        availPhysMB := NumGet(memoryStatus, 16, "UInt64") / 1048576
        info.memory := Round(availPhysMB) . "MB free of " . Round(totalPhysMB) . "MB"
    } catch {
        info.memory := "Unknown"
    }

    ; Get CPU usage
    try {
        ; This is a simple approximation of CPU usage
        startTime := A_TickCount
        startCPU := DllCall("GetTickCount64")
        Sleep(100)
        endTime := A_TickCount
        endCPU := DllCall("GetTickCount64")
        cpuTime := endCPU - startCPU
        totalTime := endTime - startTime
        info.cpu := Round((cpuTime / totalTime) * 100)
    } catch {
        info.cpu := "Unknown"
    }

    return info
}
DirSplit(path) {
    local name := "", dir := "", ext := "", nameNoExt := "", drive := ""
    SplitPath(path, &name, &dir, &ext, &nameNoExt, &drive)
    return [dir, name, ext, nameNoExt, drive]
}

; Function to safely clean up temporary files with retries
CleanupTempFile(filePath) {
    if (!FileExist(filePath))
        return

    ; Try immediate deletion
    try {
        FileDelete(filePath)
        return
    } catch {
        ; If immediate deletion fails, schedule a delayed cleanup
        SetTimer(() => DelayedFileCleanup(filePath), -1000)
    }
}

; Helper function for delayed file cleanup with retries
DelayedFileCleanup(filePath, attempt := 1, maxAttempts := 5) {
    if (!FileExist(filePath))
        return

    try {
        FileDelete(filePath)
        LogError("Temporary file deleted on attempt " . attempt, "DelayedFileCleanup")
    } catch as e {
        if (attempt < maxAttempts) {
            ; Exponential backoff for retries (1s, 2s, 4s, 8s)
            delay := 1000 * (2 ** (attempt - 1))
            SetTimer(() => DelayedFileCleanup(filePath, attempt + 1, maxAttempts), -delay)
        } else {
            LogError("Failed to delete temporary file after " . maxAttempts . " attempts: " . filePath, "DelayedFileCleanup")
        }
    }
}

; === Controller Functions ===

Controller_OpenProject(jobNumber, selectedFolder, DebugMode := false) {
    global folderNames, mainGui, btnOpen, btnCancel, jobEdit, recentsData, recentJobs, recentFolders, procPID
    global processStartTime, processTimeoutMs  ; Access global variables instead of redefining

    ; Store start time for timeout tracking
    processStartTime := A_TickCount
    processTimeoutMs := 30000  ; 30 seconds timeout

    btnOpen.Enabled := false
    btnCancel.Enabled := true
    mainGui["StatusText"].Value := "Processing..."
    SafeSetProgress()
    jobEdit.Opt("BackgroundWhite")

    scriptPath := "find_project_path.py"
    if !FileExist(scriptPath)
        scriptPath := "test_find_project_path.py"

    v := ValidateAndNormalizeInputs(jobNumber, selectedFolder)
    if (!IsObject(v) || !v.Has("valid")) {
        SafeShowInlineHint("Input validation process failed unexpectedly.", "error")
        if (IsObject(jobEdit))
            jobEdit.Opt("BackgroundF7C8C8")
        LogError("ValidateAndNormalizeInputs did not return a valid object. jobNumber: '" . (IsSet(jobNumber) ? jobNumber : "N/A") . "', selectedFolder: '" . (IsSet(selectedFolder) ? selectedFolder : "N/A") . "'", "Controller_OpenProjectFromInput", "CRITICAL")
        SafeResetGUI()
        Return
    }

    if (!v.valid) {
        local errorToShow := v.Has("errorMsg") && v.errorMsg != "" ? v.errorMsg : "Invalid input provided."
        SafeShowInlineHint(errorToShow, "error")
        if (IsObject(jobEdit))
            jobEdit.Opt("BackgroundF7C8C8")
        SafeResetGUI()
        Return
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
    SafeSetProgress(30)

    comspec := A_ComSpec
    tempFile := A_Temp . "\project_quicknav_pyout.txt"

    ; Clean up any existing temp file before starting
    if (FileExist(tempFile))
        CleanupTempFile(tempFile)

    safeJob := StrReplace(jobNumber, "`"", "")
    safeJob := Trim(safeJob, " `t`n`r")

    ; Add timeout parameter to Python script
    timeoutSec := processTimeoutMs / 1000
    cmd := comspec . " /C python " . Chr(34) . scriptPath . Chr(34) . " " . Chr(34) . safeJob . Chr(34) . " --timeout " . timeoutSec . " > " . Chr(34) . tempFile . Chr(34) . " 2>&1"

    procPID := 0

    try {
        Run(cmd, , "Hide Pid", &procPID)
        LogError("Started process with PID: " . procPID . ", command: " . cmd, "Controller_OpenProject")
    } catch as e {
        mainGui["StatusText"].Value := "Error: backend launch failed"
        SafeShowInlineHint("Backend error: " . e.Message, "error")
        LogError("Failed to launch backend: " . e.Message, "Controller_OpenProject")
        SafeResetGUI()
        return
    }

    ; Create a callback object for asynchronous processing
    callback := AsyncProcessCallback(tempFile, procPID, DebugMode, selectedFolder)

    ; Start the asynchronous process monitoring
    callback.Start()
}
Controller_WaitForBackend(tempFile, pid, attempts, maxAttempts, DebugMode, selectedFolder, checkInterval := 100) {
    global mainGui, btnOpen, btnCancel, procPID, processStartTime, processTimeoutMs

    ; Update process ID
    procPID := pid
    attempts++

    ; Calculate elapsed time
    elapsedTime := A_TickCount - processStartTime

    ; Update progress bar to show time remaining
    progressPercent := Min(90, (elapsedTime / processTimeoutMs) * 100)
    SafeSetProgress(progressPercent)

    ; Check if output file exists (process completed)
    if FileExist(tempFile) {
        btnCancel.Enabled := false
        btnOpen.Enabled := true
        SafeSetProgress(100)  ; Show 100% completion

        output := ""
        fileReadSuccess := false
        retryCount := 0
        maxRetries := 5

        ; Try to read the output file with retries
        while (!fileReadSuccess && retryCount < maxRetries) {
            try {
                output := FileRead(tempFile)
                fileReadSuccess := true
            } catch as e {
                retryCount++
                if (retryCount >= maxRetries) {
                    SafeShowInlineHint("Error reading output file: " . e.Message, "error")
                    LogError("File read error: " . e.Message, "Controller_WaitForBackend")
                    SafeResetGUI()
                    return
                }
                Sleep(100)
            }
        }

        ; Schedule cleanup of temporary file
        CleanupTempFile(tempFile)

        ; Process the output
        output := Trim(output)
        if (DebugMode) {
            SafeShowInlineHint("Raw backend output: " . output, "info")
        }

        ; Handle different response types
        if (InStr(output, "ERROR:") == 1)
            return Controller_HandleBackendError(output, selectedFolder)
        else if (InStr(output, "SELECT:") == 1)
            return Controller_HandleBackendSelect(output, selectedFolder, DebugMode)
        else if (InStr(output, "SEARCH:") == 1)
            return Controller_HandleBackendSearch(output, selectedFolder, DebugMode)
        else if (InStr(output, "SUCCESS:") == 1)
            return Controller_HandleBackendSuccess(output, selectedFolder, DebugMode)
        else if (InStr(output, "TIMEOUT:") == 1) {
            mainGui["StatusText"].Value := "Backend operation timed out"
            SafeShowInlineHint("The operation took too long to complete.", "warning")
            LogError("Backend reported timeout: " . output, "Controller_WaitForBackend")
            SafeResetGUI()
            return
        } else {
            mainGui["StatusText"].Value := "Unexpected response"
            MsgBox("Unexpected response from Python backend:`n" . output, "Error", 16)
            LogError("Unexpected backend response: " . output, "Controller_WaitForBackend")
            SafeResetGUI()
            return
        }
    }
    ; Check for timeout conditions
    else if (elapsedTime >= processTimeoutMs) {
        ; Hard timeout - kill the process
        if (procPID && ProcessExist(procPID)) {
            try {
                ProcessClose(procPID)
                LogError("Killed process " . procPID . " due to timeout", "Controller_WaitForBackend")
            } catch as e {
                LogError("Failed to kill process " . procPID . ": " . e.Message, "Controller_WaitForBackend")
            }
        }

        mainGui["StatusText"].Value := "Operation timed out"
        SafeShowInlineHint("The operation timed out after " . (processTimeoutMs / 1000) . " seconds.", "error")
        LogError("Hard timeout after " . elapsedTime . "ms", "Controller_WaitForBackend")
        SafeResetGUI()
        return
    }
    ; Check for max attempts (soft timeout)
    else if (attempts > maxAttempts) {
        ; Soft timeout - kill the process
        if (procPID && ProcessExist(procPID)) {
            try {
                ProcessClose(procPID)
                LogError("Killed process " . procPID . " due to max attempts reached", "Controller_WaitForBackend")
            } catch as e {
                LogError("Failed to kill process " . procPID . ": " . e.Message, "Controller_WaitForBackend")
            }
        }

        mainGui["StatusText"].Value := "Operation timed out"
        SafeShowInlineHint("The operation timed out after " . attempts . " attempts.", "error")
        LogError("Soft timeout after " . attempts . " attempts", "Controller_WaitForBackend")
        SafeResetGUI()
        return
    }

    ; Update status message periodically
    if (Mod(attempts, 10) == 0) {
        timeRemaining := (processTimeoutMs - elapsedTime) / 1000
        mainGui["StatusText"].Value := "Searching... (" . Round(timeRemaining) . "s remaining)"
    }

    ; Continue waiting
    SetTimer(() => Controller_WaitForBackend(tempFile, procPID, attempts, maxAttempts, DebugMode, selectedFolder, checkInterval), checkInterval)
}
Controller_HandleBackendError(output, selectedFolder) {
    global mainGui
    msg := SubStr(output, 7)
    mainGui["StatusText"].Value := "Error: " . Trim(msg)
    SafeSetProgress(0)
    MsgBox(Trim(msg), "Error", 16)
    SafeResetGUI()
}
Controller_HandleBackendSelect(output, selectedFolder, DebugMode) {
    global mainGui
    global selGui := "", selResult := {SelChoice: 0}  ; Initialize these globals
    local MainProjectPath := ""  ; Initialize as local 
    
    mainGui["StatusText"].Value := "Multiple paths found"
    SafeSetProgress(100)
    strPaths := SubStr(output, 8)
    arrPaths := StrSplit(strPaths, "|")

    OkHandler(thisCtrl, *) {
        global selGui, selResult
        selResult := selGui.Submit()
        selGui.Destroy()
    }
    CloseHandler(thisGui, *) {
        global selGui, selResult
        selResult := {SelChoice: 0}
        selGui.Destroy()
    }
    
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
        SafeResetGUI()
        return
    }
    parts := StrSplit(selResult.SelChoice, ":")
    chosenIdx := parts[1]
    MainProjectPath := arrPaths[chosenIdx]
    Controller_OpenSelectedFolder(MainProjectPath, selectedFolder, DebugMode)
}
Controller_HandleBackendSearch(output, selectedFolder, DebugMode) {
    global mainGui
    global searchGui := "", searchResult := ""  ; Initialize these globals
    local MainProjectPath := ""  ; Initialize as local variable
    
    mainGui["StatusText"].Value := "Search results found"
    SafeSetProgress(100)
    strPaths := SubStr(output, 8)
    arrPaths := StrSplit(strPaths, "|")

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
    
    searchGui := Gui("+Resize", "Search Results")
    searchGui.Add("Text", "x10 y10", "Found " . arrPaths.Length . " project folders matching your search:")
    LV := searchGui.Add("ListView", "x10 y30 w600 h300 vSearchList -Multi", ["Project Number", "Project Name", "Full Path"])
    LV.ModifyCol(1, 100)
    LV.ModifyCol(2, 250)
    LV.ModifyCol(3, 250)
    for i, path in arrPaths {
        SplitPath(path, &fileName, &dirPath)
        projNum := ""  ; Initialize to empty string
        projName := ""  ; Initialize to empty string
        
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
        SafeResetGUI()
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
    if (!IsObject(v) || !v.Has("valid")) {
        SafeShowInlineHint("Input validation process failed unexpectedly.", "error")
        LogError("ValidateAndNormalizeInputs did not return a valid object. selectedFolder: '" . (IsSet(selectedFolder) ? selectedFolder : "N/A") . "'", "Controller_OpenProjectFromFolderSelection", "CRITICAL")
        SafeResetGUI()
        Return
    }

    if (!v.valid) {
        local errorToShow := v.Has("errorMsg") && v.errorMsg != "" ? v.errorMsg : "Invalid folder selection."
        SafeShowInlineHint(errorToShow, "error")
        SafeResetGUI()
        Return
    }
    MainProjectPath := Trim(MainProjectPath, " `t`n`r`"")

    if (SubStr(MainProjectPath, 1, 1) == "\" || SubStr(MainProjectPath, 1, 1) == "/")
        MainProjectPath := SubStr(MainProjectPath, 2)

    FullSubfolderPath := ""  ; Initialize this variable
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
        SafeResetGUI()
        return
    }
    mainGui["StatusText"].Value := "Opening folder: " . FullSubfolderPath

    explorerPath := "explorer.exe " . Chr(34) . FullSubfolderPath . Chr(34)

    try {
        Run(explorerPath)
        SafeShowNotification("Folder opened successfully", "success")
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
            SafeResetGUI()
            return
        }
    }
    SetTimer(() => SafeResetGUI(), -2000)
}
Controller_CancelProcessing() {
    global procPID, mainGui, btnOpen, btnCancel
    local tempFile := ""  ; Initialize this local variable

    ; Update UI to show cancellation in progress
    mainGui["StatusText"].Value := "Cancelling..."
    SafeSetProgress(0)

    ; Cancel any active timers that might be running
    SetTimer(Controller_WaitForBackend, 0)

    ; Kill the process if it exists
    if (procPID && ProcessExist(procPID)) {
        try {
            ; Try graceful termination first
            ProcessClose(procPID)
            LogError("Process " . procPID . " terminated by user", "Controller_CancelProcessing")
            SafeShowInlineHint("Process cancelled.", "info")
        } catch as e {
            ; If graceful termination fails, try forceful termination
            try {
                Run("taskkill /F /PID " . procPID, , "Hide")
                LogError("Process " . procPID . " forcefully terminated by user", "Controller_CancelProcessing")
                SafeShowInlineHint("Process forcefully terminated.", "warning")
            } catch as e2 {
                SafeShowInlineHint("Error cancelling process: " . e2.Message, "error")
                LogError("Error cancelling process: " . e2.Message, "Controller_CancelProcessing")
            }
        } finally {
            ; Reset process ID regardless of termination success
            procPID := 0
        }
    } else {
        SafeShowInlineHint("No active process to cancel.", "info")
    }

    ; Clean up temporary files
    tempFile := A_Temp . "\project_quicknav_pyout.txt"
    if (FileExist(tempFile)) {
        CleanupTempFile(tempFile)
    }

    ; Check for any Python processes that might be related and kill them
    try {
        Run("taskkill /F /IM python.exe /FI `"WINDOWTITLE eq find_project_path*`"", , "Hide")
    } catch {
        ; Ignore errors - this is just a safety measure
    }

    ; Reset the UI
    SafeResetGUI()

    ; Ensure buttons are in the correct state
    btnOpen.Enabled := true
    btnCancel.Enabled := false
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
    SafeReloadFolderRadios()
}
Controller_ReloadFolderRadios() {
    SafeReloadFolderRadios()
}

; === Asynchronous Process Handling ===
class AsyncProcessCallback {
    tempFile := ""
    pid := 0
    debugMode := false
    selectedFolder := ""
    attempts := 0
    maxAttempts := 300
    checkInterval := 100
    startTime := 0
    timeoutMs := 30000

    __New(tempFile, pid, debugMode, selectedFolder) {
        this.tempFile := tempFile
        this.pid := pid
        this.debugMode := debugMode
        this.selectedFolder := selectedFolder
        this.startTime := A_TickCount

        ; Log creation of callback
        LogError("Created async callback for PID " . pid, "AsyncProcessCallback", "INFO")
    }

    Start() {
        ; Start the timer to check for process completion
        SetTimer(ObjBindMethod(this, "CheckProcess"), this.checkInterval)
    }

    CheckProcess() {
        global mainGui, btnOpen, btnCancel, procPID

        ; Update process ID reference
        procPID := this.pid
        this.attempts++

        ; Calculate elapsed time
        elapsedTime := A_TickCount - this.startTime

        ; Update progress bar to show time remaining
        progressPercent := Min(90, (elapsedTime / this.timeoutMs) * 100)
        SafeSetProgress(progressPercent)

        ; Check if output file exists (process completed)
        if FileExist(this.tempFile) {
            ; Stop the timer
            SetTimer(ObjBindMethod(this, "CheckProcess"), 0)

            btnCancel.Enabled := false
            btnOpen.Enabled := true
            SafeSetProgress(100)  ; Show 100% completion

            this.ProcessOutput()
        }
        ; Check for timeout conditions
        else if (elapsedTime >= this.timeoutMs) {
            ; Stop the timer
            SetTimer(ObjBindMethod(this, "CheckProcess"), 0)

            ; Hard timeout - kill the process
            if (this.pid && ProcessExist(this.pid)) {
                try {
                    ProcessClose(this.pid)
                    LogError("Killed process " . this.pid . " due to timeout", "AsyncProcessCallback", "WARNING")
                } catch as e {
                    LogError("Failed to kill process " . this.pid . ": " . e.Message, "AsyncProcessCallback", "ERROR")
                }
            }

            mainGui["StatusText"].Value := "Operation timed out"
            SafeShowInlineHint("The operation timed out after " . (this.timeoutMs / 1000) . " seconds.", "error")
            LogError("Hard timeout after " . elapsedTime . "ms", "AsyncProcessCallback", "ERROR")
            SafeResetGUI()
        }
        ; Check for max attempts (soft timeout)
        else if (this.attempts > this.maxAttempts) {
            ; Stop the timer
            SetTimer(ObjBindMethod(this, "CheckProcess"), 0)

            ; Soft timeout - kill the process
            if (this.pid && ProcessExist(this.pid)) {
                try {
                    ProcessClose(this.pid)
                    LogError("Killed process " . this.pid . " due to max attempts reached", "AsyncProcessCallback", "WARNING")
                } catch as e {
                    LogError("Failed to kill process " . this.pid . ": " . e.Message, "AsyncProcessCallback", "ERROR")
                }
            }

            mainGui["StatusText"].Value := "Operation timed out"
            SafeShowInlineHint("The operation timed out after " . this.attempts . " attempts.", "error")
            LogError("Soft timeout after " . this.attempts . " attempts", "AsyncProcessCallback", "ERROR")
            SafeResetGUI()
        }

        ; Update status message periodically
        if (Mod(this.attempts, 10) == 0) {
            timeRemaining := (this.timeoutMs - elapsedTime) / 1000
            mainGui["StatusText"].Value := "Searching... (" . Round(timeRemaining) . "s remaining)"
        }
    }

    ProcessOutput() {
        global mainGui

        output := ""
        fileReadSuccess := false
        retryCount := 0
        maxRetries := 5

        ; Try to read the output file with retries
        while (!fileReadSuccess && retryCount < maxRetries) {
            try {
                output := FileRead(this.tempFile)
                fileReadSuccess := true
            } catch as e {
                retryCount++
                if (retryCount >= maxRetries) {
                    SafeShowInlineHint("Error reading output file: " . e.Message, "error")
                    LogError("File read error: " . e.Message, "AsyncProcessCallback", "ERROR")
                    SafeResetGUI()
                    return
                }
                Sleep(100)
            }
        }

        ; Schedule cleanup of temporary file
        CleanupTempFile(this.tempFile)

        ; Process the output
        output := Trim(output)
        if (this.debugMode) {
            SafeShowInlineHint("Raw backend output: " . output, "info")
        }

        ; Handle different response types
        if (InStr(output, "ERROR:") == 1)
            Controller_HandleBackendError(output, this.selectedFolder)
        else if (InStr(output, "SELECT:") == 1)
            Controller_HandleBackendSelect(output, this.selectedFolder, this.debugMode)
        else if (InStr(output, "SEARCH:") == 1)
            Controller_HandleBackendSearch(output, this.selectedFolder, this.debugMode)
        else if (InStr(output, "SUCCESS:") == 1)
            Controller_HandleBackendSuccess(output, this.selectedFolder, this.debugMode)
        else if (InStr(output, "TIMEOUT:") == 1) {
            mainGui["StatusText"].Value := "Backend operation timed out"
            SafeShowInlineHint("The operation took too long to complete.", "warning")
            LogError("Backend reported timeout: " . output, "AsyncProcessCallback", "WARNING")
            SafeResetGUI()
        } else {
            mainGui["StatusText"].Value := "Unexpected response"
            MsgBox("Unexpected response from Python backend:`n" . output, "Error", 16)
            LogError("Unexpected backend response: " . output, "AsyncProcessCallback", "ERROR")
            SafeResetGUI()
        }
    }
}

; === Settings Management ===
Controller_LoadSettings() {
    global settingsPath
    
    ; Create a default settings map
    defaults := Map(
        "theme", "Light",
        "defaultFolder", "",
        "jobInputMode", "Prompt",
        "maxRecents", 10,
        "notifDuration", 3000
    )
    
    ; Load existing settings
    settings := JSON_Load_From_File(settingsPath)
    
    ; Apply defaults for any missing keys
    for key, value in defaults {
        if (!settings.Has(key)) {
            settings[key] := value
        }
    }
    
    return settings
}

SaveAppSettings(settings) {
    global settingsPath
    return JSON_Dump_To_File(settings, settingsPath)
}

ResetAppWithConfirmation() {
    global recentDataPath, settingsPath
    
    answer := MsgBox("This will reset all settings and favorites. Continue?", "Reset Application", 4 + 48 + 256) ; Yes/No + Warning + Always On Top
    if (answer = "Yes") {
        if (FileExist(recentDataPath))
            FileDelete(recentDataPath)
        if (FileExist(settingsPath))
            FileDelete(settingsPath)
        MsgBox("Application reset complete. The application will now restart.", "Reset Successful", 64 + 256)
        ReloadApp()
    }
}

ReloadApp() {
    Reload
}