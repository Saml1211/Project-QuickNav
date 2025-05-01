#Requires AutoHotkey v2.0
#SingleInstance Force

; Debug launcher for Project QuickNav
; This script helps with debugging by:
; 1. Setting up proper error handling
; 2. Logging all errors to a debug file
; 3. Providing a debug console

; Create debug log file
debugLogPath := A_ScriptDir . "\debug.log"
FileDelete(debugLogPath)
FileAppend("Debug session started at " . FormatTime(A_Now, "yyyy-MM-dd HH:mm:ss") . "`n", debugLogPath)

; Log system info
FileAppend("System Info:`n", debugLogPath)
FileAppend("  AHK Version: " . A_AhkVersion . "`n", debugLogPath)
FileAppend("  OS Version: " . A_OSVersion . "`n", debugLogPath)
FileAppend("  Is Admin: " . A_IsAdmin . "`n", debugLogPath)
FileAppend("  Screen DPI: " . A_ScreenDPI . "`n", debugLogPath)
FileAppend("  Working Dir: " . A_WorkingDir . "`n`n", debugLogPath)

; Create debug console
debugGui := Gui("+AlwaysOnTop", "QuickNav Debug Console")
debugGui.Add("Text", "x10 y10 w480", "Debug Console - Logs will appear here")
debugLog := debugGui.Add("Edit", "x10 y30 w480 h360 ReadOnly vDebugLog")
clearBtn := debugGui.Add("Button", "x10 y400 w100", "Clear Log")
clearBtn.OnEvent("Click", (*) => debugLog.Value := "")
saveBtn := debugGui.Add("Button", "x120 y400 w100", "Save Log")
saveBtn.OnEvent("Click", (*) => FileAppend(debugLog.Value, A_Desktop . "\quicknav_debug_" . FormatTime(A_Now, "yyyyMMdd_HHmmss") . ".log"))
runBtn := debugGui.Add("Button", "x230 y400 w100", "Run Script")
runBtn.OnEvent("Click", RunMainScript)
debugGui.Show("w500 h440")

; Log a message to both the debug console and log file
LogDebug(msg) {
    global debugGui, debugLogPath
    timestamp := FormatTime(A_Now, "HH:mm:ss")
    logMsg := timestamp . ": " . msg . "`n"
    
    ; Append to debug log file
    FileAppend(logMsg, debugLogPath)
    
    ; Update debug console
    if (debugGui) {
        try {
            debugGui["DebugLog"].Value .= logMsg
            ; Auto-scroll to bottom
            SendMessage(0x115, 7, 0, debugGui["DebugLog"].Hwnd)
        }
    }
}

; Function to run the main script
RunMainScript(*) {
    global
    LogDebug("Starting main script...")
    
    ; Kill any existing instances
    try {
        DetectHiddenWindows(true)
        if WinExist("ahk_class AutoHotkey ahk_pid " . scriptPID)
            ProcessClose(scriptPID)
    } catch {}
    
    ; Run the script with output redirection
    tempFile := A_Temp . "\quicknav_output.txt"
    if FileExist(tempFile)
        FileDelete(tempFile)
    
    try {
        LogDebug("Launching script...")
        Run(A_ComSpec . " /c `"" . A_AhkPath . "`" `"" . A_ScriptDir . "\lld_navigator.ahk`" > `"" . tempFile . "`" 2>&1", , "Hide", &scriptPID)
        LogDebug("Script launched with PID: " . scriptPID)
        
        ; Set up a timer to check for output
        SetTimer(CheckScriptOutput, 500)
    } catch as e {
        LogDebug("Error launching script: " . e.Message)
    }
}

; Function to check for script output
CheckScriptOutput() {
    global tempFile
    
    if FileExist(tempFile) {
        try {
            content := FileRead(tempFile)
            if (content && content != "") {
                LogDebug("Script output: " . content)
                FileDelete(tempFile)
            }
        } catch {}
    }
}

; Initialize
scriptPID := 0
tempFile := A_Temp . "\quicknav_output.txt"
LogDebug("Debug launcher ready. Click 'Run Script' to start QuickNav.")
