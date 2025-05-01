#Requires AutoHotkey v2.0
#SingleInstance Force

; Debug helper script for Project QuickNav
; This script helps with debugging by:
; 1. Setting up proper error handling
; 2. Logging all errors to a debug file
; 3. Providing a debug console

; Set up error handling
OnError ErrorHandler

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
            SendMessage(0xB1, debugGui["DebugLog"].Value.Length, 0, debugGui["DebugLog"].Hwnd)
        }
    }
}

; Error handler function
ErrorHandler(err) {
    LogDebug("ERROR: " . err.Message)
    LogDebug("  File: " . err.File)
    LogDebug("  Line: " . err.Line)
    LogDebug("  What: " . err.What)
    LogDebug("  Stack: " . err.Stack)
    
    MsgBox("An error occurred. See debug console for details.`n`nError: " . err.Message, "QuickNav Error", 16)
    return true  ; Continue running the script
}

; Start the main script with error trapping
LogDebug("Starting main script...")
try {
    #Include lld_navigator.ahk
    LogDebug("Main script loaded successfully")
} catch as err {
    LogDebug("Failed to load main script: " . err.Message)
    MsgBox("Failed to load main script: " . err.Message, "QuickNav Error", 16)
}
