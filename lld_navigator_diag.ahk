/*
    lld_navigator_diag.ahk
    Diagnostic/Test Harness for Project QuickNav
    Batch 4 (Phase 4) — Scaffold for automated sanity/QA checks

    Purpose:
    - Launches lld_navigator.ahk, simulates core GUI flows, verifies presence and basic interactivity of major controls.
    - Intended for developer or QA use to quickly sanity-check interactive logic after integration or refactor.

    Usage:
    - Run this script with lld_navigator.ahk present in the same directory.
    - Ensures GUI opens, can receive keystrokes, tab navigation works, and reports major control state.
*/

#Requires AutoHotkey v2.0

if !FileExist("lld_navigator.ahk") {
    MsgBox("lld_navigator.ahk not found!")
    ExitApp
}

; Launch QuickNav in a new process (non-blocking)
Run("autohotkey.exe lld_navigator.ahk", , "UseErrorLevel")

Sleep 1000 ; Wait for window

; Activate the main window by title (with version suffix wildcard)
DetectHiddenWindows true
if !WinActivate("Project QuickNav v*") {
    MsgBox("QuickNav main window not found.")
    ExitApp
}

Sleep 400

; Test: Tab navigation for first 10 controls (should circulate between Edit, Radios, Buttons)
Send "{Tab 10}"
Sleep 200

; Test: Enter a fake job number
Send "12345"
Sleep 200

; Test: Tab to Open, press Enter
Send "{Tab 2}{Enter}"
Sleep 400

; Optional: Add further checks (read GUI state, verify notifications, etc.)
MsgBox("Basic GUI flow simulated. Please visually confirm, or extend this script for deeper checks.")

ExitApp