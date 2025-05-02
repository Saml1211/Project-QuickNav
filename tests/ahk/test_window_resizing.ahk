/*
Automated Integration Test: Window Resizing and Cross-Resolution Layout
Covers:
- Resizing the main window to various resolutions
- Asserting that key controls remain visible and correctly positioned
- Verifying minimum size enforcement
*/

#Requires AutoHotkey v2.0
#Include %A_ScriptDir%\..\..\lld_navigator_controller.ahk
#Include ..\..\lld_navigator.ahk

test_passed := true
test_fail(msg) {
    global test_passed
    test_passed := false
    MsgBox("TEST FAILED: " . msg, "Test Failure", 16)
}

Sleep(1000) ; Let GUI initialize

; List of test resolutions (width x height)
resolutions := [
    [340, 380],   ; Minimum size
    [800, 600],   ; Standard
    [1280, 720],  ; HD
    [1920, 1080]  ; Full HD
]

for idx, res in resolutions {
    width := res[1]
    height := res[2]
    mainGui.Move(,, width, height)
    Sleep(500) ; Allow GUI to redraw

    ; Assert main window size
    guiW := mainGui.Width
    guiH := mainGui.Height
    if (guiW != width or guiH != height) {
        test_fail("Window size mismatch at " . width . "x" . height . ": got " . guiW . "x" . guiH)
    }

    ; Assert key controls are visible
    if (!jobEdit.Visible)
        test_fail("Job input field not visible at " . width . "x" . height)
    if (!notificationPanel.Visible and notificationLabel.Visible)
        test_fail("Notification label visible but panel hidden at " . width . "x" . height)
}

; Test minimum size enforcement
mainGui.Move(,, 100, 100)
Sleep(500)
if (mainGui.Width < 340 or mainGui.Height < 380)
    test_fail("Window allowed to shrink below minimum size.")

MsgBox(test_passed ? "TEST PASSED: Window resizing and layout checks succeeded." : "TEST FAILED: See previous messages.", "Test Result", test_passed ? 64 : 16)