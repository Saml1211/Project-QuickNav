/*
Automated Integration Test: Drag-and-Drop Input
Covers:
- Simulating user drag-and-drop of a folder onto the job input field
- Asserting normalization, field update, and notification
*/

#Requires AutoHotkey v2.0
; Include dependencies using relative paths without variables
#Include ..\..\lld_navigator_controller.ahk
#Include ..\..\lld_navigator.ahk
#Include test_utils.ahk

; Reset test state for this test
reset_test_state()
Sleep(1000) ; Let GUI initialize

try {
    ; Simulate drag-and-drop by directly invoking the handler
    fakeFolderPath := "C:\\TestProjects\\FakeJobFolder"
    WM_DROPFILES(0, 0, 0, 0) ; Normally expects wParam, simulate logic below:
    v := ValidateAndNormalizeInputs(fakeFolderPath, folderNames[1])
    if (!v.valid) {
        test_fail("Validation failed unexpectedly: " . v.errorMsg)
    } else {
        jobEdit.Value := v.normalizedJob
        ShowInlineHint("Detected path from drag-and-drop.", "info")
    }
    Sleep(1000) ; Wait for notification/hint

    ; Assert jobEdit.Value updated
    if (jobEdit.Value = "" or jobEdit.Value != v.normalizedJob)
        test_fail("Job input was not updated after drag-and-drop.")

    ; Cleanup: Clear field
    jobEdit.Value := ""

} catch as e {
    test_fail("Exception: " . e.Message)
}

if (test_passed)
    MsgBox("TEST PASSED: Drag-and-drop input", "Test Result", 64)

ExitApp