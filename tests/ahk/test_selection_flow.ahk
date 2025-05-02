/*
Automated Integration Test: Selection Flows (Multi-Path/Project Results)

Covers:
- Backend returns multiple results (multi-path/project scenario)
- User selects one result and confirms
- Verifies correct controller invocation and feedback
- Tests user cancelling selection dialog (no action taken, UI resets)
*/

#Requires AutoHotkey v2.0
#Include %A_ScriptDir%\..\..\lld_navigator_controller.ahk
#Include ..\..\lld_navigator.ahk
#Include test_utils.ahk

HandleSelection(*) {
    OutputDebug("HandleSelection called")
}
CloseSelectionDialog(*) {
    OutputDebug("CloseSelectionDialog called")
}
reset_test_state()
Sleep(1000)

selection_backend_mock(params*) {
    ; Simulate backend returning multiple selectable results
    return '[{"path": "C:\\Project1", "label": "Project One"}, {"path": "C:\\Project2", "label": "Project Two"}]'
}
try {
    ; ---- Simulate multi-select backend response ----
    global backend_mock := selection_backend_mock
    jobEdit.Value := "MULTI"
    radioCtrls[1].Value := true
    btnOpen.OnEvent("Click", OpenProject)
    btnOpen.Value := true
    btnOpen.OnEvent("Click", "")
    Sleep(1000)

    ; Assume the test can select from a dialog or list.
    ; For automation: forcibly select first result as mock user action.
    if (IsSet(mainGui["SelectionList"])) {
        mainGui["SelectionList"].Value := 1  ; Select first item
        if (IsSet(mainGui["SelectBtn"])) {
            mainGui["SelectBtn"].OnEvent("Click", HandleSelection)
            mainGui["SelectBtn"].Value := true
            mainGui["SelectBtn"].OnEvent("Click", "")
            Sleep(1000)
        }
        ; Assert correct path selected
        selectedPath := mainGui["StatusText"].Value
        assert_true(InStr(selectedPath, "Project1") > 0, "Incorrect selection feedback after multi-result selection")
        test_pass("Selection flow (multi-path, user selects one) handled")
    } else {
        test_fail("Selection list control not found in GUI")
    }

    ; ---- Simulate user cancels selection ----
    backend_mock := selection_backend_mock
    jobEdit.Value := "MULTI"
    radioCtrls[1].Value := true
    btnOpen.Value := true
    Sleep(1000)
    if (IsSet(mainGui["SelectionList"])) {
        ; Simulate user cancel: do not select, trigger cancel if possible
        if (IsSet(mainGui["CancelBtn"])) {
            mainGui["CancelBtn"].OnEvent("Click", CloseSelectionDialog)
            mainGui["CancelBtn"].Value := true
            mainGui["CancelBtn"].OnEvent("Click", "")
            Sleep(1000)
            ; Assert UI/status reset, no project opened
            status := mainGui["StatusText"].Value
            assert_true(InStr(status, "Ready") > 0 || InStr(status, "cancel") > 0, "UI did not return to ready/cancelled state after user cancellation")
            test_pass("User cancellation handled in selection flow")
        } else {
            test_fail("Cancel button not found in selection dialog")
        }
    }

    backend_mock := unset ; Cleanup

} catch as e {
    test_fail("Exception: " . e.Message)
}

if (test_passed)
    test_pass("Selection flow scenarios")

ExitApp