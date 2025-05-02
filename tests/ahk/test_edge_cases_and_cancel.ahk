/*
Automated Integration Test: Edge-Case Inputs & User Cancellation

Covers:
- Submits empty and whitespace-only job/folder input
- Submits excessively long and symbol-containing input
- User cancels operation before backend is invoked
- Asserts no unwanted backend call, UI resets, error feedback provided
*/

#Requires AutoHotkey v2.0
#Include %A_ScriptDir%\..\..\lld_navigator_controller.ahk
#Include ..\..\lld_navigator.ahk
#Include test_utils.ahk

reset_test_state()
Sleep(1000)

backend_call_count := 0
backend_spy(params*) {
    global backend_call_count
    backend_call_count += 1
    return '{"path": "C:\\Dummy"}'
}

try {
    ; ---- Empty input ----
    global backend_mock := backend_spy
    jobEdit.Value := ""
    radioCtrls[1].Value := true
    btnOpen.OnEvent("Click", OpenProject)
    btnOpen.Value := true
    btnOpen.OnEvent("Click", "")
    Sleep(500)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "empty") > 0, "No error for empty input")
    assert_eq(backend_call_count, 0, "Backend was called on empty input")

    ; ---- Whitespace input ----
    jobEdit.Value := "    "
    btnOpen.Value := true
    Sleep(500)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "empty") > 0, "No error for whitespace input")
    assert_eq(backend_call_count, 0, "Backend was called on whitespace input")

    ; ---- Excessively long input ----
    jobEdit.Value := "A" . ("X" * 500)
    btnOpen.Value := true
    Sleep(500)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "too long") > 0, "No error for excessively long input")
    assert_eq(backend_call_count, 0, "Backend was called on long input")

    ; ---- Symbol input ----
    jobEdit.Value := "!@#$%^&*()"
    btnOpen.Value := true
    Sleep(500)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "invalid") > 0, "No error for symbol input")
    assert_eq(backend_call_count, 0, "Backend was called on symbol input")

    ; ---- User cancels before backend call ----
    jobEdit.Value := "CANCEL_TEST"
    radioCtrls[1].Value := true
    if (IsSet(mainGui["CancelBtn"])) {
        mainGui["CancelBtn"].OnEvent("Click", CloseMainDialog)
        mainGui["CancelBtn"].Value := true
        mainGui["CancelBtn"].OnEvent("Click", "")
        Sleep(500)
        status := mainGui["StatusText"].Value
        assert_true(InStr(status, "Ready") > 0 || InStr(status, "cancel") > 0, "UI did not reset after cancel")
        assert_eq(backend_call_count, 0, "Backend was called after user cancel")
        test_pass("User cancellation before backend invocation handled")
    }

    backend_mock := unset ; Cleanup

} catch as e {
    test_fail("Exception: " . e.Message)
}

if (test_passed)
    test_pass("Edge-case and user cancellation scenarios")

ExitApp