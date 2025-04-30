/*
Automated Integration Test: Backend Failure Scenarios

Covers:
- Backend unresponsive error handling
- Invalid backend protocol output handling
- File not found error handling
- Asserts correct error feedback and controller resilience
*/

#Requires AutoHotkey v2.0
#Include ..\..\lld_navigator_controller.ahk
#Include ..\..\lld_navigator.ahk
#Include test_utils.ahk

reset_test_state()
Sleep(1000)

backend_unresponsive(params*) {
    throw Error("Simulated backend unresponsive")
}
backend_file_not_found(params*) {
    throw Error("File not found")
}
backend_invalid_protocol(params*) {
    return "not_a_valid_protocol_json"
}

try {
    ; ---- Simulate backend unresponsive ----
    global backend_mock := backend_unresponsive
    jobEdit.Value := "54321"
    radioCtrls[1].Value := true
    btnOpen.OnEvent("Click", OpenProject)
    btnOpen.Value := true
    btnOpen.OnEvent("Click", "")
    Sleep(1000)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "unresponsive") > 0, "No error for backend unresponsive")
    test_pass("Backend unresponsive error caught")

    ; ---- Simulate invalid protocol output ----
    backend_mock := backend_invalid_protocol
    jobEdit.Value := "67890"
    radioCtrls[1].Value := true
    btnOpen.Value := true
    Sleep(1000)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "protocol") > 0, "No error for invalid protocol output")
    test_pass("Invalid protocol output error caught")

    ; ---- Simulate file not found error ----
    backend_mock := backend_file_not_found
    jobEdit.Value := "99999"
    radioCtrls[1].Value := true
    btnOpen.Value := true
    Sleep(1000)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "not found") > 0, "No error for file not found")
    test_pass("File not found error caught")

    backend_mock := unset ; Cleanup

} catch as e {
    test_fail("Exception: " . e.Message)
}

if (test_passed)
    test_pass("Backend failure scenarios")

ExitApp