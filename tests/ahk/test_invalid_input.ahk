/*
Automated Integration Test: Invalid Input
Covers:
- Entering invalid job number and subfolder
- Simulating "Open" and checking error handling
- Asserting no recents update, error notification/UI
*/

#Requires AutoHotkey v2.0
; Include dependencies using relative paths without variables
#Include ..\..\lld_navigator_controller.ahk
#Include ..\..\lld_navigator.ahk
#Include test_utils.ahk

reset_test_state()
Sleep(1000)

try {
    ; 1. Record initial recents state
    origJobs := recentJobs.Clone()
    origFolders := recentFolders.Clone()

    ; 2. Set invalid job number input
    jobEdit.Value := "abc!@#"
    radioCtrls[1].Value := true

    ; 3. Simulate Open
    btnOpen.OnEvent("Click", OpenProject)
    btnOpen.Value := true
    btnOpen.OnEvent("Click", "")
    Sleep(1000)

    ; 4. Assert error notification is visible
    ; (Check status text for "error" or red field)
    status := mainGui["StatusText"].Value
    assert_true(InStr(status, "error") > 0 || InStr(status, "Error") > 0, "No error message/status for invalid input.")

    ; 5. Assert job/folder NOT added to recents
    assert_eq(recentJobs.Length, origJobs.Length, "Recents unexpectedly changed on invalid input (jobs).")
    assert_eq(recentFolders.Length, origFolders.Length, "Recents unexpectedly changed on invalid input (folders).")

    ; 6. Cleanup: Reset input
    jobEdit.Value := ""
    mainGui["StatusText"].Value := "Ready"

} catch as e {
    test_fail("Exception: " . e.Message)
}

if (test_passed)
    test_pass("Invalid input")

ExitApp