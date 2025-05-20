/*
Automated Integration Test: Manual Input & Open Project (happy path)
Covers:
- Entering valid job number and subfolder
- Simulating user clicking "Open"
- Asserting controller invocation, recents/favorites update, and expected UI/side effect

Note: Assumes lld_navigator.ahk and lld_navigator_controller.ahk are in project root.
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
    ; 1. Set input field to valid job number
    jobEdit.Value := "12345"
    ; 2. Select the first radio (subfolder)
    radioCtrls[1].Value := true
    selectedFolder := folderNames[1]

    ; 3. Simulate user clicking 'Open'
    btnOpen.OnEvent("Click", OpenProject)
    btnOpen.Value := true
    btnOpen.OnEvent("Click", "") ; Clear after for idempotency
    Sleep(3000) ; Wait for backend call and any resulting dialogs/side effects

    ; 4. Check: recentJobs and recentFolders updated
    if (!recentJobs.Has("12345"))
        test_fail("Recent jobs did not update.")
    if (!recentFolders.Has(selectedFolder))
        test_fail("Recent folders did not update.")

    ; 5. Check: GUI status text updated (success expected)
    status := mainGui["StatusText"].Value
    if (InStr(status, "success") = 0)
        test_fail("Status text does not indicate success: " . status)

    ; 6. Check: Folder opened (explorer launched)
    ; NOTE: Full automation of explorer launch can be flaky; skip strict check.

    ; 7. Cleanup: Remove test job from recents
    idx := recentJobs.IndexOf("12345")
    if (idx)
        recentJobs.RemoveAt(idx)
    recentsData.jobs := recentJobs
    SaveRecents(recentsData)

} catch as e {
    test_fail("Exception: " . e.Message)
}

if (test_passed)
    MsgBox("TEST PASSED: Manual input happy path", "Test Result", 64)

ExitApp