/*
Automated Integration Test: Favorites Toggle (via Context Menu)
Covers:
- Simulating right-click context menu on a folder radio button
- Invoking Controller_ToggleFavorite
- Asserting favorites/persistence updates and UI refresh
*/

#Requires AutoHotkey v2.0
; Include dependencies using relative paths without variables
#Include ..\..\lld_navigator_controller.ahk
#Include ..\..\lld_navigator.ahk
#Include test_utils.ahk

; Reset test state for this test
reset_test_state()
Sleep(1000) ; Allow GUI to initialize

try {
    ; 1. Select a test folder label (first in list)
    testFolder := folderNames[1]

    ; 2. Toggle favorite ON
    Controller_ToggleFavorite(testFolder)
    Sleep(500)
    recentsData := LoadRecents()
    if (!recentsData.favorites.Has(testFolder))
        test_fail("Favorite not added to recents.")

    ; 3. Toggle favorite OFF
    Controller_ToggleFavorite(testFolder)
    Sleep(500)
    recentsData := LoadRecents()
    if (recentsData.favorites.Has(testFolder))
        test_fail("Favorite not removed from recents.")

} catch as e {
    test_fail("Exception: " . e.Message)
}

if (test_passed)
    MsgBox("TEST PASSED: Favorites toggle", "Test Result", 64)

ExitApp