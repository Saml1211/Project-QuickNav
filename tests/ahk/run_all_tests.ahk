/*
Test Runner for AHK Integration Tests

Runs all test_*.ahk scripts in this directory (excluding test_utils.ahk).
Prints a summary of pass/fail for each test.

Usage:
    autohotkey run_all_tests.ahk
*/

#Requires AutoHotkey v2.0

tests_dir := A_ScriptDir
test_files := []
Loop Files, tests_dir . "\\test_*.ahk"
{
    if (InStr(A_LoopFileName, "test_utils.ahk"))
        continue
    test_files.Push(A_LoopFileFullPath)
}

results := []
total := test_files.Length
passed := 0

for idx, test_path in test_files
{
    test_name := StrSplit(test_path, "\\")[-1]
    ToolTip "Running: " test_name
    RunWait ["autohotkey.exe", test_path], , "Hide", &exitCode
    if (exitCode = 0) {
        results.Push("PASS: " . test_name)
        passed += 1
    } else {
        results.Push("FAIL: " . test_name)
    }
}

ToolTip
MsgBox "Integration Test Results:`n" . results.Join("`n") . "`n`nPassed: " . passed . "/" . total, "AHK Integration Test Suite"

ExitApp (passed = total ? 0 : 1)