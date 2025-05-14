/*
Test Utilities for AHK Integration Tests

Includes:
- Standardized test_fail and test_pass logic
- Assertion helpers
- (Optional) Backend function stubbing/mocking support
*/

#Requires AutoHotkey v2.0

global __TEST_UTILS_AHK_LOADED__ ; Declare guard variable

if (__TEST_UTILS_AHK_LOADED__ != true) { ; Check if not already loaded/true
    global __TEST_UTILS_AHK_LOADED__ := true ; Set the flag

    global test_passed := true
    global test_msgs := []

    test_fail(msg) {
        global test_passed, test_msgs
        test_passed := false
        test_msgs.Push("FAIL: " . msg)
        MsgBox("TEST FAILED: " . msg, "Test Failure", 16)
    }

    test_pass(msg := "") {
        global test_msgs
        if (msg != "")
            test_msgs.Push("PASS: " . msg)
    }

    assert_true(expr, msg := "Assertion failed") {
        if (!expr)
            test_fail(msg)
    }

    assert_eq(actual, expected, msg := "Value mismatch") {
        if (actual != expected)
            test_fail(msg . " (expected: " . expected . ", got: " . actual . ")")
    }

    reset_test_state() {
        global test_passed, test_msgs
        test_passed := true
        test_msgs := []
    }
}

/*
Backend Mocking/Stubbing Example:
Wrap backend calls with:
  if (IsSet(backend_mock)) {
      result := backend_mock(params*)
  } else {
      result := RealBackendFunction(params*)
  }
In test, set global backend_mock := (params*) => throw Error("Simulated backend failure")
*/