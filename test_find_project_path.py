"""
Comprehensive pytest suite for find_project_path.py

- Covers valid/invalid inputs (job numbers & search terms)
- Covers error conditions (missing env, folders, permissions)
- Asserts protocol output for SUCCESS, ERROR, SELECT
- Regression checks for bugfix validation
- Tests organized by outcome category
"""

import os
import sys
import stat
import shutil
import tempfile
import subprocess
import pytest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "find_project_path.py"))

def build_sim_project_structure(base_dir):
    """
    Create a fake OneDrive - Pro AV Solutions/Project Folders/NNNNN - NNNNN hierarchy.
    """
    od_root = os.path.join(base_dir, "OneDrive - Pro AV Solutions")
    pf_root = os.path.join(od_root, "Project Folders")
    os.makedirs(pf_root, exist_ok=True)
    # Range folders
    r1 = os.path.join(pf_root, "10000 - 10999")
    r2 = os.path.join(pf_root, "17000 - 17999")
    os.makedirs(r1, exist_ok=True)
    os.makedirs(r2, exist_ok=True)
    # Project folders (simulate)
    pa = os.path.join(r1, "10123 - Project A")
    pb = os.path.join(r1, "10456 - Project B")
    t1 = os.path.join(r2, "17741 - Test Project")
    t2 = os.path.join(r2, "17742 - Another Project")
    os.makedirs(pa, exist_ok=True)
    os.makedirs(pb, exist_ok=True)
    os.makedirs(t1, exist_ok=True)
    os.makedirs(t2, exist_ok=True)
    # Subfolders in one project for variety
    for sub in [
        "1. Sales Handover", "2. BOM & CO", "3. Handover Docs",
        "4. System Designs", "5. Floor Plans", "6. Site Photos"
    ]:
        os.makedirs(os.path.join(t1, sub), exist_ok=True)
    return {
        "onedrive": od_root,
        "pfolders": pf_root,
        "ranges": [r1, r2],
        "projects": {
            "10123": pa,
            "10456": pb,
            "17741": t1,
            "17742": t2
        }
    }

@pytest.fixture
def sim_env(tmp_path, monkeypatch):
    # Simulate UserProfile environment and OneDrive structure
    env = build_sim_project_structure(str(tmp_path))
    monkeypatch.setenv("UserProfile", str(tmp_path))
    yield env
    # Cleanup: nothing, tmp_path auto-cleans

@pytest.mark.parametrize("arg, expected_prefix, expectation, msg", [
    # Valid project numbers (one match)
    ("10123", "SUCCESS:", "10123 - Project A", "Should resolve valid 5-digit project to single folder"),
    ("10456", "SUCCESS:", "10456 - Project B", "Should resolve valid 5-digit project to single folder"),
    ("17741", "SUCCESS:", "17741 - Test Project", "Should resolve valid 5-digit project to single folder"),
    # Valid project numbers (multiple match scenario: duplicate test)
    # No multiple projects in structure, so SELECT not possible for job num in this setup
    # Valid search terms (case-insensitive, partial)
    ("Project", "SELECT:", "Project A|Project B|Test Project|Another Project", "Should select all projects on common term"),
    ("Test", "SUCCESS:", "Test Project", "Should succeed for unique project name search"),
    ("Another", "SUCCESS:", "Another Project", "Should succeed for unique partial match"),
])
def test_valid_success_cases(sim_env, arg, expected_prefix, expectation, msg):
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, arg],
        env={**os.environ, "UserProfile": sim_env["onedrive"].rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1]},
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith(expected_prefix), f"{msg}: Output did not start with {expected_prefix!r}. Out: {out!r}"
    if expected_prefix == "SUCCESS:":
        # Output should contain the right folder name
        assert expectation in out, f"{msg}: Expected project folder '{expectation}' in output. Out: {out!r}"
    elif expected_prefix == "SELECT:":
        # Should include all expected names, split by |
        parts = out[len("SELECT:"):].split("|")
        hits = [p for p in parts if any(x in p for x in expectation.split("|"))]
        assert len(hits) == len(expectation.split("|")), f"{msg}: SELECT output missing some expected projects. Out: {out!r}"

@pytest.mark.parametrize("arg, error_part, msg", [
    ("", "Exactly one argument required", "Empty argument should error"),
    ("123", "argument must be a 5-digit project number", "Too short job number should error"),
    ("123456", "argument must be a 5-digit project number", "Too long job number should error"),
    ("12ab3", "argument must be a 5-digit project number", "Non-digit job number should error"),
    ("<>?:", "contains disallowed special characters", "Forbidden chars in search term should error"),
    ("A" * 101, "under 100 characters", "Too-long search term should error"),
])
def test_invalid_input_errors(sim_env, arg, error_part, msg):
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, arg] if arg else [sys.executable, SCRIPT_PATH],
        env={**os.environ, "UserProfile": sim_env["onedrive"].rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1]},
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), f"{msg}: Did not get ERROR prefix. Out: {out!r}"
    assert error_part in out, f"{msg}: Error message missing expected part '{error_part}'. Got: {out!r}"

def test_no_project_found(sim_env):
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, "99999"],
        env={**os.environ, "UserProfile": sim_env["onedrive"].rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1]},
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), "Nonexistent project should yield ERROR"
    assert "No project folder found" in out, "Error should mention 'No project folder found'"

def test_no_search_match(sim_env):
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, "UnlikelySearchTermZZZ"],
        env={**os.environ, "UserProfile": sim_env["onedrive"].rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1]},
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), "Nonexistent search should yield ERROR"
    assert "No project folders found containing 'UnlikelySearchTermZZZ'" in out, "Error message should be clear"

def test_missing_userprofile(monkeypatch):
    monkeypatch.delenv("UserProfile", raising=False)
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, "10123"],
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), "Missing UserProfile should yield ERROR"
    assert "UserProfile" in out, "Error should mention 'UserProfile'"

def test_missing_onedrive(sim_env, monkeypatch):
    # Remove OneDrive folder to simulate
    onedrive_path = sim_env["onedrive"]
    shutil.rmtree(onedrive_path)
    monkeypatch.setenv("UserProfile", onedrive_path.rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1])
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, "10123"],
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), "Missing OneDrive directory should yield ERROR"
    assert "OneDrive path not found" in out, "Error should mention OneDrive path"

def test_missing_project_folders(sim_env, monkeypatch):
    pf_path = sim_env["pfolders"]
    shutil.rmtree(pf_path)
    monkeypatch.setenv("UserProfile", sim_env["onedrive"].rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1])
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, "10123"],
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), "Missing Project Folders directory should yield ERROR"
    assert "Project Folders" in out, "Error should mention 'Project Folders'"

def test_permissions_error(sim_env, monkeypatch):
    # Remove permissions from range folder to simulate OS error
    r1 = sim_env["ranges"][0]
    os.chmod(r1, 0)
    monkeypatch.setenv("UserProfile", sim_env["onedrive"].rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1])
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, "10123"],
        capture_output=True, text=True
    )
    os.chmod(r1, stat.S_IRWXU)
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), "OS permissions error should yield ERROR"
    assert "Unable to list range folder contents" in out, "Error should mention listing error"

# Regression: input validation and error reporting (ensure strict error messages, no tracebacks)
@pytest.mark.parametrize("bad_arg, errfrag", [
    ("12a45", "must be a 5-digit project number"),
    ("", "Exactly one argument required"),
    ("-1234", "must be a 5-digit project number"),
    ("&&", "contains disallowed special characters"),
])
def test_regression_input_validation(sim_env, bad_arg, errfrag):
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, bad_arg] if bad_arg else [sys.executable, SCRIPT_PATH],
        env={**os.environ, "UserProfile": sim_env["onedrive"].rsplit("OneDrive - Pro AV Solutions", 1)[0][:-1]},
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    assert out.startswith("ERROR:"), f"Regression: Expected ERROR for input '{bad_arg}'. Out: {out!r}"
    assert errfrag in out, f"Regression: Expected '{errfrag}' in error message for '{bad_arg}'. Out: {out!r}"