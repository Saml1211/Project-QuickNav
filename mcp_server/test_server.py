# Integration tests for Project QuickNav MCP server
import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
import os # Needed for later tests

# Import the functions to test
from mcp_server.tools import navigate_project
from mcp_server.resources import list_project_folders
from mcp_server.tools import list_projects
from mcp_server.resources import list_project_codes

class TestMCPServerIntegration(unittest.TestCase):
    # --- User Preferences and History Tools Tests ---

    @patch("mcp_server.user_data.get_user_data_path")
    def test_user_preferences_crud(self, mock_get_path):
        import tempfile, os, json
        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            mock_get_path.return_value = tf.name
            # Lazy import to refresh module state per test
            from mcp_server.tools import get_user_preferences, set_user_preferences, clear_user_preferences
            from mcp_server import user_data

            # Set prefs
            prefs = {"theme": "dark", "recent_folder": "/foo/bar"}
            setres = set_user_preferences(prefs)
            self.assertEqual(setres["status"], "success")
            getres = get_user_preferences()
            self.assertEqual(getres["status"], "success")
            self.assertEqual(getres["preferences"], prefs)

            # Clear prefs
            clrres = clear_user_preferences()
            self.assertEqual(clrres["status"], "success")
            getres2 = get_user_preferences()
            self.assertEqual(getres2["preferences"], {})
        finally:
            tf.close()
            os.unlink(tf.name)

    @patch("mcp_server.user_data.get_user_data_path")
    def test_user_history_crud_and_recommend(self, mock_get_path):
        import tempfile, os, json
        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            mock_get_path.return_value = tf.name
            from mcp_server.tools import get_user_history, add_user_history_entry, clear_user_history, recommend_projects
            from mcp_server import user_data

            # Add two projects, several times
            for code in ["11111", "22222", "11111"]:
                add_user_history_entry({"action": "navigate_project", "project_code": code, "status": "success"})
            histres = get_user_history()
            self.assertEqual(histres["status"], "success")
            self.assertEqual([h["project_code"] for h in histres["history"]], ["11111", "22222", "11111"][::-1])

            # Recommend (should favor 11111)
            recres = recommend_projects(2)
            self.assertEqual(recres["status"], "success")
            self.assertIn("11111", recres["recommended_projects"])
            self.assertTrue(len(recres["recommended_projects"]) <= 2)

            # Clear and check
            clear_user_history()
            histres2 = get_user_history()
            self.assertEqual(histres2["history"], [])
        finally:
            tf.close()
            os.unlink(tf.name)

    @patch("mcp_server.user_data.get_user_data_path")
    def test_quicknav_usage_diagnostics(self, mock_get_path):
        import tempfile, os, json
        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            mock_get_path.return_value = tf.name
            from mcp_server.tools import add_user_history_entry, get_quicknav_usage_diagnostics

            # Add entries, including error
            add_user_history_entry({"action": "navigate_project", "project_code": "11111", "status": "success"})
            add_user_history_entry({"action": "navigate_project", "project_code": "22222", "status": "error", "message": "fail"})
            diag = get_quicknav_usage_diagnostics()
            self.assertEqual(diag["status"], "success")
            self.assertEqual(diag["error_count"], 1)
            self.assertEqual(diag["total_history_entries"], 2)
            self.assertTrue(any(e["status"] == "error" for e in diag["recent_entries"]))
        finally:
            tf.close()
            os.unlink(tf.name)

    @patch("mcp_server.user_data.get_user_data_path")
    @patch("subprocess.run")
    def test_navigate_project_tracks_history(self, mock_subprocess_run, mock_get_path):
        import tempfile, os, json
        tf = tempfile.NamedTemporaryFile(delete=False)
        try:
            mock_get_path.return_value = tf.name
            from mcp_server.tools import navigate_project
            # Simulate success
            mock_subprocess_run.return_value = type("obj", (object,), {"stdout": "SUCCESS:/bar/folder", "returncode": 0})
            res = navigate_project("54321")
            self.assertEqual(res["status"], "success")
            # Confirm in history
            from mcp_server.user_data import get_history
            hist = get_history()
            self.assertTrue(any(h.get("project_code") == "54321" and h.get("status") == "success" for h in hist))
        finally:
            tf.close()
            os.unlink(tf.name)

    # --- Tests for navigate_project tool ---

    def test_navigate_project_invalid_input_length(self):
        """Test navigate_project tool with input of wrong length."""
        result = navigate_project("123")
        self.assertEqual(result["status"], "error")
        self.assertIn("must be a 5-digit string", result["message"])

        result = navigate_project("123456")
        self.assertEqual(result["status"], "error")
        self.assertIn("must be a 5-digit string", result["message"])

    def test_navigate_project_invalid_input_type(self):
        """Test navigate_project tool with non-string or non-digit input."""
        result = navigate_project(12345) # Not a string
        self.assertEqual(result["status"], "error")
        self.assertIn("must be a 5-digit string", result["message"])

        result = navigate_project("abcde") # Not digits
        self.assertEqual(result["status"], "error")
        self.assertIn("must be a 5-digit string", result["message"])

    @patch('subprocess.run')
    def test_navigate_project_success(self, mock_subprocess_run):
        """Test navigate_project tool simulating SUCCESS from script."""
        proj_num = "12345"
        fake_path = "/fake/path/12345 - Project Success"
        # Mock the CompletedProcess object returned by subprocess.run
        mock_result = MagicMock()
        mock_result.stdout = f"SUCCESS:{fake_path}\n" # Add newline like real script
        mock_result.stderr = ""
        mock_result.check_returncode.return_value = None # Simulate successful exit
        mock_subprocess_run.return_value = mock_result

        result = navigate_project(proj_num)

        # Verify subprocess.run was called correctly
        mock_subprocess_run.assert_called_once_with(
            [sys.executable, "../src/find_project_path.py", proj_num],
            capture_output=True, text=True, check=True
        )
        # Verify the output parsing
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["path"], fake_path)

    @patch('subprocess.run')
    def test_navigate_project_select(self, mock_subprocess_run):
        """Test navigate_project tool simulating SELECT from script."""
        proj_num = "54321"
        path1 = "/fake/path/54321 - Project A"
        path2 = "/fake/path/54321 - Project B"
        mock_result = MagicMock()
        mock_result.stdout = f"SELECT:{path1}|{path2}\n"
        mock_result.stderr = ""
        mock_result.check_returncode.return_value = None
        mock_subprocess_run.return_value = mock_result

        result = navigate_project(proj_num)

        mock_subprocess_run.assert_called_once_with(
            [sys.executable, "../src/find_project_path.py", proj_num],
            capture_output=True, text=True, check=True
        )
        self.assertEqual(result["status"], "select")
        self.assertEqual(result["paths"], [path1, path2])

    @patch('subprocess.run')
    def test_navigate_project_script_error(self, mock_subprocess_run):
        """Test navigate_project tool simulating ERROR from script."""
        proj_num = "99999"
        error_msg = "No project folder found for that number"
        mock_result = MagicMock()
        mock_result.stdout = f"ERROR:{error_msg}\n"
        mock_result.stderr = ""
        mock_result.check_returncode.return_value = None
        mock_subprocess_run.return_value = mock_result

        result = navigate_project(proj_num)

        mock_subprocess_run.assert_called_once_with(
            [sys.executable, "../src/find_project_path.py", proj_num],
            capture_output=True, text=True, check=True
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], error_msg)

    @patch('subprocess.run')
    def test_navigate_project_script_unexpected_output(self, mock_subprocess_run):
        """Test navigate_project tool simulating unexpected output from script."""
        proj_num = "11111"
        unexpected_output = "Something went wrong internally\n"
        mock_result = MagicMock()
        mock_result.stdout = unexpected_output
        mock_result.stderr = ""
        mock_result.check_returncode.return_value = None
        mock_subprocess_run.return_value = mock_result

        result = navigate_project(proj_num)

        mock_subprocess_run.assert_called_once_with(
            [sys.executable, "../src/find_project_path.py", proj_num],
            capture_output=True, text=True, check=True
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("Unrecognized output", result["message"])

    @patch('subprocess.run')
    def test_navigate_project_subprocess_error(self, mock_subprocess_run):
        """Test navigate_project tool simulating CalledProcessError."""
        proj_num = "22222"
        error_stderr = "Traceback: file not found"
        # Simulate CalledProcessError being raised
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["python", "../src/find_project_path.py", proj_num], stderr=error_stderr
        )

        result = navigate_project(proj_num)

        mock_subprocess_run.assert_called_once_with(
            [sys.executable, "../src/find_project_path.py", proj_num],
            capture_output=True, text=True, check=True
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("QuickNav script failed", result["message"])
        self.assertIn(error_stderr, result["message"])

    # --- Tests for list_project_folders resource ---

    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('os.getcwd')
    @patch('os.path.abspath', side_effect=lambda x: x) # Mock abspath to simplify
    def test_list_project_folders_mixed(self, mock_abspath, mock_getcwd, mock_listdir, mock_isdir):
        """Test list_project_folders with a mix of files and directories."""
        fake_cwd = "/fake/project/root"
        mock_getcwd.return_value = fake_cwd
        mock_listdir.return_value = ["file1.txt", "subdir", "file2.py", ".hiddenfile"]

        # Define which names correspond to directories
        def isdir_side_effect(path):
            # Check the *basename* against 'subdir'
            return os.path.basename(path) == "subdir"
        mock_isdir.side_effect = isdir_side_effect

        result = list_project_folders()

        mock_getcwd.assert_called_once()
        mock_listdir.assert_called_once_with(fake_cwd)
        # Check isdir called for each entry
        self.assertEqual(mock_isdir.call_count, 4)

        # Check the structure and content of the result
        self.assertIn("entries", result)
        entries = result["entries"]
        self.assertEqual(len(entries), 4)

        # Check entries (order might vary, so check existence and types)
        expected_entries = {
            "file1.txt": "file",
            "subdir": "dir",
            "file2.py": "file",
            ".hiddenfile": "file"
        }
        actual_entries = {e["name"]: e["type"] for e in entries}
        self.assertEqual(actual_entries, expected_entries)

    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('os.getcwd')
    @patch('os.path.abspath', side_effect=lambda x: x)
    def test_list_project_folders_empty(self, mock_abspath, mock_getcwd, mock_listdir, mock_isdir):
        """Test list_project_folders with an empty directory."""
        fake_cwd = "/fake/empty/dir"
        mock_getcwd.return_value = fake_cwd
        mock_listdir.return_value = [] # Empty directory
        mock_isdir.return_value = False # Doesn't matter, won't be called

        result = list_project_folders()

        mock_getcwd.assert_called_once()
        mock_listdir.assert_called_once_with(fake_cwd)
        mock_isdir.assert_not_called() # Should not be called if listdir is empty

        self.assertIn("entries", result)
        self.assertEqual(result["entries"], [])

    @patch('os.listdir')
    @patch('os.getcwd')
    @patch('os.path.abspath', side_effect=lambda x: x)
    def test_list_project_folders_listdir_error(self, mock_abspath, mock_getcwd, mock_listdir):
        """Test list_project_folders when os.listdir raises an error."""
        fake_cwd = "/fake/error/dir"
        mock_getcwd.return_value = fake_cwd
        mock_listdir.side_effect = OSError("Permission denied")

        # Check that the OSError propagates up
        with self.assertRaises(OSError):
            list_project_folders()

        mock_getcwd.assert_called_once()

    # --- New tests: Error Handler and Project Listing ---

    def test_error_handler_on_resource_exception(self):
        """Resource raising an exception should return standardized error dict (not raise)."""
        # Define a dummy resource function decorated with @error_handler
        from mcp_server.server import error_handler
        @error_handler
        def always_raises():
            raise ValueError("Simulated failure")
        result = always_raises()
        self.assertIn("error", result)
        self.assertIn("message", result["error"])
        self.assertIn("ValueError", result["error"]["type"])
        self.assertIn("Simulated failure", result["error"]["message"])
        self.assertIn("traceback", result["error"])
        self.assertTrue("ValueError: Simulated failure" in result["error"]["traceback"])

    @patch.dict(os.environ, {"UserProfile": "/fake/userprofile"})
    @patch("os.path.isdir")
    @patch("os.listdir")
    def test_list_project_codes_success(self, mock_listdir, mock_isdir):
        """Test project://list resource returns sorted 5-digit codes as expected."""
        # Setup: mock OneDrive & folders
        def isdir_side_effect(path):
            # Simulate dirs: OneDrive, Project Folders, range folders, project dirs
            if path in [
                "/fake/userprofile/OneDrive - Pro AV Solutions",
                "/fake/userprofile/OneDrive - Pro AV Solutions/Project Folders",
                "/fake/userprofile/OneDrive - Pro AV Solutions/Project Folders/11000 - 11999"
            ]:
                return True
            # Project dirs
            if path.startswith("/fake/userprofile/OneDrive - Pro AV Solutions/Project Folders/11000 - 11999/"):
                return True
            return False

        mock_isdir.side_effect = isdir_side_effect

        # Range listing
        def listdir_side_effect(path):
            if path == "/fake/userprofile/OneDrive - Pro AV Solutions/Project Folders":
                return ["11000 - 11999", "bogus"]
            if path == "/fake/userprofile/OneDrive - Pro AV Solutions/Project Folders/11000 - 11999":
                return [
                    "11001 - Foo",
                    "11002 - Bar",
                    "not_a_project"
                ]
            return []

        mock_listdir.side_effect = listdir_side_effect

        from mcp_server.resources import list_project_codes
        result = list_project_codes()
        self.assertIsInstance(result, dict)
        self.assertIn("project_codes", result)
        self.assertEqual(result["project_codes"], ["11001", "11002"])

    @patch("mcp_server.resources.list_project_codes")
    def test_list_projects_tool(self, mock_list_project_codes):
        """Test list_projects tool returns underlying resource result."""
        fake_result = {"project_codes": ["12345", "11111"]}
        mock_list_project_codes.return_value = fake_result
        result = list_projects()
        self.assertEqual(result, fake_result)
        mock_listdir.assert_called_once_with(fake_cwd)

if __name__ == "__main__":
    unittest.main()
