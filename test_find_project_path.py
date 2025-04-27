import unittest
from unittest.mock import patch, MagicMock
import find_project_path
import os
import sys

class TestFindProjectPath(unittest.TestCase):

    @patch('find_project_path.print_and_exit')
    def test_validate_proj_num_valid(self, mock_print_and_exit):
        """Test validate_proj_num with valid 5-digit numbers."""
        self.assertEqual(find_project_path.validate_proj_num("12345"), "12345")
        self.assertEqual(find_project_path.validate_proj_num("00000"), "00000")
        self.assertEqual(find_project_path.validate_proj_num("99999"), "99999")
        mock_print_and_exit.assert_not_called() # Ensure exit wasn't called for valid cases

    @patch('find_project_path.print_and_exit')
    def test_validate_proj_num_invalid_length(self, mock_print_and_exit):
        """Test validate_proj_num with invalid lengths."""
        find_project_path.validate_proj_num("1234")
        mock_print_and_exit.assert_called_once_with("ERROR:Invalid argument (must be 5-digit project number)")
        mock_print_and_exit.reset_mock()

        find_project_path.validate_proj_num("123456")
        mock_print_and_exit.assert_called_once_with("ERROR:Invalid argument (must be 5-digit project number)")

    @patch('find_project_path.print_and_exit')
    def test_validate_proj_num_invalid_chars(self, mock_print_and_exit):
        """Test validate_proj_num with non-digit characters."""
        find_project_path.validate_proj_num("1234a")
        mock_print_and_exit.assert_called_once_with("ERROR:Invalid argument (must be 5-digit project number)")
        mock_print_and_exit.reset_mock()

        find_project_path.validate_proj_num("abcde")
        mock_print_and_exit.assert_called_once_with("ERROR:Invalid argument (must be 5-digit project number)")
        mock_print_and_exit.reset_mock()

        find_project_path.validate_proj_num("12 34")
        mock_print_and_exit.assert_called_once_with("ERROR:Invalid argument (must be 5-digit project number)")

    # --- Start of previously inserted methods, now indented ---
    @patch('find_project_path.print_and_exit')
    @patch('os.path.isdir')
    @patch('os.environ.get')
    def test_get_onedrive_folder_success(self, mock_getenv, mock_isdir, mock_print_and_exit):
        """Test get_onedrive_folder finds the correct path."""
        mock_getenv.return_value = '/Users/testuser'
        mock_isdir.return_value = True
        # Use os.path.join for platform independence in expected path
        expected_path = os.path.join('/Users/testuser', 'OneDrive - Pro AV Solutions')
        self.assertEqual(find_project_path.get_onedrive_folder(), expected_path)
        mock_getenv.assert_called_once_with("UserProfile")
        mock_isdir.assert_called_once_with(expected_path)
        mock_print_and_exit.assert_not_called()

    @patch('find_project_path.print_and_exit')
    @patch('os.environ.get')
    def test_get_onedrive_folder_no_userprofile(self, mock_getenv, mock_print_and_exit):
        """Test get_onedrive_folder when UserProfile env var is missing."""
        mock_getenv.return_value = None
        find_project_path.get_onedrive_folder()
        mock_getenv.assert_called_once_with("UserProfile")
        mock_print_and_exit.assert_called_once_with("ERROR:UserProfile environment variable not found")

    @patch('find_project_path.print_and_exit')
    @patch('os.path.isdir')
    @patch('os.environ.get')
    def test_get_onedrive_folder_not_found(self, mock_getenv, mock_isdir, mock_print_and_exit):
        """Test get_onedrive_folder when the directory doesn't exist."""
        mock_getenv.return_value = '/Users/testuser'
        mock_isdir.return_value = False
        find_project_path.get_onedrive_folder()
        expected_path = os.path.join('/Users/testuser', 'OneDrive - Pro AV Solutions')
        mock_isdir.assert_called_once_with(expected_path)
        mock_print_and_exit.assert_called_once_with("ERROR:OneDrive folder not found")

    @patch('find_project_path.print_and_exit')
    @patch('os.path.isdir')
    @patch('os.path.join', side_effect=os.path.join) # Mock join but keep functionality
    def test_get_project_folders_success(self, mock_join, mock_isdir, mock_print_and_exit):
        """Test get_project_folders finds the correct path."""
        mock_isdir.return_value = True
        onedrive_path = '/path/to/onedrive'
        expected_path = os.path.join(onedrive_path, 'Project Folders')
        self.assertEqual(find_project_path.get_project_folders(onedrive_path), expected_path)
        mock_isdir.assert_called_once_with(expected_path)
        mock_print_and_exit.assert_not_called()

    @patch('find_project_path.print_and_exit')
    @patch('os.path.isdir')
    @patch('os.path.join', side_effect=os.path.join) # Mock join but keep functionality
    def test_get_project_folders_not_found(self, mock_join, mock_isdir, mock_print_and_exit):
        """Test get_project_folders when the directory doesn't exist."""
        mock_isdir.return_value = False
        onedrive_path = '/path/to/onedrive'
        expected_path = os.path.join(onedrive_path, 'Project Folders')
        find_project_path.get_project_folders(onedrive_path)
        mock_isdir.assert_called_once_with(expected_path)
        mock_print_and_exit.assert_called_once_with("ERROR:Project Folders not found")

    @patch('find_project_path.print_and_exit')
    @patch('os.path.isdir')
    @patch('os.path.join', side_effect=os.path.join) # Mock join but keep functionality
    def test_get_range_folder_success(self, mock_join, mock_isdir, mock_print_and_exit):
        """Test get_range_folder finds the correct path."""
        mock_isdir.return_value = True
        pf_path = '/path/to/Project Folders'
        proj_num = "12345"
        expected_path = os.path.join(pf_path, '12000 - 12999')
        self.assertEqual(find_project_path.get_range_folder(proj_num, pf_path), expected_path)
        mock_isdir.assert_called_once_with(expected_path)
        mock_print_and_exit.assert_not_called()

        # Test edge case 00000
        mock_isdir.reset_mock()
        mock_print_and_exit.reset_mock()
        mock_isdir.return_value = True
        proj_num = "00123"
        expected_path = os.path.join(pf_path, '0 - 999')
        self.assertEqual(find_project_path.get_range_folder(proj_num, pf_path), expected_path)
        mock_isdir.assert_called_once_with(expected_path)
        mock_print_and_exit.assert_not_called()


    @patch('find_project_path.print_and_exit')
    @patch('os.path.isdir')
    @patch('os.path.join', side_effect=os.path.join) # Mock join but keep functionality
    def test_get_range_folder_not_found(self, mock_join, mock_isdir, mock_print_and_exit):
        """Test get_range_folder when the directory doesn't exist."""
        mock_isdir.return_value = False
        pf_path = '/path/to/Project Folders'
        proj_num = "12345"
        expected_path = os.path.join(pf_path, '12000 - 12999')
        find_project_path.get_range_folder(proj_num, pf_path)
        mock_isdir.assert_called_once_with(expected_path)
        mock_print_and_exit.assert_called_once_with("ERROR:Range folder not found")

# --- End of previously inserted methods ---

    @patch('find_project_path.print_and_exit')
    @patch('os.path.abspath', side_effect=lambda x: x) # Mock abspath to just return input
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_search_project_dirs_no_match(self, mock_listdir, mock_isdir, mock_abspath, mock_print_and_exit):
        """Test search_project_dirs when no directories match."""
        mock_listdir.return_value = ["11111 - Project A", "12345 - File.txt", "Some Other Folder"]
        mock_isdir.side_effect = lambda path: "File.txt" not in path # Only dirs are dirs
        range_path = '/path/to/12000 - 12999'
        proj_num = "12345"

        matches = find_project_path.search_project_dirs(proj_num, range_path)

        self.assertEqual(matches, [])
        mock_listdir.assert_called_once_with(range_path)
        mock_print_and_exit.assert_not_called()

    @patch('find_project_path.print_and_exit')
    @patch('os.path.abspath', side_effect=lambda x: x)
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_search_project_dirs_one_match(self, mock_listdir, mock_isdir, mock_abspath, mock_print_and_exit):
        """Test search_project_dirs when one directory matches."""
        proj_num = "12345"
        range_path = '/path/to/12000 - 12999'
        expected_dir_name = f"{proj_num} - Project B"
        expected_full_path = os.path.join(range_path, expected_dir_name)

        mock_listdir.return_value = ["11111 - Project A", expected_dir_name, "Some Other Folder"]
        # Mock isdir to return True only for the expected directory path
        mock_isdir.side_effect = lambda path: path == expected_full_path

        matches = find_project_path.search_project_dirs(proj_num, range_path)

        self.assertEqual(matches, [expected_full_path])
        mock_listdir.assert_called_once_with(range_path)
        # Check isdir was called for all entries returned by listdir
        self.assertEqual(mock_isdir.call_count, 3)
        mock_abspath.assert_called_once_with(expected_full_path)
        mock_print_and_exit.assert_not_called()

    @patch('find_project_path.print_and_exit')
    @patch('os.path.abspath', side_effect=lambda x: x)
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_search_project_dirs_multiple_matches(self, mock_listdir, mock_isdir, mock_abspath, mock_print_and_exit):
        """Test search_project_dirs when multiple directories match."""
        proj_num = "12345"
        range_path = '/path/to/12000 - 12999'
        match1_name = f"{proj_num} - Project B"
        match2_name = f"{proj_num} - Project B Variant"
        match1_path = os.path.join(range_path, match1_name)
        match2_path = os.path.join(range_path, match2_name)

        mock_listdir.return_value = ["11111 - Project A", match1_name, match2_name, "Some File"]
        # Mock isdir to return True only for the matching directory paths
        mock_isdir.side_effect = lambda path: path in [match1_path, match2_path]

        matches = find_project_path.search_project_dirs(proj_num, range_path)

        self.assertEqual(matches, [match1_path, match2_path])
        mock_listdir.assert_called_once_with(range_path)
        self.assertEqual(mock_isdir.call_count, 4)
        # Check abspath was called for both matches
        self.assertEqual(mock_abspath.call_count, 2)
        self.assertIn(unittest.mock.call(match1_path), mock_abspath.call_args_list)
        self.assertIn(unittest.mock.call(match2_path), mock_abspath.call_args_list)
        mock_print_and_exit.assert_not_called()

    @patch('find_project_path.print_and_exit')
    @patch('os.listdir')
    def test_search_project_dirs_listdir_error(self, mock_listdir, mock_print_and_exit):
        """Test search_project_dirs when os.listdir raises an exception."""
        mock_listdir.side_effect = OSError("Permission denied")
        range_path = '/path/to/12000 - 12999'
        proj_num = "12345"

        find_project_path.search_project_dirs(proj_num, range_path)

        mock_listdir.assert_called_once_with(range_path)
        mock_print_and_exit.assert_called_once_with("ERROR:Unable to list range folder contents")
# --- Tests for main() ---

    @patch('find_project_path.print_and_exit')
    @patch('sys.argv', ['find_project_path.py']) # No args
    def test_main_no_args(self, mock_print_and_exit):
        """Test main() with no arguments."""
        find_project_path.main()
        mock_print_and_exit.assert_called_once_with("ERROR:Exactly one argument required (5-digit project number)")

    @patch('find_project_path.print_and_exit')
    @patch('sys.argv', ['find_project_path.py', '123', '456']) # Too many args
    def test_main_too_many_args(self, mock_print_and_exit):
        """Test main() with too many arguments."""
        find_project_path.main()
        mock_print_and_exit.assert_called_once_with("ERROR:Exactly one argument required (5-digit project number)")

    # Mock all functions called by main
    @patch('find_project_path.print_and_exit')
    @patch('find_project_path.search_project_dirs')
    @patch('find_project_path.get_range_folder')
    @patch('find_project_path.get_project_folders')
    @patch('find_project_path.get_onedrive_folder')
    @patch('find_project_path.validate_proj_num')
    @patch('sys.argv', ['find_project_path.py', '12345'])
    def test_main_success_one_match(self, mock_validate, mock_get_od, mock_get_pf, mock_get_rf, mock_search, mock_print_and_exit):
        """Test main() successful execution with one match found."""
        proj_num = "12345"
        onedrive_path = "/fake/onedrive"
        pf_path = "/fake/onedrive/Project Folders"
        range_path = "/fake/onedrive/Project Folders/12000 - 12999"
        match_path = f"{range_path}/{proj_num} - The Project"

        mock_validate.return_value = proj_num
        mock_get_od.return_value = onedrive_path
        mock_get_pf.return_value = pf_path
        mock_get_rf.return_value = range_path
        mock_search.return_value = [match_path]

        find_project_path.main()

        mock_validate.assert_called_once_with("12345")
        mock_get_od.assert_called_once()
        mock_get_pf.assert_called_once_with(onedrive_path)
        mock_get_rf.assert_called_once_with(proj_num, pf_path)
        mock_search.assert_called_once_with(proj_num, range_path)
        mock_print_and_exit.assert_called_once_with(f"SUCCESS:{match_path}")

    @patch('find_project_path.print_and_exit')
    @patch('find_project_path.search_project_dirs')
    @patch('find_project_path.get_range_folder')
    @patch('find_project_path.get_project_folders')
    @patch('find_project_path.get_onedrive_folder')
    @patch('find_project_path.validate_proj_num')
    @patch('sys.argv', ['find_project_path.py', '54321'])
    def test_main_success_multiple_matches(self, mock_validate, mock_get_od, mock_get_pf, mock_get_rf, mock_search, mock_print_and_exit):
        """Test main() successful execution with multiple matches found."""
        proj_num = "54321"
        onedrive_path = "/fake/onedrive"
        pf_path = "/fake/onedrive/Project Folders"
        range_path = "/fake/onedrive/Project Folders/54000 - 54999"
        match_path1 = f"{range_path}/{proj_num} - The Project"
        match_path2 = f"{range_path}/{proj_num} - Another Project"

        mock_validate.return_value = proj_num
        mock_get_od.return_value = onedrive_path
        mock_get_pf.return_value = pf_path
        mock_get_rf.return_value = range_path
        mock_search.return_value = [match_path1, match_path2]

        find_project_path.main()

        mock_validate.assert_called_once_with("54321")
        mock_get_od.assert_called_once()
        mock_get_pf.assert_called_once_with(onedrive_path)
        mock_get_rf.assert_called_once_with(proj_num, pf_path)
        mock_search.assert_called_once_with(proj_num, range_path)
        mock_print_and_exit.assert_called_once_with(f"SELECT:{match_path1}|{match_path2}")

    @patch('find_project_path.print_and_exit')
    @patch('find_project_path.search_project_dirs')
    @patch('find_project_path.get_range_folder')
    @patch('find_project_path.get_project_folders')
    @patch('find_project_path.get_onedrive_folder')
    @patch('find_project_path.validate_proj_num')
    @patch('sys.argv', ['find_project_path.py', '11111'])
    def test_main_no_matches(self, mock_validate, mock_get_od, mock_get_pf, mock_get_rf, mock_search, mock_print_and_exit):
        """Test main() execution when no matches are found."""
        proj_num = "11111"
        onedrive_path = "/fake/onedrive"
        pf_path = "/fake/onedrive/Project Folders"
        range_path = "/fake/onedrive/Project Folders/11000 - 11999"

        mock_validate.return_value = proj_num
        mock_get_od.return_value = onedrive_path
        mock_get_pf.return_value = pf_path
        mock_get_rf.return_value = range_path
        mock_search.return_value = [] # No matches

        find_project_path.main()

        mock_validate.assert_called_once_with("11111")
        mock_get_od.assert_called_once()
        mock_get_pf.assert_called_once_with(onedrive_path)
        mock_get_rf.assert_called_once_with(proj_num, pf_path)
        mock_search.assert_called_once_with(proj_num, range_path)
        mock_print_and_exit.assert_called_once_with("ERROR:No project folder found for that number")

    @patch('find_project_path.print_and_exit')
    @patch('find_project_path.validate_proj_num')
    @patch('sys.argv', ['find_project_path.py', 'abcde'])
    def test_main_invalid_proj_num(self, mock_validate, mock_print_and_exit):
        """Test main() execution when validate_proj_num fails (implicitly calls print_and_exit)."""
        # Set up the mock for validate_proj_num to call the mocked print_and_exit
        mock_validate.side_effect = lambda x: mock_print_and_exit("ERROR:Invalid argument (must be 5-digit project number)")

        find_project_path.main()

        mock_validate.assert_called_once_with("abcde")
        # The assertion is on the *mocked* print_and_exit, called *by* the mocked validate_proj_num
        mock_print_and_exit.assert_called_once_with("ERROR:Invalid argument (must be 5-digit project number)")

    # Example for a failure in one of the get_* functions
    @patch('find_project_path.print_and_exit')
    @patch('find_project_path.get_onedrive_folder')
    @patch('find_project_path.validate_proj_num')
    @patch('sys.argv', ['find_project_path.py', '22222'])
    def test_main_get_onedrive_fails(self, mock_validate, mock_get_od, mock_print_and_exit):
        """Test main() execution when get_onedrive_folder fails."""
        proj_num = "22222"
        mock_validate.return_value = proj_num
        # Simulate get_onedrive_folder calling print_and_exit
        mock_get_od.side_effect = lambda: mock_print_and_exit("ERROR:UserProfile environment variable not found")

        find_project_path.main()

        mock_validate.assert_called_once_with(proj_num)
        mock_get_od.assert_called_once()
        mock_print_and_exit.assert_called_once_with("ERROR:UserProfile environment variable not found")
if __name__ == '__main__':
    unittest.main()