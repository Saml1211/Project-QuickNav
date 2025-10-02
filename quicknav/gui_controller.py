"""
GUI Controller for Project QuickNav

This module provides the business logic layer between the GUI and the
backend Python modules. It handles project navigation, document search,
and training data generation while providing async operation support.

The controller integrates with:
- find_project_path.py for project resolution
- doc_navigator.py for document search
- cli.py for command-line compatibility
"""

import os
import sys
import re
import subprocess
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class GuiController:
    """Controller class for GUI backend integration."""

    def __init__(self, settings_manager=None):
        self.settings = settings_manager
        self._last_operation_time = None
        self._operation_cache = {}

        # Document type mapping from GUI to backend
        self.doc_type_mapping = {
            "lld": "lld",
            "hld": "hld",
            "change_order": "change_order",
            "sales_po": "sales_po",
            "floor_plans": "floor_plans",
            "scope": "scope",
            "qa_itp": "qa_itp",
            "swms": "swms",
            "supplier_quotes": "supplier_quotes",
            "photos": "photos"
        }

        # Initialize backend components
        self._init_backend()

    def _init_backend(self):
        """Initialize backend components."""
        try:
            # Import backend modules
            import sys
            from pathlib import Path

            # Add src to path for doc_navigator_functions
            src_path = Path(__file__).parent.parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from . import find_project_path
            from . import cli

            # Import the document navigation function
            try:
                from doc_navigator_functions import navigate_to_document
                self.doc_backend = type('DocBackend', (), {'navigate_to_document': staticmethod(navigate_to_document)})()
            except ImportError:
                logger.warning("doc_navigator_functions not available, using CLI fallback")
                self.doc_backend = None

            self.project_backend = find_project_path
            self.cli_backend = cli

            logger.info("Backend components initialized successfully")

        except ImportError as e:
            logger.warning(f"Failed to import backend modules: {e}")
            # Fallback to command-line interface
            self.project_backend = None
            self.doc_backend = None
            self.cli_backend = None

    def navigate_to_project(self, project_input: str, selected_folder: str,
                          debug_mode: bool = False, training_data: bool = False) -> Dict[str, Any]:
        """
        Navigate to a project folder.

        Args:
            project_input: Project number or search term
            selected_folder: Selected subfolder name
            debug_mode: Enable debug output
            training_data: Generate training data

        Returns:
            Dictionary with operation result
        """
        try:
            logger.info(f"Navigating to project: {project_input}, folder: {selected_folder}")

            # Use direct backend if available, otherwise CLI
            if self.project_backend:
                result = self._navigate_project_direct(
                    project_input, selected_folder, debug_mode, training_data
                )
            else:
                result = self._navigate_project_cli(
                    project_input, selected_folder, debug_mode, training_data
                )

            # Cache successful results
            if result.get('status') == 'SUCCESS':
                self._cache_result('project', project_input, result)

            return result

        except Exception as e:
            logger.exception("Project navigation failed")
            return {
                'status': 'ERROR',
                'error': str(e)
            }

    def navigate_to_document(self, project_input: str, doc_type: str,
                           version_filter: str = "Auto (Latest/Best)",
                           room_filter: str = "", co_filter: str = "",
                           include_archive: bool = False, choose_mode: bool = False,
                           debug_mode: bool = False, training_data: bool = False) -> Dict[str, Any]:
        """
        Navigate to project documents.

        Args:
            project_input: Project number or search term
            doc_type: Type of document to find
            version_filter: Version filtering option
            room_filter: Room number filter
            co_filter: Change order filter
            include_archive: Include archived documents
            choose_mode: Return all matches for selection
            debug_mode: Enable debug output
            training_data: Generate training data

        Returns:
            Dictionary with operation result
        """
        try:
            logger.info(f"Navigating to documents: {project_input}, type: {doc_type}")

            # Use direct backend if available, otherwise CLI
            if self.doc_backend:
                result = self._navigate_document_direct(
                    project_input, doc_type, version_filter, room_filter,
                    co_filter, include_archive, choose_mode, debug_mode, training_data
                )
            else:
                result = self._navigate_document_cli(
                    project_input, doc_type, version_filter, room_filter,
                    co_filter, include_archive, choose_mode, debug_mode, training_data
                )

            return result

        except Exception as e:
            logger.exception("Document navigation failed")
            return {
                'status': 'ERROR',
                'error': str(e)
            }

    def _navigate_project_direct(self, project_input: str, selected_folder: str,
                               debug_mode: bool, training_data: bool) -> Dict[str, Any]:
        """Navigate to project using direct backend calls."""
        try:
            # Get OneDrive folder and project folders
            onedrive_folder = self.project_backend.get_onedrive_folder()
            pfolder = self.project_backend.get_project_folders(onedrive_folder)

            # Check if it's a 5-digit project number
            if re.fullmatch(r"\d{5}", project_input):
                # Project number search
                proj_num = project_input
                range_folder = self.project_backend.get_range_folder(proj_num, pfolder)
                matches = self.project_backend.search_project_dirs(proj_num, range_folder)
            else:
                # Name search
                matches = self.project_backend.search_by_name(project_input, pfolder)

            # Process results
            if not matches:
                return {
                    'status': 'ERROR',
                    'error': f"No project found for '{project_input}'"
                }
            elif len(matches) == 1:
                # Single match - generate training data if requested
                if training_data:
                    self._generate_training_data(matches[0])

                return {
                    'status': 'SUCCESS',
                    'path': matches[0],
                    'folder': selected_folder
                }
            else:
                # Multiple matches
                status = 'SELECT' if re.fullmatch(r"\d{5}", project_input) else 'SEARCH'
                return {
                    'status': status,
                    'paths': matches,
                    'folder': selected_folder
                }

        except Exception as e:
            logger.exception("Direct project navigation failed")
            raise

    def _navigate_project_cli(self, project_input: str, selected_folder: str,
                            debug_mode: bool, training_data: bool) -> Dict[str, Any]:
        """Navigate to project using CLI interface."""
        try:
            # Build command
            cmd = [sys.executable, '-m', 'quicknav.cli', 'project', project_input]

            if training_data:
                cmd.append('--training-data')

            # Set custom root if configured
            env = os.environ.copy()
            if self.settings and hasattr(self.settings, 'get_custom_roots'):
                custom_roots = self.settings.get_custom_roots()
                if custom_roots:
                    env['QUICKNAV_ROOT'] = custom_roots[0]

            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )

            if result.returncode != 0:
                raise Exception(f"CLI command failed: {result.stderr}")

            # Parse output
            output = result.stdout.strip()
            return self._parse_project_output(output, selected_folder)

        except subprocess.TimeoutExpired:
            raise Exception("Project search timed out")
        except Exception as e:
            logger.exception("CLI project navigation failed")
            raise

    def _navigate_document_direct(self, project_input: str, doc_type: str,
                                version_filter: str, room_filter: str, co_filter: str,
                                include_archive: bool, choose_mode: bool,
                                debug_mode: bool, training_data: bool) -> Dict[str, Any]:
        """Navigate to documents using direct backend calls."""
        try:
            # First resolve project path
            onedrive_folder = self.project_backend.get_onedrive_folder()
            pfolder = self.project_backend.get_project_folders(onedrive_folder)

            if re.fullmatch(r"\d{5}", project_input):
                proj_num = project_input
                range_folder = self.project_backend.get_range_folder(proj_num, pfolder)
                matches = self.project_backend.search_project_dirs(proj_num, range_folder)
            else:
                matches = self.project_backend.search_by_name(project_input, pfolder)

            if not matches:
                return {
                    'status': 'ERROR',
                    'error': f"No project found for '{project_input}'"
                }
            elif len(matches) > 1:
                return {
                    'status': 'ERROR',
                    'error': "Multiple projects found, please be more specific"
                }

            project_path = matches[0]

            # Extract project code
            project_code = None
            project_name = os.path.basename(project_path)
            match = re.match(r"^(\d{5}) - ", project_name)
            if match:
                project_code = match.group(1)

            # Determine selection mode
            selection_mode = 'auto'
            if version_filter == "Latest Version":
                selection_mode = 'latest'
            elif choose_mode:
                selection_mode = 'choose'

            # Parse filters
            room_filter_int = None
            if room_filter and room_filter.isdigit():
                room_filter_int = int(room_filter)

            co_filter_int = None
            if co_filter and co_filter.isdigit():
                co_filter_int = int(co_filter)

            # Navigate to document
            result = self.doc_backend.navigate_to_document(
                project_path=project_path,
                doc_type=doc_type,
                selection_mode=selection_mode,
                project_code=project_code,
                room_filter=room_filter_int,
                co_filter=co_filter_int,
                exclude_archive=not include_archive
            )

            return self._parse_document_result(result)

        except Exception as e:
            logger.exception("Direct document navigation failed")
            raise

    def _navigate_document_cli(self, project_input: str, doc_type: str,
                             version_filter: str, room_filter: str, co_filter: str,
                             include_archive: bool, choose_mode: bool,
                             debug_mode: bool, training_data: bool) -> Dict[str, Any]:
        """Navigate to documents using CLI interface."""
        try:
            # Build command
            cmd = [
                sys.executable, '-m', 'quicknav.cli', 'doc',
                project_input, '--type', doc_type
            ]

            # Add selection mode
            if version_filter == "Latest Version":
                cmd.append('--latest')
            elif choose_mode:
                cmd.append('--choose')

            # Add filters
            if room_filter and room_filter.isdigit():
                cmd.extend(['--room', room_filter])

            if co_filter and co_filter.isdigit():
                cmd.extend(['--co', co_filter])

            if include_archive:
                cmd.append('--include-archive')

            if training_data:
                cmd.append('--training-data')

            # Set custom root if configured
            env = os.environ.copy()
            if self.settings and hasattr(self.settings, 'get_custom_roots'):
                custom_roots = self.settings.get_custom_roots()
                if custom_roots:
                    env['QUICKNAV_ROOT'] = custom_roots[0]

            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=60
            )

            if result.returncode != 0:
                raise Exception(f"CLI command failed: {result.stderr}")

            # Parse output
            output = result.stdout.strip()
            return self._parse_document_result(output)

        except subprocess.TimeoutExpired:
            raise Exception("Document search timed out")
        except Exception as e:
            logger.exception("CLI document navigation failed")
            raise

    def _parse_project_output(self, output: str, selected_folder: str) -> Dict[str, Any]:
        """Parse project navigation output."""
        lines = output.split('\n')
        status_line = lines[0] if lines else ""

        if status_line.startswith("ERROR:"):
            return {
                'status': 'ERROR',
                'error': status_line[6:].strip()
            }
        elif status_line.startswith("SUCCESS:"):
            return {
                'status': 'SUCCESS',
                'path': status_line[8:].strip(),
                'folder': selected_folder
            }
        elif status_line.startswith("SELECT:"):
            paths = status_line[7:].strip().split('|')
            return {
                'status': 'SELECT',
                'paths': paths,
                'folder': selected_folder
            }
        elif status_line.startswith("SEARCH:"):
            paths = status_line[7:].strip().split('|')
            return {
                'status': 'SEARCH',
                'paths': paths,
                'folder': selected_folder
            }
        else:
            return {
                'status': 'ERROR',
                'error': "Unexpected response from backend"
            }

    def _parse_document_result(self, result: Union[str, Dict]) -> Dict[str, Any]:
        """Parse document navigation result."""
        if isinstance(result, dict):
            return result

        # Parse string result from CLI
        if result.startswith("ERROR:"):
            return {
                'status': 'ERROR',
                'error': result[6:].strip()
            }
        elif result.startswith("SUCCESS:"):
            return {
                'status': 'SUCCESS',
                'path': result[8:].strip()
            }
        elif result.startswith("SELECT:"):
            paths = result[7:].strip().split('|')
            return {
                'status': 'SELECT',
                'paths': paths
            }
        else:
            return {
                'status': 'ERROR',
                'error': "Unexpected response from backend"
            }

    def _generate_training_data(self, project_path: str) -> Optional[str]:
        """Generate training data for a project."""
        try:
            if not self.project_backend:
                logger.warning("Training data generation not available without direct backend")
                return None

            # Use the backend function to discover documents
            documents = self.project_backend.discover_documents(project_path)
            project_name = os.path.basename(project_path)

            # Create training data
            training_data = []
            for doc_path in documents:
                document_name = os.path.basename(doc_path)
                training_entry = {
                    "project_folder": project_name,
                    "document_path": doc_path,
                    "document_name": document_name,
                    "extracted_info": {}
                }
                training_data.append(training_entry)

            # Save to file
            filename = self.project_backend.get_training_data_filename(project_path)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Generated training data: {len(training_data)} documents -> {filename}")
            return filename

        except Exception as e:
            logger.exception("Training data generation failed")
            return None

    def _cache_result(self, operation_type: str, key: str, result: Dict[str, Any]):
        """Cache operation result for performance."""
        cache_key = f"{operation_type}:{key}"
        self._operation_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }

        # Keep cache size reasonable
        if len(self._operation_cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self._operation_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            for old_key, _ in sorted_items[:50]:
                del self._operation_cache[old_key]

    def get_cached_result(self, operation_type: str, key: str, max_age_minutes: int = 5) -> Optional[Dict[str, Any]]:
        """Get cached result if available and recent."""
        cache_key = f"{operation_type}:{key}"

        if cache_key in self._operation_cache:
            cached = self._operation_cache[cache_key]
            age = (datetime.now() - cached['timestamp']).total_seconds() / 60

            if age <= max_age_minutes:
                logger.debug(f"Using cached result for {cache_key}")
                return cached['result']
            else:
                # Remove expired cache
                del self._operation_cache[cache_key]

        return None

    def validate_project_input(self, project_input: str) -> Dict[str, Any]:
        """
        Validate project input and provide feedback.

        Args:
            project_input: The input to validate

        Returns:
            Dictionary with validation result
        """
        result = {
            'valid': True,
            'message': '',
            'type': 'unknown'
        }

        if not project_input or not project_input.strip():
            result['valid'] = False
            result['message'] = "Project input cannot be empty"
            return result

        project_input = project_input.strip()

        # Check length
        if len(project_input) > 100:
            result['valid'] = False
            result['message'] = "Project input too long (max 100 characters)"
            return result

        # Check for invalid filesystem characters
        invalid_chars = '<>:"|?*'
        if any(char in project_input for char in invalid_chars):
            result['valid'] = False
            result['message'] = "Project input contains invalid characters"
            return result

        # Determine type
        if re.fullmatch(r"\d{5}", project_input):
            result['type'] = 'project_number'
            result['message'] = f"Valid project number: {project_input}"
        elif project_input.isdigit():
            if len(project_input) < 5:
                result['valid'] = False
                result['message'] = "Project number must be exactly 5 digits"
            else:
                result['valid'] = False
                result['message'] = "Project number must be exactly 5 digits"
        else:
            result['type'] = 'search_term'
            result['message'] = f"Search term: {project_input}"

        return result

    def get_recent_projects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently accessed projects.

        Args:
            limit: Maximum number of projects to return

        Returns:
            List of recent project information
        """
        recent_projects = []

        # Look through cache for recent successful project operations
        for cache_key, cached in self._operation_cache.items():
            if cache_key.startswith('project:') and cached['result'].get('status') == 'SUCCESS':
                project_path = cached['result'].get('path')
                if project_path and os.path.exists(project_path):
                    project_name = os.path.basename(project_path)

                    # Extract project number if available
                    match = re.match(r"^(\d{5}) - (.+)$", project_name)
                    if match:
                        proj_num = match.group(1)
                        proj_name = match.group(2)
                    else:
                        proj_num = "N/A"
                        proj_name = project_name

                    recent_projects.append({
                        'project_number': proj_num,
                        'project_name': proj_name,
                        'full_path': project_path,
                        'last_accessed': cached['timestamp']
                    })

        # Sort by last accessed time
        recent_projects.sort(key=lambda x: x['last_accessed'], reverse=True)

        return recent_projects[:limit]

    def search_suggestions(self, partial_input: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial input.

        Args:
            partial_input: Partial project input
            limit: Maximum number of suggestions

        Returns:
            List of search suggestions
        """
        suggestions = []

        # Get recent projects that match
        recent_projects = self.get_recent_projects(20)

        partial_lower = partial_input.lower()

        for project in recent_projects:
            # Check if project number starts with input
            if project['project_number'].startswith(partial_input):
                suggestions.append(project['project_number'])

            # Check if project name contains input
            elif partial_lower in project['project_name'].lower():
                suggestions.append(f"{project['project_number']} - {project['project_name']}")

        return suggestions[:limit]

    def get_document_types(self) -> List[Dict[str, str]]:
        """Get available document types."""
        return [
            {"key": "lld", "label": "Low-Level Design (LLD)"},
            {"key": "hld", "label": "High-Level Design (HLD)"},
            {"key": "change_order", "label": "Change Orders"},
            {"key": "sales_po", "label": "Sales & PO Reports"},
            {"key": "floor_plans", "label": "Floor Plans"},
            {"key": "scope", "label": "Scope Documents"},
            {"key": "qa_itp", "label": "QA/ITP Reports"},
            {"key": "swms", "label": "SWMS"},
            {"key": "supplier_quotes", "label": "Supplier Quotes"},
            {"key": "photos", "label": "Site Photos"}
        ]

    def get_version_filters(self) -> List[str]:
        """Get available version filter options."""
        return [
            "Auto (Latest/Best)",
            "Latest Version",
            "As-Built Only",
            "Initial Version",
            "All Versions"
        ]

    def cleanup(self):
        """Clean up resources."""
        # Clear cache
        self._operation_cache.clear()

        logger.info("Controller cleanup completed")