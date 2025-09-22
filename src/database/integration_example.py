"""
Integration Example: Project QuickNav Database Analytics

This module demonstrates how to integrate the analytics database with the existing
Project QuickNav codebase, showing practical usage patterns and performance optimization.

Key Integration Points:
1. Document discovery and metadata extraction
2. User activity tracking during navigation
3. AI conversation logging
4. Performance monitoring and optimization
5. Analytics dashboard data preparation
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing QuickNav modules
try:
    from src.find_project_path import (
        get_onedrive_folder, get_project_folders, search_project_dirs,
        search_by_name, discover_documents, get_range_folder
    )
    from quicknav.doc_navigator import DocumentParser
    from quicknav.gui_settings import SettingsManager
except ImportError as e:
    print(f"Warning: Could not import QuickNav modules: {e}")

# Import our database components
from database_manager import DatabaseManager, get_database_manager
from analytics_queries import AnalyticsQueries


class QuickNavAnalyticsIntegration:
    """
    Integration layer between Project QuickNav and the analytics database.
    Provides instrumented versions of core navigation functions with analytics tracking.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db = get_database_manager(db_path)
        self.analytics = AnalyticsQueries(self.db)
        self.doc_parser = DocumentParser()
        self.current_session_id = None

        # Initialize settings for user preferences
        try:
            self.settings = SettingsManager()
        except:
            self.settings = None

    def start_user_session(self, app_version: str = "2.0.0") -> str:
        """Start a new user session and return session ID."""
        import platform

        self.current_session_id = self.db.start_session(
            user_id='default_user',
            app_version=app_version,
            os_platform=platform.system()
        )

        if self.current_session_id:
            print(f"Started session: {self.current_session_id}")

        return self.current_session_id

    def end_user_session(self):
        """End the current user session."""
        if self.current_session_id:
            self.db.end_session(self.current_session_id)
            print(f"Ended session: {self.current_session_id}")
            self.current_session_id = None

    # =====================================================
    # INSTRUMENTED NAVIGATION FUNCTIONS
    # =====================================================

    def navigate_to_project(self, project_input: str, training_data: bool = False,
                           input_method: str = "keyboard") -> Dict[str, Any]:
        """
        Enhanced project navigation with full analytics tracking.

        Args:
            project_input: Project code or search term
            training_data: Whether to generate training data
            input_method: How the user initiated this navigation

        Returns:
            Navigation result with analytics metadata
        """
        start_time = time.time()
        activity_id = None

        try:
            # Start activity tracking
            if self.current_session_id:
                activity_id = self.db.record_activity(
                    session_id=self.current_session_id,
                    activity_type='navigate',
                    search_query=project_input,
                    input_method=input_method,
                    ui_component='main_search'
                )

            # Perform the navigation using existing logic
            onedrive_folder = get_onedrive_folder()
            pfolder = get_project_folders(onedrive_folder)

            # Check if input is a 5-digit project number
            import re
            if re.fullmatch(r"\d{5}", project_input):
                # Project number navigation
                result = self._navigate_by_project_number(
                    project_input, pfolder, training_data
                )
            else:
                # Search term navigation
                result = self._navigate_by_search_term(
                    project_input, pfolder, training_data
                )

            # Calculate response time
            response_time = int((time.time() - start_time) * 1000)

            # Update activity with results
            if self.current_session_id and activity_id:
                # Update the activity record with results
                update_query = """
                UPDATE user_activities
                SET search_results_count = ?, response_time_ms = ?,
                    success = ?, project_id = ?, action_details = ?
                WHERE activity_id = ?
                """

                action_details = {
                    'navigation_type': 'project_number' if re.fullmatch(r"\d{5}", project_input) else 'search_term',
                    'result_type': result['type'],
                    'training_data_requested': training_data
                }

                self.db.execute_write(update_query, [
                    result.get('count', 0),
                    response_time,
                    result['success'],
                    result.get('project_id'),
                    json.dumps(action_details),
                    activity_id
                ])

            # Store/update project information in database
            if result['success'] and result.get('projects'):
                for project_path in result['projects']:
                    self._upsert_project_from_path(project_path)

            result['response_time_ms'] = response_time
            result['activity_id'] = activity_id

            return result

        except Exception as e:
            # Log error and update activity
            response_time = int((time.time() - start_time) * 1000)

            if self.current_session_id and activity_id:
                update_query = """
                UPDATE user_activities
                SET response_time_ms = ?, success = 0, error_message = ?
                WHERE activity_id = ?
                """
                self.db.execute_write(update_query, [response_time, str(e), activity_id])

            return {
                'success': False,
                'type': 'error',
                'message': str(e),
                'response_time_ms': response_time,
                'activity_id': activity_id
            }

    def _navigate_by_project_number(self, proj_num: str, pfolder: str,
                                   training_data: bool) -> Dict[str, Any]:
        """Navigate by project number with database integration."""
        try:
            range_folder = get_range_folder(proj_num, pfolder)
            matches = search_project_dirs(proj_num, range_folder)

            if not matches:
                return {
                    'success': False,
                    'type': 'no_results',
                    'message': f"No project folder found for number {proj_num}",
                    'count': 0
                }
            elif len(matches) == 1:
                project_path = matches[0]
                project_id = proj_num

                # Generate training data if requested
                if training_data:
                    self._generate_training_data(project_path, project_id)

                return {
                    'success': True,
                    'type': 'single_result',
                    'project_path': project_path,
                    'project_id': project_id,
                    'projects': [project_path],
                    'count': 1
                }
            else:
                return {
                    'success': True,
                    'type': 'multiple_results',
                    'projects': matches,
                    'count': len(matches),
                    'message': f"Multiple projects found for {proj_num}"
                }

        except Exception as e:
            return {
                'success': False,
                'type': 'error',
                'message': str(e),
                'count': 0
            }

    def _navigate_by_search_term(self, search_term: str, pfolder: str,
                                training_data: bool) -> Dict[str, Any]:
        """Navigate by search term with database integration."""
        try:
            matches = search_by_name(search_term, pfolder)

            if not matches:
                return {
                    'success': False,
                    'type': 'no_results',
                    'message': f"No project folders found containing '{search_term}'",
                    'count': 0
                }
            elif len(matches) == 1:
                project_path = matches[0]
                # Extract project ID from path
                project_id = self._extract_project_id_from_path(project_path)

                # Generate training data if requested
                if training_data:
                    self._generate_training_data(project_path, project_id)

                return {
                    'success': True,
                    'type': 'single_result',
                    'project_path': project_path,
                    'project_id': project_id,
                    'projects': [project_path],
                    'count': 1
                }
            else:
                return {
                    'success': True,
                    'type': 'multiple_results',
                    'projects': matches,
                    'count': len(matches),
                    'message': f"Multiple projects found for '{search_term}'"
                }

        except Exception as e:
            return {
                'success': False,
                'type': 'error',
                'message': str(e),
                'count': 0
            }

    def open_document(self, document_path: str, document_id: str = None) -> Dict[str, Any]:
        """
        Track document opening with analytics.

        Args:
            document_path: Full path to the document
            document_id: Database document ID if known

        Returns:
            Result of document opening
        """
        start_time = time.time()

        try:
            # Record document access
            if self.current_session_id:
                project_id = self._extract_project_id_from_path(document_path)

                activity_id = self.db.record_activity(
                    session_id=self.current_session_id,
                    activity_type='document_open',
                    project_id=project_id,
                    document_id=document_id,
                    input_method='click',
                    ui_component='document_list',
                    action_details={'document_path': document_path}
                )

                # Update document last_accessed timestamp
                if document_id:
                    update_query = """
                    UPDATE documents
                    SET last_accessed = CURRENT_TIMESTAMP
                    WHERE document_id = ?
                    """
                    self.db.execute_write(update_query, [document_id])

            # Simulate document opening (replace with actual implementation)
            import subprocess
            import platform

            if platform.system() == 'Windows':
                os.startfile(document_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', document_path])
            else:  # Linux
                subprocess.call(['xdg-open', document_path])

            response_time = int((time.time() - start_time) * 1000)

            return {
                'success': True,
                'document_path': document_path,
                'response_time_ms': response_time
            }

        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)

            return {
                'success': False,
                'error': str(e),
                'response_time_ms': response_time
            }

    # =====================================================
    # DOCUMENT ANALYSIS AND STORAGE
    # =====================================================

    def analyze_and_store_documents(self, project_path: str, project_id: str) -> int:
        """
        Analyze all documents in a project and store metadata in database.

        Args:
            project_path: Full path to project folder
            project_id: Project identifier

        Returns:
            Number of documents processed
        """
        documents = discover_documents(project_path)
        processed_count = 0

        for doc_path in documents:
            try:
                doc_metadata = self._extract_document_metadata(doc_path, project_id)
                if self.db.upsert_document(doc_metadata):
                    processed_count += 1
            except Exception as e:
                print(f"Error processing document {doc_path}: {e}")

        return processed_count

    def _extract_document_metadata(self, file_path: str, project_id: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from a document file."""
        file_path_obj = Path(file_path)
        filename = file_path_obj.name

        # Parse filename using DocumentParser
        parsed_metadata = self.doc_parser.parse_filename(filename)

        # Get file statistics
        try:
            file_stats = file_path_obj.stat()
            file_size = file_stats.st_size
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)
        except:
            file_size = 0
            modified_time = None

        # Calculate content hash
        try:
            with open(file_path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()
        except:
            content_hash = None

        # Determine document type and folder category
        document_type = self._classify_document_type(filename, file_path)
        folder_category = self._determine_folder_category(file_path)

        return {
            'project_id': project_id,
            'file_path': str(file_path),
            'filename': filename,
            'file_extension': file_path_obj.suffix.lower(),
            'file_size_bytes': file_size,
            'document_type': document_type,
            'folder_category': folder_category,
            'version_string': parsed_metadata.get('version_string'),
            'version_numeric': parsed_metadata.get('version_numeric'),
            'version_type': parsed_metadata.get('version_type'),
            'status_tags': parsed_metadata.get('status_tags', []),
            'status_weight': parsed_metadata.get('status_weight'),
            'document_date': parsed_metadata.get('document_date'),
            'date_format': parsed_metadata.get('date_format'),
            'modified_at': modified_time.isoformat() if modified_time else None,
            'content_hash': content_hash,
            # Content analysis would be implemented here
            'word_count': None,
            'page_count': None,
            'content_preview': None,
            'classification_confidence': None
        }

    def _classify_document_type(self, filename: str, file_path: str) -> str:
        """Classify document type based on filename and path."""
        filename_lower = filename.lower()
        path_lower = file_path.lower()

        # Document type classification logic
        if any(term in filename_lower for term in ['lld', 'low level', 'detailed design']):
            return 'lld'
        elif any(term in filename_lower for term in ['hld', 'high level', 'system design']):
            return 'hld'
        elif any(term in filename_lower for term in ['co', 'change order', 'variation']):
            return 'co'
        elif any(term in filename_lower for term in ['floor plan', 'floorplan', 'layout']):
            return 'floor_plan'
        elif any(term in filename_lower for term in ['scope', 'sow', 'statement of work']):
            return 'scope'
        elif any(term in filename_lower for term in ['quote', 'quotation', 'estimate']):
            return 'quote'
        elif any(term in filename_lower for term in ['invoice', 'receipt', 'purchase']):
            return 'financial'
        else:
            return 'other'

    def _determine_folder_category(self, file_path: str) -> str:
        """Determine folder category from file path."""
        path_parts = Path(file_path).parts

        # Find the subfolder within the project
        for part in path_parts:
            part_lower = part.lower()
            if 'system design' in part_lower:
                return 'System Designs'
            elif 'sales handover' in part_lower:
                return 'Sales Handover'
            elif 'bom' in part_lower or 'order' in part_lower:
                return 'BOM & Orders'
            elif 'handover' in part_lower and 'customer' in part_lower:
                return 'Customer Handover'
            elif 'floor plan' in part_lower:
                return 'Floor Plans'
            elif 'photo' in part_lower or 'image' in part_lower:
                return 'Site Photos'

        return 'Other'

    def _upsert_project_from_path(self, project_path: str):
        """Extract project information from path and store in database."""
        try:
            project_name = Path(project_path).name
            # Extract project code from folder name (assumes format "12345 - Project Name")
            import re
            match = re.match(r'^(\d{5})\s*-\s*(.+)$', project_name)

            if match:
                project_code = match.group(1)
                project_title = match.group(2).strip()

                # Determine range folder
                project_num = int(project_code)
                range_start = int(project_num // 1000) * 1000
                range_end = range_start + 999
                range_folder = f"{range_start} - {range_end}"

                # Store project
                self.db.upsert_project(
                    project_id=project_code,
                    project_code=project_code,
                    project_name=project_title,
                    full_path=project_path,
                    range_folder=range_folder,
                    metadata={'discovered_at': datetime.now().isoformat()}
                )

                # Analyze and store documents
                self.analyze_and_store_documents(project_path, project_code)

        except Exception as e:
            print(f"Error upserting project from path {project_path}: {e}")

    def _extract_project_id_from_path(self, path: str) -> Optional[str]:
        """Extract project ID from a file or folder path."""
        import re
        # Look for 5-digit project code in path
        match = re.search(r'\b(\d{5})\b', path)
        return match.group(1) if match else None

    def _generate_training_data(self, project_path: str, project_id: str):
        """Generate training data for ML analysis."""
        try:
            # This would integrate with the existing training data generation
            from src.find_project_path import discover_documents

            documents = discover_documents(project_path)
            training_data = []

            for doc_path in documents:
                document_name = os.path.basename(doc_path)
                training_entry = {
                    "project_folder": os.path.basename(project_path),
                    "document_path": doc_path,
                    "document_name": document_name,
                    "extracted_info": self.doc_parser.parse_filename(document_name)
                }
                training_data.append(training_entry)

            # Save to file (existing logic)
            training_dir = "C:/Users/SamLyndon/Projects/Work/av-project-analysis-tools/training_data"
            os.makedirs(training_dir, exist_ok=True)
            filename = os.path.join(training_dir, f"training_data_{project_id}.json")

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

            print(f"Training data generated: {filename}")

        except Exception as e:
            print(f"Error generating training data: {e}")

    # =====================================================
    # ANALYTICS AND REPORTING
    # =====================================================

    def get_user_dashboard_data(self) -> Dict[str, Any]:
        """Get data for user dashboard display."""
        try:
            # Recent activity
            recent_activity = self.analytics.get_recent_activity(days=7)

            # Popular projects
            popular_projects = self.analytics.get_popular_projects(days=30, limit=5)

            # Performance metrics
            performance = self.analytics.get_performance_bottlenecks(days=7)

            # User engagement
            engagement = self.analytics.get_user_engagement_metrics(days=30)

            return {
                'recent_activity': recent_activity,
                'popular_projects': popular_projects,
                'performance_metrics': performance,
                'engagement_metrics': engagement,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error generating dashboard data: {e}")
            return {}

    def get_project_insights(self, project_id: str) -> Dict[str, Any]:
        """Get detailed insights for a specific project."""
        try:
            # Project basic info
            project = self.db.get_project(project_id)

            # Latest documents
            latest_docs = self.db.get_latest_documents(project_id)

            # Document versions
            doc_versions = {}
            for doc in latest_docs:
                doc_type = doc['document_type']
                if doc_type:
                    versions = self.db.get_document_versions(project_id, doc_type)
                    doc_versions[doc_type] = versions

            # Access patterns (would need to be implemented)
            access_query = """
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as access_count,
                COUNT(DISTINCT session_id) as unique_users
            FROM user_activities
            WHERE project_id = ? AND timestamp > datetime('now', '-90 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            """
            access_patterns = self.db.execute_query(access_query, [project_id])

            return {
                'project': project,
                'latest_documents': latest_docs,
                'document_versions': doc_versions,
                'access_patterns': access_patterns,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error generating project insights for {project_id}: {e}")
            return {}


# =====================================================
# USAGE EXAMPLES
# =====================================================

def example_usage():
    """Demonstrate the analytics integration in practice."""

    # Initialize the integration
    integration = QuickNavAnalyticsIntegration()

    # Start a user session
    session_id = integration.start_user_session("2.0.0")

    try:
        # Example 1: Navigate to a project
        print("=== Example 1: Project Navigation ===")
        result = integration.navigate_to_project("17741", input_method="keyboard")
        print(f"Navigation result: {result}")

        # Example 2: Search for projects
        print("\n=== Example 2: Project Search ===")
        search_result = integration.navigate_to_project("Test Project", input_method="ai_chat")
        print(f"Search result: {search_result}")

        # Example 3: Open a document (simulated)
        if result['success'] and result['type'] == 'single_result':
            project_path = result['project_path']
            # Find documents in the project
            documents = discover_documents(project_path)
            if documents:
                doc_result = integration.open_document(documents[0])
                print(f"Document open result: {doc_result}")

        # Example 4: Get analytics dashboard data
        print("\n=== Example 4: Analytics Dashboard ===")
        dashboard_data = integration.get_user_dashboard_data()
        print(f"Dashboard metrics: {json.dumps(dashboard_data, indent=2, default=str)}")

        # Example 5: Get project insights
        if result['success'] and result.get('project_id'):
            print(f"\n=== Example 5: Project Insights for {result['project_id']} ===")
            insights = integration.get_project_insights(result['project_id'])
            print(f"Project insights: {json.dumps(insights, indent=2, default=str)}")

        # Example 6: Performance analysis
        print("\n=== Example 6: Performance Analysis ===")
        performance = integration.analytics.get_performance_bottlenecks(days=7)
        print(f"Performance bottlenecks: {json.dumps(performance, indent=2, default=str)}")

    finally:
        # End the session
        integration.end_user_session()

    # Database statistics
    print("\n=== Database Statistics ===")
    stats = integration.db.get_database_stats()
    print(f"Database stats: {json.dumps(stats, indent=2, default=str)}")


if __name__ == "__main__":
    # Run the example
    example_usage()