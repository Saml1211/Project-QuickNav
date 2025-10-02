"""
Module-level convenience functions for document navigation.

These functions provide a simpler interface for GUI integration
without requiring instantiation of the DocumentNavigator class.
"""

import os
import re
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def navigate_to_document(project_path: str, doc_type: str,
                        selection_mode: str = 'auto',
                        project_code: Optional[str] = None,
                        room_filter: Optional[int] = None,
                        co_filter: Optional[int] = None,
                        exclude_archive: bool = True) -> Dict[str, Any]:
    """
    Navigate to a specific document type in a project.

    This is a convenience function for GUI integration that performs
    simple document search without requiring the full DocumentNavigator class.

    Args:
        project_path: Full path to the project directory
        doc_type: Type of document ('lld', 'hld', 'change_order', etc.)
        selection_mode: 'auto' (pick best), 'latest' (newest), or 'choose' (return all)
        project_code: Optional project code for additional filtering
        room_filter: Optional room number filter
        co_filter: Optional change order number filter
        exclude_archive: Exclude archived documents

    Returns:
        Dictionary with:
            - status: 'SUCCESS', 'SELECT', or 'ERROR'
            - path: Document path (if SUCCESS)
            - paths: List of paths (if SELECT)
            - error: Error message (if ERROR)
    """
    try:
        # Validate project path exists
        if not os.path.exists(project_path):
            return {
                'status': 'ERROR',
                'error': f"Project path does not exist: {project_path}"
            }

        # Map doc types to search patterns
        doc_type_patterns = {
            'lld': [r'\blld\b', r'low.?level.?design', r'detailed.?design'],
            'hld': [r'\bhld\b', r'high.?level.?design', r'system.?design'],
            'change_order': [r'change.?order', r'\bco\b', r'variation'],
            'sales_po': [r'purchase.?order', r'\bpo\b', r'sales.?order'],
            'floor_plans': [r'floor.?plan', r'layout', r'\bfp\b'],
            'scope': [r'scope', r'sow', r'statement.?of.?work'],
            'qa_itp': [r'\bqa\b', r'\bitp\b', r'inspection', r'test.?plan'],
            'swms': [r'\bswms\b', r'safe.?work', r'method.?statement'],
            'supplier_quotes': [r'quote', r'quotation', r'supplier'],
            'photos': [r'photo', r'image', r'picture'],
            'cad': [r'\bcad\b', r'drawing', r'dwg', r'dxf']
        }

        patterns = doc_type_patterns.get(doc_type, [doc_type])

        # Search for matching documents
        matching_docs = []
        for root, dirs, files in os.walk(project_path):
            # Skip archive folders if requested
            if exclude_archive and 'archive' in root.lower():
                continue

            for file in files:
                # Skip hidden files and non-documents
                if file.startswith('.') or file.startswith('~'):
                    continue

                file_lower = file.lower()

                # Check if file matches document type
                matches_type = any(re.search(pattern, file_lower, re.IGNORECASE) for pattern in patterns)
                if not matches_type:
                    continue

                # Apply room filter if specified
                if room_filter:
                    if not re.search(rf'\b{room_filter}\b', file):
                        continue

                # Apply CO filter if specified
                if co_filter:
                    if not re.search(rf'\bco[_\s-]*{co_filter}\b', file_lower):
                        continue

                full_path = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(full_path)
                    size = os.path.getsize(full_path)
                    matching_docs.append({
                        'path': full_path,
                        'filename': file,
                        'mtime': mtime,
                        'size': size
                    })
                except OSError as e:
                    logger.warning(f"Could not access file {full_path}: {e}")
                    continue

        if not matching_docs:
            return {
                'status': 'ERROR',
                'error': f"No {doc_type} documents found in project"
            }

        # Sort by modification time (newest first)
        matching_docs.sort(key=lambda x: x['mtime'], reverse=True)

        # Handle selection mode
        if selection_mode == 'choose' or (selection_mode != 'auto' and len(matching_docs) > 1):
            # Choose mode or multiple matches with non-auto mode - return all
            return {
                'status': 'SELECT',
                'paths': [doc['path'] for doc in matching_docs]
            }
        else:
            # Auto mode or single match - return the newest/best
            return {
                'status': 'SUCCESS',
                'path': matching_docs[0]['path']
            }

    except Exception as e:
        logger.exception("Document navigation failed")
        return {
            'status': 'ERROR',
            'error': str(e)
        }
