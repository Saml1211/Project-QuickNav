"""
Document Navigation Module for Project QuickNav

This module provides comprehensive document discovery, parsing, and ranking
functionality for various project document types including:
- Low/High-Level Design (LLD/HLD) documents
- Change Orders and Sales PO Reports
- Floor Plans and Site Photos
- Scope Documents and QA-ITP Reports
- SWMS and Supplier Quotes

Features include:
- Intelligent version/revision parsing (REV 100, 1.03, Rev D, etc.)
- Document type classification and folder mapping
- Weighted ranking and selection algorithms
- Custom root resolution with fallback handling
- Training data integration for corpus validation
"""

import os
import re
import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DocumentParser:
    """Parses document metadata including versions, dates, tags, and project codes."""

    def __init__(self):
        # Version patterns in order of preference: REV > period-based > other
        self.version_patterns = [
            # Primary REV-based patterns (highest priority)
            (r'REV\s*(\d{3})', 'rev_numeric', lambda m: int(m.group(1))),
            (r'REV[\s_]*(\d+)', 'rev_numeric', lambda m: int(m.group(1))),
            (r'Rev\s*(\d{3})', 'rev_numeric', lambda m: int(m.group(1))),
            (r'Rev[\s_]*(\d+)', 'rev_numeric', lambda m: int(m.group(1))),

            # Period-based patterns (secondary priority)
            (r'Rev\.(\d+)\.(\d+)', 'period', lambda m: (int(m.group(1)), int(m.group(2)))),
            (r'(\d+)\.(\d+)', 'period', lambda m: (int(m.group(1)), int(m.group(2)))),

            # Other patterns (lowest priority)
            (r'\((\d+)\)$', 'parenthetical', lambda m: int(m.group(1))),
            (r'_\(([A-Z])\)', 'letter', lambda m: ord(m.group(1)) - ord('A') + 1),
            (r'Rev\.([A-Z])', 'letter', lambda m: ord(m.group(1)) - ord('A') + 1),
        ]

        # Status/tag patterns with priority weights
        self.status_patterns = {
            'AS-BUILT': 2.0,
            'AS BUILT': 2.0,
            'ASBUILT': 2.0,
            'SIGNED': 1.5,
            'FINAL': 1.0,
            'APPROVED': 0.8,
            'DRAFT': -0.5,
            'OLD': -1.0,
        }

        # Date patterns with formats
        self.date_patterns = [
            (r'(\d{2})\.(\d{2})\.(\d{2})', 'DD.MM.YY'),
            (r'(\d{6})', 'YYMMDD'),
            (r'Received\s+(\d{2})-(\d{2})-(\d{4})', 'DD-MM-YYYY'),
            (r'(\d{2})-(\d{2})-(\d{4})', 'DD-MM-YYYY'),
            (r'(\d{2}/\d{2}/\d{4})', 'DD/MM/YYYY'),
        ]

        # Project code pattern (5-digit)
        self.project_code_pattern = re.compile(r'\b(\d{5})\b')

        # Change Order patterns
        self.co_patterns = [
            re.compile(r'\bCO\s*(\d+)\b', re.IGNORECASE),
            re.compile(r'\bChange Order\s*(\d+)\b', re.IGNORECASE),
        ]

        # Room patterns
        self.room_pattern = re.compile(r'Room\s*(\d+)', re.IGNORECASE)

        # Sheet patterns for floor plans
        self.sheet_pattern = re.compile(r'SHEET\s*(\d+)', re.IGNORECASE)

    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse a filename to extract metadata including version, status, dates, etc.

        Args:
            filename: The filename to parse

        Returns:
            Dictionary containing parsed metadata
        """
        result = {
            'original_name': filename,
            'series': self._extract_series(filename),
            'version': None,
            'version_type': None,
            'version_raw': None,
            'status_tags': set(),
            'project_code': None,
            'co_number': None,
            'room_number': None,
            'sheet_number': None,
            'dates': [],
            'is_as_built': False,
            'is_initial': False,
            'archive_indicators': []
        }

        # Extract project code
        project_match = self.project_code_pattern.search(filename)
        if project_match:
            result['project_code'] = project_match.group(1)

        # Extract version information
        version_info = self._parse_version(filename)
        result.update(version_info)

        # Extract status tags
        result['status_tags'] = self._extract_status_tags(filename)
        result['is_as_built'] = any(tag in result['status_tags']
                                  for tag in ['AS-BUILT', 'AS BUILT', 'ASBUILT'])

        # Extract Change Order number
        co_match = self._extract_co_number(filename)
        if co_match:
            result['co_number'] = co_match

        # Extract room number
        room_match = self.room_pattern.search(filename)
        if room_match:
            result['room_number'] = int(room_match.group(1))

        # Extract sheet number
        sheet_match = self.sheet_pattern.search(filename)
        if sheet_match:
            result['sheet_number'] = int(sheet_match.group(1))

        # Extract dates
        result['dates'] = self._extract_dates(filename)

        # Check for archive indicators
        result['archive_indicators'] = self._extract_archive_indicators(filename)

        # Determine if this is an initial version
        result['is_initial'] = self._is_initial_version(result['version'], result['version_type'])

        return result

    def _extract_series(self, filename: str) -> str:
        """Extract the document series by removing version and status information."""
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]

        # Remove version patterns
        for pattern, _, _ in self.version_patterns:
            name_without_ext = re.sub(pattern, '', name_without_ext, flags=re.IGNORECASE)

        # Remove status tags
        for status in self.status_patterns:
            name_without_ext = re.sub(re.escape(status), '', name_without_ext, flags=re.IGNORECASE)

        # Remove dates
        for pattern, _ in self.date_patterns:
            name_without_ext = re.sub(pattern, '', name_without_ext)

        # Clean up extra spaces and separators
        series = re.sub(r'\s*[-_\s]+\s*', ' ', name_without_ext).strip()
        series = re.sub(r'\s+', ' ', series)

        return series

    def _parse_version(self, filename: str) -> Dict[str, Any]:
        """Parse version information from filename."""
        for pattern, version_type, extractor in self.version_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    version = extractor(match)
                    return {
                        'version': version,
                        'version_type': version_type,
                        'version_raw': match.group(0)
                    }
                except (ValueError, IndexError):
                    continue

        return {'version': None, 'version_type': None, 'version_raw': None}

    def _extract_status_tags(self, filename: str) -> set:
        """Extract status tags from filename."""
        tags = set()
        for status in self.status_patterns:
            if re.search(re.escape(status), filename, re.IGNORECASE):
                tags.add(status)
        return tags

    def _extract_co_number(self, filename: str) -> Optional[int]:
        """Extract Change Order number from filename."""
        for pattern in self.co_patterns:
            match = pattern.search(filename)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_dates(self, filename: str) -> List[datetime]:
        """Extract dates from filename."""
        dates = []
        for pattern, format_type in self.date_patterns:
            matches = re.finditer(pattern, filename)
            for match in matches:
                try:
                    date_obj = self._parse_date(match, format_type)
                    if date_obj:
                        dates.append(date_obj)
                except (ValueError, IndexError):
                    continue
        return dates

    def _parse_date(self, match: re.Match, format_type: str) -> Optional[datetime]:
        """Parse a date match object based on format type."""
        try:
            if format_type == 'DD.MM.YY':
                day, month, year = match.groups()
                year = int(year)
                if year < 70:  # Y2K window
                    year += 2000
                else:
                    year += 1900
                return datetime(year, int(month), int(day))

            elif format_type == 'YYMMDD':
                date_str = match.group(1)
                year = int(date_str[:2])
                if year < 70:
                    year += 2000
                else:
                    year += 1900
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                return datetime(year, month, day)

            elif format_type in ['DD-MM-YYYY', 'DD/MM/YYYY']:
                if format_type == 'DD-MM-YYYY' and 'Received' in match.group(0):
                    day, month, year = match.groups()
                    return datetime(int(year), int(month), int(day))
                else:
                    # Handle DD-MM-YYYY or DD/MM/YYYY
                    date_str = match.group(1) if len(match.groups()) == 1 else match.group(0)
                    parts = re.split(r'[-/]', date_str)
                    if len(parts) == 3:
                        day, month, year = map(int, parts)
                        return datetime(year, month, day)

        except (ValueError, IndexError):
            pass

        return None

    def _extract_archive_indicators(self, filename: str) -> List[str]:
        """Extract indicators that this is an archived document."""
        indicators = []
        archive_terms = ['ARCHIVE', 'OLD DRAWINGS', 'OLD', 'BACKUP', 'PREV']

        for term in archive_terms:
            if re.search(re.escape(term), filename, re.IGNORECASE):
                indicators.append(term)

        return indicators

    def _is_initial_version(self, version: Any, version_type: str) -> bool:
        """Determine if this is an initial version."""
        if version is None:
            return False

        if version_type == 'rev_numeric':
            return version == 100 or version == 1
        elif version_type == 'period':
            return version == (1, 0) or version == (1, 1)
        elif version_type in ['parenthetical', 'letter']:
            return version == 1

        return False

    def compare_versions(self, version1: Any, type1: str, version2: Any, type2: str) -> int:
        """
        Compare two versions, returning -1, 0, or 1.

        Returns:
            -1 if version1 < version2
             0 if version1 == version2
             1 if version1 > version2
        """
        # Type priority: rev_numeric > period > parenthetical/letter
        type_priority = {
            'rev_numeric': 3,
            'period': 2,
            'parenthetical': 1,
            'letter': 1
        }

        priority1 = type_priority.get(type1, 0)
        priority2 = type_priority.get(type2, 0)

        if priority1 != priority2:
            return 1 if priority1 > priority2 else -1

        # Same type comparison
        if type1 == 'rev_numeric':
            return self._compare_numeric(version1, version2)
        elif type1 == 'period':
            return self._compare_period(version1, version2)
        elif type1 in ['parenthetical', 'letter']:
            return self._compare_numeric(version1, version2)

        return 0

    def _compare_numeric(self, v1: int, v2: int) -> int:
        """Compare numeric versions."""
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        return 0

    def _compare_period(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
        """Compare period-based versions (major.minor)."""
        major1, minor1 = v1
        major2, minor2 = v2

        if major1 != major2:
            return 1 if major1 > major2 else -1
        if minor1 != minor2:
            return 1 if minor1 > minor2 else -1
        return 0


class DocumentTypeClassifier:
    """Classifies documents into types and maps them to appropriate folders."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default document type configuration."""
        return {
            "doc_types": [
                {
                    "type": "lld",
                    "label": "Low-Level Design",
                    "folders": ["4. System Designs"],
                    "exts": [".vsdx", ".vsd", ".pdf"],
                    "name_includes_any": ["LLD", "Low Level"],
                    "exclude_folders": ["OLD DRAWINGS", "ARCHIVE"],
                    "priority": 100
                },
                {
                    "type": "hld",
                    "label": "High-Level Design",
                    "folders": ["4. System Designs"],
                    "exts": [".vsdx", ".vsd", ".pdf"],
                    "name_includes_any": ["HLD", "High Level"],
                    "exclude_folders": ["OLD DRAWINGS", "ARCHIVE"],
                    "priority": 95
                },
                {
                    "type": "change_order",
                    "label": "Change Orders",
                    "folders": ["2. BOM & Orders", "2. BOM & Orders/Sales & Change Orders"],
                    "exts": [".pdf"],
                    "name_includes_any": ["Change Order", "CO"],
                    "exclude_folders": ["ARCHIVE"],
                    "priority": 90
                },
                {
                    "type": "sales_po",
                    "label": "Sales and PO Reports",
                    "folders": ["2. BOM & Orders"],
                    "exts": [".pdf"],
                    "name_includes_any": ["Sales & PO Report"],
                    "priority": 80
                },
                {
                    "type": "floor_plans",
                    "label": "Floor Plans",
                    "folders": ["5. Floor Plans", "1. Sales Handover/Floor Plans"],
                    "exts": [".pdf"],
                    "name_includes_any": ["GA PLAN", "SHEET", "A-", "Floor Plan"],
                    "exclude_folders": ["OLD DRAWINGS", "ARCHIVE"],
                    "priority": 85
                },
                {
                    "type": "scope",
                    "label": "Scope and Handover",
                    "folders": ["1. Sales Handover"],
                    "exts": [".pdf", ".docx"],
                    "name_includes_any": ["Project Handover Document", "Scope", "Proposal"],
                    "priority": 70
                },
                {
                    "type": "qa_itp",
                    "label": "QA and ITP Reports",
                    "folders": ["3. PMO", "QA-ITP Reports"],
                    "exts": [".pdf"],
                    "name_includes_any": ["QA", "ITP", "Quality"],
                    "priority": 75
                },
                {
                    "type": "swms",
                    "label": "SWMS",
                    "folders": ["3. PMO"],
                    "exts": [".pdf"],
                    "name_includes_any": ["SWMS"],
                    "priority": 78
                },
                {
                    "type": "supplier_quotes",
                    "label": "Supplier Quotes",
                    "folders": ["1. Sales Handover"],
                    "exts": [".pdf", ".xlsx"],
                    "name_includes_any": ["Quote", "Supplier"],
                    "priority": 65
                },
                {
                    "type": "photos",
                    "label": "Site Photos",
                    "folders": ["6. Site Photos"],
                    "exts": [".jpg", ".jpeg", ".png", ".heic", ".tiff"],
                    "priority": 60
                }
            ],
            "ranking_weights": {
                "match_project_code": 5.0,
                "in_preferred_folder": 2.0,
                "not_in_archive_or_old": 1.5,
                "status_as_built": 2.0,
                "status_signed": 1.5,
                "status_draft": -0.5,
                "archive_penalty": -2.0,
                "newer_version": 3.0,
                "newer_date": 1.0,
                "room_match": 1.5,
                "ext_preference": 0.5
            }
        }

    def classify_document(self, filepath: str, doc_type: Optional[str] = None) -> Optional[Dict]:
        """
        Classify a document and return its type configuration.

        Args:
            filepath: Path to the document
            doc_type: Optional specific type to match against

        Returns:
            Document type configuration dict or None if no match
        """
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()

        for type_config in self.config["doc_types"]:
            # If specific type requested, only check that one
            if doc_type and type_config["type"] != doc_type:
                continue

            # Check file extension
            if file_ext not in type_config["exts"]:
                continue

            # Check name patterns
            name_includes = type_config.get("name_includes_any", [])
            if name_includes:
                if not any(pattern.lower() in filename.lower() for pattern in name_includes):
                    continue

            # Check exclusion patterns
            exclude_folders = type_config.get("exclude_folders", [])
            if exclude_folders:
                if any(exclude.lower() in filepath.lower() for exclude in exclude_folders):
                    continue

            return type_config

        return None

    def get_document_folders(self, project_path: str, doc_type: str) -> List[str]:
        """
        Get the list of folders where documents of this type should be found.

        Args:
            project_path: Base project directory path
            doc_type: Document type identifier

        Returns:
            List of full folder paths to search
        """
        folders = []

        for type_config in self.config["doc_types"]:
            if type_config["type"] == doc_type:
                for folder_rel in type_config["folders"]:
                    folder_path = os.path.join(project_path, folder_rel)
                    if os.path.isdir(folder_path):
                        folders.append(folder_path)
                break

        return folders


def get_custom_roots() -> List[str]:
    """
    Get list of custom root paths from environment and configuration.

    Returns:
        List of root paths to try in order
    """
    roots = []

    # Check environment variable
    env_root = os.environ.get('QUICKNAV_ROOT')
    if env_root:
        roots.append(env_root)

    # Check settings file
    try:
        settings_path = os.path.join(os.environ.get('APPDATA', ''), 'QuickNav', 'settings.json')
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                custom_roots = settings.get('custom_roots', [])
                for root in custom_roots:
                    # Expand environment variables
                    expanded_root = os.path.expandvars(root)
                    roots.append(expanded_root)
    except Exception:
        logger.warning("Failed to load custom roots from settings")

    # Add default OneDrive path
    user_profile = os.environ.get("UserProfile")
    if user_profile:
        onedrive_path = os.path.join(user_profile, "OneDrive - Pro AV Solutions")
        roots.append(onedrive_path)

    return roots


def resolve_project_root() -> str:
    """
    Resolve the project root directory using custom roots and fallbacks.

    Returns:
        Path to the Project Folders directory

    Raises:
        FileNotFoundError: If no valid root is found
    """
    # Import here to avoid circular imports
    from .find_project_path import get_project_folders, setup_test_environment

    roots = get_custom_roots()

    for root in roots:
        try:
            if os.path.isdir(root):
                # Try to get project folders from this root
                project_folders = get_project_folders(root)
                if os.path.isdir(project_folders):
                    logger.info(f"Using project root: {project_folders}")
                    return project_folders
        except Exception as e:
            logger.debug(f"Failed to use root {root}: {e}")
            continue

    # Fall back to test environment
    logger.warning("No valid project root found, using test environment")
    return setup_test_environment()


class DocumentScanner:
    """Scans project directories for documents with metadata extraction."""

    def __init__(self, parser: DocumentParser, classifier: DocumentTypeClassifier):
        self.parser = parser
        self.classifier = classifier

    def scan_documents(self, project_path: str, doc_type: Optional[str] = None,
                      exclude_archive: bool = True, max_depth: int = 5) -> List[Dict]:
        """
        Scan for documents in a project directory.

        Args:
            project_path: Path to the project directory
            doc_type: Optional specific document type to filter for
            exclude_archive: Whether to exclude archived documents
            max_depth: Maximum directory depth to traverse

        Returns:
            List of document metadata dictionaries
        """
        documents = []

        # Get folders to search based on document type
        if doc_type:
            search_folders = self.classifier.get_document_folders(project_path, doc_type)
            if not search_folders:
                # If no specific folders found, search entire project
                search_folders = [project_path]
        else:
            search_folders = [project_path]

        for folder in search_folders:
            if not os.path.isdir(folder):
                continue

            try:
                docs = self._scan_folder_recursive(
                    folder, doc_type, exclude_archive, max_depth, 0
                )
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Error scanning folder {folder}: {e}")

        return documents

    def _scan_folder_recursive(self, folder: str, doc_type: Optional[str],
                              exclude_archive: bool, max_depth: int,
                              current_depth: int) -> List[Dict]:
        """Recursively scan a folder for documents."""
        documents = []

        if current_depth > max_depth:
            return documents

        try:
            entries = os.listdir(folder)
        except (PermissionError, FileNotFoundError):
            return documents

        for entry in entries:
            full_path = os.path.join(folder, entry)

            if os.path.isfile(full_path):
                # Check if this is a document we're interested in
                type_config = self.classifier.classify_document(full_path, doc_type)
                if type_config:
                    # Parse document metadata
                    metadata = self.parser.parse_filename(entry)

                    # Add file system metadata
                    metadata.update({
                        'filepath': full_path,
                        'filename': entry,
                        'doc_type': type_config['type'],
                        'doc_type_config': type_config,
                        'folder_path': folder,
                        'relative_path': os.path.relpath(full_path, folder),
                        'file_size': os.path.getsize(full_path),
                        'mod_time': datetime.fromtimestamp(os.path.getmtime(full_path)),
                        'in_archive': self._is_in_archive_folder(full_path),
                        'in_preferred_folder': self._is_in_preferred_folder(folder, type_config)
                    })

                    # Skip archived documents if requested
                    if exclude_archive and (metadata['in_archive'] or metadata['archive_indicators']):
                        continue

                    documents.append(metadata)

            elif os.path.isdir(full_path):
                # Skip deep nesting unless it's room-specific
                if current_depth < max_depth - 1 or self._is_room_folder(entry):
                    subdocs = self._scan_folder_recursive(
                        full_path, doc_type, exclude_archive, max_depth, current_depth + 1
                    )
                    documents.extend(subdocs)

        return documents

    def _is_in_archive_folder(self, filepath: str) -> bool:
        """Check if file is in an archive/old drawings folder."""
        archive_indicators = ['ARCHIVE', 'OLD DRAWINGS', 'OLD', 'BACKUP']
        path_upper = filepath.upper()
        return any(indicator in path_upper for indicator in archive_indicators)

    def _is_in_preferred_folder(self, folder: str, type_config: Dict) -> bool:
        """Check if folder matches the preferred folders for this document type."""
        folder_name = os.path.basename(folder)
        preferred_folders = type_config.get('folders', [])
        return any(preferred in folder for preferred in preferred_folders)

    def _is_room_folder(self, folder_name: str) -> bool:
        """Check if this is a room-specific folder."""
        return bool(re.match(r'Room\s*\d+', folder_name, re.IGNORECASE))


class DocumentRanker:
    """Ranks and selects documents based on configurable weights."""

    def __init__(self, classifier: DocumentTypeClassifier, parser: DocumentParser):
        self.classifier = classifier
        self.parser = parser
        self.weights = classifier.config.get('ranking_weights', {})

    def rank_documents(self, documents: List[Dict], project_code: Optional[str] = None,
                      room_filter: Optional[int] = None) -> List[Dict]:
        """
        Rank documents by relevance and quality.

        Args:
            documents: List of document metadata
            project_code: Optional project code for boosting matches
            room_filter: Optional room number for filtering

        Returns:
            Sorted list of documents with ranking scores
        """
        if not documents:
            return []

        # Filter by room if specified
        if room_filter is not None:
            documents = [doc for doc in documents
                        if doc.get('room_number') == room_filter]

        # Calculate scores
        for doc in documents:
            doc['ranking_score'] = self._calculate_score(doc, project_code)

        # Sort by score (descending) and then by modification time (descending)
        documents.sort(key=lambda x: (-x['ranking_score'], -x['mod_time'].timestamp()))

        return documents

    def _calculate_score(self, doc: Dict, project_code: Optional[str]) -> float:
        """Calculate ranking score for a document."""
        score = 0.0

        # Project code match
        if project_code and doc.get('project_code') == project_code:
            score += self.weights.get('match_project_code', 5.0)

        # Preferred folder bonus
        if doc.get('in_preferred_folder'):
            score += self.weights.get('in_preferred_folder', 2.0)

        # Archive penalty
        if doc.get('in_archive'):
            score += self.weights.get('archive_penalty', -2.0)

        # Status bonuses/penalties
        status_tags = doc.get('status_tags', set())
        if any(tag in status_tags for tag in ['AS-BUILT', 'AS BUILT', 'ASBUILT']):
            score += self.weights.get('status_as_built', 2.0)
        elif 'SIGNED' in status_tags:
            score += self.weights.get('status_signed', 1.5)
        elif 'DRAFT' in status_tags:
            score += self.weights.get('status_draft', -0.5)

        # Version bonus (newer versions get higher scores)
        if doc.get('version') is not None:
            score += self._calculate_version_bonus(doc)

        # Date bonus (more recent dates get higher scores)
        if doc.get('dates'):
            most_recent = max(doc['dates'])
            days_old = (datetime.now() - most_recent).days
            # Bonus decreases with age, max 1.0 points
            date_bonus = max(0, self.weights.get('newer_date', 1.0) * (1 - days_old / 365))
            score += date_bonus

        # Extension preference (prioritize native formats)
        ext = os.path.splitext(doc['filename'])[1].lower()
        if ext in ['.vsdx', '.vsd']:  # Native Visio formats
            score += self.weights.get('ext_preference', 0.5)

        return score

    def _calculate_version_bonus(self, doc: Dict) -> float:
        """Calculate version-based bonus score."""
        version = doc.get('version')
        version_type = doc.get('version_type')

        if version is None:
            return 0.0

        base_bonus = self.weights.get('newer_version', 3.0)

        # REV-based versions get full bonus based on number
        if version_type == 'rev_numeric':
            if version >= 200:
                return base_bonus * 1.0  # Full bonus for 200+
            elif version >= 100:
                return base_bonus * 0.8  # Reduced bonus for 100-199
            else:
                return base_bonus * 0.5  # Minimal bonus for < 100

        # Period-based versions
        elif version_type == 'period':
            major, minor = version
            return base_bonus * (0.5 + major * 0.2 + minor * 0.1)

        # Other version types get minimal bonus
        else:
            return base_bonus * 0.3

    def group_by_series(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Group documents by their series (base name without version/status)."""
        groups = {}

        for doc in documents:
            series = doc.get('series', doc['filename'])
            if series not in groups:
                groups[series] = []
            groups[series].append(doc)

        # Sort each group by version
        for series, docs in groups.items():
            docs.sort(key=lambda x: self._get_version_sort_key(x), reverse=True)

        return groups

    def _get_version_sort_key(self, doc: Dict) -> Tuple:
        """Get sort key for version comparison."""
        version = doc.get('version')
        version_type = doc.get('version_type')
        mod_time = doc.get('mod_time', datetime.min)

        if version is None:
            return (0, 0, mod_time.timestamp())

        # Type priority
        type_priority = {
            'rev_numeric': 3,
            'period': 2,
            'parenthetical': 1,
            'letter': 1
        }.get(version_type, 0)

        # Version value
        if version_type == 'rev_numeric':
            version_value = version
        elif version_type == 'period':
            major, minor = version
            version_value = major * 100 + minor
        else:
            version_value = version

        return (type_priority, version_value, mod_time.timestamp())

    def select_best_document(self, documents: List[Dict],
                           selection_mode: str = 'auto') -> Union[str, List[str]]:
        """
        Select the best document(s) based on ranking.

        Args:
            documents: Ranked list of documents
            selection_mode: 'auto', 'latest', or 'choose'

        Returns:
            Single path for auto selection, or list of paths for choose mode
        """
        if not documents:
            return []

        if selection_mode == 'auto':
            # Auto-select if top document is significantly better
            if len(documents) == 1:
                return documents[0]['filepath']

            top_score = documents[0]['ranking_score']
            second_score = documents[1]['ranking_score'] if len(documents) > 1 else 0

            # Auto-select if top score is 30% higher than second
            if top_score > second_score * 1.3:
                return documents[0]['filepath']
            else:
                # Return top candidates for user selection
                return [doc['filepath'] for doc in documents[:5]]

        elif selection_mode == 'latest':
            # Group by series and return latest from each
            groups = self.group_by_series(documents)
            latest_docs = []
            for series_docs in groups.values():
                if series_docs:
                    latest_docs.append(series_docs[0]['filepath'])
            return latest_docs

        else:  # choose mode
            return [doc['filepath'] for doc in documents]


def navigate_to_document(project_path: str, doc_type: str,
                        selection_mode: str = 'auto',
                        project_code: Optional[str] = None,
                        room_filter: Optional[int] = None,
                        co_filter: Optional[int] = None,
                        exclude_archive: bool = True) -> Union[str, List[str]]:
    """
    Main function to navigate to project documents.

    Args:
        project_path: Path to the project directory
        doc_type: Type of document to find
        selection_mode: 'auto', 'latest', or 'choose'
        project_code: Optional project code for filtering
        room_filter: Optional room number filter
        co_filter: Optional change order number filter
        exclude_archive: Whether to exclude archived documents

    Returns:
        Path to selected document or list of candidate paths
    """
    # Initialize components
    parser = DocumentParser()
    classifier = DocumentTypeClassifier()
    scanner = DocumentScanner(parser, classifier)
    ranker = DocumentRanker(classifier, parser)

    # Scan for documents
    documents = scanner.scan_documents(
        project_path, doc_type, exclude_archive
    )

    if not documents:
        return f"ERROR:No {doc_type} documents found in project"

    # Apply additional filters
    if co_filter is not None:
        documents = [doc for doc in documents
                    if doc.get('co_number') == co_filter]

    if not documents:
        return f"ERROR:No documents found matching filter criteria"

    # Rank documents
    ranked_docs = ranker.rank_documents(documents, project_code, room_filter)

    # Select best document(s)
    result = ranker.select_best_document(ranked_docs, selection_mode)

    if isinstance(result, str):
        if result.startswith("ERROR:"):
            return result
        return f"SUCCESS:{result}"
    else:
        return "SELECT:" + "|".join(result)