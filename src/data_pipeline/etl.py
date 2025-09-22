"""
ETL Pipeline for Project QuickNav

Implements batch processing pipeline for document indexing, metadata extraction,
and feature engineering. Integrates with existing find_project_path.py and
doc_navigator.py modules.
"""

import os
import hashlib
import json
import time
import asyncio
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .config import get_config
from ..find_project_path import (
    get_onedrive_folder,
    get_project_folders,
    discover_documents
)
from ..doc_navigator import DocumentParser, DocumentTypeClassifier

logger = logging.getLogger(__name__)


class FileChangeDetector:
    """Detects changes in project files for incremental processing"""

    def __init__(self, state_file: str = "data/file_state.json"):
        self.state_file = state_file
        self.file_state = self._load_state()

    def _load_state(self) -> Dict[str, Dict]:
        """Load previous file state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load file state: {e}")
        return {}

    def _save_state(self):
        """Save current file state to disk"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.file_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save file state: {e}")

    def _get_file_signature(self, filepath: str) -> Dict[str, Any]:
        """Get file signature for change detection"""
        try:
            stat = os.stat(filepath)
            with open(filepath, 'rb') as f:
                # Read first 1KB for fast hashing
                chunk = f.read(1024)
                file_hash = hashlib.md5(chunk).hexdigest()

            return {
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'hash': file_hash
            }
        except Exception as e:
            logger.debug(f"Failed to get signature for {filepath}: {e}")
            return {}

    def get_changes_since(self, last_sync_time: datetime) -> Iterator[Dict]:
        """Get file changes since last sync"""
        project_root = get_project_folders(get_onedrive_folder())

        for root, dirs, files in os.walk(project_root):
            # Skip archive and temp directories
            dirs[:] = [d for d in dirs if not any(skip in d.upper()
                      for skip in ['ARCHIVE', 'OLD DRAWINGS', 'TEMP', '.GIT'])]

            for file in files:
                filepath = os.path.join(root, file)

                # Skip non-document files
                if not self._is_document_file(filepath):
                    continue

                try:
                    current_sig = self._get_file_signature(filepath)
                    if not current_sig:
                        continue

                    previous_sig = self.file_state.get(filepath, {})

                    # Determine change type
                    change_type = None
                    if not previous_sig:
                        change_type = 'created'
                    elif (current_sig['size'] != previous_sig.get('size') or
                          current_sig['modified'] != previous_sig.get('modified') or
                          current_sig['hash'] != previous_sig.get('hash')):
                        change_type = 'modified'

                    if change_type:
                        yield {
                            'path': filepath,
                            'type': change_type,
                            'timestamp': datetime.fromtimestamp(current_sig['modified']),
                            'size': current_sig['size']
                        }

                        # Update state
                        self.file_state[filepath] = current_sig

                except Exception as e:
                    logger.debug(f"Error processing {filepath}: {e}")

        # Check for deleted files
        for filepath in list(self.file_state.keys()):
            if not os.path.exists(filepath):
                yield {
                    'path': filepath,
                    'type': 'deleted',
                    'timestamp': datetime.utcnow(),
                    'size': 0
                }
                del self.file_state[filepath]

        self._save_state()

    def _is_document_file(self, filepath: str) -> bool:
        """Check if file is a document we care about"""
        config = get_config()
        ext = os.path.splitext(filepath)[1].lower()
        return ext in config.etl.supported_extensions

    def mark_processed(self, filepath: str):
        """Mark file as processed"""
        if os.path.exists(filepath):
            self.file_state[filepath] = self._get_file_signature(filepath)
            self._save_state()


class OneDriveExtractor:
    """Extracts project and document metadata from OneDrive structure"""

    def __init__(self):
        self.config = get_config()
        self.change_detector = FileChangeDetector()
        self.parser = DocumentParser()
        self.classifier = DocumentTypeClassifier()

    def extract_incremental(self, last_sync_time: datetime) -> Iterator[Dict]:
        """Extract only changed files since last sync"""
        logger.info(f"Starting incremental extraction since {last_sync_time}")

        changes = self.change_detector.get_changes_since(last_sync_time)
        processed_count = 0

        for change in changes:
            try:
                if change['type'] == 'deleted':
                    yield {
                        'record_type': 'document_deletion',
                        'file_path': change['path'],
                        'timestamp': change['timestamp']
                    }
                else:
                    metadata = self._extract_file_metadata(change['path'])
                    if metadata:
                        metadata.update({
                            'change_type': change['type'],
                            'change_timestamp': change['timestamp']
                        })
                        yield metadata

                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} file changes")

            except Exception as e:
                logger.error(f"Error processing change {change}: {e}")

        logger.info(f"Incremental extraction completed: {processed_count} changes")

    def extract_full_scan(self) -> Iterator[Dict]:
        """Full project structure scan for initial load"""
        logger.info("Starting full project scan")

        project_root = get_project_folders(get_onedrive_folder())
        processed_count = 0

        # Extract project-level metadata
        for project_path in self._scan_project_directories(project_root):
            try:
                project_data = self._extract_project_metadata(project_path)
                if project_data:
                    yield project_data
                    processed_count += 1

                # Extract document metadata for this project
                documents = discover_documents(project_path)
                for doc_path in documents:
                    try:
                        doc_data = self._extract_file_metadata(doc_path)
                        if doc_data:
                            yield doc_data
                            processed_count += 1

                        if processed_count % 500 == 0:
                            logger.info(f"Processed {processed_count} items")

                    except Exception as e:
                        logger.error(f"Error processing document {doc_path}: {e}")

            except Exception as e:
                logger.error(f"Error processing project {project_path}: {e}")

        logger.info(f"Full scan completed: {processed_count} items processed")

    def _scan_project_directories(self, project_root: str) -> Iterator[str]:
        """Scan for project directories"""
        try:
            for range_folder in os.listdir(project_root):
                range_path = os.path.join(project_root, range_folder)

                if not os.path.isdir(range_path):
                    continue

                # Skip non-range folders
                if not range_folder.replace(' ', '').replace('-', '').isdigit():
                    continue

                for project_folder in os.listdir(range_path):
                    project_path = os.path.join(range_path, project_folder)

                    if os.path.isdir(project_path):
                        yield project_path

        except Exception as e:
            logger.error(f"Error scanning project directories: {e}")

    def _extract_project_metadata(self, project_path: str) -> Dict[str, Any]:
        """Extract metadata for a project directory"""
        project_name = os.path.basename(project_path)

        # Extract project ID from folder name (e.g., "17741 - Test Project")
        project_id = None
        if ' - ' in project_name:
            potential_id = project_name.split(' - ')[0]
            if potential_id.isdigit() and len(potential_id) == 5:
                project_id = potential_id

        # Scan folder structure
        folder_structure = self._analyze_folder_structure(project_path)

        # Get project statistics
        stats = self._get_project_statistics(project_path)

        return {
            'record_type': 'project',
            'project_id': project_id,
            'project_name': project_name,
            'project_path': project_path,
            'folder_structure': folder_structure,
            'statistics': stats,
            'created_date': datetime.fromtimestamp(os.path.getctime(project_path)),
            'modified_date': datetime.fromtimestamp(os.path.getmtime(project_path)),
            'extracted_at': datetime.utcnow()
        }

    def _extract_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract metadata for a single file"""
        try:
            # Basic file info
            stat = os.stat(file_path)
            filename = os.path.basename(file_path)

            # Parse filename metadata using existing parser
            parsed_metadata = self.parser.parse_filename(filename)

            # Classify document type
            doc_classification = self.classifier.classify_document(file_path)

            # Extract project ID from path
            project_id = self._extract_project_id_from_path(file_path)

            return {
                'record_type': 'document',
                'document_id': self._generate_document_id(file_path),
                'project_id': project_id,
                'file_path': file_path,
                'file_name': filename,
                'file_extension': os.path.splitext(filename)[1].lower(),
                'file_size': stat.st_size,
                'created_date': datetime.fromtimestamp(stat.st_ctime),
                'modified_date': datetime.fromtimestamp(stat.st_mtime),
                'parsed_metadata': parsed_metadata,
                'doc_classification': doc_classification,
                'relative_path': os.path.relpath(file_path,
                                               get_project_folders(get_onedrive_folder())),
                'extracted_at': datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Failed to extract metadata for {file_path}: {e}")
            return None

    def _analyze_folder_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze project folder structure"""
        structure = {
            'folders': [],
            'depth': 0,
            'has_standard_structure': False
        }

        standard_folders = [
            '1. Sales Handover',
            '2. BOM & Orders',
            '3. PMO',
            '4. System Designs',
            '5. Floor Plans',
            '6. Site Photos'
        ]

        try:
            for root, dirs, files in os.walk(project_path):
                depth = root.replace(project_path, '').count(os.sep)
                structure['depth'] = max(structure['depth'], depth)

                for folder in dirs:
                    folder_info = {
                        'name': folder,
                        'path': os.path.relpath(os.path.join(root, folder), project_path),
                        'depth': depth + 1
                    }
                    structure['folders'].append(folder_info)

            # Check for standard folder structure
            top_level_folders = [f['name'] for f in structure['folders'] if f['depth'] == 1]
            standard_folder_count = sum(1 for sf in standard_folders
                                      if any(sf in tlf for tlf in top_level_folders))
            structure['has_standard_structure'] = standard_folder_count >= 3

        except Exception as e:
            logger.error(f"Error analyzing folder structure for {project_path}: {e}")

        return structure

    def _get_project_statistics(self, project_path: str) -> Dict[str, Any]:
        """Get project-level statistics"""
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'document_count': 0,
            'doc_types': {},
            'last_activity': None
        }

        try:
            latest_mtime = 0

            for root, dirs, files in os.walk(project_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        stat = os.stat(file_path)
                        stats['total_files'] += 1
                        stats['total_size_bytes'] += stat.st_size
                        latest_mtime = max(latest_mtime, stat.st_mtime)

                        # Check if it's a document
                        if self.change_detector._is_document_file(file_path):
                            stats['document_count'] += 1

                            # Count document types
                            doc_type = self.classifier.classify_document(file_path)
                            if doc_type:
                                type_name = doc_type['type']
                                stats['doc_types'][type_name] = stats['doc_types'].get(type_name, 0) + 1

                    except Exception as e:
                        logger.debug(f"Error getting stats for {file_path}: {e}")

            if latest_mtime > 0:
                stats['last_activity'] = datetime.fromtimestamp(latest_mtime)

        except Exception as e:
            logger.error(f"Error getting statistics for {project_path}: {e}")

        return stats

    def _extract_project_id_from_path(self, file_path: str) -> Optional[str]:
        """Extract project ID from file path"""
        path_parts = Path(file_path).parts

        for part in path_parts:
            if ' - ' in part:
                potential_id = part.split(' - ')[0]
                if potential_id.isdigit() and len(potential_id) == 5:
                    return potential_id
        return None

    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        # Use hash of normalized path for consistency
        normalized_path = os.path.normpath(file_path).lower()
        return hashlib.md5(normalized_path.encode()).hexdigest()


class DocumentTransformer:
    """Transforms raw document data into analytics-ready format"""

    def __init__(self):
        self.config = get_config()

    def transform_batch(self, raw_records: List[Dict]) -> List[Dict]:
        """Transform a batch of raw records"""
        transformed = []

        for record in raw_records:
            try:
                if record['record_type'] == 'project':
                    transformed_record = self._transform_project(record)
                elif record['record_type'] == 'document':
                    transformed_record = self._transform_document(record)
                elif record['record_type'] == 'document_deletion':
                    transformed_record = self._transform_deletion(record)
                else:
                    logger.warning(f"Unknown record type: {record['record_type']}")
                    continue

                if transformed_record:
                    transformed.append(transformed_record)

            except Exception as e:
                logger.error(f"Error transforming record: {e}")

        return transformed

    def _transform_project(self, record: Dict) -> Dict:
        """Transform project record"""
        return {
            'project_id': record['project_id'],
            'project_name': record['project_name'],
            'project_path': record['project_path'],
            'folder_structure': json.dumps(record['folder_structure']),
            'total_files': record['statistics']['total_files'],
            'total_size_bytes': record['statistics']['total_size_bytes'],
            'document_count': record['statistics']['document_count'],
            'doc_types_json': json.dumps(record['statistics']['doc_types']),
            'has_standard_structure': record['folder_structure']['has_standard_structure'],
            'folder_depth': record['folder_structure']['depth'],
            'created_date': record['created_date'],
            'modified_date': record['modified_date'],
            'last_activity': record['statistics'].get('last_activity'),
            'last_updated': datetime.utcnow()
        }

    def _transform_document(self, record: Dict) -> Dict:
        """Transform document record"""
        # Extract features from parsed metadata
        parsed = record.get('parsed_metadata', {})
        classification = record.get('doc_classification', {})

        return {
            'document_id': record['document_id'],
            'project_id': record['project_id'],
            'file_path': record['file_path'],
            'file_name': record['file_name'],
            'file_extension': record['file_extension'],
            'file_size': record['file_size'],
            'doc_type': classification.get('type', 'unknown') if classification else 'unknown',
            'doc_type_priority': classification.get('priority', 0) if classification else 0,

            # Parsed metadata fields
            'version_raw': parsed.get('version_raw'),
            'version_type': parsed.get('version_type'),
            'version_numeric': self._normalize_version(parsed.get('version'), parsed.get('version_type')),
            'status_tags': json.dumps(list(parsed.get('status_tags', set()))),
            'is_as_built': parsed.get('is_as_built', False),
            'is_initial': parsed.get('is_initial', False),
            'project_code': parsed.get('project_code'),
            'co_number': parsed.get('co_number'),
            'room_number': parsed.get('room_number'),
            'sheet_number': parsed.get('sheet_number'),
            'dates_json': json.dumps([d.isoformat() for d in parsed.get('dates', [])]),
            'archive_indicators': json.dumps(parsed.get('archive_indicators', [])),

            # File system metadata
            'relative_path': record['relative_path'],
            'folder_name': os.path.dirname(record['relative_path']),
            'created_date': record['created_date'],
            'modified_date': record['modified_date'],
            'extracted_at': record['extracted_at'],
            'last_updated': datetime.utcnow()
        }

    def _transform_deletion(self, record: Dict) -> Dict:
        """Transform deletion record"""
        return {
            'file_path': record['file_path'],
            'deleted_at': record['timestamp'],
            'operation': 'delete'
        }

    def _normalize_version(self, version: Any, version_type: str) -> Optional[float]:
        """Normalize version to numeric value for comparison"""
        if version is None:
            return None

        try:
            if version_type == 'rev_numeric':
                return float(version)
            elif version_type == 'period':
                major, minor = version
                return float(f"{major}.{minor:02d}")
            elif version_type in ['parenthetical', 'letter']:
                return float(version)
            else:
                return None
        except (ValueError, TypeError):
            return None


class DuckDBLoader:
    """Loads transformed data into DuckDB analytics database"""

    def __init__(self, db_path: str = None):
        self.config = get_config()
        self.db_path = db_path or self.config.database.duckdb_path
        self.db = None
        self._setup_database()

    def _setup_database(self):
        """Setup database connection and schema"""
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            self.db = duckdb.connect(self.db_path)

            # Enable WAL mode for better concurrency
            if self.config.database.enable_wal_mode:
                self.db.execute("PRAGMA journal_mode=WAL")

            self._create_schema()
            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise

    def _create_schema(self):
        """Create database schema"""
        schema_sql = """
        -- Create schema
        CREATE SCHEMA IF NOT EXISTS quicknav;

        -- Projects table
        CREATE TABLE IF NOT EXISTS quicknav.projects (
            project_id VARCHAR PRIMARY KEY,
            project_name VARCHAR NOT NULL,
            project_path VARCHAR NOT NULL,
            folder_structure TEXT,
            total_files INTEGER DEFAULT 0,
            total_size_bytes BIGINT DEFAULT 0,
            document_count INTEGER DEFAULT 0,
            doc_types_json TEXT,
            has_standard_structure BOOLEAN DEFAULT FALSE,
            folder_depth INTEGER DEFAULT 0,
            created_date TIMESTAMP,
            modified_date TIMESTAMP,
            last_activity TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Documents table
        CREATE TABLE IF NOT EXISTS quicknav.documents (
            document_id VARCHAR PRIMARY KEY,
            project_id VARCHAR,
            file_path VARCHAR NOT NULL,
            file_name VARCHAR NOT NULL,
            file_extension VARCHAR,
            file_size BIGINT DEFAULT 0,
            doc_type VARCHAR DEFAULT 'unknown',
            doc_type_priority INTEGER DEFAULT 0,
            version_raw VARCHAR,
            version_type VARCHAR,
            version_numeric DOUBLE,
            status_tags TEXT,
            is_as_built BOOLEAN DEFAULT FALSE,
            is_initial BOOLEAN DEFAULT FALSE,
            project_code VARCHAR,
            co_number INTEGER,
            room_number INTEGER,
            sheet_number INTEGER,
            dates_json TEXT,
            archive_indicators TEXT,
            relative_path VARCHAR,
            folder_name VARCHAR,
            created_date TIMESTAMP,
            modified_date TIMESTAMP,
            extracted_at TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES quicknav.projects(project_id)
        );

        -- Indices for performance
        CREATE INDEX IF NOT EXISTS idx_projects_id ON quicknav.projects(project_id);
        CREATE INDEX IF NOT EXISTS idx_projects_updated ON quicknav.projects(last_updated);
        CREATE INDEX IF NOT EXISTS idx_documents_project ON quicknav.documents(project_id);
        CREATE INDEX IF NOT EXISTS idx_documents_type ON quicknav.documents(doc_type);
        CREATE INDEX IF NOT EXISTS idx_documents_updated ON quicknav.documents(last_updated);
        CREATE INDEX IF NOT EXISTS idx_documents_path ON quicknav.documents(file_path);
        """

        self.db.executescript(schema_sql)

    def load_batch(self, records: List[Dict], table_type: str):
        """Load batch of records with upsert logic"""
        if not records:
            return

        try:
            if table_type == 'projects':
                self._load_projects(records)
            elif table_type == 'documents':
                self._load_documents(records)
            elif table_type == 'deletions':
                self._handle_deletions(records)
            else:
                raise ValueError(f"Unknown table type: {table_type}")

            logger.info(f"Loaded {len(records)} {table_type} records")

        except Exception as e:
            logger.error(f"Failed to load batch: {e}")
            raise

    def _load_projects(self, records: List[Dict]):
        """Load project records"""
        df = pd.DataFrame(records)

        # Use DuckDB's INSERT OR REPLACE for upsert
        self.db.execute("""
            INSERT OR REPLACE INTO quicknav.projects
            SELECT * FROM df
        """)

    def _load_documents(self, records: List[Dict]):
        """Load document records"""
        df = pd.DataFrame(records)

        # Use DuckDB's INSERT OR REPLACE for upsert
        self.db.execute("""
            INSERT OR REPLACE INTO quicknav.documents
            SELECT * FROM df
        """)

    def _handle_deletions(self, records: List[Dict]):
        """Handle document deletions"""
        for record in records:
            self.db.execute("""
                DELETE FROM quicknav.documents
                WHERE file_path = ?
            """, [record['file_path']])

    def create_analytics_views(self):
        """Create materialized views for analytics"""
        views_sql = """
        -- Project summary view
        CREATE OR REPLACE VIEW quicknav.project_summary AS
        SELECT
            p.project_id,
            p.project_name,
            p.project_path,
            p.total_files,
            p.total_size_bytes,
            p.document_count,
            p.has_standard_structure,
            p.folder_depth,
            p.last_activity,
            COUNT(d.document_id) as indexed_documents,
            COUNT(DISTINCT d.doc_type) as doc_types_count,
            MAX(d.modified_date) as latest_document_date,
            SUM(CASE WHEN d.is_as_built THEN 1 ELSE 0 END) as as_built_count
        FROM quicknav.projects p
        LEFT JOIN quicknav.documents d ON p.project_id = d.project_id
        GROUP BY p.project_id, p.project_name, p.project_path,
                 p.total_files, p.total_size_bytes, p.document_count,
                 p.has_standard_structure, p.folder_depth, p.last_activity;

        -- Document type analysis view
        CREATE OR REPLACE VIEW quicknav.document_type_analysis AS
        SELECT
            doc_type,
            COUNT(*) as document_count,
            AVG(file_size) as avg_file_size,
            SUM(file_size) as total_size,
            COUNT(DISTINCT project_id) as projects_with_type,
            AVG(version_numeric) as avg_version,
            SUM(CASE WHEN is_as_built THEN 1 ELSE 0 END) as as_built_count
        FROM quicknav.documents
        WHERE doc_type != 'unknown'
        GROUP BY doc_type
        ORDER BY document_count DESC;

        -- Recent activity view
        CREATE OR REPLACE VIEW quicknav.recent_activity AS
        SELECT
            'project' as activity_type,
            project_id as entity_id,
            project_name as entity_name,
            last_updated as activity_date
        FROM quicknav.projects
        WHERE last_updated > CURRENT_DATE - INTERVAL '7 days'
        UNION ALL
        SELECT
            'document' as activity_type,
            document_id as entity_id,
            file_name as entity_name,
            last_updated as activity_date
        FROM quicknav.documents
        WHERE last_updated > CURRENT_DATE - INTERVAL '7 days'
        ORDER BY activity_date DESC;
        """

        self.db.executescript(views_sql)

    def get_load_statistics(self) -> Dict[str, Any]:
        """Get loading statistics"""
        stats = {}

        try:
            # Project statistics
            project_stats = self.db.execute("""
                SELECT
                    COUNT(*) as total_projects,
                    SUM(total_files) as total_files,
                    SUM(total_size_bytes) as total_size_bytes,
                    AVG(document_count) as avg_documents_per_project
                FROM quicknav.projects
            """).fetchone()

            stats['projects'] = {
                'total_projects': project_stats[0],
                'total_files': project_stats[1],
                'total_size_bytes': project_stats[2],
                'avg_documents_per_project': round(project_stats[3], 2) if project_stats[3] else 0
            }

            # Document statistics
            doc_stats = self.db.execute("""
                SELECT
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT doc_type) as unique_doc_types,
                    COUNT(DISTINCT project_id) as projects_with_documents,
                    AVG(file_size) as avg_file_size
                FROM quicknav.documents
            """).fetchone()

            stats['documents'] = {
                'total_documents': doc_stats[0],
                'unique_doc_types': doc_stats[1],
                'projects_with_documents': doc_stats[2],
                'avg_file_size': round(doc_stats[3], 2) if doc_stats[3] else 0
            }

            # Recent activity
            recent_activity = self.db.execute("""
                SELECT COUNT(*)
                FROM quicknav.documents
                WHERE last_updated > CURRENT_DATE - INTERVAL '24 hours'
            """).fetchone()

            stats['recent_activity'] = {
                'documents_updated_24h': recent_activity[0]
            }

        except Exception as e:
            logger.error(f"Error getting load statistics: {e}")

        return stats

    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()


class ETLPipeline:
    """Main ETL pipeline orchestrator"""

    def __init__(self):
        self.config = get_config()
        self.extractor = OneDriveExtractor()
        self.transformer = DocumentTransformer()
        self.loader = DuckDBLoader()

    def run_incremental(self, last_sync_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Run incremental ETL pipeline"""
        if last_sync_time is None:
            last_sync_time = datetime.utcnow() - timedelta(hours=1)

        logger.info(f"Starting incremental ETL pipeline since {last_sync_time}")
        start_time = time.time()

        stats = {
            'start_time': datetime.utcnow(),
            'records_processed': 0,
            'projects_updated': 0,
            'documents_updated': 0,
            'deletions_processed': 0,
            'errors': 0
        }

        try:
            # Extract changes
            raw_records = list(self.extractor.extract_incremental(last_sync_time))
            logger.info(f"Extracted {len(raw_records)} changed records")

            # Process in batches
            batch_size = self.config.etl.batch_size
            for i in range(0, len(raw_records), batch_size):
                batch = raw_records[i:i + batch_size]

                try:
                    # Group by record type
                    projects = [r for r in batch if r['record_type'] == 'project']
                    documents = [r for r in batch if r['record_type'] == 'document']
                    deletions = [r for r in batch if r['record_type'] == 'document_deletion']

                    # Transform and load each type
                    if projects:
                        transformed_projects = self.transformer.transform_batch(projects)
                        self.loader.load_batch(transformed_projects, 'projects')
                        stats['projects_updated'] += len(transformed_projects)

                    if documents:
                        transformed_documents = self.transformer.transform_batch(documents)
                        self.loader.load_batch(transformed_documents, 'documents')
                        stats['documents_updated'] += len(transformed_documents)

                    if deletions:
                        transformed_deletions = self.transformer.transform_batch(deletions)
                        self.loader.load_batch(transformed_deletions, 'deletions')
                        stats['deletions_processed'] += len(transformed_deletions)

                    stats['records_processed'] += len(batch)

                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    stats['errors'] += 1

            # Update analytics views
            self.loader.create_analytics_views()

            stats['duration_seconds'] = time.time() - start_time
            stats['end_time'] = datetime.utcnow()
            stats['success'] = True

            logger.info(f"Incremental ETL completed: {stats}")

        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)
            raise

        return stats

    def run_full_scan(self) -> Dict[str, Any]:
        """Run full scan ETL pipeline"""
        logger.info("Starting full scan ETL pipeline")
        start_time = time.time()

        stats = {
            'start_time': datetime.utcnow(),
            'records_processed': 0,
            'projects_loaded': 0,
            'documents_loaded': 0,
            'errors': 0
        }

        try:
            # Extract all data
            raw_records = []
            for record in self.extractor.extract_full_scan():
                raw_records.append(record)

                # Process in chunks to avoid memory issues
                if len(raw_records) >= self.config.etl.batch_size:
                    self._process_batch(raw_records, stats)
                    raw_records.clear()

            # Process remaining records
            if raw_records:
                self._process_batch(raw_records, stats)

            # Create analytics views
            self.loader.create_analytics_views()

            stats['duration_seconds'] = time.time() - start_time
            stats['end_time'] = datetime.utcnow()
            stats['success'] = True
            stats['load_statistics'] = self.loader.get_load_statistics()

            logger.info(f"Full scan ETL completed: {stats}")

        except Exception as e:
            logger.error(f"Full scan ETL failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)
            raise

        return stats

    def _process_batch(self, raw_records: List[Dict], stats: Dict):
        """Process a batch of raw records"""
        try:
            # Group by record type
            projects = [r for r in raw_records if r['record_type'] == 'project']
            documents = [r for r in raw_records if r['record_type'] == 'document']

            # Transform and load
            if projects:
                transformed_projects = self.transformer.transform_batch(projects)
                self.loader.load_batch(transformed_projects, 'projects')
                stats['projects_loaded'] += len(transformed_projects)

            if documents:
                transformed_documents = self.transformer.transform_batch(documents)
                self.loader.load_batch(transformed_documents, 'documents')
                stats['documents_loaded'] += len(transformed_documents)

            stats['records_processed'] += len(raw_records)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            stats['errors'] += 1

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'config': self.config.etl.__dict__,
            'database_stats': self.loader.get_load_statistics(),
            'last_run': None  # TODO: Implement run tracking
        }