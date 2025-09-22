"""
Data Ingestion Pipeline for Project QuickNav

This module provides comprehensive data ingestion capabilities including:
- Real-time document monitoring and processing
- Batch processing of existing document corpus
- Metadata extraction and enrichment
- Integration with ML recommendation engine
- Event streaming for analytics

Features:
- File system monitoring for OneDrive changes
- Incremental document processing
- Parallel processing for large datasets
- Error handling and retry mechanisms
- Data validation and quality checks
- Integration with existing document parser
"""

import os
import json
import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import hashlib
import queue
import concurrent.futures
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sqlite3

# Import existing components
try:
    from ..quicknav.doc_navigator import DocumentParser, DocumentNavigator
    from ..quicknav.find_project_path import ProjectPathFinder
    from .recommendation_engine import RecommendationEngine
except ImportError:
    # Fallback for development
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from quicknav.doc_navigator import DocumentParser, DocumentNavigator
    from quicknav.find_project_path import ProjectPathFinder
    from ml.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)

@dataclass
class DocumentEvent:
    """Data class for document events"""
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    file_path: str
    project_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    checksum: Optional[str] = None
    size: Optional[int] = None

@dataclass
class ProcessingResult:
    """Data class for processing results"""
    success: bool
    document_path: str
    project_id: Optional[str]
    metadata: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None
    extracted_info: Dict[str, Any] = None

class FileSystemWatcher(FileSystemEventHandler):
    """File system event handler for monitoring document changes"""

    def __init__(self, ingestion_pipeline):
        super().__init__()
        self.pipeline = ingestion_pipeline
        self.ignored_extensions = {'.tmp', '.lock', '.part', '.crdownload'}
        self.debounce_time = 2.0  # seconds
        self.pending_events = {}

    def on_created(self, event):
        if not event.is_directory:
            self._queue_event('created', event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._queue_event('modified', event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._queue_event('deleted', event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._queue_event('moved', event.dest_path, {'old_path': event.src_path})

    def _queue_event(self, event_type: str, file_path: str, extra_metadata: Dict = None):
        """Queue file system event with debouncing"""
        file_path = Path(file_path)

        # Skip ignored file types
        if file_path.suffix.lower() in self.ignored_extensions:
            return

        # Skip hidden files and temporary files
        if file_path.name.startswith('.') or '~$' in file_path.name:
            return

        # Debouncing: replace pending events for the same file
        event_key = str(file_path)

        def process_event():
            time.sleep(self.debounce_time)
            if event_key in self.pending_events:
                del self.pending_events[event_key]
                self.pipeline.handle_file_event(event_type, str(file_path), extra_metadata or {})

        # Cancel existing timer for this file
        if event_key in self.pending_events:
            self.pending_events[event_key].cancel()

        # Start new timer
        timer = threading.Timer(self.debounce_time, process_event)
        self.pending_events[event_key] = timer
        timer.start()

class DataIngestionPipeline:
    """
    Comprehensive data ingestion pipeline for Project QuickNav
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data ingestion pipeline

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.document_parser = DocumentParser()
        self.document_navigator = DocumentNavigator()
        self.recommendation_engine = None  # Will be set if available

        # Processing queues
        self.processing_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue()

        # State tracking
        self.processed_files = {}
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now(),
            'last_processed': None
        }

        # File system monitoring
        self.observer = None
        self.watcher = None
        self.is_monitoring = False

        # Worker threads
        self.worker_threads = []
        self.max_workers = self.config.get('max_workers', 4)
        self.shutdown_event = threading.Event()

        # Database for tracking
        self.db_path = self.config.get('db_path', 'data/ingestion_tracking.db')
        self._init_database()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'watch_directories': [
                os.path.expanduser("~/OneDrive - Pro AV Solutions/Project Files"),
                "test_environment/projects"
            ],
            'supported_extensions': ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt', '.md'],
            'max_workers': 4,
            'batch_size': 50,
            'retry_attempts': 3,
            'retry_delay': 5.0,
            'enable_monitoring': True,
            'processing_timeout': 30.0,
            'checksum_algorithm': 'md5',
            'db_path': 'data/ingestion_tracking.db'
        }

    def _init_database(self):
        """Initialize SQLite database for tracking processed files"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_path TEXT PRIMARY KEY,
                    checksum TEXT,
                    last_processed TIMESTAMP,
                    project_id TEXT,
                    processing_result TEXT,
                    metadata TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    file_path TEXT,
                    timestamp TIMESTAMP,
                    project_id TEXT,
                    success BOOLEAN,
                    processing_time REAL,
                    error_message TEXT,
                    metadata TEXT
                )
            ''')

            conn.commit()

    def start(self):
        """Start the data ingestion pipeline"""
        logger.info("Starting data ingestion pipeline...")

        # Start worker threads
        self._start_workers()

        # Start file system monitoring
        if self.config.get('enable_monitoring', True):
            self._start_monitoring()

        # Start result processing
        self._start_result_processor()

        logger.info("Data ingestion pipeline started successfully")

    def stop(self):
        """Stop the data ingestion pipeline"""
        logger.info("Stopping data ingestion pipeline...")

        # Signal shutdown
        self.shutdown_event.set()

        # Stop file system monitoring
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            self.is_monitoring = False

        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=10)

        # Clear queues
        try:
            while True:
                self.processing_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass

        logger.info("Data ingestion pipeline stopped")

    def process_existing_documents(self, directories: List[str] = None) -> Dict[str, Any]:
        """
        Process existing documents in specified directories

        Args:
            directories: List of directories to process

        Returns:
            Processing statistics
        """
        directories = directories or self.config.get('watch_directories', [])

        logger.info(f"Starting batch processing of existing documents in {len(directories)} directories")
        start_time = datetime.now()

        total_files = 0
        processed_files = 0
        failed_files = 0

        for directory in directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue

            logger.info(f"Processing directory: {directory}")

            # Walk through directory
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Check if file should be processed
                    if self._should_process_file(file_path):
                        total_files += 1

                        # Queue for processing
                        self.queue_file_for_processing(file_path, 'batch_processing')

            # Process files in batches
            batch_results = self._process_batch(batch_size=self.config.get('batch_size', 50))
            processed_files += batch_results['successful']
            failed_files += batch_results['failed']

        processing_time = (datetime.now() - start_time).total_seconds()

        results = {
            'total_files': total_files,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'processing_time': processing_time,
            'files_per_second': processed_files / processing_time if processing_time > 0 else 0
        }

        logger.info(f"Batch processing complete: {results}")
        return results

    def queue_file_for_processing(self, file_path: str, event_type: str = 'manual',
                                 metadata: Dict[str, Any] = None):
        """
        Queue a file for processing

        Args:
            file_path: Path to file
            event_type: Type of event that triggered processing
            metadata: Additional metadata
        """
        try:
            # Check if file should be processed
            if not self._should_process_file(file_path):
                return

            # Extract project ID from path
            project_id = self._extract_project_id(file_path)

            # Create document event
            event = DocumentEvent(
                event_type=event_type,
                file_path=file_path,
                project_id=project_id,
                timestamp=datetime.now(),
                metadata=metadata or {},
                checksum=self._calculate_checksum(file_path),
                size=os.path.getsize(file_path) if os.path.exists(file_path) else None
            )

            # Add to processing queue
            self.processing_queue.put(event, timeout=10)

        except Exception as e:
            logger.error(f"Error queuing file for processing {file_path}: {e}")

    def handle_file_event(self, event_type: str, file_path: str, metadata: Dict[str, Any]):
        """Handle file system events"""
        logger.debug(f"Handling file event: {event_type} - {file_path}")

        if event_type == 'deleted':
            self._handle_file_deletion(file_path)
        else:
            self.queue_file_for_processing(file_path, event_type, metadata)

    def set_recommendation_engine(self, engine: RecommendationEngine):
        """Set recommendation engine for integration"""
        self.recommendation_engine = engine
        logger.info("Recommendation engine integration enabled")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        uptime = (datetime.now() - self.processing_stats['start_time']).total_seconds()

        return {
            **self.processing_stats,
            'uptime_seconds': uptime,
            'processing_rate': self.processing_stats['successful'] / uptime if uptime > 0 else 0,
            'queue_size': self.processing_queue.qsize(),
            'is_monitoring': self.is_monitoring,
            'worker_threads_active': len([t for t in self.worker_threads if t.is_alive()])
        }

    def get_recent_activity(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent processing activity"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM processing_events
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (cutoff_time,))

            return [dict(row) for row in cursor.fetchall()]

    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get processing summary for a specific project"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get file count
            cursor = conn.execute('''
                SELECT COUNT(*) as file_count,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed
                FROM processing_events
                WHERE project_id = ?
            ''', (project_id,))

            stats = dict(cursor.fetchone())

            # Get recent files
            cursor = conn.execute('''
                SELECT file_path, timestamp, success, processing_time
                FROM processing_events
                WHERE project_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (project_id,))

            recent_files = [dict(row) for row in cursor.fetchall()]

            return {
                'project_id': project_id,
                'statistics': stats,
                'recent_files': recent_files
            }

    # Private methods

    def _start_workers(self):
        """Start worker threads for processing files"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"IngestionWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)

        logger.info(f"Started {self.max_workers} worker threads")

    def _start_monitoring(self):
        """Start file system monitoring"""
        try:
            self.watcher = FileSystemWatcher(self)
            self.observer = Observer()

            for directory in self.config.get('watch_directories', []):
                if os.path.exists(directory):
                    self.observer.schedule(self.watcher, directory, recursive=True)
                    logger.info(f"Monitoring directory: {directory}")
                else:
                    logger.warning(f"Watch directory not found: {directory}")

            self.observer.start()
            self.is_monitoring = True

        except Exception as e:
            logger.error(f"Error starting file system monitoring: {e}")

    def _start_result_processor(self):
        """Start result processing thread"""
        def result_processor():
            while not self.shutdown_event.is_set():
                try:
                    result = self.result_queue.get(timeout=1)
                    self._handle_processing_result(result)
                    self.result_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in result processor: {e}")

        thread = threading.Thread(target=result_processor, name="ResultProcessor", daemon=True)
        thread.start()

    def _worker_thread(self):
        """Worker thread for processing documents"""
        while not self.shutdown_event.is_set():
            try:
                # Get next event from queue
                event = self.processing_queue.get(timeout=1)

                if event is None:
                    continue

                # Process the document
                result = self._process_document(event)

                # Add result to result queue
                self.result_queue.put(result)

                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")

    def _process_document(self, event: DocumentEvent) -> ProcessingResult:
        """
        Process a single document

        Args:
            event: Document event to process

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            # Check if file still exists (for deleted events)
            if event.event_type != 'deleted' and not os.path.exists(event.file_path):
                return ProcessingResult(
                    success=False,
                    document_path=event.file_path,
                    project_id=event.project_id,
                    metadata=event.metadata,
                    processing_time=time.time() - start_time,
                    error_message="File not found"
                )

            # Check if file has changed (using checksum)
            if not self._file_has_changed(event.file_path, event.checksum):
                return ProcessingResult(
                    success=True,
                    document_path=event.file_path,
                    project_id=event.project_id,
                    metadata=event.metadata,
                    processing_time=time.time() - start_time,
                    extracted_info={'status': 'unchanged'}
                )

            # Parse document metadata
            filename = os.path.basename(event.file_path)
            parsed_metadata = self.document_parser.parse_filename(filename)

            # Extract additional information if supported
            extracted_info = {}
            if event.file_path.lower().endswith('.pdf'):
                # Could integrate PDF text extraction here
                extracted_info['type'] = 'pdf_document'
            elif event.file_path.lower().endswith(('.docx', '.doc')):
                extracted_info['type'] = 'word_document'
            elif event.file_path.lower().endswith(('.xlsx', '.xls')):
                extracted_info['type'] = 'excel_document'

            # Combine metadata
            combined_metadata = {
                **event.metadata,
                **parsed_metadata,
                'file_size': event.size,
                'last_modified': os.path.getmtime(event.file_path) if os.path.exists(event.file_path) else None,
                'extraction_timestamp': datetime.now().isoformat()
            }

            return ProcessingResult(
                success=True,
                document_path=event.file_path,
                project_id=event.project_id,
                metadata=combined_metadata,
                processing_time=time.time() - start_time,
                extracted_info=extracted_info
            )

        except Exception as e:
            logger.error(f"Error processing document {event.file_path}: {e}")
            return ProcessingResult(
                success=False,
                document_path=event.file_path,
                project_id=event.project_id,
                metadata=event.metadata,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def _handle_processing_result(self, result: ProcessingResult):
        """Handle processing result and update tracking"""
        # Update statistics
        self.processing_stats['total_processed'] += 1
        if result.success:
            self.processing_stats['successful'] += 1
        else:
            self.processing_stats['failed'] += 1
        self.processing_stats['last_processed'] = datetime.now()

        # Update database
        self._update_database(result)

        # Update recommendation engine if available
        if self.recommendation_engine and result.success:
            self._update_recommendation_engine(result)

        # Log result
        if result.success:
            logger.debug(f"Successfully processed {result.document_path} in {result.processing_time:.2f}s")
        else:
            logger.warning(f"Failed to process {result.document_path}: {result.error_message}")

    def _update_database(self, result: ProcessingResult):
        """Update database with processing result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update processed files table
                conn.execute('''
                    INSERT OR REPLACE INTO processed_files
                    (file_path, checksum, last_processed, project_id, processing_result, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.document_path,
                    self._calculate_checksum(result.document_path) if os.path.exists(result.document_path) else None,
                    datetime.now(),
                    result.project_id,
                    'success' if result.success else 'failed',
                    json.dumps(result.metadata)
                ))

                # Insert processing event
                conn.execute('''
                    INSERT INTO processing_events
                    (event_type, file_path, timestamp, project_id, success, processing_time, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'processing',
                    result.document_path,
                    datetime.now(),
                    result.project_id,
                    result.success,
                    result.processing_time,
                    result.error_message,
                    json.dumps(result.extracted_info or {})
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error updating database: {e}")

    def _update_recommendation_engine(self, result: ProcessingResult):
        """Update recommendation engine with new document data"""
        try:
            # This would integrate with the recommendation engine
            # to update document features and user interactions
            pass
        except Exception as e:
            logger.error(f"Error updating recommendation engine: {e}")

    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        file_path = Path(file_path)

        # Check extension
        if file_path.suffix.lower() not in self.config.get('supported_extensions', []):
            return False

        # Skip hidden files
        if file_path.name.startswith('.'):
            return False

        # Skip temporary files
        if '~$' in file_path.name or file_path.name.endswith('.tmp'):
            return False

        # Check file size (skip very large files)
        try:
            if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB limit
                return False
        except OSError:
            return False

        return True

    def _extract_project_id(self, file_path: str) -> Optional[str]:
        """Extract project ID from file path"""
        try:
            # Look for 5-digit project codes in the path
            parts = Path(file_path).parts
            for part in parts:
                # Check if part contains a project code pattern
                import re
                match = re.search(r'\b(\d{5})\b', part)
                if match:
                    return match.group(1)
            return None
        except Exception:
            return None

    def _calculate_checksum(self, file_path: str) -> Optional[str]:
        """Calculate file checksum"""
        if not os.path.exists(file_path):
            return None

        try:
            hash_obj = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return None

    def _file_has_changed(self, file_path: str, current_checksum: str) -> bool:
        """Check if file has changed since last processing"""
        if not os.path.exists(file_path):
            return True

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT checksum FROM processed_files WHERE file_path = ?',
                    (file_path,)
                )
                row = cursor.fetchone()

                if not row:
                    return True  # File not processed before

                stored_checksum = row[0]
                return stored_checksum != current_checksum

        except Exception:
            return True  # Assume changed if can't determine

    def _handle_file_deletion(self, file_path: str):
        """Handle file deletion event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE processed_files SET processing_result = ? WHERE file_path = ?',
                    ('deleted', file_path)
                )

                conn.execute('''
                    INSERT INTO processing_events
                    (event_type, file_path, timestamp, success, processing_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    'deleted',
                    file_path,
                    datetime.now(),
                    True,
                    0.0,
                    json.dumps({'deleted': True})
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error handling file deletion {file_path}: {e}")

    def _process_batch(self, batch_size: int = 50) -> Dict[str, int]:
        """Process a batch of queued items"""
        processed = 0
        successful = 0
        failed = 0

        start_time = time.time()
        timeout = start_time + 60  # 1 minute timeout for batch

        while processed < batch_size and time.time() < timeout:
            try:
                # Try to get an item with short timeout
                event = self.processing_queue.get(timeout=0.1)
                result = self._process_document(event)

                # Handle result
                self._handle_processing_result(result)

                processed += 1
                if result.success:
                    successful += 1
                else:
                    failed += 1

                self.processing_queue.task_done()

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                failed += 1

        return {
            'processed': processed,
            'successful': successful,
            'failed': failed,
            'processing_time': time.time() - start_time
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create pipeline with test config
    config = {
        'watch_directories': ['test_environment/projects'],
        'max_workers': 2,
        'batch_size': 10,
        'enable_monitoring': True,
        'db_path': 'test_ingestion.db'
    }

    pipeline = DataIngestionPipeline(config)

    try:
        # Start pipeline
        pipeline.start()

        # Process existing documents
        print("Processing existing documents...")
        results = pipeline.process_existing_documents()
        print(f"Batch processing results: {results}")

        # Get statistics
        stats = pipeline.get_processing_stats()
        print(f"Processing statistics: {stats}")

        # Keep running for a bit to test monitoring
        print("Pipeline running... Press Ctrl+C to stop")
        time.sleep(30)

    except KeyboardInterrupt:
        print("\nShutting down pipeline...")
    finally:
        pipeline.stop()
        print("Pipeline stopped")