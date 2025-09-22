"""
Document Monitor - Real-time OneDrive folder monitoring for document changes

Features:
- Cross-platform file system monitoring (Windows, macOS, Linux)
- Efficient directory watching with minimal CPU overhead
- Intelligent filtering for relevant document types
- Batch processing of file events to reduce noise
- Network-aware monitoring with OneDrive sync detection
- Graceful handling of temporary files and conflicts
- Integration with existing document parser and database
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

# Cross-platform file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("Watchdog not available, falling back to polling")

from ...doc_navigator import DocumentParser, DocumentTypeClassifier
from ...database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class FileEventType(Enum):
    """Types of file system events."""
    CREATED = "created"
    MODIFIED = "modified"
    MOVED = "moved"
    DELETED = "deleted"


@dataclass
class DocumentEvent:
    """Represents a document file system event."""
    event_type: FileEventType
    file_path: str
    project_id: Optional[str] = None
    document_type: Optional[str] = None
    timestamp: float = None
    old_path: Optional[str] = None  # For move events
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    is_temporary: bool = False
    is_onedrive_conflict: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class DocumentFileHandler(FileSystemEventHandler):
    """Watchdog file system event handler for document monitoring."""

    def __init__(self, monitor: 'DocumentMonitor'):
        super().__init__()
        self.monitor = monitor

    def on_created(self, event):
        if not event.is_directory:
            asyncio.create_task(self.monitor._handle_file_event(
                FileEventType.CREATED, event.src_path
            ))

    def on_modified(self, event):
        if not event.is_directory:
            asyncio.create_task(self.monitor._handle_file_event(
                FileEventType.MODIFIED, event.src_path
            ))

    def on_moved(self, event):
        if not event.is_directory:
            asyncio.create_task(self.monitor._handle_file_event(
                FileEventType.MOVED, event.dest_path, old_path=event.src_path
            ))

    def on_deleted(self, event):
        if not event.is_directory:
            asyncio.create_task(self.monitor._handle_file_event(
                FileEventType.DELETED, event.src_path
            ))


class DocumentMonitor:
    """
    Real-time document monitoring system for OneDrive project folders.

    Monitors file system changes and processes document updates through
    the pipeline for indexing, analysis, and ML feature extraction.
    """

    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

        # Configuration
        self.watch_paths = config.get('watch_paths', [])
        self.document_extensions = set(config.get('document_extensions', [
            '.pdf', '.docx', '.doc', '.rtf', '.vsdx', '.vsd',
            '.xlsx', '.xls', '.pptx', '.ppt', '.txt'
        ]))
        self.batch_interval = config.get('batch_interval', 5.0)  # seconds
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
        self.ignore_patterns = config.get('ignore_patterns', [
            r'~\$.*',  # Office temp files
            r'\.tmp$', r'\.temp$',  # Temporary files
            r'Desktop\.ini$', r'Thumbs\.db$',  # Windows system files
            r'\.DS_Store$',  # macOS system files
            r'.*conflict.*',  # OneDrive conflicts
        ])

        # Components
        self.parser = DocumentParser()
        self.classifier = DocumentTypeClassifier()

        # State management
        self.running = False
        self.observers: List[Any] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.file_cache: Dict[str, Dict[str, Any]] = {}  # Cache file metadata
        self.pending_events: List[DocumentEvent] = []

        # Background tasks
        self.processor_task: Optional[asyncio.Task] = None
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.cache_cleanup_task: Optional[asyncio.Task] = None

        # Callbacks for external integration
        self.event_callbacks: List[Callable[[DocumentEvent], None]] = []

        # Performance metrics
        self.metrics = {
            'events_processed': 0,
            'documents_indexed': 0,
            'errors': 0,
            'last_activity': time.time(),
            'processing_time_ms': 0.0
        }

        # Auto-detect OneDrive paths if not specified
        if not self.watch_paths:
            self.watch_paths = self._detect_onedrive_paths()

    def _detect_onedrive_paths(self) -> List[str]:
        """Auto-detect OneDrive project folders to monitor."""
        paths = []

        # Check common OneDrive locations
        user_profile = os.environ.get('USERPROFILE') or os.path.expanduser('~')

        potential_paths = [
            os.path.join(user_profile, 'OneDrive - Pro AV Solutions', 'Project Files'),
            os.path.join(user_profile, 'OneDrive', 'Project Files'),
            os.path.join(user_profile, 'OneDrive - Pro AV Solutions'),
            os.path.join(user_profile, 'OneDrive')
        ]

        for path in potential_paths:
            if os.path.isdir(path):
                # Check if it contains project-like folders
                try:
                    entries = os.listdir(path)
                    project_folders = [e for e in entries if os.path.isdir(os.path.join(path, e))
                                     and any(char.isdigit() for char in e)]

                    if project_folders:
                        paths.append(path)
                        logger.info(f"Auto-detected OneDrive path: {path}")
                        break
                except (PermissionError, OSError):
                    continue

        return paths

    async def start(self):
        """Start the document monitoring system."""
        if self.running:
            logger.warning("Document monitor already running")
            return

        logger.info("Starting document monitor...")

        if not self.watch_paths:
            logger.warning("No watch paths configured or detected")
            return

        try:
            # Start file system monitoring
            if WATCHDOG_AVAILABLE:
                await self._start_watchdog_monitoring()
            else:
                await self._start_polling_monitoring()

            # Start background processing tasks
            self.processor_task = asyncio.create_task(self._event_processor_loop())
            self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            self.cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())

            self.running = True
            logger.info(f"Document monitor started, watching {len(self.watch_paths)} paths")

        except Exception as e:
            logger.error(f"Failed to start document monitor: {e}")
            await self.shutdown()
            raise

    async def _start_watchdog_monitoring(self):
        """Start watchdog-based file system monitoring."""
        handler = DocumentFileHandler(self)

        for watch_path in self.watch_paths:
            if not os.path.exists(watch_path):
                logger.warning(f"Watch path does not exist: {watch_path}")
                continue

            observer = Observer()
            observer.schedule(handler, watch_path, recursive=True)
            observer.start()
            self.observers.append(observer)

            logger.info(f"Started monitoring: {watch_path}")

    async def _start_polling_monitoring(self):
        """Start polling-based file system monitoring (fallback)."""
        # TODO: Implement polling-based monitoring for systems without watchdog
        logger.warning("Polling-based monitoring not yet implemented")

    async def shutdown(self):
        """Shutdown the document monitoring system."""
        if not self.running:
            return

        logger.info("Shutting down document monitor...")
        self.running = False

        # Stop file system observers
        for observer in self.observers:
            observer.stop()
            observer.join(timeout=5)

        # Cancel background tasks
        for task in [self.processor_task, self.batch_processor_task, self.cache_cleanup_task]:
            if task:
                task.cancel()

        # Process remaining events
        try:
            await asyncio.wait_for(self._process_pending_events(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Timeout processing remaining events during shutdown")

        logger.info("Document monitor shutdown complete")

    async def _handle_file_event(self, event_type: FileEventType, file_path: str,
                                old_path: Optional[str] = None):
        """Handle a file system event."""
        try:
            # Filter out non-document files
            if not self._is_document_file(file_path):
                return

            # Filter out temporary/system files
            if self._is_filtered_file(file_path):
                return

            # Create document event
            event = DocumentEvent(
                event_type=event_type,
                file_path=file_path,
                old_path=old_path,
                is_temporary=self._is_temporary_file(file_path),
                is_onedrive_conflict=self._is_onedrive_conflict(file_path)
            )

            # Add file metadata
            if event_type != FileEventType.DELETED and os.path.exists(file_path):
                try:
                    stat = os.stat(file_path)
                    event.file_size = stat.st_size

                    # Skip very large files
                    if event.file_size > self.max_file_size_mb * 1024 * 1024:
                        logger.debug(f"Skipping large file: {file_path} ({event.file_size / 1024 / 1024:.1f} MB)")
                        return

                    # Calculate file hash for change detection
                    event.file_hash = await self._calculate_file_hash(file_path)

                except (OSError, PermissionError) as e:
                    logger.debug(f"Cannot access file {file_path}: {e}")
                    return

            # Extract project information
            event.project_id, event.document_type = self._extract_project_info(file_path)

            # Add to event queue
            await self.event_queue.put(event)

        except Exception as e:
            logger.error(f"Error handling file event {event_type} for {file_path}: {e}")
            self.metrics['errors'] += 1

    def _is_document_file(self, file_path: str) -> bool:
        """Check if file is a document type we care about."""
        ext = Path(file_path).suffix.lower()
        return ext in self.document_extensions

    def _is_filtered_file(self, file_path: str) -> bool:
        """Check if file should be filtered out."""
        import re
        filename = os.path.basename(file_path)

        for pattern in self.ignore_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        return False

    def _is_temporary_file(self, file_path: str) -> bool:
        """Check if file is a temporary file."""
        filename = os.path.basename(file_path)
        temp_indicators = ['~$', '.tmp', '.temp', '.lock']
        return any(indicator in filename.lower() for indicator in temp_indicators)

    def _is_onedrive_conflict(self, file_path: str) -> bool:
        """Check if file is a OneDrive conflict file."""
        filename = os.path.basename(file_path)
        conflict_indicators = ['-conflict-', ' conflict ', 'conflicted copy']
        return any(indicator in filename.lower() for indicator in conflict_indicators)

    def _extract_project_info(self, file_path: str) -> tuple[Optional[str], Optional[str]]:
        """Extract project ID and document type from file path."""
        try:
            # Extract project code from path
            path_parts = Path(file_path).parts
            project_id = None

            for part in path_parts:
                # Look for 5-digit project codes
                import re
                match = re.search(r'\b(\d{5})\b', part)
                if match:
                    project_id = match.group(1)
                    break

            # Classify document type
            type_config = self.classifier.classify_document(file_path)
            document_type = type_config['type'] if type_config else None

            return project_id, document_type

        except Exception as e:
            logger.debug(f"Error extracting project info from {file_path}: {e}")
            return None, None

    async def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate MD5 hash of file for change detection."""
        try:
            hash_md5 = hashlib.md5()

            # Read file in chunks to handle large files
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)

            return hash_md5.hexdigest()

        except Exception as e:
            logger.debug(f"Error calculating hash for {file_path}: {e}")
            return None

    async def _event_processor_loop(self):
        """Main event processing loop."""
        logger.info("Event processor started")

        while self.running:
            try:
                # Get event with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Add to pending events for batch processing
                self.pending_events.append(event)
                self.metrics['events_processed'] += 1
                self.metrics['last_activity'] = time.time()

                # Notify external callbacks
                for callback in self.event_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.warning(f"Event callback failed: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processor error: {e}")
                self.metrics['errors'] += 1
                await asyncio.sleep(1)

    async def _batch_processor_loop(self):
        """Batch process accumulated events to reduce database load."""
        logger.info("Batch processor started")

        while self.running:
            try:
                await asyncio.sleep(self.batch_interval)

                if not self.pending_events:
                    continue

                # Process batch of events
                events_to_process = self.pending_events.copy()
                self.pending_events.clear()

                start_time = time.time()
                await self._process_event_batch(events_to_process)
                processing_time = (time.time() - start_time) * 1000

                self.metrics['processing_time_ms'] = processing_time
                logger.debug(f"Processed {len(events_to_process)} events in {processing_time:.1f}ms")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                self.metrics['errors'] += 1

    async def _process_event_batch(self, events: List[DocumentEvent]):
        """Process a batch of document events."""
        # Group events by file to handle duplicates
        file_events: Dict[str, DocumentEvent] = {}

        for event in events:
            # Keep the most recent event for each file
            if (event.file_path not in file_events or
                event.timestamp > file_events[event.file_path].timestamp):
                file_events[event.file_path] = event

        # Process unique file events
        for event in file_events.values():
            try:
                await self._process_single_event(event)
                self.metrics['documents_indexed'] += 1
            except Exception as e:
                logger.error(f"Error processing event for {event.file_path}: {e}")
                self.metrics['errors'] += 1

    async def _process_single_event(self, event: DocumentEvent):
        """Process a single document event."""
        if event.event_type == FileEventType.DELETED:
            await self._handle_document_deletion(event)
        else:
            await self._handle_document_update(event)

    async def _handle_document_update(self, event: DocumentEvent):
        """Handle document creation or modification."""
        if not os.path.exists(event.file_path):
            return

        # Check if file has actually changed
        cache_key = event.file_path
        cached_info = self.file_cache.get(cache_key)

        if cached_info and cached_info.get('hash') == event.file_hash:
            logger.debug(f"File unchanged, skipping: {event.file_path}")
            return

        try:
            # Parse document metadata
            filename = os.path.basename(event.file_path)
            parsed_metadata = self.parser.parse_filename(filename)

            # Get file stats
            stat = os.stat(event.file_path)

            # Prepare document data for database
            document_data = {
                'project_id': event.project_id,
                'file_path': event.file_path,
                'filename': filename,
                'file_extension': Path(event.file_path).suffix.lower(),
                'file_size_bytes': event.file_size,
                'document_type': event.document_type,
                'folder_category': self._extract_folder_category(event.file_path),
                'version_string': parsed_metadata.get('version_raw'),
                'version_numeric': self._normalize_version(parsed_metadata.get('version')),
                'version_type': parsed_metadata.get('version_type'),
                'status_tags': json.dumps(list(parsed_metadata.get('status_tags', set()))),
                'status_weight': self._calculate_status_weight(parsed_metadata.get('status_tags', set())),
                'document_date': self._extract_document_date(parsed_metadata.get('dates', [])),
                'modified_at': datetime.fromtimestamp(stat.st_mtime),
                'content_hash': event.file_hash,
                'classification_confidence': 0.8 if event.document_type else 0.0
            }

            # Store in database
            success = await self.db_manager.upsert_document(document_data)

            if success:
                # Update cache
                self.file_cache[cache_key] = {
                    'hash': event.file_hash,
                    'processed_at': time.time(),
                    'document_id': document_data.get('document_id')
                }

                logger.debug(f"Indexed document: {event.file_path}")
            else:
                logger.warning(f"Failed to index document: {event.file_path}")

        except Exception as e:
            logger.error(f"Error processing document update for {event.file_path}: {e}")
            raise

    async def _handle_document_deletion(self, event: DocumentEvent):
        """Handle document deletion."""
        try:
            # Mark document as inactive rather than deleting
            await self.db_manager.execute_write(
                "UPDATE documents SET is_active = 0 WHERE file_path = ?",
                [event.file_path]
            )

            # Remove from cache
            if event.file_path in self.file_cache:
                del self.file_cache[event.file_path]

            logger.debug(f"Marked document as inactive: {event.file_path}")

        except Exception as e:
            logger.error(f"Error handling document deletion for {event.file_path}: {e}")

    def _extract_folder_category(self, file_path: str) -> Optional[str]:
        """Extract folder category from file path."""
        path_parts = Path(file_path).parts

        # Look for numbered folder patterns
        for part in path_parts:
            if '. ' in part and part[0].isdigit():
                return part

        return None

    def _normalize_version(self, version: Any) -> Optional[int]:
        """Normalize version to integer for database storage."""
        if version is None:
            return None

        if isinstance(version, int):
            return version
        elif isinstance(version, tuple):
            # Convert (major, minor) to single integer
            major, minor = version
            return major * 1000 + minor
        else:
            return None

    def _calculate_status_weight(self, status_tags: Set[str]) -> float:
        """Calculate status weight for document ranking."""
        weight = 0.0

        status_weights = {
            'AS-BUILT': 2.0,
            'AS BUILT': 2.0,
            'ASBUILT': 2.0,
            'SIGNED': 1.5,
            'FINAL': 1.0,
            'APPROVED': 0.8,
            'DRAFT': -0.5,
            'OLD': -1.0,
        }

        for tag in status_tags:
            weight += status_weights.get(tag.upper(), 0.0)

        return weight

    def _extract_document_date(self, dates: List[datetime]) -> Optional[datetime]:
        """Extract the most relevant date from parsed dates."""
        if not dates:
            return None

        # Return the most recent date
        return max(dates)

    async def _cache_cleanup_loop(self):
        """Periodically clean up the file cache."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                current_time = time.time()
                cache_ttl = 86400  # 24 hours

                # Remove old cache entries
                expired_keys = [
                    key for key, value in self.file_cache.items()
                    if current_time - value.get('processed_at', 0) > cache_ttl
                ]

                for key in expired_keys:
                    del self.file_cache[key]

                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _process_pending_events(self):
        """Process any remaining pending events."""
        if self.pending_events:
            logger.info(f"Processing {len(self.pending_events)} remaining events...")
            await self._process_event_batch(self.pending_events)
            self.pending_events.clear()

    # Public API methods

    def register_event_callback(self, callback: Callable[[DocumentEvent], None]):
        """Register a callback for document events."""
        self.event_callbacks.append(callback)

    async def get_status(self) -> Dict[str, Any]:
        """Get monitor status and metrics."""
        return {
            "running": self.running,
            "watch_paths": self.watch_paths,
            "metrics": self.metrics.copy(),
            "cache_size": len(self.file_cache),
            "pending_events": len(self.pending_events),
            "queue_size": self.event_queue.qsize(),
            "observers": len(self.observers)
        }

    async def force_rescan(self, path: Optional[str] = None):
        """Force a rescan of specified path or all watched paths."""
        paths_to_scan = [path] if path else self.watch_paths

        for scan_path in paths_to_scan:
            if not os.path.exists(scan_path):
                continue

            logger.info(f"Starting forced rescan of: {scan_path}")

            # Walk through directory and simulate creation events
            for root, dirs, files in os.walk(scan_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    if self._is_document_file(file_path) and not self._is_filtered_file(file_path):
                        await self._handle_file_event(FileEventType.CREATED, file_path)

            logger.info(f"Completed forced rescan of: {scan_path}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        healthy = True
        issues = []

        # Check if monitoring is active
        if not self.running:
            healthy = False
            issues.append("Monitor not running")

        # Check watch paths
        for path in self.watch_paths:
            if not os.path.exists(path):
                healthy = False
                issues.append(f"Watch path not accessible: {path}")

        # Check recent activity
        time_since_activity = time.time() - self.metrics['last_activity']
        if time_since_activity > 3600:  # 1 hour
            issues.append(f"No activity for {time_since_activity / 3600:.1f} hours")

        # Check error rate
        total_events = self.metrics['events_processed']
        if total_events > 0:
            error_rate = self.metrics['errors'] / total_events
            if error_rate > 0.1:  # 10% error rate
                healthy = False
                issues.append(f"High error rate: {error_rate:.1%}")

        return {
            "healthy": healthy,
            "issues": issues,
            "metrics": self.metrics.copy()
        }