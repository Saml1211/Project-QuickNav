"""
Streaming Pipeline for Project QuickNav

Implements real-time event processing for user behavior tracking,
document access patterns, and system performance monitoring.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import threading
from queue import Queue, Empty
import weakref

from .config import get_config
from .etl import DuckDBLoader

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event structure for streaming pipeline"""
    event_id: str
    event_type: str
    timestamp: datetime
    session_id: str
    user_id: Optional[str] = None
    data: Dict[str, Any] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EventQueue:
    """Thread-safe event queue with batching capabilities"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue = Queue(maxsize=max_size)
        self._stats = {
            'events_queued': 0,
            'events_dropped': 0,
            'queue_size': 0
        }

    def put(self, event: Event, block: bool = False) -> bool:
        """Add event to queue"""
        try:
            self._queue.put(event, block=block)
            self._stats['events_queued'] += 1
            self._stats['queue_size'] = self._queue.qsize()
            return True
        except:
            self._stats['events_dropped'] += 1
            logger.warning(f"Event queue full, dropping event: {event.event_type}")
            return False

    def get_batch(self, max_size: int, timeout: float = 1.0) -> List[Event]:
        """Get a batch of events"""
        batch = []
        end_time = time.time() + timeout

        while len(batch) < max_size and time.time() < end_time:
            try:
                remaining_timeout = max(0, end_time - time.time())
                event = self._queue.get(timeout=remaining_timeout)
                batch.append(event)
                self._stats['queue_size'] = self._queue.qsize()
            except Empty:
                break

        return batch

    def size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        return self._stats.copy()


class EventProcessor:
    """Processes streaming events with configurable handlers"""

    def __init__(self, event_queue: EventQueue):
        self.event_queue = event_queue
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.stats = {
            'events_processed': 0,
            'processing_errors': 0,
            'handler_executions': defaultdict(int)
        }

    def register_handler(self, event_type: str, handler: Callable[[Event], None]):
        """Register event handler for specific event type"""
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")

    def register_global_handler(self, handler: Callable[[Event], None]):
        """Register handler for all event types"""
        self.handlers['*'].append(handler)
        logger.info("Registered global event handler")

    async def process_events(self, batch_size: int = 100, flush_interval: float = 30.0):
        """Process events in batches"""
        logger.info(f"Starting event processor (batch_size={batch_size}, flush_interval={flush_interval})")

        last_flush = time.time()

        while True:
            try:
                # Get batch of events
                events = self.event_queue.get_batch(batch_size, timeout=1.0)

                if events or (time.time() - last_flush > flush_interval):
                    if events:
                        await self._process_batch(events)
                    last_flush = time.time()

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1.0)

    async def _process_batch(self, events: List[Event]):
        """Process a batch of events"""
        try:
            for event in events:
                await self._process_single_event(event)
                self.stats['events_processed'] += 1

            logger.debug(f"Processed batch of {len(events)} events")

        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
            self.stats['processing_errors'] += 1

    async def _process_single_event(self, event: Event):
        """Process a single event"""
        try:
            # Call specific handlers
            if event.event_type in self.handlers:
                for handler in self.handlers[event.event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                        self.stats['handler_executions'][event.event_type] += 1
                    except Exception as e:
                        logger.error(f"Error in handler for {event.event_type}: {e}")

            # Call global handlers
            for handler in self.handlers['*']:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                    self.stats['handler_executions']['global'] += 1
                except Exception as e:
                    logger.error(f"Error in global handler: {e}")

        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'events_processed': self.stats['events_processed'],
            'processing_errors': self.stats['processing_errors'],
            'handler_executions': dict(self.stats['handler_executions']),
            'queue_stats': self.event_queue.get_stats()
        }


class UserBehaviorTracker:
    """Tracks user behavior and interaction patterns"""

    def __init__(self, event_queue: EventQueue):
        self.event_queue = event_queue
        self.session_data = {}
        self.user_sessions = defaultdict(set)

    def start_session(self, session_id: str, user_id: str = None) -> str:
        """Start a new user session"""
        if not session_id:
            session_id = str(uuid.uuid4())

        self.session_data[session_id] = {
            'user_id': user_id,
            'start_time': datetime.utcnow(),
            'events': [],
            'projects_accessed': set(),
            'documents_opened': set(),
            'search_queries': []
        }

        if user_id:
            self.user_sessions[user_id].add(session_id)

        self._emit_event('session_start', session_id, user_id, {
            'start_time': datetime.utcnow().isoformat()
        })

        return session_id

    def end_session(self, session_id: str):
        """End a user session"""
        if session_id in self.session_data:
            session = self.session_data[session_id]
            duration = (datetime.utcnow() - session['start_time']).total_seconds()

            self._emit_event('session_end', session_id, session.get('user_id'), {
                'duration_seconds': duration,
                'events_count': len(session['events']),
                'projects_accessed': len(session['projects_accessed']),
                'documents_opened': len(session['documents_opened']),
                'search_queries': len(session['search_queries'])
            })

            # Cleanup
            user_id = session.get('user_id')
            if user_id and session_id in self.user_sessions[user_id]:
                self.user_sessions[user_id].remove(session_id)

            del self.session_data[session_id]

    def track_project_search(self, session_id: str, query: str, results: List[str],
                           response_time_ms: float):
        """Track project search event"""
        self._emit_event('project_search', session_id, None, {
            'query': query,
            'result_count': len(results),
            'response_time_ms': response_time_ms,
            'results': results[:10]  # Limit to top 10 results
        })

        # Update session data
        if session_id in self.session_data:
            self.session_data[session_id]['search_queries'].append({
                'query': query,
                'timestamp': datetime.utcnow(),
                'result_count': len(results)
            })

    def track_project_access(self, session_id: str, project_id: str, project_path: str):
        """Track project access event"""
        self._emit_event('project_access', session_id, None, {
            'project_id': project_id,
            'project_path': project_path
        })

        # Update session data
        if session_id in self.session_data:
            self.session_data[session_id]['projects_accessed'].add(project_id)

    def track_document_open(self, session_id: str, document_path: str, doc_type: str,
                          project_id: str = None):
        """Track document open event"""
        self._emit_event('document_open', session_id, None, {
            'document_path': document_path,
            'doc_type': doc_type,
            'project_id': project_id
        })

        # Update session data
        if session_id in self.session_data:
            self.session_data[session_id]['documents_opened'].add(document_path)

    def track_ai_interaction(self, session_id: str, query: str, response_type: str,
                           processing_time_ms: float):
        """Track AI assistant interaction"""
        self._emit_event('ai_interaction', session_id, None, {
            'query': query,
            'response_type': response_type,
            'processing_time_ms': processing_time_ms
        })

    def track_error(self, session_id: str, error_type: str, error_message: str,
                   context: Dict[str, Any] = None):
        """Track error event"""
        self._emit_event('error', session_id, None, {
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        })

    def _emit_event(self, event_type: str, session_id: str, user_id: str = None,
                   data: Dict[str, Any] = None):
        """Emit an event to the event queue"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            user_id=user_id,
            data=data or {}
        )

        # Update session events
        if session_id in self.session_data:
            self.session_data[session_id]['events'].append(event_type)

        self.event_queue.put(event)

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of session activity"""
        if session_id not in self.session_data:
            return None

        session = self.session_data[session_id]
        current_time = datetime.utcnow()
        duration = (current_time - session['start_time']).total_seconds()

        return {
            'session_id': session_id,
            'user_id': session.get('user_id'),
            'start_time': session['start_time'].isoformat(),
            'duration_seconds': duration,
            'events_count': len(session['events']),
            'projects_accessed': len(session['projects_accessed']),
            'documents_opened': len(session['documents_opened']),
            'search_queries': len(session['search_queries']),
            'recent_events': session['events'][-10:],  # Last 10 events
            'is_active': duration < 1800  # Active if less than 30 minutes
        }


class StreamingAnalyzer:
    """Real-time analysis of streaming events"""

    def __init__(self):
        self.config = get_config()
        self.window_size = 300  # 5 minute windows
        self.event_windows = defaultdict(lambda: deque(maxlen=1000))
        self.metrics = defaultdict(float)
        self.last_analysis = time.time()

    async def analyze_event(self, event: Event):
        """Analyze a single event for real-time insights"""
        current_time = time.time()
        window_key = int(current_time // self.window_size)

        # Add to current window
        self.event_windows[window_key].append(event)

        # Update metrics
        self._update_metrics(event)

        # Periodic analysis
        if current_time - self.last_analysis > 60:  # Every minute
            await self._perform_analysis()
            self.last_analysis = current_time

    def _update_metrics(self, event: Event):
        """Update real-time metrics"""
        self.metrics['total_events'] += 1
        self.metrics[f'events_{event.event_type}'] += 1

        if event.event_type == 'project_search':
            response_time = event.data.get('response_time_ms', 0)
            self.metrics['total_search_time'] += response_time
            self.metrics['search_count'] += 1
            self.metrics['avg_search_time'] = (
                self.metrics['total_search_time'] / self.metrics['search_count']
            )

        elif event.event_type == 'error':
            self.metrics['error_count'] += 1
            self.metrics['error_rate'] = (
                self.metrics['error_count'] / self.metrics['total_events']
            )

    async def _perform_analysis(self):
        """Perform periodic analysis of event patterns"""
        try:
            current_window = int(time.time() // self.window_size)

            # Analyze recent windows
            recent_events = []
            for window_key in range(current_window - 5, current_window + 1):
                recent_events.extend(self.event_windows.get(window_key, []))

            if recent_events:
                analysis = self._analyze_event_patterns(recent_events)
                logger.info(f"Real-time analysis: {analysis}")

                # Check for anomalies
                await self._check_anomalies(analysis)

        except Exception as e:
            logger.error(f"Error in streaming analysis: {e}")

    def _analyze_event_patterns(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze patterns in recent events"""
        if not events:
            return {}

        event_types = defaultdict(int)
        response_times = []
        error_events = []
        user_activity = defaultdict(int)

        for event in events:
            event_types[event.event_type] += 1

            if event.user_id:
                user_activity[event.user_id] += 1

            if event.event_type == 'project_search':
                response_time = event.data.get('response_time_ms', 0)
                if response_time > 0:
                    response_times.append(response_time)

            elif event.event_type == 'error':
                error_events.append(event)

        analysis = {
            'total_events': len(events),
            'event_type_distribution': dict(event_types),
            'unique_users': len(user_activity),
            'most_active_user': max(user_activity.items(), key=lambda x: x[1]) if user_activity else None,
        }

        if response_times:
            analysis['search_performance'] = {
                'avg_response_time_ms': sum(response_times) / len(response_times),
                'max_response_time_ms': max(response_times),
                'min_response_time_ms': min(response_times),
                'slow_searches': sum(1 for rt in response_times if rt > 5000)
            }

        if error_events:
            error_types = defaultdict(int)
            for error_event in error_events:
                error_type = error_event.data.get('error_type', 'unknown')
                error_types[error_type] += 1

            analysis['error_analysis'] = {
                'total_errors': len(error_events),
                'error_rate': len(error_events) / len(events),
                'error_types': dict(error_types)
            }

        return analysis

    async def _check_anomalies(self, analysis: Dict[str, Any]):
        """Check for anomalies and alert if necessary"""
        thresholds = self.config.monitoring.alert_thresholds

        # Check error rate
        error_rate = analysis.get('error_analysis', {}).get('error_rate', 0)
        if error_rate > thresholds.get('pipeline_failure_rate', 0.05):
            logger.warning(f"High error rate detected: {error_rate:.2%}")

        # Check response times
        search_perf = analysis.get('search_performance', {})
        avg_response_time = search_perf.get('avg_response_time_ms', 0)
        if avg_response_time > thresholds.get('avg_response_time_ms', 5000):
            logger.warning(f"Slow response times detected: {avg_response_time:.0f}ms")

        # Check for unusual patterns
        event_distribution = analysis.get('event_type_distribution', {})
        total_events = analysis.get('total_events', 0)

        if total_events > 0:
            search_ratio = event_distribution.get('project_search', 0) / total_events
            if search_ratio > 0.8:  # More than 80% searches might indicate issues
                logger.warning(f"Unusually high search activity: {search_ratio:.2%}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current streaming metrics"""
        return dict(self.metrics)


class DatabaseEventSink:
    """Persists streaming events to database"""

    def __init__(self, loader: DuckDBLoader = None):
        self.config = get_config()
        self.loader = loader or DuckDBLoader()
        self.event_buffer = []
        self.last_flush = time.time()
        self._setup_event_tables()

    def _setup_event_tables(self):
        """Setup database tables for events"""
        schema_sql = """
        -- User interactions table
        CREATE TABLE IF NOT EXISTS quicknav.user_interactions (
            interaction_id VARCHAR PRIMARY KEY,
            event_type VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            user_id VARCHAR,
            timestamp TIMESTAMP NOT NULL,
            project_id VARCHAR,
            document_path VARCHAR,
            search_query VARCHAR,
            response_time_ms DOUBLE,
            event_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Session summary table
        CREATE TABLE IF NOT EXISTS quicknav.user_sessions (
            session_id VARCHAR PRIMARY KEY,
            user_id VARCHAR,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_seconds DOUBLE,
            events_count INTEGER DEFAULT 0,
            projects_accessed INTEGER DEFAULT 0,
            documents_opened INTEGER DEFAULT 0,
            search_queries INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indices for performance
        CREATE INDEX IF NOT EXISTS idx_interactions_session ON quicknav.user_interactions(session_id);
        CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON quicknav.user_interactions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_interactions_user ON quicknav.user_interactions(user_id);
        CREATE INDEX IF NOT EXISTS idx_interactions_type ON quicknav.user_interactions(event_type);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON quicknav.user_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_start ON quicknav.user_sessions(start_time);
        """

        self.loader.db.executescript(schema_sql)

    async def persist_event(self, event: Event):
        """Add event to buffer for persistence"""
        # Convert event to database record
        record = {
            'interaction_id': event.event_id,
            'event_type': event.event_type,
            'session_id': event.session_id,
            'user_id': event.user_id,
            'timestamp': event.timestamp,
            'project_id': event.data.get('project_id'),
            'document_path': event.data.get('document_path'),
            'search_query': event.data.get('query'),
            'response_time_ms': event.data.get('response_time_ms'),
            'event_data': json.dumps(event.data)
        }

        self.event_buffer.append(record)

        # Flush if buffer is full or time threshold reached
        if (len(self.event_buffer) >= self.config.streaming.batch_size or
            time.time() - self.last_flush > self.config.streaming.flush_interval_seconds):
            await self._flush_buffer()

    async def _flush_buffer(self):
        """Flush event buffer to database"""
        if not self.event_buffer:
            return

        try:
            # Convert to DataFrame and insert
            import pandas as pd
            df = pd.DataFrame(self.event_buffer)

            self.loader.db.execute("""
                INSERT INTO quicknav.user_interactions
                SELECT * FROM df
            """)

            logger.debug(f"Flushed {len(self.event_buffer)} events to database")

            self.event_buffer.clear()
            self.last_flush = time.time()

        except Exception as e:
            logger.error(f"Error flushing events to database: {e}")

    async def persist_session_summary(self, session_summary: Dict[str, Any]):
        """Persist session summary to database"""
        try:
            record = {
                'session_id': session_summary['session_id'],
                'user_id': session_summary.get('user_id'),
                'start_time': datetime.fromisoformat(session_summary['start_time']),
                'end_time': datetime.utcnow(),
                'duration_seconds': session_summary['duration_seconds'],
                'events_count': session_summary['events_count'],
                'projects_accessed': session_summary['projects_accessed'],
                'documents_opened': session_summary['documents_opened'],
                'search_queries': session_summary['search_queries']
            }

            self.loader.db.execute("""
                INSERT OR REPLACE INTO quicknav.user_sessions
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record['session_id'],
                record['user_id'],
                record['start_time'],
                record['end_time'],
                record['duration_seconds'],
                record['events_count'],
                record['projects_accessed'],
                record['documents_opened'],
                record['search_queries']
            ])

        except Exception as e:
            logger.error(f"Error persisting session summary: {e}")


class StreamingPipeline:
    """Main streaming pipeline orchestrator"""

    def __init__(self):
        self.config = get_config()
        self.event_queue = EventQueue(self.config.streaming.event_queue_size)
        self.event_processor = EventProcessor(self.event_queue)
        self.behavior_tracker = UserBehaviorTracker(self.event_queue)
        self.analyzer = StreamingAnalyzer()
        self.db_sink = DatabaseEventSink()

        self._setup_handlers()
        self._running = False
        self._tasks = []

    def _setup_handlers(self):
        """Setup event handlers"""
        # Database persistence
        self.event_processor.register_global_handler(self.db_sink.persist_event)

        # Real-time analysis
        self.event_processor.register_global_handler(self.analyzer.analyze_event)

        # Session end handler
        async def handle_session_end(event: Event):
            session_summary = self.behavior_tracker.get_session_summary(event.session_id)
            if session_summary:
                await self.db_sink.persist_session_summary(session_summary)

        self.event_processor.register_handler('session_end', handle_session_end)

    async def start(self):
        """Start the streaming pipeline"""
        if self._running:
            logger.warning("Streaming pipeline is already running")
            return

        logger.info("Starting streaming pipeline")
        self._running = True

        # Start event processor
        processor_task = asyncio.create_task(
            self.event_processor.process_events(
                batch_size=self.config.streaming.batch_size,
                flush_interval=self.config.streaming.flush_interval_seconds
            )
        )
        self._tasks.append(processor_task)

        # Start periodic cleanup
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._tasks.append(cleanup_task)

        logger.info("Streaming pipeline started successfully")

    async def stop(self):
        """Stop the streaming pipeline"""
        if not self._running:
            return

        logger.info("Stopping streaming pipeline")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Flush any remaining events
        await self.db_sink._flush_buffer()

        logger.info("Streaming pipeline stopped")

    async def _periodic_cleanup(self):
        """Periodic cleanup of old data"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Clean up old event windows
                current_time = time.time()
                window_size = self.analyzer.window_size
                cutoff_window = int((current_time - 86400) // window_size)  # 24 hours ago

                for window_key in list(self.analyzer.event_windows.keys()):
                    if window_key < cutoff_window:
                        del self.analyzer.event_windows[window_key]

                # Clean up old sessions
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                for session_id in list(self.behavior_tracker.session_data.keys()):
                    session = self.behavior_tracker.session_data[session_id]
                    if session['start_time'] < cutoff_time:
                        self.behavior_tracker.end_session(session_id)

                logger.debug("Performed periodic cleanup")

            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'running': self._running,
            'queue_stats': self.event_queue.get_stats(),
            'processor_stats': self.event_processor.get_stats(),
            'analyzer_metrics': self.analyzer.get_current_metrics(),
            'active_sessions': len(self.behavior_tracker.session_data),
            'config': {
                'batch_size': self.config.streaming.batch_size,
                'flush_interval': self.config.streaming.flush_interval_seconds,
                'queue_size': self.config.streaming.event_queue_size
            }
        }

    # Convenience methods for external integration
    def track_search(self, session_id: str, query: str, results: List[str],
                    response_time_ms: float):
        """Track a search operation"""
        self.behavior_tracker.track_project_search(session_id, query, results, response_time_ms)

    def track_project_access(self, session_id: str, project_id: str, project_path: str):
        """Track project access"""
        self.behavior_tracker.track_project_access(session_id, project_id, project_path)

    def track_document_open(self, session_id: str, document_path: str, doc_type: str,
                          project_id: str = None):
        """Track document opening"""
        self.behavior_tracker.track_document_open(session_id, document_path, doc_type, project_id)

    def track_ai_interaction(self, session_id: str, query: str, response_type: str,
                           processing_time_ms: float):
        """Track AI interaction"""
        self.behavior_tracker.track_ai_interaction(session_id, query, response_type, processing_time_ms)

    def track_error(self, session_id: str, error_type: str, error_message: str,
                   context: Dict[str, Any] = None):
        """Track an error"""
        self.behavior_tracker.track_error(session_id, error_type, error_message, context)

    def start_session(self, user_id: str = None) -> str:
        """Start a new user session"""
        return self.behavior_tracker.start_session(str(uuid.uuid4()), user_id)

    def end_session(self, session_id: str):
        """End a user session"""
        self.behavior_tracker.end_session(session_id)


# Global streaming pipeline instance
_streaming_pipeline: Optional[StreamingPipeline] = None


def get_streaming_pipeline() -> StreamingPipeline:
    """Get global streaming pipeline instance"""
    global _streaming_pipeline
    if _streaming_pipeline is None:
        _streaming_pipeline = StreamingPipeline()
    return _streaming_pipeline


async def start_streaming_pipeline():
    """Start the global streaming pipeline"""
    pipeline = get_streaming_pipeline()
    await pipeline.start()


async def stop_streaming_pipeline():
    """Stop the global streaming pipeline"""
    global _streaming_pipeline
    if _streaming_pipeline:
        await _streaming_pipeline.stop()
        _streaming_pipeline = None