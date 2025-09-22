"""
Production Data Pipelines for Project QuickNav

This package provides comprehensive data pipeline infrastructure for:
- Real-time document ingestion and monitoring
- Batch processing for analytics and ML features
- User activity streaming and behavior tracking
- Data quality monitoring and validation
- Performance metrics and alerting

Architecture:
- Async-first design for responsive UI integration
- Modular pipeline components with clear interfaces
- Robust error handling and retry mechanisms
- Comprehensive monitoring and observability
- Cross-platform compatibility (Windows, macOS, Linux)
"""

from .core.pipeline_manager import PipelineManager
from .core.scheduler import AsyncScheduler
from .core.worker_pool import WorkerPool
from .ingestion.document_monitor import DocumentMonitor
from .ingestion.realtime_processor import RealtimeProcessor
from .batch.analytics_processor import AnalyticsProcessor
from .batch.ml_feature_processor import MLFeatureProcessor
from .streaming.activity_streamer import ActivityStreamer
from .quality.data_validator import DataValidator
from .quality.anomaly_detector import AnomalyDetector
from .monitoring.metrics_collector import MetricsCollector
from .monitoring.alerting import AlertManager
from .config.pipeline_config import PipelineConfig

__version__ = "1.0.0"
__all__ = [
    "PipelineManager",
    "AsyncScheduler",
    "WorkerPool",
    "DocumentMonitor",
    "RealtimeProcessor",
    "AnalyticsProcessor",
    "MLFeatureProcessor",
    "ActivityStreamer",
    "DataValidator",
    "AnomalyDetector",
    "MetricsCollector",
    "AlertManager",
    "PipelineConfig"
]