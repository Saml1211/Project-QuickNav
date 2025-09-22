"""
Project QuickNav Data Pipeline

This module provides comprehensive data engineering capabilities for Project QuickNav,
including ETL/ELT pipelines, streaming data processing, feature stores, and analytics.

Key Components:
- ETL Pipeline: Batch processing for document indexing and metadata extraction
- Streaming Pipeline: Real-time user behavior tracking and event processing
- Feature Store: ML-ready features for search ranking and recommendations
- Analytics Engine: Performance monitoring and usage analytics
- Cache Layer: Multi-level caching for optimal performance

Integration:
- Seamlessly integrates with existing find_project_path.py and doc_navigator.py
- Maintains backward compatibility with current GUI and CLI interfaces
- Provides enhanced search and recommendation capabilities
"""

__version__ = "1.0.0"
__author__ = "Project QuickNav Data Engineering Team"

from .etl import ETLPipeline, OneDriveExtractor, DocumentTransformer, DuckDBLoader
from .streaming import UserBehaviorStream, EventProcessor, StreamingPipeline
from .feature_store import FeatureStore, FeatureEngineer, FeatureServer
from .analytics import AnalyticsEngine, PipelineMonitor, UsageAnalyzer
from .cache import MultiLevelCache, CacheManager
from .integration import DataPipelineIntegration, EnhancedProjectSearch

__all__ = [
    # ETL Components
    'ETLPipeline',
    'OneDriveExtractor',
    'DocumentTransformer',
    'DuckDBLoader',

    # Streaming Components
    'UserBehaviorStream',
    'EventProcessor',
    'StreamingPipeline',

    # Feature Store
    'FeatureStore',
    'FeatureEngineer',
    'FeatureServer',

    # Analytics
    'AnalyticsEngine',
    'PipelineMonitor',
    'UsageAnalyzer',

    # Caching
    'MultiLevelCache',
    'CacheManager',

    # Integration
    'DataPipelineIntegration',
    'EnhancedProjectSearch'
]