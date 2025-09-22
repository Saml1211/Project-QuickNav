"""
ML Pipeline for Project QuickNav

Production-ready machine learning pipeline for enhanced project navigation,
document ranking, and user behavior prediction.

Components:
- Feature Engineering: User behavior, document, temporal, and collaborative features
- Model Training: Recommendation systems, ranking models, intent prediction
- Model Serving: Real-time inference with fallbacks and monitoring
- A/B Testing: Framework for model evaluation and gradual rollouts
- Monitoring: Model performance tracking and drift detection

Integration:
- Database: DuckDB/SQLite for feature storage and analytics
- MCP Server: ML tools and resources exposure
- GUI: Enhanced search and recommendations
- CLI: Batch processing and model management
"""

from .core import MLPipeline, ModelConfig, ModelRegistry
from .models import (
    RecommendationSystem,
    DocumentRanker,
    UserIntentPredictor,
    AnomalyDetector
)
from .serving import ModelServer, PredictionService
from .training import ModelTrainer, ValidationFramework
from .monitoring import ModelMonitor, DriftDetector
from .ab_testing import ABTestFramework, ExperimentManager

__version__ = "1.0.0"
__all__ = [
    "MLPipeline",
    "ModelConfig",
    "ModelRegistry",
    "RecommendationSystem",
    "DocumentRanker",
    "UserIntentPredictor",
    "AnomalyDetector",
    "ModelServer",
    "PredictionService",
    "ModelTrainer",
    "ValidationFramework",
    "ModelMonitor",
    "DriftDetector",
    "ABTestFramework",
    "ExperimentManager"
]