"""
ML Pipeline Core Infrastructure

Provides the foundational components for model management, configuration,
and orchestration across the QuickNav ML pipeline.
"""

import os
import json
import time
import logging
import asyncio
import pickle
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..database.database_manager import DatabaseManager, get_database_manager
from ..data_pipeline.feature_store import FeatureStore, FeatureEngineer

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ModelStatus(Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelType(Enum):
    """Supported model types"""
    RECOMMENDATION = "recommendation"
    RANKING = "ranking"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class ModelConfig:
    """Model configuration and metadata"""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    description: str

    # Training configuration
    features: List[str]
    target_metric: str
    hyperparameters: Dict[str, Any]

    # Deployment configuration
    serving_config: Dict[str, Any]
    fallback_strategy: str = "default"
    max_batch_size: int = 100
    timeout_ms: int = 5000

    # Metadata
    created_at: datetime = None
    updated_at: datetime = None
    status: ModelStatus = ModelStatus.TRAINING
    performance_metrics: Dict[str, float] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        data = data.copy()
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class BaseModel(Generic[T]):
    """Base class for all ML models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names: List[str] = config.features.copy()
        self.metadata: Dict[str, Any] = {}

    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              validation_data: Optional[tuple] = None) -> Dict[str, float]:
        """Train the model"""
        raise NotImplementedError("Subclasses must implement train method")

    def predict(self, X: np.ndarray) -> Union[np.ndarray, List[T]]:
        """Make predictions"""
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for applicable models)"""
        raise NotImplementedError("Predict proba not available for this model type")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return {}

    def save(self, path: str) -> bool:
        """Save model to disk"""
        try:
            model_data = {
                'config': self.config.to_dict(),
                'model': self.model,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'metadata': self.metadata
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model {self.config.model_id} saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model {self.config.model_id}: {e}")
            return False

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            config = ModelConfig.from_dict(model_data['config'])
            instance = cls(config)
            instance.model = model_data['model']
            instance.is_trained = model_data['is_trained']
            instance.feature_names = model_data['feature_names']
            instance.metadata = model_data['metadata']

            logger.info(f"Model {config.model_id} loaded from {path}")
            return instance

        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise


class ModelRegistry:
    """Central registry for managing ML models"""

    def __init__(self, db_manager: DatabaseManager = None,
                 storage_path: Optional[str] = None):
        self.db = db_manager or get_database_manager()
        self.storage_path = Path(storage_path) if storage_path else self._get_default_storage_path()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, BaseModel] = {}
        self._model_locks: Dict[str, threading.RLock] = {}
        self._registry_lock = threading.RLock()

        self._setup_registry_tables()
        self._load_active_models()

    def _get_default_storage_path(self) -> Path:
        """Get default model storage path"""
        if os.name == 'nt':  # Windows
            base_path = Path(os.environ.get('APPDATA', '')) / 'QuickNav' / 'models'
        else:  # Unix-like
            base_path = Path.home() / '.local' / 'share' / 'quicknav' / 'models'
        return base_path

    def _setup_registry_tables(self):
        """Setup model registry database tables"""
        schema_sql = """
        -- Model registry table
        CREATE TABLE IF NOT EXISTS ml_model_registry (
            model_id VARCHAR PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            description TEXT,
            config_json TEXT NOT NULL,
            file_path VARCHAR,
            status VARCHAR NOT NULL,
            performance_metrics TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            deployed_at TIMESTAMP,
            deprecated_at TIMESTAMP
        );

        -- Model predictions log
        CREATE TABLE IF NOT EXISTS ml_predictions_log (
            prediction_id VARCHAR PRIMARY KEY,
            model_id VARCHAR NOT NULL,
            input_hash VARCHAR,
            prediction_data TEXT,
            confidence_score REAL,
            response_time_ms REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR,
            user_id VARCHAR,
            FOREIGN KEY (model_id) REFERENCES ml_model_registry(model_id)
        );

        -- Model performance tracking
        CREATE TABLE IF NOT EXISTS ml_model_performance (
            metric_id VARCHAR PRIMARY KEY,
            model_id VARCHAR NOT NULL,
            metric_name VARCHAR NOT NULL,
            metric_value REAL NOT NULL,
            measurement_date DATE NOT NULL,
            data_window_days INTEGER,
            metadata TEXT,
            FOREIGN KEY (model_id) REFERENCES ml_model_registry(model_id)
        );

        -- Indices
        CREATE INDEX IF NOT EXISTS idx_model_registry_type_status ON ml_model_registry(model_type, status);
        CREATE INDEX IF NOT EXISTS idx_predictions_model_time ON ml_predictions_log(model_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_model_metric ON ml_model_performance(model_id, metric_name);
        """

        try:
            self.db.execute_write(schema_sql, use_transaction=False)
            logger.info("Model registry tables initialized")
        except Exception as e:
            logger.error(f"Error setting up model registry tables: {e}")

    def register_model(self, model: BaseModel, file_path: Optional[str] = None) -> bool:
        """Register a model in the registry"""
        try:
            with self._registry_lock:
                model_id = model.config.model_id

                # Save model to storage if no file path provided
                if file_path is None:
                    file_path = str(self.storage_path / f"{model_id}.pkl")
                    if not model.save(file_path):
                        return False

                # Insert into database
                query = """
                INSERT OR REPLACE INTO ml_model_registry
                (model_id, model_name, model_type, version, description, config_json,
                 file_path, status, performance_metrics, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                params = [
                    model_id,
                    model.config.model_name,
                    model.config.model_type.value,
                    model.config.version,
                    model.config.description,
                    json.dumps(model.config.to_dict()),
                    file_path,
                    model.config.status.value,
                    json.dumps(model.config.performance_metrics),
                    model.config.created_at,
                    model.config.updated_at
                ]

                success = self.db.execute_write(query, params)

                if success:
                    # Cache model in memory
                    self._models[model_id] = model
                    self._model_locks[model_id] = threading.RLock()
                    logger.info(f"Model {model_id} registered successfully")

                return success

        except Exception as e:
            logger.error(f"Error registering model {model.config.model_id}: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Get a model from the registry"""
        with self._registry_lock:
            # Check memory cache first
            if model_id in self._models:
                return self._models[model_id]

            # Load from database
            try:
                query = "SELECT file_path FROM ml_model_registry WHERE model_id = ?"
                results = self.db.execute_query(query, [model_id])

                if not results:
                    return None

                file_path = results[0]['file_path']
                if not file_path or not os.path.exists(file_path):
                    logger.warning(f"Model file not found for {model_id}: {file_path}")
                    return None

                # Load model
                model = BaseModel.load(file_path)

                # Cache in memory
                self._models[model_id] = model
                self._model_locks[model_id] = threading.RLock()

                return model

            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}")
                return None

    def list_models(self, model_type: Optional[ModelType] = None,
                   status: Optional[ModelStatus] = None) -> List[Dict[str, Any]]:
        """List models in the registry"""
        try:
            query = "SELECT * FROM ml_model_registry WHERE 1=1"
            params = []

            if model_type:
                query += " AND model_type = ?"
                params.append(model_type.value)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY created_at DESC"

            return self.db.execute_query(query, params)

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def update_model_status(self, model_id: str, status: ModelStatus,
                           performance_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Update model status and performance metrics"""
        try:
            query = """
            UPDATE ml_model_registry
            SET status = ?, updated_at = ?
            """
            params = [status.value, datetime.utcnow()]

            if performance_metrics:
                query += ", performance_metrics = ?"
                params.append(json.dumps(performance_metrics))

            if status == ModelStatus.DEPLOYED:
                query += ", deployed_at = ?"
                params.append(datetime.utcnow())
            elif status == ModelStatus.DEPRECATED:
                query += ", deprecated_at = ?"
                params.append(datetime.utcnow())

            query += " WHERE model_id = ?"
            params.append(model_id)

            success = self.db.execute_write(query, params)

            # Update cached model config if exists
            if model_id in self._models:
                self._models[model_id].config.status = status
                self._models[model_id].config.updated_at = datetime.utcnow()
                if performance_metrics:
                    self._models[model_id].config.performance_metrics.update(performance_metrics)

            return success

        except Exception as e:
            logger.error(f"Error updating model status for {model_id}: {e}")
            return False

    def log_prediction(self, model_id: str, input_data: Any, prediction: Any,
                      confidence_score: Optional[float] = None,
                      response_time_ms: Optional[float] = None,
                      session_id: Optional[str] = None,
                      user_id: Optional[str] = None) -> bool:
        """Log a model prediction"""
        try:
            prediction_id = str(uuid.uuid4())

            # Create hash of input data for deduplication
            input_hash = hashlib.md5(str(input_data).encode()).hexdigest()

            query = """
            INSERT INTO ml_predictions_log
            (prediction_id, model_id, input_hash, prediction_data, confidence_score,
             response_time_ms, session_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = [
                prediction_id, model_id, input_hash,
                json.dumps(prediction) if prediction is not None else None,
                confidence_score, response_time_ms, session_id, user_id
            ]

            return self.db.execute_write(query, params)

        except Exception as e:
            logger.error(f"Error logging prediction for model {model_id}: {e}")
            return False

    def get_model_performance(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            # Get recent predictions
            predictions_query = """
            SELECT COUNT(*) as prediction_count,
                   AVG(response_time_ms) as avg_response_time,
                   AVG(confidence_score) as avg_confidence
            FROM ml_predictions_log
            WHERE model_id = ? AND timestamp > datetime('now', '-{} days')
            """.format(days)

            predictions_stats = self.db.execute_query(predictions_query, [model_id])

            # Get performance metrics
            metrics_query = """
            SELECT metric_name, AVG(metric_value) as avg_value,
                   COUNT(*) as measurement_count
            FROM ml_model_performance
            WHERE model_id = ? AND measurement_date > date('now', '-{} days')
            GROUP BY metric_name
            """.format(days)

            metrics_stats = self.db.execute_query(metrics_query, [model_id])

            performance = {
                'model_id': model_id,
                'period_days': days,
                'prediction_stats': predictions_stats[0] if predictions_stats else {},
                'performance_metrics': {row['metric_name']: row['avg_value']
                                      for row in metrics_stats}
            }

            return performance

        except Exception as e:
            logger.error(f"Error getting performance for model {model_id}: {e}")
            return {}

    def cleanup_old_predictions(self, days: int = 90) -> int:
        """Clean up old prediction logs"""
        try:
            query = """
            DELETE FROM ml_predictions_log
            WHERE timestamp < datetime('now', '-{} days')
            """.format(days)

            result = self.db.execute_write(query)
            logger.info(f"Cleaned up old predictions older than {days} days")
            return 1 if result else 0

        except Exception as e:
            logger.error(f"Error cleaning up old predictions: {e}")
            return 0

    def _load_active_models(self):
        """Load active models into memory cache"""
        try:
            query = """
            SELECT model_id FROM ml_model_registry
            WHERE status IN ('deployed', 'validated')
            ORDER BY updated_at DESC
            LIMIT 10
            """

            results = self.db.execute_query(query)

            for row in results:
                model_id = row['model_id']
                try:
                    model = self.get_model(model_id)
                    if model:
                        logger.debug(f"Preloaded model {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to preload model {model_id}: {e}")

        except Exception as e:
            logger.error(f"Error preloading active models: {e}")


class MLPipeline:
    """Main ML pipeline orchestrator"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or get_database_manager()
        self.feature_store = FeatureStore()
        self.feature_engineer = FeatureEngineer()
        self.model_registry = ModelRegistry(self.db)

        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MLPipeline")
        self._shutdown = False

        logger.info("ML Pipeline initialized")

    async def prepare_training_data(self, entity_type: str, feature_names: List[str],
                                  days_back: int = 90) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from feature store"""
        try:
            # Get entities with sufficient activity
            query = """
            SELECT DISTINCT entity_id
            FROM quicknav.features
            WHERE entity_type = ? AND computed_at > datetime('now', '-{} days')
            """.format(days_back)

            results = self.db.execute_query(query, [entity_type], prefer_analytics=True)
            entity_ids = [row['entity_id'] for row in results]

            if not entity_ids:
                logger.warning(f"No entities found for training data: {entity_type}")
                return np.array([]), np.array([])

            # Collect feature vectors
            feature_vectors = []
            labels = []

            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id, entity_type, feature_names)
                if len(features) == len(feature_names):
                    vector = self.feature_store.get_feature_vector(entity_id, entity_type, feature_names)
                    feature_vectors.append(vector)

                    # Create synthetic labels based on activity (for demo)
                    # In practice, these would come from actual user feedback/behavior
                    if 'popularity_score' in features:
                        label = 1 if features['popularity_score'].value > 0.5 else 0
                    else:
                        label = 0
                    labels.append(label)

            X = np.array(feature_vectors) if feature_vectors else np.array([])
            y = np.array(labels) if labels else np.array([])

            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1] if len(X.shape) > 1 else 0} features")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])

    def train_model_async(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                         validation_data: Optional[tuple] = None) -> bool:
        """Train a model asynchronously"""
        try:
            # Update status to training
            self.model_registry.update_model_status(
                model.config.model_id, ModelStatus.TRAINING
            )

            # Train the model
            start_time = time.time()
            metrics = model.train(X, y, validation_data)
            training_time = time.time() - start_time

            # Update model status and metrics
            metrics['training_time_seconds'] = training_time
            self.model_registry.update_model_status(
                model.config.model_id, ModelStatus.TRAINED, metrics
            )

            # Register the trained model
            success = self.model_registry.register_model(model)

            if success:
                logger.info(f"Model {model.config.model_id} trained successfully")
                logger.info(f"Training metrics: {metrics}")

            return success

        except Exception as e:
            logger.error(f"Error training model {model.config.model_id}: {e}")
            self.model_registry.update_model_status(
                model.config.model_id, ModelStatus.FAILED
            )
            return False

    async def batch_feature_computation(self, entity_ids: List[str],
                                      entity_type: str) -> Dict[str, bool]:
        """Compute features for multiple entities in parallel"""
        results = {}

        # Process in batches to avoid overwhelming the system
        batch_size = 20
        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i:i + batch_size]

            # Submit tasks
            tasks = []
            for entity_id in batch:
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.feature_store.compute_and_store_features,
                    entity_id, entity_type, self.feature_engineer
                )
                tasks.append((entity_id, task))

            # Collect results
            for entity_id, task in tasks:
                try:
                    success = await task
                    results[entity_id] = success
                except Exception as e:
                    logger.error(f"Error computing features for {entity_id}: {e}")
                    results[entity_id] = False

            # Small delay between batches
            await asyncio.sleep(0.1)

        return results

    def get_model_recommendations(self, model_type: ModelType,
                                status: ModelStatus = ModelStatus.DEPLOYED) -> List[str]:
        """Get recommended models for a given type and status"""
        models = self.model_registry.list_models(model_type, status)

        # Sort by performance and recency
        def score_model(model):
            perf_metrics = json.loads(model.get('performance_metrics', '{}'))
            recency_score = 1.0 / max(1, (datetime.utcnow() -
                                        datetime.fromisoformat(model['updated_at'])).days)

            # Combine performance score with recency
            perf_score = perf_metrics.get('accuracy', perf_metrics.get('f1_score', 0.5))
            return perf_score * 0.7 + recency_score * 0.3

        sorted_models = sorted(models, key=score_model, reverse=True)
        return [model['model_id'] for model in sorted_models[:5]]

    def health_check(self) -> Dict[str, Any]:
        """Perform ML pipeline health check"""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.utcnow().isoformat()
        }

        try:
            # Check database connection
            stats = self.db.get_database_stats()
            health['components']['database'] = {
                'status': 'healthy' if stats else 'unhealthy',
                'stats': stats
            }

            # Check feature store
            feature_stats = self.feature_store.get_feature_statistics()
            health['components']['feature_store'] = {
                'status': 'healthy' if feature_stats else 'unhealthy',
                'stats': feature_stats
            }

            # Check model registry
            models = self.model_registry.list_models()
            deployed_models = [m for m in models if m['status'] == 'deployed']
            health['components']['model_registry'] = {
                'status': 'healthy',
                'total_models': len(models),
                'deployed_models': len(deployed_models)
            }

            # Overall status
            component_statuses = [comp['status'] for comp in health['components'].values()]
            if any(status == 'unhealthy' for status in component_statuses):
                health['status'] = 'degraded'

        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"ML Pipeline health check failed: {e}")

        return health

    def shutdown(self):
        """Shutdown the ML pipeline"""
        self._shutdown = True
        self.executor.shutdown(wait=True)
        logger.info("ML Pipeline shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()