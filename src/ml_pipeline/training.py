"""
Model Training and Validation Framework

Provides comprehensive training infrastructure with:
- Cross-validation with temporal splits
- Hyperparameter optimization
- Model performance evaluation
- Training pipeline orchestration
- Automated feature selection
"""

import numpy as np
import pandas as pd
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Try to import sklearn components
try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .core import BaseModel, ModelConfig, ModelType, ModelStatus, ModelRegistry
from .models import RecommendationSystem, DocumentRanker, UserIntentPredictor, AnomalyDetector
from ..database.database_manager import DatabaseManager, get_database_manager
from ..data_pipeline.feature_store import FeatureStore, FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    experiment_name: str
    model_type: ModelType

    # Data configuration
    train_test_split_ratio: float = 0.8
    validation_split_ratio: float = 0.1
    temporal_split: bool = True
    min_samples: int = 100

    # Training configuration
    cross_validation_folds: int = 5
    hyperparameter_search: bool = True
    feature_selection: bool = True
    max_features: Optional[int] = None

    # Performance configuration
    target_metric: str = 'accuracy'
    min_metric_threshold: float = 0.7
    early_stopping: bool = True
    max_training_time_minutes: int = 60

    # Resource configuration
    max_parallel_jobs: int = 4
    use_gpu: bool = False
    memory_limit_gb: Optional[float] = None

    # Output configuration
    save_intermediate_models: bool = True
    generate_reports: bool = True

    def __post_init__(self):
        if self.max_features is None:
            self.max_features = 50  # Default


@dataclass
class TrainingResult:
    """Result from model training"""
    experiment_id: str
    model_id: str
    config: TrainingConfig

    # Performance metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cross_validation_metrics: Dict[str, float]

    # Training metadata
    training_duration_seconds: float
    best_hyperparameters: Dict[str, Any]
    selected_features: List[str]

    # Model artifacts
    model_path: str
    feature_importance: Dict[str, float]
    training_history: List[Dict[str, Any]]

    # Status
    status: str  # 'success', 'failed', 'timeout'
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result


class ModelTrainer:
    """Main model training orchestrator"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or get_database_manager()
        self.feature_store = FeatureStore()
        self.feature_engineer = FeatureEngineer()
        self.model_registry = ModelRegistry(self.db)

        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ModelTrainer")
        self._setup_training_tables()

    def _setup_training_tables(self):
        """Setup training experiment tracking tables"""
        schema_sql = """
        -- Training experiments table
        CREATE TABLE IF NOT EXISTS ml_training_experiments (
            experiment_id VARCHAR PRIMARY KEY,
            experiment_name VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            config_json TEXT NOT NULL,
            status VARCHAR NOT NULL,
            started_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            duration_seconds REAL,
            best_model_id VARCHAR,
            metrics_json TEXT,
            error_message TEXT
        );

        -- Training runs table (for hyperparameter search)
        CREATE TABLE IF NOT EXISTS ml_training_runs (
            run_id VARCHAR PRIMARY KEY,
            experiment_id VARCHAR NOT NULL,
            hyperparameters_json TEXT NOT NULL,
            metrics_json TEXT,
            status VARCHAR NOT NULL,
            started_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            duration_seconds REAL,
            FOREIGN KEY (experiment_id) REFERENCES ml_training_experiments(experiment_id)
        );

        -- Feature selection results
        CREATE TABLE IF NOT EXISTS ml_feature_selection (
            selection_id VARCHAR PRIMARY KEY,
            experiment_id VARCHAR NOT NULL,
            method VARCHAR NOT NULL,
            selected_features TEXT NOT NULL,
            feature_scores TEXT,
            selection_criteria TEXT,
            FOREIGN KEY (experiment_id) REFERENCES ml_training_experiments(experiment_id)
        );

        -- Indices
        CREATE INDEX IF NOT EXISTS idx_experiments_type_status ON ml_training_experiments(model_type, status);
        CREATE INDEX IF NOT EXISTS idx_runs_experiment ON ml_training_runs(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_feature_selection_experiment ON ml_feature_selection(experiment_id);
        """

        try:
            self.db.execute_write(schema_sql, use_transaction=False)
            logger.info("Training tables initialized")
        except Exception as e:
            logger.error(f"Error setting up training tables: {e}")

    async def train_model(self, config: TrainingConfig,
                         hyperparameter_grid: Optional[Dict[str, List[Any]]] = None) -> TrainingResult:
        """
        Train a model with the given configuration

        Args:
            config: Training configuration
            hyperparameter_grid: Optional hyperparameter search space

        Returns:
            Training result with metrics and model artifacts
        """
        experiment_id = str(uuid.uuid4())
        start_time = time.time()

        # Log experiment start
        await self._log_experiment_start(experiment_id, config)

        try:
            # Load and prepare training data
            logger.info(f"Loading training data for {config.model_type.value}")
            X, y, feature_names = await self._load_training_data(config)

            if len(X) < config.min_samples:
                raise ValueError(f"Insufficient training data: {len(X)} < {config.min_samples}")

            # Feature selection
            if config.feature_selection:
                logger.info("Performing feature selection")
                X, selected_features = await self._perform_feature_selection(
                    X, y, feature_names, config, experiment_id
                )
            else:
                selected_features = feature_names

            # Create data splits
            splits = await self._create_data_splits(X, y, config)

            # Hyperparameter optimization
            if config.hyperparameter_search and hyperparameter_grid:
                logger.info("Starting hyperparameter optimization")
                best_params, best_metrics = await self._hyperparameter_search(
                    splits, selected_features, config, hyperparameter_grid, experiment_id
                )
            else:
                best_params = {}
                best_metrics = {}

            # Train final model with best parameters
            logger.info("Training final model")
            final_model, final_metrics = await self._train_final_model(
                splits, selected_features, best_params, config
            )

            # Cross-validation evaluation
            cv_metrics = await self._cross_validation_evaluation(
                X, y, final_model, config
            )

            # Save model
            model_path = await self._save_trained_model(final_model, experiment_id)

            # Register model
            await self._register_trained_model(final_model, experiment_id)

            training_duration = time.time() - start_time

            # Create training result
            result = TrainingResult(
                experiment_id=experiment_id,
                model_id=final_model.config.model_id,
                config=config,
                train_metrics=final_metrics.get('train', {}),
                validation_metrics=final_metrics.get('validation', {}),
                test_metrics=final_metrics.get('test', {}),
                cross_validation_metrics=cv_metrics,
                training_duration_seconds=training_duration,
                best_hyperparameters=best_params,
                selected_features=selected_features,
                model_path=model_path,
                feature_importance=final_model.get_feature_importance(),
                training_history=[],  # Would be populated during training
                status='success'
            )

            # Log experiment completion
            await self._log_experiment_completion(experiment_id, result)

            logger.info(f"Training completed successfully: {experiment_id}")
            return result

        except Exception as e:
            logger.error(f"Training failed for experiment {experiment_id}: {e}")

            # Log failure
            await self._log_experiment_failure(experiment_id, str(e))

            # Return failed result
            return TrainingResult(
                experiment_id=experiment_id,
                model_id="",
                config=config,
                train_metrics={},
                validation_metrics={},
                test_metrics={},
                cross_validation_metrics={},
                training_duration_seconds=time.time() - start_time,
                best_hyperparameters={},
                selected_features=[],
                model_path="",
                feature_importance={},
                training_history=[],
                status='failed',
                error_message=str(e)
            )

    async def _load_training_data(self, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare training data"""
        try:
            if config.model_type == ModelType.RECOMMENDATION:
                return await self._load_recommendation_data(config)
            elif config.model_type == ModelType.RANKING:
                return await self._load_ranking_data(config)
            elif config.model_type == ModelType.CLASSIFICATION:
                return await self._load_classification_data(config)
            elif config.model_type == ModelType.ANOMALY_DETECTION:
                return await self._load_anomaly_data(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise

    async def _load_recommendation_data(self, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load data for recommendation model training"""
        # Get user-item interaction matrix
        query = """
        SELECT
            session_id as user_proxy,
            project_id,
            COUNT(*) as interaction_count,
            SUM(CASE WHEN activity_type = 'document_open' THEN 2 ELSE 1 END) as weighted_score
        FROM user_activities
        WHERE timestamp > datetime('now', '-90 days')
        AND project_id IS NOT NULL
        GROUP BY session_id, project_id
        HAVING interaction_count >= 2
        """

        interactions = self.db.execute_query(query, prefer_analytics=True)

        if not interactions:
            raise ValueError("No interaction data found for recommendation training")

        # Create user-item matrix
        user_ids = sorted(set(row['user_proxy'] for row in interactions))
        item_ids = sorted(set(row['project_id'] for row in interactions))

        user_item_matrix = np.zeros((len(user_ids), len(item_ids)))

        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        for row in interactions:
            user_idx = user_to_idx[row['user_proxy']]
            item_idx = item_to_idx[row['project_id']]
            user_item_matrix[user_idx, item_idx] = row['weighted_score']

        # For recommendation systems, we'll use the matrix as both X and y
        # y could be binary (interaction/no interaction) or rating values
        X = user_item_matrix
        y = (user_item_matrix > 0).astype(int)  # Binary labels

        feature_names = [f"item_{item_id}" for item_id in item_ids]

        return X, y.flatten(), feature_names

    async def _load_ranking_data(self, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load data for document ranking model training"""
        # Get document features and user interactions for ranking
        query = """
        SELECT
            d.document_id,
            d.project_id,
            COUNT(ua.activity_id) as access_count,
            AVG(ua.response_time_ms) as avg_response_time,
            d.file_size_bytes,
            d.version_numeric,
            CASE WHEN d.status_tags LIKE '%as_built%' THEN 1 ELSE 0 END as is_as_built,
            d.status_weight
        FROM documents d
        LEFT JOIN user_activities ua ON ua.project_id = d.project_id
        WHERE d.updated_at > datetime('now', '-60 days')
        GROUP BY d.document_id
        HAVING access_count > 0
        """

        documents = self.db.execute_query(query, prefer_analytics=True)

        if not documents:
            raise ValueError("No document data found for ranking training")

        # Prepare feature matrix
        features = []
        labels = []

        for doc in documents:
            doc_features = [
                float(doc['access_count']),
                float(doc['avg_response_time'] or 0),
                float(doc['file_size_bytes'] or 0) / (1024 * 1024),  # MB
                float(doc['version_numeric'] or 0),
                float(doc['is_as_built']),
                float(doc['status_weight'] or 0)
            ]
            features.append(doc_features)

            # Label: high access count = relevant (1), low = not relevant (0)
            label = 1 if doc['access_count'] > 5 else 0
            labels.append(label)

        feature_names = [
            'access_count', 'avg_response_time', 'file_size_mb',
            'version_numeric', 'is_as_built', 'status_weight'
        ]

        return np.array(features), np.array(labels), feature_names

    async def _load_classification_data(self, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load data for intent classification training"""
        # Get session data for intent prediction
        query = """
        SELECT
            s.session_id,
            COUNT(ua.activity_id) as total_interactions,
            COUNT(DISTINCT ua.project_id) as unique_projects,
            COUNT(CASE WHEN ua.activity_type = 'project_search' THEN 1 END) as search_count,
            COUNT(CASE WHEN ua.activity_type = 'document_open' THEN 1 END) as document_opens,
            AVG(ua.response_time_ms) as avg_response_time,
            (julianday(MAX(ua.timestamp)) - julianday(MIN(ua.timestamp))) * 1440 as session_duration_minutes
        FROM user_sessions s
        JOIN user_activities ua ON s.session_id = ua.session_id
        WHERE s.session_start > datetime('now', '-30 days')
        GROUP BY s.session_id
        HAVING total_interactions >= 3
        """

        sessions = self.db.execute_query(query, prefer_analytics=True)

        if not sessions:
            raise ValueError("No session data found for classification training")

        features = []
        labels = []

        for session in sessions:
            # Extract features
            total_interactions = session['total_interactions']
            session_features = [
                float(total_interactions),
                float(session['unique_projects']),
                float(session['search_count']) / max(1, total_interactions),  # search ratio
                float(session['document_opens']) / max(1, total_interactions),  # document ratio
                float(session['avg_response_time'] or 0),
                float(session['session_duration_minutes'] or 0)
            ]
            features.append(session_features)

            # Create synthetic labels based on patterns (in practice, these would come from user feedback)
            search_ratio = session['search_count'] / max(1, total_interactions)
            doc_ratio = session['document_opens'] / max(1, total_interactions)

            if search_ratio > 0.5:
                label = 'search'
            elif doc_ratio > 0.6 and session['session_duration_minutes'] > 10:
                label = 'work'
            elif session['unique_projects'] > 3:
                label = 'compare'
            else:
                label = 'browse'

            labels.append(label)

        feature_names = [
            'total_interactions', 'unique_projects', 'search_ratio',
            'document_ratio', 'avg_response_time', 'session_duration_minutes'
        ]

        # Encode labels
        if SKLEARN_AVAILABLE:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
        else:
            # Manual encoding
            label_map = {'browse': 0, 'search': 1, 'work': 2, 'compare': 3}
            encoded_labels = np.array([label_map.get(label, 0) for label in labels])

        return np.array(features), encoded_labels, feature_names

    async def _load_anomaly_data(self, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load data for anomaly detection training"""
        # Get system metrics and user behavior data
        query = """
        SELECT
            DATE(ua.timestamp) as date,
            COUNT(ua.activity_id) as daily_interactions,
            COUNT(DISTINCT ua.session_id) as unique_sessions,
            COUNT(DISTINCT ua.project_id) as unique_projects,
            AVG(ua.response_time_ms) as avg_response_time,
            COUNT(CASE WHEN ua.success = 0 THEN 1 END) as error_count,
            COUNT(CASE WHEN ua.activity_type = 'project_search' THEN 1 END) as search_count
        FROM user_activities ua
        WHERE ua.timestamp > datetime('now', '-90 days')
        GROUP BY DATE(ua.timestamp)
        ORDER BY date
        """

        daily_stats = self.db.execute_query(query, prefer_analytics=True)

        if not daily_stats:
            raise ValueError("No activity data found for anomaly detection training")

        features = []

        for stats in daily_stats:
            daily_features = [
                float(stats['daily_interactions']),
                float(stats['unique_sessions']),
                float(stats['unique_projects']),
                float(stats['avg_response_time'] or 0),
                float(stats['error_count']),
                float(stats['search_count'])
            ]
            features.append(daily_features)

        feature_names = [
            'daily_interactions', 'unique_sessions', 'unique_projects',
            'avg_response_time', 'error_count', 'search_count'
        ]

        # For anomaly detection, we don't have labels (unsupervised)
        # But we'll create dummy labels for compatibility
        X = np.array(features)
        y = np.zeros(len(features))  # Dummy labels

        return X, y, feature_names

    async def _perform_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                       feature_names: List[str], config: TrainingConfig,
                                       experiment_id: str) -> Tuple[np.ndarray, List[str]]:
        """Perform feature selection"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn not available, skipping feature selection")
                return X, feature_names

            n_features = min(config.max_features or len(feature_names), len(feature_names))

            if len(feature_names) <= n_features:
                return X, feature_names

            # Use SelectKBest with appropriate scoring function
            if config.model_type in [ModelType.CLASSIFICATION]:
                selector = SelectKBest(score_func=f_classif, k=n_features)
            else:
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features)

            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]

            # Log feature selection results
            feature_scores = dict(zip(feature_names, selector.scores_))
            await self._log_feature_selection(
                experiment_id, 'SelectKBest', selected_features, feature_scores
            )

            logger.info(f"Selected {len(selected_features)} features from {len(feature_names)}")
            return X_selected, selected_features

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return X, feature_names

    async def _create_data_splits(self, X: np.ndarray, y: np.ndarray,
                                config: TrainingConfig) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Create train/validation/test splits"""
        try:
            if SKLEARN_AVAILABLE:
                if config.temporal_split and len(X) > 100:
                    # Temporal split for time series data
                    n_train = int(len(X) * config.train_test_split_ratio)
                    n_val = int(len(X) * config.validation_split_ratio)

                    X_train = X[:n_train]
                    y_train = y[:n_train]

                    X_val = X[n_train:n_train + n_val]
                    y_val = y[n_train:n_train + n_val]

                    X_test = X[n_train + n_val:]
                    y_test = y[n_train + n_val:]
                else:
                    # Random split
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=1 - config.train_test_split_ratio, random_state=42
                    )

                    val_size = config.validation_split_ratio / config.train_test_split_ratio
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size, random_state=42
                    )
            else:
                # Manual split
                n_total = len(X)
                n_train = int(n_total * config.train_test_split_ratio)
                n_val = int(n_total * config.validation_split_ratio)

                X_train = X[:n_train]
                y_train = y[:n_train]

                X_val = X[n_train:n_train + n_val]
                y_val = y[n_train:n_train + n_val]

                X_test = X[n_train + n_val:]
                y_test = y[n_train + n_val:]

            return {
                'train': (X_train, y_train),
                'validation': (X_val, y_val),
                'test': (X_test, y_test)
            }

        except Exception as e:
            logger.error(f"Error creating data splits: {e}")
            raise

    async def _hyperparameter_search(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                   selected_features: List[str], config: TrainingConfig,
                                   hyperparameter_grid: Dict[str, List[Any]],
                                   experiment_id: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform hyperparameter search"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn not available, using default hyperparameters")
                return {}, {}

            param_combinations = list(ParameterGrid(hyperparameter_grid))
            best_params = {}
            best_score = -np.inf
            best_metrics = {}

            # Limit number of combinations for reasonable training time
            max_combinations = min(len(param_combinations), 20)
            param_combinations = param_combinations[:max_combinations]

            logger.info(f"Testing {len(param_combinations)} hyperparameter combinations")

            for i, params in enumerate(param_combinations):
                run_id = f"{experiment_id}_run_{i}"

                try:
                    # Create model with these parameters
                    model_config = ModelConfig(
                        model_id=f"temp_{run_id}",
                        model_name=f"temp_{config.experiment_name}",
                        model_type=config.model_type,
                        version="temp",
                        description="Temporary model for hyperparameter search",
                        features=selected_features,
                        target_metric=config.target_metric,
                        hyperparameters=params,
                        serving_config={}
                    )

                    model = self._create_model_instance(model_config)

                    # Train and evaluate
                    X_train, y_train = splits['train']
                    X_val, y_val = splits['validation']

                    train_metrics = model.train(X_train, y_train, (X_val, y_val))

                    # Evaluate on validation set
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val)
                        y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else (y_pred_proba > 0.5).astype(int)
                    else:
                        y_pred = model.predict(X_val)

                    # Calculate metrics
                    if config.model_type == ModelType.CLASSIFICATION:
                        score = accuracy_score(y_val, y_pred)
                        metrics = {
                            'accuracy': score,
                            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                            'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0)
                        }
                    else:
                        # For other types, use MSE or custom metrics
                        score = -mean_squared_error(y_val, y_pred)  # Negative because we want to maximize
                        metrics = {
                            'mse': -score,
                            'mae': mean_absolute_error(y_val, y_pred)
                        }

                    # Log run
                    await self._log_training_run(experiment_id, run_id, params, metrics)

                    # Update best parameters
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_metrics = metrics.copy()

                except Exception as e:
                    logger.error(f"Error in hyperparameter run {run_id}: {e}")
                    await self._log_training_run(experiment_id, run_id, params, {}, error=str(e))

            logger.info(f"Best hyperparameters: {best_params}")
            logger.info(f"Best score: {best_score}")

            return best_params, best_metrics

        except Exception as e:
            logger.error(f"Error in hyperparameter search: {e}")
            return {}, {}

    async def _train_final_model(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               selected_features: List[str], best_params: Dict[str, Any],
                               config: TrainingConfig) -> Tuple[BaseModel, Dict[str, Dict[str, float]]]:
        """Train the final model with best parameters"""
        try:
            # Create final model configuration
            model_config = ModelConfig(
                model_id=str(uuid.uuid4()),
                model_name=config.experiment_name,
                model_type=config.model_type,
                version="1.0",
                description=f"Model trained from experiment {config.experiment_name}",
                features=selected_features,
                target_metric=config.target_metric,
                hyperparameters=best_params,
                serving_config={
                    'max_batch_size': 100,
                    'timeout_ms': 5000,
                    'fallback_strategy': 'default'
                }
            )

            model = self._create_model_instance(model_config)

            # Train on combined train + validation data
            X_train, y_train = splits['train']
            X_val, y_val = splits['validation']
            X_test, y_test = splits['test']

            # Combine train and validation for final training
            X_train_final = np.vstack([X_train, X_val])
            y_train_final = np.hstack([y_train, y_val])

            # Train final model
            train_metrics = model.train(X_train_final, y_train_final)

            # Evaluate on all splits
            metrics = {
                'train': self._evaluate_model(model, X_train_final, y_train_final, config),
                'test': self._evaluate_model(model, X_test, y_test, config)
            }

            # Add training metrics
            metrics['train'].update(train_metrics)

            return model, metrics

        except Exception as e:
            logger.error(f"Error training final model: {e}")
            raise

    def _create_model_instance(self, config: ModelConfig) -> BaseModel:
        """Create model instance based on type"""
        if config.model_type == ModelType.RECOMMENDATION:
            return RecommendationSystem(config)
        elif config.model_type == ModelType.RANKING:
            return DocumentRanker(config)
        elif config.model_type == ModelType.CLASSIFICATION:
            return UserIntentPredictor(config)
        elif config.model_type == ModelType.ANOMALY_DETECTION:
            return AnomalyDetector(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def _evaluate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                       config: TrainingConfig) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if len(X) == 0:
                return {}

            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
                if y_pred_proba.ndim > 1:
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X)
                if hasattr(y_pred, '__len__') and len(y_pred) > 0:
                    if isinstance(y_pred[0], (list, tuple)):
                        # Handle complex prediction outputs (e.g., recommendations)
                        y_pred = np.array([1.0 if pred else 0.0 for pred in y_pred])
                    else:
                        y_pred = np.array(y_pred)

            # Calculate metrics based on model type
            if config.model_type == ModelType.CLASSIFICATION and SKLEARN_AVAILABLE:
                metrics = {
                    'accuracy': float(accuracy_score(y, y_pred)),
                    'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0))
                }
            elif SKLEARN_AVAILABLE:
                # Regression or other metrics
                metrics = {
                    'mse': float(mean_squared_error(y, y_pred)),
                    'mae': float(mean_absolute_error(y, y_pred)),
                    'r2': float(r2_score(y, y_pred)) if len(set(y)) > 1 else 0.0
                }
            else:
                # Basic metrics without sklearn
                if len(set(y)) > 1:  # Classification-like
                    accuracy = np.mean(y == y_pred) if len(y) == len(y_pred) else 0.0
                    metrics = {'accuracy': float(accuracy)}
                else:
                    # Regression-like
                    mse = np.mean((y - y_pred) ** 2) if len(y) == len(y_pred) else 0.0
                    metrics = {'mse': float(mse)}

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

    async def _cross_validation_evaluation(self, X: np.ndarray, y: np.ndarray,
                                         model: BaseModel, config: TrainingConfig) -> Dict[str, float]:
        """Perform cross-validation evaluation"""
        try:
            if not SKLEARN_AVAILABLE or len(X) < config.cross_validation_folds:
                return {}

            if config.temporal_split:
                cv = TimeSeriesSplit(n_splits=config.cross_validation_folds)
            else:
                from sklearn.model_selection import StratifiedKFold, KFold
                if config.model_type == ModelType.CLASSIFICATION:
                    cv = StratifiedKFold(n_splits=config.cross_validation_folds, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=config.cross_validation_folds, shuffle=True, random_state=42)

            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                try:
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                    # Create new model instance for this fold
                    fold_model = self._create_model_instance(model.config)

                    # Train on fold
                    fold_model.train(X_train_fold, y_train_fold)

                    # Evaluate on validation fold
                    fold_metrics = self._evaluate_model(fold_model, X_val_fold, y_val_fold, config)

                    # Use target metric for CV score
                    if config.target_metric in fold_metrics:
                        cv_scores.append(fold_metrics[config.target_metric])

                except Exception as e:
                    logger.error(f"Error in CV fold {fold}: {e}")

            if cv_scores:
                return {
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                    'cv_scores': cv_scores
                }

            return {}

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}

    async def _save_trained_model(self, model: BaseModel, experiment_id: str) -> str:
        """Save trained model to disk"""
        try:
            # Create model storage directory
            model_dir = Path(self.model_registry.storage_path) / experiment_id
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / f"{model.config.model_id}.pkl"

            if model.save(str(model_path)):
                return str(model_path)
            else:
                raise Exception("Failed to save model")

        except Exception as e:
            logger.error(f"Error saving trained model: {e}")
            raise

    async def _register_trained_model(self, model: BaseModel, experiment_id: str):
        """Register trained model in the registry"""
        try:
            # Update model status
            model.config.status = ModelStatus.TRAINED

            # Register in model registry
            success = self.model_registry.register_model(model)

            if not success:
                raise Exception("Failed to register model in registry")

        except Exception as e:
            logger.error(f"Error registering trained model: {e}")
            raise

    # Logging methods
    async def _log_experiment_start(self, experiment_id: str, config: TrainingConfig):
        """Log experiment start"""
        query = """
        INSERT INTO ml_training_experiments
        (experiment_id, experiment_name, model_type, config_json, status, started_at)
        VALUES (?, ?, ?, ?, 'running', ?)
        """

        params = [
            experiment_id,
            config.experiment_name,
            config.model_type.value,
            json.dumps(asdict(config)),
            datetime.utcnow()
        ]

        self.db.execute_write(query, params)

    async def _log_experiment_completion(self, experiment_id: str, result: TrainingResult):
        """Log experiment completion"""
        query = """
        UPDATE ml_training_experiments
        SET status = ?, completed_at = ?, duration_seconds = ?,
            best_model_id = ?, metrics_json = ?
        WHERE experiment_id = ?
        """

        params = [
            'completed',
            datetime.utcnow(),
            result.training_duration_seconds,
            result.model_id,
            json.dumps(result.to_dict()),
            experiment_id
        ]

        self.db.execute_write(query, params)

    async def _log_experiment_failure(self, experiment_id: str, error_message: str):
        """Log experiment failure"""
        query = """
        UPDATE ml_training_experiments
        SET status = 'failed', completed_at = ?, error_message = ?
        WHERE experiment_id = ?
        """

        params = [datetime.utcnow(), error_message, experiment_id]
        self.db.execute_write(query, params)

    async def _log_training_run(self, experiment_id: str, run_id: str,
                              hyperparameters: Dict[str, Any], metrics: Dict[str, float],
                              error: Optional[str] = None):
        """Log individual training run"""
        query = """
        INSERT INTO ml_training_runs
        (run_id, experiment_id, hyperparameters_json, metrics_json, status,
         started_at, completed_at, duration_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        status = 'failed' if error else 'completed'
        now = datetime.utcnow()

        params = [
            run_id,
            experiment_id,
            json.dumps(hyperparameters),
            json.dumps(metrics) if not error else None,
            status,
            now,
            now,
            0.0  # Duration would be calculated in real implementation
        ]

        self.db.execute_write(query, params)

    async def _log_feature_selection(self, experiment_id: str, method: str,
                                   selected_features: List[str], feature_scores: Dict[str, float]):
        """Log feature selection results"""
        query = """
        INSERT INTO ml_feature_selection
        (selection_id, experiment_id, method, selected_features, feature_scores, selection_criteria)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        params = [
            str(uuid.uuid4()),
            experiment_id,
            method,
            json.dumps(selected_features),
            json.dumps(feature_scores),
            json.dumps({'method': method, 'n_features': len(selected_features)})
        ]

        self.db.execute_write(query, params)


class ValidationFramework:
    """Framework for model validation and testing"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or get_database_manager()
        self.model_registry = ModelRegistry(self.db)

    async def validate_model(self, model: BaseModel, validation_type: str = 'standard') -> Dict[str, Any]:
        """
        Validate a trained model

        Args:
            model: Trained model to validate
            validation_type: Type of validation ('standard', 'comprehensive', 'production')

        Returns:
            Validation results
        """
        validation_results = {
            'model_id': model.config.model_id,
            'validation_type': validation_type,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'passed',
            'checks': {}
        }

        try:
            # Basic model checks
            validation_results['checks']['model_trained'] = model.is_trained
            validation_results['checks']['has_features'] = len(model.feature_names) > 0
            validation_results['checks']['config_valid'] = self._validate_model_config(model.config)

            # Performance validation
            if validation_type in ['comprehensive', 'production']:
                perf_results = await self._validate_performance(model)
                validation_results['checks']['performance'] = perf_results

            # Integration validation
            if validation_type == 'production':
                integration_results = await self._validate_integration(model)
                validation_results['checks']['integration'] = integration_results

            # Determine overall status
            all_passed = all(
                check if isinstance(check, bool) else check.get('passed', False)
                for check in validation_results['checks'].values()
            )

            validation_results['status'] = 'passed' if all_passed else 'failed'

            # Update model status based on validation
            if validation_results['status'] == 'passed':
                self.model_registry.update_model_status(
                    model.config.model_id, ModelStatus.VALIDATED
                )

        except Exception as e:
            logger.error(f"Error validating model {model.config.model_id}: {e}")
            validation_results['status'] = 'error'
            validation_results['error'] = str(e)

        return validation_results

    def _validate_model_config(self, config: ModelConfig) -> bool:
        """Validate model configuration"""
        try:
            checks = [
                bool(config.model_id),
                bool(config.model_name),
                config.model_type in ModelType,
                bool(config.version),
                len(config.features) > 0,
                bool(config.target_metric),
                isinstance(config.hyperparameters, dict)
            ]
            return all(checks)
        except:
            return False

    async def _validate_performance(self, model: BaseModel) -> Dict[str, Any]:
        """Validate model performance"""
        results = {'passed': True, 'details': {}}

        try:
            # Get model performance metrics
            performance = self.model_registry.get_model_performance(model.config.model_id)

            if performance and 'performance_metrics' in performance:
                metrics = performance['performance_metrics']

                # Check if metrics meet minimum thresholds
                if model.config.model_type == ModelType.CLASSIFICATION:
                    accuracy = metrics.get('accuracy', 0)
                    results['details']['accuracy'] = accuracy
                    results['passed'] = accuracy >= 0.7  # Minimum 70% accuracy

                elif model.config.model_type == ModelType.RANKING:
                    # Check ranking-specific metrics
                    mse = metrics.get('mse', float('inf'))
                    results['details']['mse'] = mse
                    results['passed'] = mse < 1.0  # Maximum MSE of 1.0

                # Check prediction stats
                pred_stats = performance.get('prediction_stats', {})
                avg_response_time = pred_stats.get('avg_response_time', 0)
                results['details']['avg_response_time_ms'] = avg_response_time

                # Response time should be under 1 second
                if avg_response_time > 1000:
                    results['passed'] = False
                    results['details']['slow_predictions'] = True

        except Exception as e:
            logger.error(f"Error validating performance: {e}")
            results['passed'] = False
            results['error'] = str(e)

        return results

    async def _validate_integration(self, model: BaseModel) -> Dict[str, Any]:
        """Validate model integration capabilities"""
        results = {'passed': True, 'details': {}}

        try:
            # Test model loading/saving
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                save_success = model.save(tmp_file.name)
                results['details']['can_save'] = save_success

                if save_success:
                    try:
                        loaded_model = BaseModel.load(tmp_file.name)
                        results['details']['can_load'] = True
                        results['details']['config_preserved'] = (
                            loaded_model.config.model_id == model.config.model_id
                        )
                    except:
                        results['details']['can_load'] = False
                        results['passed'] = False
                else:
                    results['passed'] = False

            # Test prediction interface
            try:
                # Create dummy input data
                dummy_input = np.random.random((1, len(model.feature_names)))
                prediction = model.predict(dummy_input)
                results['details']['can_predict'] = prediction is not None
            except:
                results['details']['can_predict'] = False
                results['passed'] = False

            # Test feature importance (if available)
            try:
                importance = model.get_feature_importance()
                results['details']['has_feature_importance'] = len(importance) > 0
            except:
                results['details']['has_feature_importance'] = False

        except Exception as e:
            logger.error(f"Error validating integration: {e}")
            results['passed'] = False
            results['error'] = str(e)

        return results

    async def benchmark_model(self, model: BaseModel, benchmark_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Benchmark model performance"""
        benchmark_results = {
            'model_id': model.config.model_id,
            'timestamp': datetime.utcnow().isoformat(),
            'benchmarks': {}
        }

        try:
            # Prediction latency benchmark
            latency_results = await self._benchmark_latency(model)
            benchmark_results['benchmarks']['latency'] = latency_results

            # Memory usage benchmark
            memory_results = await self._benchmark_memory(model)
            benchmark_results['benchmarks']['memory'] = memory_results

            # Accuracy benchmark (if benchmark data provided)
            if benchmark_data:
                accuracy_results = await self._benchmark_accuracy(model, benchmark_data)
                benchmark_results['benchmarks']['accuracy'] = accuracy_results

        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            benchmark_results['error'] = str(e)

        return benchmark_results

    async def _benchmark_latency(self, model: BaseModel) -> Dict[str, float]:
        """Benchmark prediction latency"""
        try:
            # Create test data
            test_sizes = [1, 10, 100]
            latency_results = {}

            for size in test_sizes:
                dummy_data = np.random.random((size, len(model.feature_names)))

                # Warm up
                model.predict(dummy_data[:1])

                # Measure latency
                start_time = time.time()
                for _ in range(10):  # 10 iterations
                    model.predict(dummy_data)
                end_time = time.time()

                avg_latency_ms = (end_time - start_time) * 1000 / 10
                latency_results[f'batch_size_{size}_ms'] = avg_latency_ms

            return latency_results

        except Exception as e:
            logger.error(f"Error benchmarking latency: {e}")
            return {}

    async def _benchmark_memory(self, model: BaseModel) -> Dict[str, Any]:
        """Benchmark memory usage"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Measure baseline memory
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Make predictions to measure memory usage
            dummy_data = np.random.random((100, len(model.feature_names)))

            model.predict(dummy_data)
            peak_memory = process.memory_info().rss / (1024 * 1024)  # MB

            return {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - baseline_memory
            }

        except Exception as e:
            logger.error(f"Error benchmarking memory: {e}")
            return {}

    async def _benchmark_accuracy(self, model: BaseModel, benchmark_data: Dict) -> Dict[str, float]:
        """Benchmark model accuracy on benchmark dataset"""
        try:
            X_benchmark = benchmark_data['X']
            y_benchmark = benchmark_data['y']

            # Make predictions
            y_pred = model.predict(X_benchmark)

            # Calculate metrics
            if SKLEARN_AVAILABLE and model.config.model_type == ModelType.CLASSIFICATION:
                return {
                    'accuracy': float(accuracy_score(y_benchmark, y_pred)),
                    'precision': float(precision_score(y_benchmark, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_benchmark, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_benchmark, y_pred, average='weighted', zero_division=0))
                }
            else:
                # Basic accuracy for other types
                accuracy = np.mean(y_benchmark == y_pred) if len(y_benchmark) == len(y_pred) else 0.0
                return {'accuracy': float(accuracy)}

        except Exception as e:
            logger.error(f"Error benchmarking accuracy: {e}")
            return {}