"""
A/B Testing Framework for ML Models

Provides comprehensive A/B testing infrastructure for:
- Model comparison and evaluation
- Gradual rollout strategies
- Statistical significance testing
- Performance monitoring during experiments
- Automated decision making for model deployment
"""

import numpy as np
import pandas as pd
import json
import time
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import hashlib

# Try to import scipy for statistical tests
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .core import BaseModel, ModelRegistry, ModelStatus
from ..database.database_manager import DatabaseManager, get_database_manager

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """A/B test experiment status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class DecisionRule(Enum):
    """Decision rules for experiment conclusions"""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    BUSINESS_METRIC = "business_metric"
    CUSTOM_THRESHOLD = "custom_threshold"
    MANUAL = "manual"


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiment"""
    experiment_name: str
    description: str

    # Models being tested
    control_model_id: str
    treatment_model_ids: List[str]

    # Traffic allocation
    control_traffic_percent: float
    treatment_traffic_split: Dict[str, float]  # model_id -> percentage

    # Experiment criteria
    primary_metric: str
    secondary_metrics: List[str]
    success_criteria: Dict[str, Any]

    # Duration and sample size
    min_sample_size: int
    max_duration_days: int
    min_duration_days: int = 7

    # Statistical parameters
    significance_level: float = 0.05
    power: float = 0.8
    minimum_detectable_effect: float = 0.05

    # Decision rules
    decision_rule: DecisionRule = DecisionRule.STATISTICAL_SIGNIFICANCE
    auto_promote_winner: bool = False
    early_stopping_enabled: bool = True

    # Safety guardrails
    error_rate_threshold: float = 0.1
    latency_threshold_ms: float = 5000
    min_confidence_score: float = 0.3

    def __post_init__(self):
        # Validate traffic allocation
        total_traffic = self.control_traffic_percent + sum(self.treatment_traffic_split.values())
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 100%, got {total_traffic}")


@dataclass
class ExperimentResult:
    """Result from A/B test experiment"""
    experiment_id: str
    status: ExperimentStatus
    conclusion: str

    # Statistical results
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, Dict[str, float]]  # model_id -> metrics
    statistical_tests: Dict[str, Dict[str, Any]]

    # Recommendation
    recommended_model: str
    confidence_level: float
    winner_determined: bool

    # Sample sizes and duration
    total_samples: int
    samples_per_variant: Dict[str, int]
    experiment_duration_days: float

    # Metadata
    started_at: datetime
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExperimentTracker:
    """Tracks experiment assignments and outcomes"""

    def __init__(self):
        self.assignments: Dict[str, str] = {}  # user_id -> model_id
        self.outcomes: deque = deque(maxlen=10000)  # Recent outcomes
        self.lock = threading.RLock()

    def assign_user(self, user_id: str, experiment_id: str, config: ExperimentConfig) -> str:
        """Assign user to experiment variant"""
        with self.lock:
            # Check if user already assigned
            assignment_key = f"{experiment_id}:{user_id}"
            if assignment_key in self.assignments:
                return self.assignments[assignment_key]

            # Hash user ID for consistent assignment
            user_hash = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
            assignment_percentage = (user_hash % 100) + 1

            # Determine assignment based on traffic allocation
            cumulative_percent = 0.0

            # Control group
            cumulative_percent += config.control_traffic_percent
            if assignment_percentage <= cumulative_percent:
                assigned_model = config.control_model_id
            else:
                # Treatment groups
                for model_id, percent in config.treatment_traffic_split.items():
                    cumulative_percent += percent
                    if assignment_percentage <= cumulative_percent:
                        assigned_model = model_id
                        break
                else:
                    # Fallback to control
                    assigned_model = config.control_model_id

            self.assignments[assignment_key] = assigned_model
            return assigned_model

    def record_outcome(self, experiment_id: str, user_id: str, model_id: str,
                      metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Record experiment outcome"""
        with self.lock:
            outcome = {
                'experiment_id': experiment_id,
                'user_id': user_id,
                'model_id': model_id,
                'metrics': metrics,
                'timestamp': timestamp or datetime.utcnow()
            }
            self.outcomes.append(outcome)

    def get_experiment_data(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all outcomes for an experiment"""
        with self.lock:
            return [outcome for outcome in self.outcomes
                   if outcome['experiment_id'] == experiment_id]


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests"""

    def __init__(self):
        self.confidence_level = 0.95

    def analyze_experiment(self, experiment_data: List[Dict[str, Any]],
                          config: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
        """Analyze experiment results"""
        results = {}

        try:
            # Group data by model
            model_data = defaultdict(list)
            for outcome in experiment_data:
                model_id = outcome['model_id']
                model_data[model_id].append(outcome)

            # Analyze primary metric
            primary_results = self._analyze_metric(
                model_data, config.primary_metric, config.control_model_id
            )
            results[config.primary_metric] = primary_results

            # Analyze secondary metrics
            for metric in config.secondary_metrics:
                secondary_results = self._analyze_metric(
                    model_data, metric, config.control_model_id
                )
                results[metric] = secondary_results

        except Exception as e:
            logger.error(f"Error analyzing experiment: {e}")
            results['error'] = str(e)

        return results

    def _analyze_metric(self, model_data: Dict[str, List[Dict]],
                       metric_name: str, control_model_id: str) -> Dict[str, Any]:
        """Analyze a specific metric across models"""
        results = {
            'metric': metric_name,
            'control_model': control_model_id,
            'comparisons': {}
        }

        try:
            # Extract control data
            control_values = []
            if control_model_id in model_data:
                for outcome in model_data[control_model_id]:
                    if metric_name in outcome['metrics']:
                        control_values.append(outcome['metrics'][metric_name])

            if not control_values:
                results['error'] = f"No control data for metric {metric_name}"
                return results

            results['control_stats'] = self._compute_descriptive_stats(control_values)

            # Compare each treatment to control
            for model_id, outcomes in model_data.items():
                if model_id == control_model_id:
                    continue

                treatment_values = []
                for outcome in outcomes:
                    if metric_name in outcome['metrics']:
                        treatment_values.append(outcome['metrics'][metric_name])

                if not treatment_values:
                    continue

                comparison = self._compare_groups(
                    control_values, treatment_values, f"control_vs_{model_id}"
                )
                results['comparisons'][model_id] = comparison

        except Exception as e:
            logger.error(f"Error analyzing metric {metric_name}: {e}")
            results['error'] = str(e)

        return results

    def _compute_descriptive_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute descriptive statistics"""
        if not values:
            return {}

        values_array = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array, ddof=1)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'q25': float(np.percentile(values_array, 25)),
            'q75': float(np.percentile(values_array, 75))
        }

    def _compare_groups(self, control: List[float], treatment: List[float],
                       comparison_name: str) -> Dict[str, Any]:
        """Compare two groups statistically"""
        result = {
            'comparison': comparison_name,
            'control_stats': self._compute_descriptive_stats(control),
            'treatment_stats': self._compute_descriptive_stats(treatment)
        }

        try:
            control_array = np.array(control)
            treatment_array = np.array(treatment)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control) - 1) * np.var(control, ddof=1) +
                                 (len(treatment) - 1) * np.var(treatment, ddof=1)) /
                                (len(control) + len(treatment) - 2))

            if pooled_std > 0:
                cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
                result['effect_size'] = float(cohens_d)

            # Relative improvement
            control_mean = np.mean(control_array)
            treatment_mean = np.mean(treatment_array)
            if control_mean != 0:
                relative_improvement = (treatment_mean - control_mean) / abs(control_mean)
                result['relative_improvement'] = float(relative_improvement)

            # Statistical tests
            if SCIPY_AVAILABLE:
                # T-test (assuming normal distribution)
                t_stat, t_p_value = ttest_ind(treatment_array, control_array)
                result['t_test'] = {
                    'statistic': float(t_stat),
                    'p_value': float(t_p_value),
                    'significant': t_p_value < 0.05
                }

                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_p_value = mannwhitneyu(treatment_array, control_array, alternative='two-sided')
                    result['mann_whitney_u'] = {
                        'statistic': float(u_stat),
                        'p_value': float(u_p_value),
                        'significant': u_p_value < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Mann-Whitney U test failed: {e}")

            else:
                # Simple t-test without scipy
                result['simple_comparison'] = self._simple_t_test(control_array, treatment_array)

            # Confidence interval for difference in means
            result['confidence_interval'] = self._confidence_interval_diff(
                control_array, treatment_array
            )

        except Exception as e:
            logger.error(f"Error in statistical comparison: {e}")
            result['error'] = str(e)

        return result

    def _simple_t_test(self, control: np.ndarray, treatment: np.ndarray) -> Dict[str, Any]:
        """Simple t-test implementation without scipy"""
        try:
            n1, n2 = len(control), len(treatment)
            mean1, mean2 = np.mean(control), np.mean(treatment)
            var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)

            # Pooled standard error
            pooled_se = np.sqrt(var1/n1 + var2/n2)

            if pooled_se > 0:
                t_stat = (mean2 - mean1) / pooled_se
                df = n1 + n2 - 2

                # Rough p-value approximation
                # For large samples, t-distribution approaches normal
                if df > 30:
                    p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
                else:
                    # Very rough approximation
                    p_value = 0.05 if abs(t_stat) > 2 else 0.5

                return {
                    'statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'method': 'simple_approximation'
                }

        except Exception as e:
            logger.error(f"Error in simple t-test: {e}")

        return {'error': 'Could not compute t-test'}

    def _normal_cdf(self, x: float) -> float:
        """Rough approximation of normal CDF"""
        return 0.5 * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def _confidence_interval_diff(self, control: np.ndarray, treatment: np.ndarray,
                                 confidence: float = 0.95) -> Dict[str, float]:
        """Confidence interval for difference in means"""
        try:
            n1, n2 = len(control), len(treatment)
            mean1, mean2 = np.mean(control), np.mean(treatment)
            var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)

            diff = mean2 - mean1
            se_diff = np.sqrt(var1/n1 + var2/n2)

            # Use z-score for large samples, t-score approximation for small
            if n1 + n2 > 30:
                critical_value = 1.96  # 95% confidence
            else:
                critical_value = 2.0  # Rough approximation

            margin_error = critical_value * se_diff

            return {
                'difference': float(diff),
                'lower_bound': float(diff - margin_error),
                'upper_bound': float(diff + margin_error),
                'confidence_level': confidence
            }

        except Exception as e:
            logger.error(f"Error computing confidence interval: {e}")
            return {}

    def calculate_required_sample_size(self, config: ExperimentConfig) -> int:
        """Calculate required sample size for experiment"""
        try:
            # Simplified sample size calculation
            # In practice, this would use power analysis formulas

            alpha = config.significance_level
            beta = 1 - config.power
            effect_size = config.minimum_detectable_effect

            # Very rough approximation
            # n = 2 * (z_alpha/2 + z_beta)^2 / effect_size^2

            z_alpha = 1.96 if alpha == 0.05 else 2.58  # rough approximation
            z_beta = 0.84 if beta == 0.2 else 1.28

            n_per_group = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)

            # Total sample size across all groups
            n_groups = 1 + len(config.treatment_model_ids)  # control + treatments
            total_n = int(n_per_group * n_groups)

            return max(total_n, config.min_sample_size)

        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            return config.min_sample_size


class ABTestFramework:
    """Main A/B testing framework"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or get_database_manager()
        self.model_registry = ModelRegistry(self.db)
        self.tracker = ExperimentTracker()
        self.analyzer = StatisticalAnalyzer()

        self._setup_ab_test_tables()

    def _setup_ab_test_tables(self):
        """Setup A/B testing database tables"""
        schema_sql = """
        -- A/B test experiments table
        CREATE TABLE IF NOT EXISTS ab_test_experiments (
            experiment_id VARCHAR PRIMARY KEY,
            experiment_name VARCHAR NOT NULL,
            description TEXT,
            config_json TEXT NOT NULL,
            status VARCHAR NOT NULL,
            control_model_id VARCHAR NOT NULL,
            treatment_model_ids TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            conclusion TEXT,
            winner_model_id VARCHAR,
            confidence_level REAL,
            metadata_json TEXT
        );

        -- A/B test assignments table
        CREATE TABLE IF NOT EXISTS ab_test_assignments (
            assignment_id VARCHAR PRIMARY KEY,
            experiment_id VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL,
            model_id VARCHAR NOT NULL,
            assigned_at TIMESTAMP NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES ab_test_experiments(experiment_id),
            UNIQUE(experiment_id, user_id)
        );

        -- A/B test outcomes table
        CREATE TABLE IF NOT EXISTS ab_test_outcomes (
            outcome_id VARCHAR PRIMARY KEY,
            experiment_id VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL,
            model_id VARCHAR NOT NULL,
            primary_metric_value REAL,
            secondary_metrics_json TEXT,
            prediction_data TEXT,
            timestamp TIMESTAMP NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES ab_test_experiments(experiment_id)
        );

        -- A/B test results table
        CREATE TABLE IF NOT EXISTS ab_test_results (
            result_id VARCHAR PRIMARY KEY,
            experiment_id VARCHAR NOT NULL,
            analysis_date DATE NOT NULL,
            statistical_results_json TEXT NOT NULL,
            recommendation TEXT,
            confidence_level REAL,
            sample_sizes_json TEXT,
            FOREIGN KEY (experiment_id) REFERENCES ab_test_experiments(experiment_id)
        );

        -- Indices
        CREATE INDEX IF NOT EXISTS idx_experiments_status ON ab_test_experiments(status);
        CREATE INDEX IF NOT EXISTS idx_assignments_experiment_user ON ab_test_assignments(experiment_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_outcomes_experiment_timestamp ON ab_test_outcomes(experiment_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_results_experiment_date ON ab_test_results(experiment_id, analysis_date);
        """

        try:
            self.db.execute_write(schema_sql, use_transaction=False)
            logger.info("A/B testing tables initialized")
        except Exception as e:
            logger.error(f"Error setting up A/B testing tables: {e}")

    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment"""
        experiment_id = str(uuid.uuid4())

        try:
            # Validate models exist
            for model_id in [config.control_model_id] + config.treatment_model_ids:
                model = self.model_registry.get_model(model_id)
                if not model:
                    raise ValueError(f"Model not found: {model_id}")

            # Calculate required sample size
            required_sample_size = self.analyzer.calculate_required_sample_size(config)
            logger.info(f"Required sample size for experiment: {required_sample_size}")

            # Insert experiment record
            query = """
            INSERT INTO ab_test_experiments
            (experiment_id, experiment_name, description, config_json, status,
             control_model_id, treatment_model_ids, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = [
                experiment_id,
                config.experiment_name,
                config.description,
                json.dumps(asdict(config)),
                ExperimentStatus.DRAFT.value,
                config.control_model_id,
                json.dumps(config.treatment_model_ids),
                datetime.utcnow(),
                json.dumps({'required_sample_size': required_sample_size})
            ]

            success = self.db.execute_write(query, params)

            if success:
                logger.info(f"Created experiment: {experiment_id}")
                return experiment_id
            else:
                raise Exception("Failed to create experiment record")

        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment"""
        try:
            # Update experiment status
            query = """
            UPDATE ab_test_experiments
            SET status = ?, started_at = ?
            WHERE experiment_id = ? AND status = ?
            """

            params = [
                ExperimentStatus.RUNNING.value,
                datetime.utcnow(),
                experiment_id,
                ExperimentStatus.DRAFT.value
            ]

            success = self.db.execute_write(query, params)

            if success:
                logger.info(f"Started experiment: {experiment_id}")

            return success

        except Exception as e:
            logger.error(f"Error starting experiment {experiment_id}: {e}")
            return False

    async def assign_user_to_experiment(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign user to experiment variant and return model ID"""
        try:
            # Get experiment configuration
            experiment = await self._get_experiment(experiment_id)
            if not experiment or experiment['status'] != ExperimentStatus.RUNNING.value:
                return None

            config_dict = json.loads(experiment['config_json'])
            config = ExperimentConfig(**config_dict)

            # Check existing assignment
            existing_assignment = await self._get_user_assignment(experiment_id, user_id)
            if existing_assignment:
                return existing_assignment['model_id']

            # Assign user to variant
            assigned_model_id = self.tracker.assign_user(user_id, experiment_id, config)

            # Record assignment in database
            await self._record_assignment(experiment_id, user_id, assigned_model_id)

            return assigned_model_id

        except Exception as e:
            logger.error(f"Error assigning user to experiment: {e}")
            return None

    async def record_experiment_outcome(self, experiment_id: str, user_id: str, model_id: str,
                                      primary_metric_value: float,
                                      secondary_metrics: Optional[Dict[str, float]] = None,
                                      prediction_data: Optional[Dict[str, Any]] = None):
        """Record experiment outcome"""
        try:
            outcome_id = str(uuid.uuid4())

            query = """
            INSERT INTO ab_test_outcomes
            (outcome_id, experiment_id, user_id, model_id, primary_metric_value,
             secondary_metrics_json, prediction_data, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = [
                outcome_id,
                experiment_id,
                user_id,
                model_id,
                primary_metric_value,
                json.dumps(secondary_metrics) if secondary_metrics else None,
                json.dumps(prediction_data) if prediction_data else None,
                datetime.utcnow()
            ]

            success = self.db.execute_write(query, params)

            if success:
                # Also record in tracker for real-time analysis
                metrics = {'primary': primary_metric_value}
                if secondary_metrics:
                    metrics.update(secondary_metrics)

                self.tracker.record_outcome(experiment_id, user_id, model_id, metrics)

        except Exception as e:
            logger.error(f"Error recording experiment outcome: {e}")

    async def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Analyze experiment and determine results"""
        try:
            # Get experiment data
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return None

            config_dict = json.loads(experiment['config_json'])
            config = ExperimentConfig(**config_dict)

            # Get experiment outcomes
            outcomes = await self._get_experiment_outcomes(experiment_id)

            if not outcomes:
                logger.warning(f"No outcomes found for experiment {experiment_id}")
                return None

            # Perform statistical analysis
            statistical_results = self.analyzer.analyze_experiment(outcomes, config)

            # Determine winner and confidence
            winner_info = self._determine_winner(statistical_results, config)

            # Calculate sample sizes
            sample_sizes = self._calculate_sample_sizes(outcomes)

            # Create result
            started_at = datetime.fromisoformat(experiment['started_at']) if experiment['started_at'] else datetime.utcnow()
            duration_days = (datetime.utcnow() - started_at).days

            result = ExperimentResult(
                experiment_id=experiment_id,
                status=ExperimentStatus(experiment['status']),
                conclusion=winner_info['conclusion'],
                control_metrics=self._extract_model_metrics(outcomes, config.control_model_id),
                treatment_metrics=self._extract_treatment_metrics(outcomes, config.treatment_model_ids),
                statistical_tests=statistical_results,
                recommended_model=winner_info['recommended_model'],
                confidence_level=winner_info['confidence_level'],
                winner_determined=winner_info['winner_determined'],
                total_samples=len(outcomes),
                samples_per_variant=sample_sizes,
                experiment_duration_days=duration_days,
                started_at=started_at
            )

            # Store analysis results
            await self._store_analysis_results(experiment_id, result, statistical_results)

            return result

        except Exception as e:
            logger.error(f"Error analyzing experiment {experiment_id}: {e}")
            return None

    async def complete_experiment(self, experiment_id: str, conclusion: str,
                                winner_model_id: Optional[str] = None) -> bool:
        """Complete an A/B test experiment"""
        try:
            query = """
            UPDATE ab_test_experiments
            SET status = ?, completed_at = ?, conclusion = ?, winner_model_id = ?
            WHERE experiment_id = ?
            """

            params = [
                ExperimentStatus.COMPLETED.value,
                datetime.utcnow(),
                conclusion,
                winner_model_id,
                experiment_id
            ]

            success = self.db.execute_write(query, params)

            if success:
                logger.info(f"Completed experiment {experiment_id}: {conclusion}")

                # If auto-promotion is enabled and there's a clear winner
                if winner_model_id:
                    await self._handle_winner_promotion(experiment_id, winner_model_id)

            return success

        except Exception as e:
            logger.error(f"Error completing experiment {experiment_id}: {e}")
            return False

    def _determine_winner(self, statistical_results: Dict[str, Dict[str, Any]],
                         config: ExperimentConfig) -> Dict[str, Any]:
        """Determine experiment winner based on statistical results"""
        try:
            primary_results = statistical_results.get(config.primary_metric, {})
            comparisons = primary_results.get('comparisons', {})

            if not comparisons:
                return {
                    'winner_determined': False,
                    'recommended_model': config.control_model_id,
                    'confidence_level': 0.0,
                    'conclusion': 'Insufficient data for analysis'
                }

            # Find best performing treatment
            best_model = config.control_model_id
            best_improvement = 0.0
            best_confidence = 0.0
            significant_results = []

            for model_id, comparison in comparisons.items():
                # Check statistical significance
                is_significant = False
                p_value = 1.0

                if 't_test' in comparison:
                    is_significant = comparison['t_test']['significant']
                    p_value = comparison['t_test']['p_value']
                elif 'mann_whitney_u' in comparison:
                    is_significant = comparison['mann_whitney_u']['significant']
                    p_value = comparison['mann_whitney_u']['p_value']
                elif 'simple_comparison' in comparison:
                    is_significant = comparison['simple_comparison']['significant']
                    p_value = comparison['simple_comparison']['p_value']

                relative_improvement = comparison.get('relative_improvement', 0.0)

                if is_significant and relative_improvement > config.minimum_detectable_effect:
                    significant_results.append({
                        'model_id': model_id,
                        'improvement': relative_improvement,
                        'p_value': p_value,
                        'confidence': 1 - p_value
                    })

                    if relative_improvement > best_improvement:
                        best_model = model_id
                        best_improvement = relative_improvement
                        best_confidence = 1 - p_value

            # Determine conclusion
            if significant_results:
                if config.decision_rule == DecisionRule.STATISTICAL_SIGNIFICANCE:
                    winner_determined = best_confidence >= (1 - config.significance_level)
                    conclusion = f"Treatment {best_model} shows significant improvement of {best_improvement:.2%}"
                else:
                    winner_determined = True
                    conclusion = f"Best performing model: {best_model}"
            else:
                winner_determined = False
                best_model = config.control_model_id
                best_confidence = 0.0
                conclusion = "No statistically significant difference found"

            return {
                'winner_determined': winner_determined,
                'recommended_model': best_model,
                'confidence_level': best_confidence,
                'conclusion': conclusion
            }

        except Exception as e:
            logger.error(f"Error determining winner: {e}")
            return {
                'winner_determined': False,
                'recommended_model': config.control_model_id,
                'confidence_level': 0.0,
                'conclusion': f'Error in analysis: {str(e)}'
            }

    def _extract_model_metrics(self, outcomes: List[Dict], model_id: str) -> Dict[str, float]:
        """Extract metrics for a specific model"""
        model_outcomes = [o for o in outcomes if o['model_id'] == model_id]

        if not model_outcomes:
            return {}

        primary_values = [o['primary_metric_value'] for o in model_outcomes if o['primary_metric_value'] is not None]

        if not primary_values:
            return {}

        return {
            'count': len(model_outcomes),
            'mean': float(np.mean(primary_values)),
            'std': float(np.std(primary_values)),
            'median': float(np.median(primary_values))
        }

    def _extract_treatment_metrics(self, outcomes: List[Dict], treatment_model_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract metrics for all treatment models"""
        treatment_metrics = {}

        for model_id in treatment_model_ids:
            treatment_metrics[model_id] = self._extract_model_metrics(outcomes, model_id)

        return treatment_metrics

    def _calculate_sample_sizes(self, outcomes: List[Dict]) -> Dict[str, int]:
        """Calculate sample sizes per variant"""
        sample_sizes = defaultdict(int)

        for outcome in outcomes:
            sample_sizes[outcome['model_id']] += 1

        return dict(sample_sizes)

    async def _handle_winner_promotion(self, experiment_id: str, winner_model_id: str):
        """Handle automatic promotion of winning model"""
        try:
            # Update winner model status to deployed
            self.model_registry.update_model_status(winner_model_id, ModelStatus.DEPLOYED)

            # Optionally deprecate losing models
            experiment = await self._get_experiment(experiment_id)
            if experiment:
                config_dict = json.loads(experiment['config_json'])
                all_models = [config_dict['control_model_id']] + config_dict['treatment_model_ids']

                for model_id in all_models:
                    if model_id != winner_model_id:
                        self.model_registry.update_model_status(model_id, ModelStatus.DEPRECATED)

            logger.info(f"Promoted winner model {winner_model_id} from experiment {experiment_id}")

        except Exception as e:
            logger.error(f"Error promoting winner model: {e}")

    # Database helper methods
    async def _get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment from database"""
        query = "SELECT * FROM ab_test_experiments WHERE experiment_id = ?"
        results = self.db.execute_query(query, [experiment_id])
        return results[0] if results else None

    async def _get_user_assignment(self, experiment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user assignment from database"""
        query = """
        SELECT * FROM ab_test_assignments
        WHERE experiment_id = ? AND user_id = ?
        """
        results = self.db.execute_query(query, [experiment_id, user_id])
        return results[0] if results else None

    async def _record_assignment(self, experiment_id: str, user_id: str, model_id: str):
        """Record user assignment in database"""
        query = """
        INSERT OR REPLACE INTO ab_test_assignments
        (assignment_id, experiment_id, user_id, model_id, assigned_at)
        VALUES (?, ?, ?, ?, ?)
        """

        params = [
            str(uuid.uuid4()),
            experiment_id,
            user_id,
            model_id,
            datetime.utcnow()
        ]

        self.db.execute_write(query, params)

    async def _get_experiment_outcomes(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get experiment outcomes from database"""
        query = """
        SELECT user_id, model_id, primary_metric_value, secondary_metrics_json, timestamp
        FROM ab_test_outcomes
        WHERE experiment_id = ?
        ORDER BY timestamp
        """

        results = self.db.execute_query(query, [experiment_id])

        # Convert to format expected by analyzer
        outcomes = []
        for row in results:
            outcome = {
                'user_id': row['user_id'],
                'model_id': row['model_id'],
                'primary_metric_value': row['primary_metric_value'],
                'timestamp': row['timestamp']
            }

            # Parse secondary metrics
            if row['secondary_metrics_json']:
                secondary_metrics = json.loads(row['secondary_metrics_json'])
                outcome['metrics'] = {'primary': row['primary_metric_value']}
                outcome['metrics'].update(secondary_metrics)
            else:
                outcome['metrics'] = {'primary': row['primary_metric_value']}

            outcomes.append(outcome)

        return outcomes

    async def _store_analysis_results(self, experiment_id: str, result: ExperimentResult,
                                    statistical_results: Dict[str, Any]):
        """Store analysis results in database"""
        query = """
        INSERT OR REPLACE INTO ab_test_results
        (result_id, experiment_id, analysis_date, statistical_results_json,
         recommendation, confidence_level, sample_sizes_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            str(uuid.uuid4()),
            experiment_id,
            datetime.utcnow().date(),
            json.dumps(statistical_results),
            result.conclusion,
            result.confidence_level,
            json.dumps(result.samples_per_variant)
        ]

        self.db.execute_write(query, params)


class ExperimentManager:
    """High-level experiment management"""

    def __init__(self, ab_framework: ABTestFramework):
        self.ab_framework = ab_framework

    async def run_model_comparison(self, experiment_name: str, control_model_id: str,
                                 treatment_model_ids: List[str],
                                 primary_metric: str = 'accuracy',
                                 duration_days: int = 14) -> str:
        """Run a standard model comparison experiment"""

        config = ExperimentConfig(
            experiment_name=experiment_name,
            description=f"Comparing {control_model_id} vs {', '.join(treatment_model_ids)}",
            control_model_id=control_model_id,
            treatment_model_ids=treatment_model_ids,
            control_traffic_percent=50.0,
            treatment_traffic_split={model_id: 50.0 / len(treatment_model_ids)
                                   for model_id in treatment_model_ids},
            primary_metric=primary_metric,
            secondary_metrics=['response_time', 'confidence_score'],
            success_criteria={primary_metric: {'min_improvement': 0.05}},
            min_sample_size=100,
            max_duration_days=duration_days,
            auto_promote_winner=True
        )

        experiment_id = await self.ab_framework.create_experiment(config)
        await self.ab_framework.start_experiment(experiment_id)

        return experiment_id

    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment status"""
        try:
            experiment = await self.ab_framework._get_experiment(experiment_id)
            if not experiment:
                return {'error': 'Experiment not found'}

            # Get current analysis
            analysis_result = await self.ab_framework.analyze_experiment(experiment_id)

            status = {
                'experiment_id': experiment_id,
                'name': experiment['experiment_name'],
                'status': experiment['status'],
                'started_at': experiment['started_at'],
                'completed_at': experiment['completed_at'],
                'conclusion': experiment['conclusion']
            }

            if analysis_result:
                status.update({
                    'total_samples': analysis_result.total_samples,
                    'duration_days': analysis_result.experiment_duration_days,
                    'recommended_model': analysis_result.recommended_model,
                    'confidence_level': analysis_result.confidence_level,
                    'winner_determined': analysis_result.winner_determined
                })

            return status

        except Exception as e:
            logger.error(f"Error getting experiment status: {e}")
            return {'error': str(e)}

    async def auto_manage_experiments(self):
        """Automatically manage running experiments"""
        try:
            # Get running experiments
            query = "SELECT experiment_id FROM ab_test_experiments WHERE status = 'running'"
            running_experiments = self.db.execute_query(query)

            for exp in running_experiments:
                experiment_id = exp['experiment_id']

                try:
                    # Analyze experiment
                    result = await self.ab_framework.analyze_experiment(experiment_id)

                    if result and result.winner_determined:
                        # Check if experiment should be concluded
                        if (result.experiment_duration_days >= 7 and
                            result.total_samples >= 100 and
                            result.confidence_level >= 0.95):

                            await self.ab_framework.complete_experiment(
                                experiment_id,
                                result.conclusion,
                                result.recommended_model
                            )

                            logger.info(f"Auto-completed experiment {experiment_id}")

                except Exception as e:
                    logger.error(f"Error auto-managing experiment {experiment_id}: {e}")

        except Exception as e:
            logger.error(f"Error in auto-management: {e}")