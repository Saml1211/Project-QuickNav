"""
Configuration management for Project QuickNav Data Pipeline

Provides centralized configuration for all pipeline components with
environment-specific overrides and validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    duckdb_path: str = "data/quicknav.duckdb"
    sqlite_path: str = "data/quicknav.db"
    connection_pool_size: int = 10
    query_timeout_seconds: int = 30
    enable_wal_mode: bool = True
    vacuum_interval_hours: int = 24


@dataclass
class ETLConfig:
    """ETL pipeline configuration"""
    batch_size: int = 1000
    max_workers: int = 4
    scan_interval_minutes: int = 60
    enable_incremental: bool = True
    max_file_size_mb: int = 100
    supported_extensions: list = None
    exclude_patterns: list = None

    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf', '.docx', '.doc', '.vsdx', '.vsd', '.xlsx', '.pptx']
        if self.exclude_patterns is None:
            self.exclude_patterns = ['~$*', '*.tmp', '**/OLD DRAWINGS/**', '**/ARCHIVE/**']


@dataclass
class StreamingConfig:
    """Streaming pipeline configuration"""
    event_queue_size: int = 10000
    batch_size: int = 100
    flush_interval_seconds: int = 30
    enable_compression: bool = True
    retention_days: int = 90
    max_memory_mb: int = 512


@dataclass
class FeatureStoreConfig:
    """Feature store configuration"""
    cache_size_mb: int = 256
    feature_ttl_hours: int = 24
    compute_interval_minutes: int = 15
    enable_realtime_features: bool = True
    ml_model_path: str = "models/document_ranking.pkl"


@dataclass
class CacheConfig:
    """Caching configuration"""
    l1_cache_size_mb: int = 64
    l2_cache_size_mb: int = 256
    l3_cache_size_mb: int = 1024
    default_ttl_seconds: int = 3600
    enable_compression: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    metrics_retention_days: int = 30
    alert_thresholds: dict = None
    log_level: str = "INFO"
    enable_profiling: bool = False

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'pipeline_failure_rate': 0.05,  # 5% failure rate threshold
                'avg_response_time_ms': 5000,   # 5 second response time threshold
                'queue_depth': 1000,            # Event queue depth threshold
                'memory_usage_percent': 85      # Memory usage threshold
            }


@dataclass
class PipelineConfig:
    """Main pipeline configuration container"""
    database: DatabaseConfig
    etl: ETLConfig
    streaming: StreamingConfig
    feature_store: FeatureStoreConfig
    cache: CacheConfig
    monitoring: MonitoringConfig

    # Environment settings
    environment: str = "development"
    debug: bool = False
    project_root: Optional[str] = None

    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            return cls(
                database=DatabaseConfig(**config_data.get('database', {})),
                etl=ETLConfig(**config_data.get('etl', {})),
                streaming=StreamingConfig(**config_data.get('streaming', {})),
                feature_store=FeatureStoreConfig(**config_data.get('feature_store', {})),
                cache=CacheConfig(**config_data.get('cache', {})),
                monitoring=MonitoringConfig(**config_data.get('monitoring', {})),
                environment=config_data.get('environment', 'development'),
                debug=config_data.get('debug', False),
                project_root=config_data.get('project_root')
            )
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls.default()

    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """Load configuration from environment variables"""
        config = cls.default()

        # Override with environment variables
        env_mappings = {
            'QUICKNAV_ENVIRONMENT': 'environment',
            'QUICKNAV_DEBUG': 'debug',
            'QUICKNAV_PROJECT_ROOT': 'project_root',
            'QUICKNAV_DB_PATH': 'database.duckdb_path',
            'QUICKNAV_LOG_LEVEL': 'monitoring.log_level',
            'QUICKNAV_BATCH_SIZE': 'etl.batch_size',
            'QUICKNAV_CACHE_SIZE_MB': 'cache.l1_cache_size_mb'
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value:
                cls._set_nested_attr(config, config_path, env_value)

        return config

    @classmethod
    def default(cls) -> 'PipelineConfig':
        """Create default configuration"""
        return cls(
            database=DatabaseConfig(),
            etl=ETLConfig(),
            streaming=StreamingConfig(),
            feature_store=FeatureStoreConfig(),
            cache=CacheConfig(),
            monitoring=MonitoringConfig()
        )

    @staticmethod
    def _set_nested_attr(obj: Any, path: str, value: str):
        """Set nested attribute from dot-separated path"""
        attrs = path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)

        # Type conversion based on original value type
        original_value = getattr(obj, attrs[-1])
        if isinstance(original_value, bool):
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(original_value, int):
            value = int(value)
        elif isinstance(original_value, float):
            value = float(value)

        setattr(obj, attrs[-1], value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []

        # Validate database paths
        db_dir = Path(self.database.duckdb_path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create database directory {db_dir}: {e}")

        # Validate ETL settings
        if self.etl.batch_size <= 0:
            errors.append("ETL batch_size must be positive")

        if self.etl.max_workers <= 0:
            errors.append("ETL max_workers must be positive")

        # Validate streaming settings
        if self.streaming.event_queue_size <= 0:
            errors.append("Streaming event_queue_size must be positive")

        # Validate cache settings
        if self.cache.l1_cache_size_mb <= 0:
            errors.append("Cache l1_cache_size_mb must be positive")

        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return False

        return True

    def setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.monitoring.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data/quicknav_pipeline.log', mode='a')
            ]
        )

        if self.debug:
            logging.getLogger('quicknav').setLevel(logging.DEBUG)


# Global configuration instance
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        # Try to load from file first, then environment, then default
        config_path = os.environ.get('QUICKNAV_CONFIG_PATH', 'config/pipeline.json')
        if os.path.exists(config_path):
            _config = PipelineConfig.from_file(config_path)
        else:
            _config = PipelineConfig.from_env()

        # Validate configuration
        if not _config.validate():
            raise ValueError("Invalid pipeline configuration")

        # Setup logging
        _config.setup_logging()

    return _config


def set_config(config: PipelineConfig):
    """Set global configuration instance"""
    global _config
    _config = config


def reset_config():
    """Reset global configuration instance"""
    global _config
    _config = None


# Configuration presets for different environments
DEVELOPMENT_CONFIG = PipelineConfig(
    database=DatabaseConfig(
        duckdb_path="data/dev_quicknav.duckdb",
        enable_wal_mode=False
    ),
    etl=ETLConfig(
        batch_size=100,
        scan_interval_minutes=5,
        max_workers=2
    ),
    streaming=StreamingConfig(
        batch_size=10,
        flush_interval_seconds=5
    ),
    feature_store=FeatureStoreConfig(
        compute_interval_minutes=1
    ),
    monitoring=MonitoringConfig(
        log_level="DEBUG",
        enable_profiling=True
    ),
    environment="development",
    debug=True
)

PRODUCTION_CONFIG = PipelineConfig(
    database=DatabaseConfig(
        duckdb_path="/data/quicknav.duckdb",
        connection_pool_size=20,
        vacuum_interval_hours=6
    ),
    etl=ETLConfig(
        batch_size=5000,
        max_workers=8,
        scan_interval_minutes=30
    ),
    streaming=StreamingConfig(
        event_queue_size=50000,
        batch_size=1000,
        flush_interval_seconds=60
    ),
    feature_store=FeatureStoreConfig(
        cache_size_mb=1024,
        compute_interval_minutes=5
    ),
    cache=CacheConfig(
        l1_cache_size_mb=256,
        l2_cache_size_mb=1024,
        l3_cache_size_mb=4096
    ),
    monitoring=MonitoringConfig(
        log_level="INFO",
        metrics_retention_days=90
    ),
    environment="production",
    debug=False
)

TESTING_CONFIG = PipelineConfig(
    database=DatabaseConfig(
        duckdb_path=":memory:",
        enable_wal_mode=False
    ),
    etl=ETLConfig(
        batch_size=10,
        max_workers=1,
        scan_interval_minutes=1
    ),
    streaming=StreamingConfig(
        event_queue_size=100,
        batch_size=5,
        flush_interval_seconds=1
    ),
    cache=CacheConfig(
        l1_cache_size_mb=16,
        l2_cache_size_mb=32
    ),
    monitoring=MonitoringConfig(
        log_level="WARNING",
        enable_metrics=False
    ),
    environment="testing",
    debug=False
)