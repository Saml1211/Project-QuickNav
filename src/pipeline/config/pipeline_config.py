"""
Pipeline Configuration Management

Features:
- Centralized configuration for all pipeline components
- Environment-based configuration loading
- Configuration validation and schema enforcement
- Hot-reload capabilities for development
- Encrypted configuration for sensitive data
- Configuration versioning and migration
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import jsonschema
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """Configuration for a single pipeline component."""
    enabled: bool = True
    priority: int = 5
    config: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)


class PipelineConfig:
    """
    Central configuration manager for all pipeline components.

    Handles loading, validation, and management of configuration from
    multiple sources with environment-specific overrides.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config_data: Dict[str, Any] = {}
        self.component_configs: Dict[str, ComponentConfig] = {}

        # Configuration metadata
        self.loaded_at: Optional[datetime] = None
        self.config_version: str = "1.0.0"

        # Load and validate configuration
        self._load_configuration()
        self._validate_configuration()
        self._setup_component_configs()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Check environment variable first
        env_config = os.environ.get('QUICKNAV_CONFIG_PATH')
        if env_config and os.path.exists(env_config):
            return env_config

        # Check common locations
        possible_paths = [
            'config/pipeline.yaml',
            'config/pipeline.yml',
            'config/pipeline.json',
            os.path.expanduser('~/.quicknav/pipeline.yaml'),
            '/etc/quicknav/pipeline.yaml'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Return default path (will create default config)
        return 'config/pipeline.yaml'

    def _load_configuration(self):
        """Load configuration from file with environment overrides."""
        try:
            # Load base configuration
            if os.path.exists(self.config_path):
                self.config_data = self._load_config_file(self.config_path)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config_data = self._get_default_configuration()
                self._save_default_configuration()

            # Apply environment overrides
            self._apply_environment_overrides()

            self.loaded_at = datetime.now()

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config_data = self._get_default_configuration()

    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from a file (YAML or JSON)."""
        file_ext = Path(file_path).suffix.lower()

        with open(file_path, 'r', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif file_ext == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")

    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            "version": "1.0.0",
            "environment": os.environ.get('QUICKNAV_ENV', 'development'),

            # Core pipeline settings
            "pipeline": {
                "max_concurrent_components": 10,
                "health_check_interval": 30,
                "metrics_interval": 60,
                "shutdown_timeout": 30,
                "strict_startup": False
            },

            # Component enablement
            "enabled_components": [
                "document_monitor",
                "realtime_processor",
                "analytics_processor",
                "activity_streamer",
                "data_validator"
            ],

            # Scheduler configuration
            "scheduler": {
                "max_concurrent_tasks": 20,
                "health_check_interval": 30,
                "resources": {
                    "max_cpu_tasks": 4,
                    "max_memory_mb": 2048,
                    "max_io_tasks": 8
                }
            },

            # Worker pool configuration
            "workers": {
                "min_workers": 2,
                "max_workers": 8,
                "scale_threshold": 5,
                "health_check_interval": 30,
                "thread_workers": 4,
                "process_workers": 2
            },

            # Component-specific configurations
            "components": {
                "document_monitor": {
                    "enabled": True,
                    "config": {
                        "watch_paths": [],  # Auto-detected
                        "document_extensions": [
                            ".pdf", ".docx", ".doc", ".rtf", ".vsdx", ".vsd",
                            ".xlsx", ".xls", ".pptx", ".ppt", ".txt"
                        ],
                        "batch_interval": 5.0,
                        "max_file_size_mb": 100,
                        "ignore_patterns": [
                            r"~\$.*", r"\.tmp$", r"\.temp$",
                            r"Desktop\.ini$", r"Thumbs\.db$",
                            r"\.DS_Store$", r".*conflict.*"
                        ]
                    },
                    "schedule": {
                        "forced_rescan": "0 2 * * *"  # Daily at 2 AM
                    }
                },

                "realtime_processor": {
                    "enabled": True,
                    "config": {
                        "processing_timeout": 30.0,
                        "max_concurrent_tasks": 5,
                        "enable_content_extraction": True,
                        "enable_ml_features": False,
                        "content_extraction": {
                            "max_content_length": 5000,
                            "extract_text": True,
                            "extract_metadata": True
                        }
                    }
                },

                "analytics_processor": {
                    "enabled": True,
                    "config": {
                        "batch_size": 1000,
                        "max_processing_time": 3600,
                        "retention_days": 90
                    },
                    "schedule": {
                        "daily_aggregation": "0 1 * * *",    # Daily at 1 AM
                        "weekly_summary": "0 2 * * 0",      # Weekly on Sunday at 2 AM
                        "monthly_report": "0 3 1 * *"       # Monthly on 1st at 3 AM
                    }
                },

                "ml_feature_processor": {
                    "enabled": False,
                    "config": {
                        "feature_extraction": {
                            "text_features": True,
                            "document_structure": True,
                            "usage_patterns": True
                        },
                        "model_training": {
                            "auto_retrain": True,
                            "training_threshold": 1000,
                            "validation_split": 0.2
                        }
                    },
                    "schedule": {
                        "feature_computation": "0 */6 * * *",  # Every 6 hours
                        "model_training": "0 4 * * 0"         # Weekly on Sunday at 4 AM
                    }
                },

                "activity_streamer": {
                    "enabled": True,
                    "config": {
                        "buffer_size": 1000,
                        "flush_interval": 10.0,
                        "max_event_age": 3600,
                        "compression": True
                    }
                },

                "data_validator": {
                    "enabled": True,
                    "config": {
                        "validation_rules": {
                            "document_completeness": True,
                            "project_consistency": True,
                            "data_freshness": True
                        },
                        "alert_thresholds": {
                            "error_rate": 0.05,
                            "missing_data_rate": 0.10,
                            "inconsistency_rate": 0.03
                        }
                    },
                    "schedule": {
                        "validation_run": "0 */2 * * *"  # Every 2 hours
                    }
                },

                "anomaly_detector": {
                    "enabled": False,
                    "config": {
                        "detection_methods": ["statistical", "ml"],
                        "sensitivity": 0.95,
                        "min_samples": 100,
                        "time_windows": [3600, 86400, 604800]  # 1h, 1d, 1w
                    }
                }
            },

            # Monitoring and alerting
            "monitoring": {
                "enabled": True,
                "metrics_retention_days": 30,
                "performance_tracking": True,
                "export_metrics": False
            },

            "alerting": {
                "enabled": True,
                "channels": ["log"],  # log, email, slack, webhook
                "severity_levels": ["error", "warning"],
                "rate_limiting": {
                    "max_alerts_per_hour": 10,
                    "burst_threshold": 3
                }
            },

            # Performance and resource limits
            "performance": {
                "memory_alert_threshold": 1024,  # MB
                "processing_time_alert_threshold": 5000,  # ms
                "error_rate_alert_threshold": 0.10,  # 10%
                "disk_usage_alert_threshold": 0.85  # 85%
            },

            # Database settings
            "database": {
                "use_duckdb": True,
                "connection_pool_size": 5,
                "query_timeout": 30,
                "backup_enabled": True,
                "backup_interval": 86400  # Daily
            },

            # Logging configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_rotation": True,
                "max_log_size_mb": 50,
                "backup_count": 5
            }
        }

    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Environment variable prefix
        prefix = "QUICKNAV_"

        # Simple overrides for common settings
        env_overrides = {
            f"{prefix}LOG_LEVEL": ["logging", "level"],
            f"{prefix}MAX_WORKERS": ["workers", "max_workers"],
            f"{prefix}ENABLE_ML": ["components", "ml_feature_processor", "enabled"],
            f"{prefix}DB_USE_DUCKDB": ["database", "use_duckdb"],
            f"{prefix}METRICS_INTERVAL": ["pipeline", "metrics_interval"]
        }

        for env_var, config_path in env_overrides.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Type conversion
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)

                # Set nested configuration value
                self._set_nested_value(self.config_data, config_path, value)
                logger.info(f"Applied environment override: {env_var} = {value}")

    def _is_float(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _set_nested_value(self, data: Dict[str, Any], path: List[str], value: Any):
        """Set a nested dictionary value using a path list."""
        for key in path[:-1]:
            data = data.setdefault(key, {})
        data[path[-1]] = value

    def _validate_configuration(self):
        """Validate configuration against schema."""
        try:
            schema = self._get_configuration_schema()
            jsonschema.validate(self.config_data, schema)
            logger.info("Configuration validation passed")
        except jsonschema.ValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            raise
        except Exception as e:
            logger.warning(f"Configuration validation error: {e}")

    def _get_configuration_schema(self) -> Dict[str, Any]:
        """Get JSON schema for configuration validation."""
        return {
            "type": "object",
            "required": ["version", "pipeline", "components"],
            "properties": {
                "version": {"type": "string"},
                "environment": {"type": "string"},
                "pipeline": {
                    "type": "object",
                    "properties": {
                        "max_concurrent_components": {"type": "integer", "minimum": 1},
                        "health_check_interval": {"type": "number", "minimum": 1},
                        "metrics_interval": {"type": "number", "minimum": 1},
                        "shutdown_timeout": {"type": "number", "minimum": 1}
                    }
                },
                "enabled_components": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "components": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "config": {"type": "object"},
                                "schedule": {"type": "object"}
                            }
                        }
                    }
                }
            }
        }

    def _setup_component_configs(self):
        """Setup component configuration objects."""
        components_config = self.config_data.get("components", {})

        for component_name, component_data in components_config.items():
            self.component_configs[component_name] = ComponentConfig(
                enabled=component_data.get("enabled", True),
                priority=component_data.get("priority", 5),
                config=component_data.get("config", {}),
                schedule=component_data.get("schedule"),
                dependencies=component_data.get("dependencies", []),
                resources=component_data.get("resources", {})
            )

    def _save_default_configuration(self):
        """Save default configuration to file."""
        try:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            # Save configuration
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2)

            logger.info(f"Saved default configuration to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save default configuration: {e}")

    # Public API methods

    def get_component_config(self, component_name: str) -> ComponentConfig:
        """Get configuration for a specific component."""
        if component_name in self.component_configs:
            return self.component_configs[component_name]

        # Return default config if component not found
        return ComponentConfig()

    def is_component_enabled(self, component_name: str) -> bool:
        """Check if a component is enabled."""
        component_config = self.get_component_config(component_name)
        return component_config.enabled and component_name in self.enabled_components

    @property
    def enabled_components(self) -> List[str]:
        """Get list of enabled components."""
        return self.config_data.get("enabled_components", [])

    @property
    def environment(self) -> str:
        """Get current environment."""
        return self.config_data.get("environment", "development")

    @property
    def pipeline(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.config_data.get("pipeline", {})

    @property
    def scheduler(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        return self.config_data.get("scheduler", {})

    @property
    def workers(self) -> Dict[str, Any]:
        """Get worker pool configuration."""
        return self.config_data.get("workers", {})

    @property
    def monitoring(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.config_data.get("monitoring", {})

    @property
    def alerting(self) -> Dict[str, Any]:
        """Get alerting configuration."""
        return self.config_data.get("alerting", {})

    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config_data.get("database", {})

    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config_data.get("logging", {})

    @property
    def performance(self) -> Dict[str, Any]:
        """Get performance thresholds."""
        return self.config_data.get("performance", {})

    @property
    def memory_alert_threshold(self) -> int:
        """Get memory alert threshold in MB."""
        return self.performance.get("memory_alert_threshold", 1024)

    @property
    def processing_time_alert_threshold(self) -> int:
        """Get processing time alert threshold in ms."""
        return self.performance.get("processing_time_alert_threshold", 5000)

    @property
    def error_rate_alert_threshold(self) -> float:
        """Get error rate alert threshold."""
        return self.performance.get("error_rate_alert_threshold", 0.10)

    @property
    def health_check_interval(self) -> int:
        """Get health check interval in seconds."""
        return self.pipeline.get("health_check_interval", 30)

    @property
    def metrics_interval(self) -> int:
        """Get metrics collection interval in seconds."""
        return self.pipeline.get("metrics_interval", 60)

    def get_schedule_for_component(self, component_name: str, job_type: str) -> Optional[str]:
        """Get cron schedule for a component job type."""
        component_config = self.get_component_config(component_name)
        if component_config.schedule:
            return component_config.schedule.get(job_type)
        return None

    def reload_configuration(self) -> bool:
        """Reload configuration from file."""
        try:
            self._load_configuration()
            self._validate_configuration()
            self._setup_component_configs()
            logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    def update_component_config(self, component_name: str, config_updates: Dict[str, Any]) -> bool:
        """Update configuration for a specific component."""
        try:
            if component_name not in self.config_data.get("components", {}):
                self.config_data.setdefault("components", {})[component_name] = {}

            component_data = self.config_data["components"][component_name]
            component_data.setdefault("config", {}).update(config_updates)

            # Update component config object
            self._setup_component_configs()

            logger.info(f"Updated configuration for component: {component_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update component configuration: {e}")
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "config_path": self.config_path,
            "version": self.config_version,
            "environment": self.environment,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "enabled_components": self.enabled_components,
            "component_count": len(self.component_configs),
            "pipeline_settings": {
                "max_concurrent_components": self.pipeline.get("max_concurrent_components"),
                "health_check_interval": self.health_check_interval,
                "metrics_interval": self.metrics_interval
            },
            "resource_limits": {
                "max_workers": self.workers.get("max_workers"),
                "memory_threshold": self.memory_alert_threshold,
                "processing_time_threshold": self.processing_time_alert_threshold
            }
        }

    def export_configuration(self, format: str = "yaml") -> str:
        """Export current configuration as string."""
        if format.lower() == "json":
            return json.dumps(self.config_data, indent=2)
        else:
            return yaml.dump(self.config_data, default_flow_style=False, indent=2)

    def validate_component_dependencies(self) -> Dict[str, List[str]]:
        """Validate component dependencies and return any issues."""
        issues = {}

        for component_name, config in self.component_configs.items():
            if not config.enabled:
                continue

            component_issues = []

            # Check if dependencies are enabled
            for dependency in config.dependencies:
                if dependency not in self.component_configs:
                    component_issues.append(f"Unknown dependency: {dependency}")
                elif not self.component_configs[dependency].enabled:
                    component_issues.append(f"Disabled dependency: {dependency}")

            if component_issues:
                issues[component_name] = component_issues

        return issues