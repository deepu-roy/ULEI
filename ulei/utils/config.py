"""
Configuration loader utility for YAML parsing and validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from ulei.core.schemas import EvaluationSuite
from ulei.utils.errors import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader with YAML parsing and validation."""

    def __init__(self, env_prefix: str = "ULEI_"):
        """Initialize configuration loader.

        Args:
            env_prefix: Prefix for environment variable substitution
        """
        self.env_prefix = env_prefix
        self.logger = logging.getLogger("ulei.utils.config")

    def load_suite(self, config_path: Union[str, Path]) -> EvaluationSuite:
        """Load and validate an evaluation suite from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Validated EvaluationSuite instance

        Raises:
            ConfigurationError: If loading or validation fails
        """
        try:
            path = Path(config_path)
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {path}")

            # Load YAML content
            with open(path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            if not raw_config:
                raise ConfigurationError("Configuration file is empty")

            # Perform environment variable substitution
            processed_config = self._substitute_env_vars(raw_config)

            # Validate and create EvaluationSuite
            suite = EvaluationSuite(**processed_config)

            self.logger.info(f"Loaded configuration from {config_path}")
            return suite

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}") from e
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

    def load_dict(self, config_dict: Dict[str, Any]) -> EvaluationSuite:
        """Load and validate an evaluation suite from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Validated EvaluationSuite instance

        Raises:
            ConfigurationError: If validation fails
        """
        try:
            # Perform environment variable substitution
            processed_config = self._substitute_env_vars(config_dict)

            # Validate and create EvaluationSuite
            suite = EvaluationSuite(**processed_config)

            self.logger.debug("Loaded configuration from dictionary")
            return suite

        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to process configuration: {e}") from e

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration.

        Args:
            config: Configuration data (dict, list, or string)

        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string_vars(config)
        else:
            return config

    def _substitute_string_vars(self, text: str) -> str:
        """Substitute environment variables in a string.

        Supports formats:
        - ${VAR_NAME} - Required variable (raises error if not found)
        - ${VAR_NAME:default} - Optional variable with default value

        Args:
            text: String potentially containing variable references

        Returns:
            String with variables substituted

        Raises:
            ConfigurationError: If required variable is not found
        """
        import re

        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*):?([^}]*)\}"

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else None

            # Check for prefixed environment variable
            env_var = os.getenv(f"{self.env_prefix}{var_name}")
            if env_var is not None:
                return env_var

            # Check for unprefixed environment variable
            env_var = os.getenv(var_name)
            if env_var is not None:
                return env_var

            # Use default if provided
            if default_value is not None:
                return default_value

            # Required variable not found
            raise ConfigurationError(
                f"Required environment variable not found: {var_name} "
                f"(tried {self.env_prefix}{var_name} and {var_name})"
            )

        try:
            return re.sub(pattern, replace_var, text)
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to substitute variables in '{text}': {e}") from e

    def validate_suite(self, suite: EvaluationSuite) -> Dict[str, Any]:
        """Validate a loaded evaluation suite for runtime requirements.

        Args:
            suite: EvaluationSuite to validate

        Returns:
            Dictionary with validation results and warnings
        """
        results: Dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        # Check for provider configurations
        for metric in suite.metrics:
            provider_name = metric.provider
            if provider_name and provider_name not in suite.providers:
                results["warnings"].append(
                    f"Metric '{metric.name}' specifies provider '{provider_name}' "
                    f"but no configuration found in providers section"
                )

        # Check threshold configurations
        metric_names = {m.name for m in suite.metrics}
        for threshold_metric in suite.thresholds:
            if threshold_metric not in metric_names:
                results["warnings"].append(
                    f"Threshold defined for '{threshold_metric}' but no matching metric found"
                )

        # Validate budget settings
        if suite.budget_limit and suite.budget_limit <= 0:
            results["errors"].append("Budget limit must be positive")
            results["valid"] = False

        # Validate parallel workers
        if suite.parallel_workers <= 0:
            results["errors"].append("Parallel workers must be positive")
            results["valid"] = False
        elif suite.parallel_workers > 20:
            results["warnings"].append(
                f"High number of parallel workers ({suite.parallel_workers}). "
                "Consider rate limiting to avoid API throttling"
            )

        # Validate retry policy
        if suite.retry_policy.max_retries < 0:
            results["errors"].append("Max retries cannot be negative")
            results["valid"] = False

        if suite.retry_policy.backoff_factor <= 0:
            results["errors"].append("Backoff factor must be positive")
            results["valid"] = False

        # Check output formats
        supported_formats = ["json", "html", "junit"]
        for output_format in suite.output_formats:
            if output_format not in supported_formats:
                results["warnings"].append(
                    f"Unknown output format '{output_format}'. Supported: {supported_formats}"
                )

        return results

    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration dictionary.

        Returns:
            Default configuration suitable for basic evaluation
        """
        return {
            "name": "default_evaluation",
            "description": "Default evaluation configuration",
            "metrics": [{"name": "faithfulness", "provider": "ragas", "config": {}}],
            "providers": {
                "ragas": {"api_key": "${OPENAI_API_KEY}", "model": "gpt-3.5-turbo", "timeout": 30}
            },
            "provider_priority": ["ragas"],
            "thresholds": {"faithfulness": 0.8},
            "budget_limit": 10.0,
            "output_formats": ["json", "html"],
            "parallel_workers": 2,
            "retry_policy": {"max_retries": 3, "backoff_factor": 2.0, "timeout": 60},
            "output_dir": "./evaluation_results",
        }

    def save_config(self, suite: EvaluationSuite, output_path: Union[str, Path]) -> None:
        """Save evaluation suite to YAML file.

        Args:
            suite: EvaluationSuite to save
            output_path: Path where to save the configuration

        Raises:
            ConfigurationError: If saving fails
        """
        try:
            path = Path(output_path)

            # Create output directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dictionary (use model_dump if available, otherwise dict)
            if hasattr(suite, "model_dump"):
                config_dict = suite.model_dump()
            else:
                config_dict = suite.dict()

            # Write YAML
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)

            self.logger.info(f"Saved configuration to {output_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}") from e


# Global configuration loader instance
_global_loader: Optional[ConfigLoader] = None


def get_config_loader(env_prefix: str = "ULEI_") -> ConfigLoader:
    """Get global configuration loader instance.

    Args:
        env_prefix: Environment variable prefix

    Returns:
        ConfigLoader instance
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader(env_prefix)
    return _global_loader


def load_suite_from_file(config_path: str) -> EvaluationSuite:
    """Load evaluation suite from YAML file using global loader.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded EvaluationSuite
    """
    loader = get_config_loader()
    return loader.load_suite(config_path)


def load_suite_from_dict(config_dict: Dict[str, Any]) -> EvaluationSuite:
    """Load evaluation suite from dictionary using global loader.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Loaded EvaluationSuite
    """
    loader = get_config_loader()
    return loader.load_dict(config_dict)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If file cannot be loaded
    """
    import json

    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                result = yaml.safe_load(f)
                return result if isinstance(result, dict) else {}
            elif path.suffix == ".json":
                result = json.load(f)
                return result if isinstance(result, dict) else {}
            else:
                # Try YAML first, then JSON
                content = f.read()
                try:
                    result = yaml.safe_load(content)
                    return result if isinstance(result, dict) else {}
                except yaml.YAMLError:
                    result = json.loads(content)
                    return result if isinstance(result, dict) else {}
    except Exception as e:
        raise ConfigurationError(f"Error loading config from {path}: {e}")
