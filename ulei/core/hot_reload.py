"""
Configuration hot-reload system for ULEI custom metrics.

Provides dynamic configuration reloading without service restart,
including file watching, validation, and graceful metric updates.
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ulei.core.custom_metrics import (
    CustomMetricRegistry,
    MetricConfiguration,
    metric_registry,
)
from ulei.core.validation import MetricValidator, ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


class ReloadEvent(Enum):
    """Types of reload events."""

    FILE_MODIFIED = "file_modified"
    FILE_CREATED = "file_created"
    FILE_DELETED = "file_deleted"
    MANUAL_RELOAD = "manual_reload"
    VALIDATION_FAILED = "validation_failed"
    RELOAD_SUCCESS = "reload_success"


@dataclass
class ReloadResult:
    """Result of configuration reload operation."""

    success: bool
    event_type: ReloadEvent
    config_path: str
    metric_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    old_config_hash: Optional[str] = None
    new_config_hash: Optional[str] = None


class ConfigurationWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes."""

    def __init__(self, hot_reloader: "ConfigurationHotReloader"):
        """
        Initialize configuration watcher.

        Args:
            hot_reloader: Hot reloader instance
        """
        self.hot_reloader = hot_reloader

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self._is_config_file(event.src_path):
            logger.info(f"Configuration file modified: {event.src_path}")
            asyncio.create_task(
                self.hot_reloader._handle_file_event(event.src_path, ReloadEvent.FILE_MODIFIED)
            )

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self._is_config_file(event.src_path):
            logger.info(f"Configuration file created: {event.src_path}")
            asyncio.create_task(
                self.hot_reloader._handle_file_event(event.src_path, ReloadEvent.FILE_CREATED)
            )

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory and self._is_config_file(event.src_path):
            logger.info(f"Configuration file deleted: {event.src_path}")
            asyncio.create_task(
                self.hot_reloader._handle_file_event(event.src_path, ReloadEvent.FILE_DELETED)
            )

    def _is_config_file(self, file_path: str) -> bool:
        """Check if file is a configuration file."""
        path = Path(file_path)
        return path.suffix.lower() in [".yml", ".yaml", ".json"] and path.stem.endswith("_metric")


class ConfigurationCache:
    """Cache for configuration files and their states."""

    def __init__(self):
        """Initialize configuration cache."""
        self.configs: Dict[str, MetricConfiguration] = {}
        self.file_hashes: Dict[str, str] = {}
        self.load_times: Dict[str, datetime] = {}

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of configuration file."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def has_changed(self, file_path: str) -> bool:
        """Check if configuration file has changed."""
        current_hash = self._calculate_file_hash(file_path)
        previous_hash = self.file_hashes.get(file_path, "")
        return current_hash != previous_hash

    def update_cache(self, file_path: str, config: MetricConfiguration):
        """Update cache with new configuration."""
        self.configs[file_path] = config
        self.file_hashes[file_path] = self._calculate_file_hash(file_path)
        self.load_times[file_path] = datetime.utcnow()

    def remove_cache(self, file_path: str):
        """Remove configuration from cache."""
        self.configs.pop(file_path, None)
        self.file_hashes.pop(file_path, None)
        self.load_times.pop(file_path, None)

    def get_config(self, file_path: str) -> Optional[MetricConfiguration]:
        """Get cached configuration."""
        return self.configs.get(file_path)

    def get_all_configs(self) -> Dict[str, MetricConfiguration]:
        """Get all cached configurations."""
        return self.configs.copy()


class ConfigurationHotReloader:
    """Hot reloader for metric configurations."""

    def __init__(
        self,
        registry: CustomMetricRegistry,
        validator: Optional[MetricValidator] = None,
        watch_directories: Optional[List[str]] = None,
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
        reload_delay_seconds: float = 1.0,
    ):
        """
        Initialize configuration hot reloader.

        Args:
            registry: Metric registry to update
            validator: Validator for configurations
            watch_directories: Directories to watch for changes
            validation_level: Validation strictness level
            reload_delay_seconds: Delay before reload to debounce events
        """
        self.registry = registry
        self.validator = validator or MetricValidator(validation_level)
        self.watch_directories = watch_directories or []
        self.reload_delay_seconds = reload_delay_seconds

        # Internal state
        self.cache = ConfigurationCache()
        self.observers: List[Observer] = []
        self.reload_callbacks: List[Callable[[ReloadResult], None]] = []
        self.pending_reloads: Dict[str, float] = {}  # file_path -> scheduled_time
        self.running = False

        # Statistics
        self.reload_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_reload_time: Optional[datetime] = None

    def add_reload_callback(self, callback: Callable[[ReloadResult], None]):
        """Add callback to be called on reload events."""
        self.reload_callbacks.append(callback)

    def add_watch_directory(self, directory: str):
        """Add directory to watch for configuration changes."""
        if directory not in self.watch_directories:
            self.watch_directories.append(directory)

            if self.running:
                self._start_watching_directory(directory)

    def remove_watch_directory(self, directory: str):
        """Remove directory from watch list."""
        if directory in self.watch_directories:
            self.watch_directories.remove(directory)

            # Stop watching this directory
            self._stop_watching_directory(directory)

    def start(self):
        """Start the hot reloader."""
        if self.running:
            logger.warning("Hot reloader already running")
            return

        logger.info("Starting configuration hot reloader")
        self.running = True

        # Start watching directories
        for directory in self.watch_directories:
            self._start_watching_directory(directory)

        # Start background task for processing pending reloads
        asyncio.create_task(self._process_pending_reloads())

    def stop(self):
        """Stop the hot reloader."""
        if not self.running:
            return

        logger.info("Stopping configuration hot reloader")
        self.running = False

        # Stop all observers
        for observer in self.observers:
            observer.stop()
            observer.join()

        self.observers.clear()

    def _start_watching_directory(self, directory: str):
        """Start watching a specific directory."""
        if not os.path.exists(directory):
            logger.warning(f"Watch directory does not exist: {directory}")
            return

        observer = Observer()
        observer.schedule(ConfigurationWatcher(self), directory, recursive=True)
        observer.start()
        self.observers.append(observer)

        logger.info(f"Started watching directory: {directory}")

    def _stop_watching_directory(self, directory: str):
        """Stop watching a specific directory."""
        # This is simplified - in a real implementation, you'd track
        # which observer corresponds to which directory
        pass

    async def _handle_file_event(self, file_path: str, event_type: ReloadEvent):
        """Handle file system events."""
        # Debounce rapid file changes
        current_time = time.time()
        self.pending_reloads[file_path] = current_time + self.reload_delay_seconds

        logger.debug(f"Scheduled reload for {file_path} in {self.reload_delay_seconds}s")

    async def _process_pending_reloads(self):
        """Background task to process pending reloads."""
        while self.running:
            try:
                current_time = time.time()
                ready_reloads = []

                # Find files ready for reload
                for file_path, scheduled_time in list(self.pending_reloads.items()):
                    if current_time >= scheduled_time:
                        ready_reloads.append(file_path)
                        del self.pending_reloads[file_path]

                # Process ready reloads
                for file_path in ready_reloads:
                    await self._reload_configuration(file_path)

                await asyncio.sleep(0.1)  # Check every 100ms

            except Exception as e:
                logger.error(f"Error in reload processing: {e}")
                await asyncio.sleep(1)

    async def reload_all_configurations(self) -> List[ReloadResult]:
        """Manually reload all configurations."""
        results = []

        # Reload configurations from watch directories
        for directory in self.watch_directories:
            if os.path.exists(directory):
                config_files = self._find_config_files(directory)
                for config_file in config_files:
                    result = await self._reload_configuration(
                        config_file, ReloadEvent.MANUAL_RELOAD
                    )
                    results.append(result)

        return results

    async def reload_configuration(self, file_path: str) -> ReloadResult:
        """Manually reload a specific configuration."""
        return await self._reload_configuration(file_path, ReloadEvent.MANUAL_RELOAD)

    def _find_config_files(self, directory: str) -> List[str]:
        """Find all configuration files in a directory."""
        config_files = []

        for root, _dirs, files in os.walk(directory):
            for file in files:
                if (
                    file.endswith("_metric.yml")
                    or file.endswith("_metric.yaml")
                    or file.endswith("_metric.json")
                ):
                    config_files.append(os.path.join(root, file))

        return config_files

    async def _reload_configuration(
        self, file_path: str, event_type: ReloadEvent = ReloadEvent.FILE_MODIFIED
    ) -> ReloadResult:
        """Reload a specific configuration file."""
        self.reload_count += 1
        start_time = time.time()

        try:
            # Check if file exists (might be deleted)
            if not os.path.exists(file_path):
                if event_type == ReloadEvent.FILE_DELETED:
                    return await self._handle_config_deletion(file_path)
                else:
                    raise FileNotFoundError(f"Configuration file not found: {file_path}")

            # Check if file actually changed
            if not self.cache.has_changed(file_path):
                logger.debug(f"Configuration file unchanged, skipping reload: {file_path}")
                return ReloadResult(
                    success=True,
                    event_type=event_type,
                    config_path=file_path,
                    error_message="File unchanged",
                )

            old_hash = self.cache.file_hashes.get(file_path)

            # Load and parse configuration
            config = await self._load_configuration_file(file_path)

            # Validate configuration
            validation_result = await self._validate_configuration(config, file_path)

            if not validation_result.valid:
                self.error_count += 1

                # Call callbacks for validation failure
                result = ReloadResult(
                    success=False,
                    event_type=ReloadEvent.VALIDATION_FAILED,
                    config_path=file_path,
                    metric_name=config.name,
                    error_message=f"Validation failed: {len(validation_result.issues)} issues",
                    validation_result=validation_result,
                    old_config_hash=old_hash,
                )

                await self._notify_callbacks(result)
                return result

            # Update registry
            await self._update_registry(config, file_path)

            # Update cache
            new_hash = self.cache._calculate_file_hash(file_path)
            self.cache.update_cache(file_path, config)

            self.success_count += 1
            self.last_reload_time = datetime.utcnow()

            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Successfully reloaded metric '{config.name}' from {file_path} ({execution_time:.1f}ms)"
            )

            # Notify callbacks
            result = ReloadResult(
                success=True,
                event_type=ReloadEvent.RELOAD_SUCCESS,
                config_path=file_path,
                metric_name=config.name,
                validation_result=validation_result,
                old_config_hash=old_hash,
                new_config_hash=new_hash,
            )

            await self._notify_callbacks(result)
            return result

        except Exception as e:
            self.error_count += 1
            execution_time = (time.time() - start_time) * 1000

            error_msg = f"Failed to reload configuration from {file_path}: {e}"
            logger.error(f"{error_msg} ({execution_time:.1f}ms)")

            result = ReloadResult(
                success=False, event_type=event_type, config_path=file_path, error_message=str(e)
            )

            await self._notify_callbacks(result)
            return result

    async def _handle_config_deletion(self, file_path: str) -> ReloadResult:
        """Handle deletion of configuration file."""
        cached_config = self.cache.get_config(file_path)

        if cached_config:
            # Remove from registry
            if cached_config.name in self.registry.configurations:
                del self.registry.configurations[cached_config.name]
                logger.info(f"Removed metric '{cached_config.name}' due to config deletion")

            # Remove from cache
            self.cache.remove_cache(file_path)

        result = ReloadResult(
            success=True,
            event_type=ReloadEvent.FILE_DELETED,
            config_path=file_path,
            metric_name=cached_config.name if cached_config else None,
        )

        await self._notify_callbacks(result)
        return result

    async def _load_configuration_file(self, file_path: str) -> MetricConfiguration:
        """Load configuration from file."""
        import json

        import yaml

        path = Path(file_path)

        with open(file_path) as f:
            if path.suffix.lower() in [".yml", ".yaml"]:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {path.suffix}")

        return MetricConfiguration(**data)

    async def _validate_configuration(
        self, config: MetricConfiguration, file_path: str
    ) -> ValidationResult:
        """Validate configuration before loading."""
        # Create a temporary metric instance for validation
        try:
            # Load metric implementation
            metric_class = self.registry._load_metric_implementation(config)
            temp_metric = metric_class(config)

            # Validate the metric
            return await self.validator.validate_metric(temp_metric)

        except Exception as e:
            # Return validation failure
            from ulei.core.validation import ValidationCategory, ValidationIssue

            return ValidationResult(
                metric_name=config.name,
                valid=False,
                issues=[
                    ValidationIssue(
                        category=ValidationCategory.IMPLEMENTATION,
                        severity="error",
                        message=f"Failed to load metric implementation: {e}",
                    )
                ],
            )

    async def _update_registry(self, config: MetricConfiguration, file_path: str):
        """Update metric registry with new configuration."""
        try:
            # Load metric implementation
            metric_class = self.registry._load_metric_implementation(config)

            # Register or update metric
            self.registry.register_metric(metric_class, config)

        except Exception as e:
            raise Exception(f"Failed to update registry: {e}")

    async def _notify_callbacks(self, result: ReloadResult):
        """Notify all registered callbacks."""
        for callback in self.reload_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")

    def get_reload_statistics(self) -> Dict[str, Any]:
        """Get hot reload statistics."""
        return {
            "running": self.running,
            "watch_directories": self.watch_directories,
            "total_reloads": self.reload_count,
            "successful_reloads": self.success_count,
            "failed_reloads": self.error_count,
            "success_rate": self.success_count / self.reload_count if self.reload_count > 0 else 0,
            "last_reload_time": self.last_reload_time.isoformat()
            if self.last_reload_time
            else None,
            "cached_configs": len(self.cache.configs),
            "pending_reloads": len(self.pending_reloads),
            "active_observers": len(self.observers),
        }

    def get_cached_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about cached configurations."""
        result = {}

        for file_path, config in self.cache.get_all_configs().items():
            result[file_path] = {
                "metric_name": config.name,
                "version": config.version,
                "file_hash": self.cache.file_hashes.get(file_path),
                "load_time": self.cache.load_times.get(file_path).isoformat()
                if file_path in self.cache.load_times
                else None,
                "type": config.type.value,
                "complexity": config.complexity.value,
            }

        return result


# Global hot reloader instance
hot_reloader: Optional[ConfigurationHotReloader] = None


def initialize_hot_reloader(
    registry: Optional[CustomMetricRegistry] = None,
    watch_directories: Optional[List[str]] = None,
    **kwargs,
) -> ConfigurationHotReloader:
    """
    Initialize global hot reloader.

    Args:
        registry: Metric registry (uses global if None)
        watch_directories: Directories to watch
        **kwargs: Additional hot reloader options

    Returns:
        Hot reloader instance
    """
    global hot_reloader

    if hot_reloader is not None:
        logger.warning("Hot reloader already initialized")
        return hot_reloader

    registry = registry or metric_registry
    watch_directories = watch_directories or ["./metrics", "./config/metrics"]

    hot_reloader = ConfigurationHotReloader(
        registry=registry, watch_directories=watch_directories, **kwargs
    )

    return hot_reloader


def get_hot_reloader() -> Optional[ConfigurationHotReloader]:
    """Get global hot reloader instance."""
    return hot_reloader
