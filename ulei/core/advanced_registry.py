"""
Enhanced metric registry system for ULEI.

Provides advanced discovery, versioning, lifecycle management,
and metadata handling for custom evaluation metrics.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ulei.core.custom_metrics import (
    CustomMetricRegistry,
    MetricComplexity,
    MetricConfiguration,
    MetricType,
)
from ulei.core.validation import MetricValidator

logger = logging.getLogger(__name__)


class MetricStatus(Enum):
    """Lifecycle status of metrics."""

    ACTIVE = "active"  # Ready for use
    DEPRECATED = "deprecated"  # Still available but discouraged
    EXPERIMENTAL = "experimental"  # Under development
    DISABLED = "disabled"  # Temporarily disabled
    ARCHIVED = "archived"  # No longer available


class RegistryEvent(Enum):
    """Registry events for tracking."""

    METRIC_REGISTERED = "metric_registered"
    METRIC_UPDATED = "metric_updated"
    METRIC_DEPRECATED = "metric_deprecated"
    METRIC_DISABLED = "metric_disabled"
    METRIC_REMOVED = "metric_removed"
    VERSION_ADDED = "version_added"
    USAGE_RECORDED = "usage_recorded"


@dataclass
class MetricVersion:
    """Information about a metric version."""

    version: str
    config: MetricConfiguration
    status: MetricStatus = MetricStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    deprecated_at: Optional[datetime] = None
    disabled_at: Optional[datetime] = None
    changelog: Optional[str] = None
    breaking_changes: bool = False

    def is_available(self) -> bool:
        """Check if this version is available for use."""
        return self.status in [MetricStatus.ACTIVE, MetricStatus.EXPERIMENTAL]


@dataclass
class MetricUsage:
    """Tracks usage statistics for a metric."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time_ms: float = 0.0
    last_used: Optional[datetime] = None
    average_execution_time_ms: float = 0.0

    # Usage patterns
    daily_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hourly_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_patterns: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_usage(
        self, success: bool, execution_time_ms: float, error_type: Optional[str] = None
    ):
        """Record a usage event."""
        now = datetime.utcnow()

        self.total_calls += 1
        self.total_execution_time_ms += execution_time_ms
        self.last_used = now

        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error_type:
                self.error_patterns[error_type] += 1

        # Update averages
        if self.total_calls > 0:
            self.average_execution_time_ms = self.total_execution_time_ms / self.total_calls

        # Update daily/hourly usage
        day_key = now.strftime("%Y-%m-%d")
        hour_key = now.strftime("%Y-%m-%d-%H")

        self.daily_usage[day_key] += 1
        self.hourly_usage[hour_key] += 1

        # Cleanup old usage data (keep last 30 days)
        cutoff_date = now - timedelta(days=30)
        cutoff_day = cutoff_date.strftime("%Y-%m-%d")
        cutoff_hour = cutoff_date.strftime("%Y-%m-%d-%H")

        self.daily_usage = {k: v for k, v in self.daily_usage.items() if k >= cutoff_day}
        self.hourly_usage = {k: v for k, v in self.hourly_usage.items() if k >= cutoff_hour}

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    def get_recent_usage(self, days: int = 7) -> int:
        """Get usage count for recent days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cutoff_day = cutoff_date.strftime("%Y-%m-%d")

        return sum(count for day, count in self.daily_usage.items() if day >= cutoff_day)


@dataclass
class MetricEntry:
    """Complete metric registry entry."""

    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    # Versioning
    versions: Dict[str, MetricVersion] = field(default_factory=dict)
    current_version: Optional[str] = None
    latest_version: Optional[str] = None

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: MetricStatus = MetricStatus.ACTIVE

    # Usage tracking
    usage: MetricUsage = field(default_factory=MetricUsage)

    # Relationships
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    alternatives: Set[str] = field(default_factory=set)

    # Metadata
    author: Optional[str] = None
    license: Optional[str] = None
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None

    def add_version(self, version: str, config: MetricConfiguration):
        """Add a new version of the metric."""
        metric_version = MetricVersion(
            version=version, config=config, changelog=f"Version {version} of {self.name}"
        )

        self.versions[version] = metric_version
        self.latest_version = version

        if not self.current_version or version > self.current_version:
            self.current_version = version

        self.updated_at = datetime.utcnow()

    def get_version(self, version: Optional[str] = None) -> Optional[MetricVersion]:
        """Get specific version or current version."""
        if version is None:
            version = self.current_version

        if version is None:
            return None

        return self.versions.get(version)

    def get_available_versions(self) -> List[str]:
        """Get list of available versions."""
        return [version for version, info in self.versions.items() if info.is_available()]

    def deprecate_version(self, version: str, reason: Optional[str] = None):
        """Deprecate a specific version."""
        if version in self.versions:
            self.versions[version].status = MetricStatus.DEPRECATED
            self.versions[version].deprecated_at = datetime.utcnow()
            if reason:
                self.versions[version].changelog = f"Deprecated: {reason}"


class AdvancedMetricRegistry(CustomMetricRegistry):
    """Enhanced metric registry with advanced features."""

    def __init__(self, validator: Optional[MetricValidator] = None):
        """
        Initialize advanced metric registry.

        Args:
            validator: Metric validator for registration
        """
        super().__init__()
        self.validator = validator

        # Enhanced registry storage
        self.entries: Dict[str, MetricEntry] = {}
        self.categories: Dict[str, Set[str]] = defaultdict(set)
        self.tags: Dict[str, Set[str]] = defaultdict(set)

        # Event tracking
        self.events: List[Dict[str, Any]] = []
        self.event_callbacks: List[callable] = []

        # Registry statistics
        self.registration_count = 0
        self.update_count = 0
        self.usage_count = 0

        # Background maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        self._start_maintenance_task()

    def _start_maintenance_task(self):
        """Start background maintenance task."""
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while True:
            try:
                await self._cleanup_old_data()
                await self._update_statistics()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in registry maintenance: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _cleanup_old_data(self):
        """Clean up old event data and usage statistics."""
        # Remove events older than 30 days
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.events = [
            event
            for event in self.events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]

    async def _update_statistics(self):
        """Update registry statistics."""
        # This could compute derived statistics, health metrics, etc.
        pass

    def register_metric_advanced(
        self,
        metric_class: type,
        config: MetricConfiguration,
        version: str = "1.0.0",
        category: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        **metadata,
    ) -> bool:
        """
        Register metric with advanced metadata.

        Args:
            metric_class: Metric implementation class
            config: Metric configuration
            version: Version string
            category: Metric category
            tags: Tags for discovery
            **metadata: Additional metadata

        Returns:
            True if registration successful
        """
        try:
            # Validate metric if validator available
            if self.validator:
                temp_metric = metric_class(config)
                validation_result = asyncio.run(self.validator.validate_metric(temp_metric))

                if not validation_result.valid:
                    logger.error(
                        f"Metric validation failed for {config.name}: {validation_result.issues}"
                    )
                    return False

            # Get or create registry entry
            if config.name in self.entries:
                entry = self.entries[config.name]
                entry.add_version(version, config)
                event_type = RegistryEvent.METRIC_UPDATED
                self.update_count += 1
            else:
                entry = MetricEntry(
                    name=config.name,
                    display_name=config.display_name,
                    description=config.description,
                    category=category or config.category,
                    tags=tags or set(config.tags),
                    author=metadata.get("author") or config.author,
                    **{k: v for k, v in metadata.items() if k not in ["author"]},
                )
                entry.add_version(version, config)
                self.entries[config.name] = entry
                event_type = RegistryEvent.METRIC_REGISTERED
                self.registration_count += 1

            # Update parent registry
            super().register_metric(metric_class, config)

            # Update indices
            self._update_indices(entry)

            # Record event
            self._record_event(
                event_type,
                {
                    "metric_name": config.name,
                    "version": version,
                    "category": entry.category,
                    "tags": list(entry.tags),
                },
            )

            logger.info(f"Registered metric {config.name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Failed to register metric {config.name}: {e}")
            return False

    def _update_indices(self, entry: MetricEntry):
        """Update category and tag indices."""
        if entry.category:
            self.categories[entry.category].add(entry.name)

        for tag in entry.tags:
            self.tags[tag].add(entry.name)

    def _record_event(self, event_type: RegistryEvent, data: Dict[str, Any]):
        """Record registry event."""
        event = {"type": event_type.value, "timestamp": datetime.utcnow().isoformat(), "data": data}

        self.events.append(event)

        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def discover_metrics(
        self,
        category: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        metric_type: Optional[MetricType] = None,
        complexity: Optional[MetricComplexity] = None,
        status: Optional[MetricStatus] = None,
        search_term: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Discover metrics with advanced filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            metric_type: Filter by metric type
            complexity: Filter by complexity
            status: Filter by status
            search_term: Search in name/description

        Returns:
            List of metric information
        """
        results = []

        for entry in self.entries.values():
            # Apply filters
            if category and entry.category != category:
                continue

            if tags and not entry.tags.intersection(tags):
                continue

            if status and entry.status != status:
                continue

            if search_term:
                search_lower = search_term.lower()
                if (
                    search_lower not in entry.name.lower()
                    and search_lower not in (entry.description or "").lower()
                ):
                    continue

            # Get current version for additional filtering
            current_version = entry.get_version()
            if current_version:
                if metric_type and current_version.config.type != metric_type:
                    continue

                if complexity and current_version.config.complexity != complexity:
                    continue

            # Build result
            result = {
                "name": entry.name,
                "display_name": entry.display_name,
                "description": entry.description,
                "category": entry.category,
                "tags": list(entry.tags),
                "status": entry.status.value,
                "current_version": entry.current_version,
                "latest_version": entry.latest_version,
                "available_versions": entry.get_available_versions(),
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "usage_stats": {
                    "total_calls": entry.usage.total_calls,
                    "success_rate": entry.usage.get_success_rate(),
                    "average_execution_time_ms": entry.usage.average_execution_time_ms,
                    "recent_usage": entry.usage.get_recent_usage(),
                },
            }

            if current_version:
                result.update(
                    {
                        "type": current_version.config.type.value,
                        "complexity": current_version.config.complexity.value,
                        "timeout_seconds": current_version.config.timeout_seconds,
                    }
                )

            results.append(result)

        # Sort by usage and name
        results.sort(key=lambda x: (-x["usage_stats"]["total_calls"], x["name"]))

        return results

    def get_metric_details(
        self, name: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a metric.

        Args:
            name: Metric name
            version: Specific version (None for current)

        Returns:
            Detailed metric information
        """
        if name not in self.entries:
            return None

        entry = self.entries[name]
        metric_version = entry.get_version(version)

        if not metric_version:
            return None

        return {
            "name": entry.name,
            "display_name": entry.display_name,
            "description": entry.description,
            "category": entry.category,
            "tags": list(entry.tags),
            "status": entry.status.value,
            # Version information
            "version": metric_version.version,
            "version_status": metric_version.status.value,
            "version_created": metric_version.created_at.isoformat(),
            "changelog": metric_version.changelog,
            "breaking_changes": metric_version.breaking_changes,
            # Configuration
            "config": metric_version.config.dict(),
            # Usage statistics
            "usage": {
                "total_calls": entry.usage.total_calls,
                "successful_calls": entry.usage.successful_calls,
                "failed_calls": entry.usage.failed_calls,
                "success_rate": entry.usage.get_success_rate(),
                "average_execution_time_ms": entry.usage.average_execution_time_ms,
                "last_used": entry.usage.last_used.isoformat() if entry.usage.last_used else None,
                "recent_usage_7d": entry.usage.get_recent_usage(7),
                "recent_usage_30d": entry.usage.get_recent_usage(30),
                "error_patterns": dict(entry.usage.error_patterns),
            },
            # Relationships
            "dependencies": list(entry.dependencies),
            "dependents": list(entry.dependents),
            "alternatives": list(entry.alternatives),
            # Metadata
            "author": entry.author,
            "license": entry.license,
            "documentation_url": entry.documentation_url,
            "source_url": entry.source_url,
            # Available versions
            "all_versions": list(entry.versions.keys()),
            "available_versions": entry.get_available_versions(),
        }

    def record_usage(
        self,
        metric_name: str,
        success: bool,
        execution_time_ms: float,
        error_type: Optional[str] = None,
    ):
        """Record metric usage for statistics."""
        if metric_name in self.entries:
            entry = self.entries[metric_name]
            entry.usage.record_usage(success, execution_time_ms, error_type)
            self.usage_count += 1

            # Record usage event
            self._record_event(
                RegistryEvent.USAGE_RECORDED,
                {
                    "metric_name": metric_name,
                    "success": success,
                    "execution_time_ms": execution_time_ms,
                    "error_type": error_type,
                },
            )

    def deprecate_metric(
        self,
        name: str,
        version: Optional[str] = None,
        reason: Optional[str] = None,
        alternative: Optional[str] = None,
    ) -> bool:
        """
        Deprecate a metric or specific version.

        Args:
            name: Metric name
            version: Specific version (None for all versions)
            reason: Deprecation reason
            alternative: Recommended alternative metric

        Returns:
            True if successful
        """
        if name not in self.entries:
            return False

        entry = self.entries[name]

        if version:
            # Deprecate specific version
            entry.deprecate_version(version, reason)
        else:
            # Deprecate entire metric
            entry.status = MetricStatus.DEPRECATED
            for ver in entry.versions.values():
                ver.status = MetricStatus.DEPRECATED
                ver.deprecated_at = datetime.utcnow()

        if alternative:
            entry.alternatives.add(alternative)

        self._record_event(
            RegistryEvent.METRIC_DEPRECATED,
            {"metric_name": name, "version": version, "reason": reason, "alternative": alternative},
        )

        logger.info(f"Deprecated metric {name}" + (f" v{version}" if version else ""))
        return True

    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        total_metrics = len(self.entries)
        active_metrics = sum(1 for e in self.entries.values() if e.status == MetricStatus.ACTIVE)
        deprecated_metrics = sum(
            1 for e in self.entries.values() if e.status == MetricStatus.DEPRECATED
        )

        # Category distribution
        category_counts = {cat: len(metrics) for cat, metrics in self.categories.items()}

        # Tag distribution
        tag_counts = {tag: len(metrics) for tag, metrics in self.tags.items()}

        # Usage statistics
        total_usage = sum(e.usage.total_calls for e in self.entries.values())
        total_success = sum(e.usage.successful_calls for e in self.entries.values())

        # Most used metrics
        most_used = sorted(
            [(name, entry.usage.total_calls) for name, entry in self.entries.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "total_metrics": total_metrics,
            "active_metrics": active_metrics,
            "deprecated_metrics": deprecated_metrics,
            "experimental_metrics": sum(
                1 for e in self.entries.values() if e.status == MetricStatus.EXPERIMENTAL
            ),
            "disabled_metrics": sum(
                1 for e in self.entries.values() if e.status == MetricStatus.DISABLED
            ),
            "total_registrations": self.registration_count,
            "total_updates": self.update_count,
            "total_usage_events": self.usage_count,
            "total_usage_calls": total_usage,
            "overall_success_rate": (total_success / total_usage * 100) if total_usage > 0 else 0,
            "categories": category_counts,
            "tags": tag_counts,
            "most_used_metrics": most_used,
            "recent_events": len(
                [
                    e
                    for e in self.events
                    if datetime.fromisoformat(e["timestamp"])
                    > datetime.utcnow() - timedelta(days=7)
                ]
            ),
        }

    def export_registry(self, include_usage: bool = True) -> Dict[str, Any]:
        """Export registry data for backup or migration."""
        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "metrics": {},
        }

        for name, entry in self.entries.items():
            metric_data = {
                "name": entry.name,
                "display_name": entry.display_name,
                "description": entry.description,
                "category": entry.category,
                "tags": list(entry.tags),
                "status": entry.status.value,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "current_version": entry.current_version,
                "latest_version": entry.latest_version,
                "author": entry.author,
                "license": entry.license,
                "documentation_url": entry.documentation_url,
                "source_url": entry.source_url,
                "dependencies": list(entry.dependencies),
                "dependents": list(entry.dependents),
                "alternatives": list(entry.alternatives),
                "versions": {},
            }

            # Export versions
            for version, version_info in entry.versions.items():
                version_data = {
                    "version": version_info.version,
                    "status": version_info.status.value,
                    "created_at": version_info.created_at.isoformat(),
                    "deprecated_at": version_info.deprecated_at.isoformat()
                    if version_info.deprecated_at
                    else None,
                    "changelog": version_info.changelog,
                    "breaking_changes": version_info.breaking_changes,
                    "config": version_info.config.dict(),
                }
                metric_data["versions"][version] = version_data

            # Include usage data if requested
            if include_usage:
                metric_data["usage"] = {
                    "total_calls": entry.usage.total_calls,
                    "successful_calls": entry.usage.successful_calls,
                    "failed_calls": entry.usage.failed_calls,
                    "total_execution_time_ms": entry.usage.total_execution_time_ms,
                    "last_used": entry.usage.last_used.isoformat()
                    if entry.usage.last_used
                    else None,
                    "average_execution_time_ms": entry.usage.average_execution_time_ms,
                    "error_patterns": dict(entry.usage.error_patterns),
                }

            export_data["metrics"][name] = metric_data

        return export_data

    def add_event_callback(self, callback: callable):
        """Add callback for registry events."""
        self.event_callbacks.append(callback)

    def remove_event_callback(self, callback: callable):
        """Remove event callback."""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
