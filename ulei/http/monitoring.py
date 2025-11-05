"""
Enhanced health check endpoint and comprehensive service monitoring.

Provides detailed health status, dependency checks, and system diagnostics
for online shadow evaluation service monitoring and operational visibility.
"""

import asyncio
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import psutil

from ulei.http.alerts import AlertManager
from ulei.http.queue import EventQueue
from ulei.http.scheduler import BackgroundScheduler
from ulei.reporters.prometheus import PrometheusExporter

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class DependencyStatus:
    """Status of an external dependency."""

    name: str
    type: str  # "database", "api", "queue", etc.
    status: HealthStatus
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    last_checked: Optional[datetime] = None


class HealthMonitor:
    """Comprehensive health monitoring for ULEI service."""

    def __init__(
        self,
        event_queue: Optional[EventQueue] = None,
        alert_manager: Optional[AlertManager] = None,
        scheduler: Optional[BackgroundScheduler] = None,
        prometheus_exporter: Optional[PrometheusExporter] = None,
    ):
        """
        Initialize health monitor.

        Args:
            event_queue: Event queue to monitor
            alert_manager: Alert manager to check
            scheduler: Background scheduler to monitor
            prometheus_exporter: Metrics exporter to check
        """
        self.event_queue = event_queue
        self.alert_manager = alert_manager
        self.scheduler = scheduler
        self.prometheus_exporter = prometheus_exporter

        self.start_time = datetime.utcnow()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.dependencies: Dict[str, DependencyStatus] = {}

        # Monitoring thresholds
        self.memory_threshold_mb = 1000  # 1GB
        self.cpu_threshold_percent = 80
        self.disk_threshold_percent = 85
        self.queue_utilization_threshold = 0.8

    async def perform_comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all system components.

        Returns:
            Complete health status report
        """
        start_time = time.time()

        # Perform individual health checks
        checks = await asyncio.gather(
            self._check_system_resources(),
            self._check_event_queue(),
            self._check_scheduler(),
            self._check_alert_manager(),
            self._check_prometheus_exporter(),
            self._check_dependencies(),
            return_exceptions=True,
        )

        # Collect results
        health_checks = {}
        for check_result in checks:
            if isinstance(check_result, Exception):
                logger.error(f"Health check failed: {check_result}")
                continue
            if isinstance(check_result, dict):
                health_checks.update(check_result)

        # Determine overall status
        overall_status = self._determine_overall_status(health_checks)

        # Calculate total check duration
        total_duration_ms = (time.time() - start_time) * 1000

        # Build comprehensive response
        response = {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "checks": {
                name: self._health_check_to_dict(check) for name, check in health_checks.items()
            },
            "dependencies": {
                name: self._dependency_to_dict(dep) for name, dep in self.dependencies.items()
            },
            "system_info": self._get_system_info(),
            "performance_metrics": await self._get_performance_metrics(),
            "total_check_duration_ms": total_duration_ms,
        }

        # Store checks for monitoring
        self.health_checks.update(health_checks)

        return response

    async def _check_system_resources(self) -> Dict[str, HealthCheck]:
        """Check system resource utilization."""
        checks = {}

        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)

            if memory_mb > self.memory_threshold_mb:
                memory_status = (
                    HealthStatus.DEGRADED
                    if memory_mb < self.memory_threshold_mb * 1.5
                    else HealthStatus.UNHEALTHY
                )
                memory_message = f"High memory usage: {memory_mb:.1f}MB ({memory.percent:.1f}%)"
            else:
                memory_status = HealthStatus.HEALTHY
                memory_message = f"Memory usage normal: {memory_mb:.1f}MB ({memory.percent:.1f}%)"

            checks["memory"] = HealthCheck(
                name="Memory Usage",
                status=memory_status,
                message=memory_message,
                details={
                    "used_mb": memory_mb,
                    "total_mb": memory.total / (1024 * 1024),
                    "percent": memory.percent,
                    "available_mb": memory.available / (1024 * 1024),
                },
                timestamp=datetime.utcnow(),
            )

            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)

            if cpu_percent > self.cpu_threshold_percent:
                cpu_status = HealthStatus.DEGRADED if cpu_percent < 95 else HealthStatus.UNHEALTHY
                cpu_message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                cpu_status = HealthStatus.HEALTHY
                cpu_message = f"CPU usage normal: {cpu_percent:.1f}%"

            checks["cpu"] = HealthCheck(
                name="CPU Usage",
                status=cpu_status,
                message=cpu_message,
                details={
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_avg": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
                },
                timestamp=datetime.utcnow(),
            )

            # Disk check
            disk = psutil.disk_usage("/")

            if disk.percent > self.disk_threshold_percent:
                disk_status = HealthStatus.DEGRADED if disk.percent < 95 else HealthStatus.CRITICAL
                disk_message = f"High disk usage: {disk.percent:.1f}%"
            else:
                disk_status = HealthStatus.HEALTHY
                disk_message = f"Disk usage normal: {disk.percent:.1f}%"

            checks["disk"] = HealthCheck(
                name="Disk Usage",
                status=disk_status,
                message=disk_message,
                details={
                    "used_gb": disk.used / (1024**3),
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                },
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            checks["system_resources"] = HealthCheck(
                name="System Resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {e}",
                timestamp=datetime.utcnow(),
            )

        return checks

    async def _check_event_queue(self) -> Dict[str, HealthCheck]:
        """Check event queue health."""
        if not self.event_queue:
            return {}

        try:
            stats = self.event_queue.get_statistics()

            # Check queue utilization
            utilization = stats.get("utilization", 0)

            if utilization > self.queue_utilization_threshold:
                if utilization > 0.95:
                    status = HealthStatus.CRITICAL
                    message = f"Queue nearly full: {utilization:.1%}"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Queue utilization high: {utilization:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Queue utilization normal: {utilization:.1%}"

            # Check for high failure rate
            failed_size = stats.get("failed_size", 0)
            current_size = stats.get("current_size", 0)

            if failed_size > 0 and current_size > 0:
                failure_rate = failed_size / (failed_size + current_size)
                if failure_rate > 0.1:  # 10% failure rate
                    status = max(status, HealthStatus.DEGRADED)
                    message += f" (failure rate: {failure_rate:.1%})"

            return {
                "event_queue": HealthCheck(
                    name="Event Queue",
                    status=status,
                    message=message,
                    details=stats,
                    timestamp=datetime.utcnow(),
                )
            }

        except Exception as e:
            logger.error(f"Error checking event queue: {e}")
            return {
                "event_queue": HealthCheck(
                    name="Event Queue",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check event queue: {e}",
                    timestamp=datetime.utcnow(),
                )
            }

    async def _check_scheduler(self) -> Dict[str, HealthCheck]:
        """Check background scheduler health."""
        if not self.scheduler:
            return {}

        try:
            task_status = self.scheduler.get_task_status()

            # Check if scheduler is running
            if not task_status.get("scheduler_running", False):
                return {
                    "scheduler": HealthCheck(
                        name="Background Scheduler",
                        status=HealthStatus.UNHEALTHY,
                        message="Scheduler is not running",
                        details=task_status,
                        timestamp=datetime.utcnow(),
                    )
                }

            # Check for overdue tasks
            overdue_tasks = [
                task for task in task_status.get("tasks", []) if task.get("overdue", False)
            ]

            # Check for tasks with high error rates
            error_tasks = [
                task for task in task_status.get("tasks", []) if task.get("error_count", 0) > 3
            ]

            if overdue_tasks or error_tasks:
                status = HealthStatus.DEGRADED
                message = f"Scheduler issues: {len(overdue_tasks)} overdue, {len(error_tasks)} with errors"
            else:
                status = HealthStatus.HEALTHY
                message = f"Scheduler healthy: {task_status.get('enabled_tasks', 0)} active tasks"

            return {
                "scheduler": HealthCheck(
                    name="Background Scheduler",
                    status=status,
                    message=message,
                    details=task_status,
                    timestamp=datetime.utcnow(),
                )
            }

        except Exception as e:
            logger.error(f"Error checking scheduler: {e}")
            return {
                "scheduler": HealthCheck(
                    name="Background Scheduler",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check scheduler: {e}",
                    timestamp=datetime.utcnow(),
                )
            }

    async def _check_alert_manager(self) -> Dict[str, HealthCheck]:
        """Check alert manager health."""
        if not self.alert_manager:
            return {}

        try:
            # Test webhook connectivity if configured
            webhook_status = await self.alert_manager.test_webhook()

            alert_stats = self.alert_manager.get_alert_statistics()

            if not alert_stats.get("webhook_configured", False):
                status = HealthStatus.DEGRADED
                message = "Alert manager: No webhook configured"
            elif not webhook_status:
                status = HealthStatus.DEGRADED
                message = "Alert manager: Webhook unreachable"
            else:
                status = HealthStatus.HEALTHY
                message = "Alert manager healthy"

            return {
                "alert_manager": HealthCheck(
                    name="Alert Manager",
                    status=status,
                    message=message,
                    details=alert_stats,
                    timestamp=datetime.utcnow(),
                )
            }

        except Exception as e:
            logger.error(f"Error checking alert manager: {e}")
            return {
                "alert_manager": HealthCheck(
                    name="Alert Manager",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check alert manager: {e}",
                    timestamp=datetime.utcnow(),
                )
            }

    async def _check_prometheus_exporter(self) -> Dict[str, HealthCheck]:
        """Check Prometheus exporter health."""
        if not self.prometheus_exporter:
            return {}

        try:
            metrics_summary = self.prometheus_exporter.get_metrics_summary()

            # Check if metrics are being updated
            last_updated = metrics_summary.get("last_updated", 0)
            time_since_update = time.time() - last_updated

            if time_since_update > 300:  # 5 minutes
                status = HealthStatus.DEGRADED
                message = f"Metrics stale: {time_since_update:.0f}s since last update"
            else:
                status = HealthStatus.HEALTHY
                message = (
                    f"Metrics exporter healthy: {metrics_summary.get('total_metrics', 0)} metrics"
                )

            return {
                "prometheus_exporter": HealthCheck(
                    name="Prometheus Exporter",
                    status=status,
                    message=message,
                    details=metrics_summary,
                    timestamp=datetime.utcnow(),
                )
            }

        except Exception as e:
            logger.error(f"Error checking Prometheus exporter: {e}")
            return {
                "prometheus_exporter": HealthCheck(
                    name="Prometheus Exporter",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check Prometheus exporter: {e}",
                    timestamp=datetime.utcnow(),
                )
            }

    async def _check_dependencies(self) -> Dict[str, HealthCheck]:
        """Check external dependencies."""
        # This would check external services like:
        # - Provider APIs (OpenAI, Anthropic, etc.)
        # - Databases
        # - External storage
        # - Other microservices

        # For now, return placeholder
        return {}

    def _determine_overall_status(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall health status from individual checks."""
        if not checks:
            return HealthStatus.HEALTHY

        statuses = [check.status for check in checks.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor() or "unknown",
            "pid": psutil.Process().pid,
            "start_time": self.start_time.isoformat(),
        }

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            process = psutil.Process()

            return {
                "memory_info": {
                    "rss_mb": process.memory_info().rss / (1024 * 1024),
                    "vms_mb": process.memory_info().vms / (1024 * 1024),
                    "percent": process.memory_percent(),
                },
                "cpu_info": {
                    "percent": process.cpu_percent(),
                    "times": process.cpu_times()._asdict(),
                },
                "io_counters": process.io_counters()._asdict()
                if hasattr(process, "io_counters")
                else None,
                "num_threads": process.num_threads(),
                "connections": len(process.connections()) if hasattr(process, "connections") else 0,
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    def _health_check_to_dict(self, check: HealthCheck) -> Dict[str, Any]:
        """Convert HealthCheck to dictionary."""
        return {
            "name": check.name,
            "status": check.status.value,
            "message": check.message,
            "details": check.details,
            "duration_ms": check.duration_ms,
            "timestamp": check.timestamp.isoformat() if check.timestamp else None,
        }

    def _dependency_to_dict(self, dependency: DependencyStatus) -> Dict[str, Any]:
        """Convert DependencyStatus to dictionary."""
        return {
            "name": dependency.name,
            "type": dependency.type,
            "status": dependency.status.value,
            "response_time_ms": dependency.response_time_ms,
            "error_message": dependency.error_message,
            "last_checked": dependency.last_checked.isoformat()
            if dependency.last_checked
            else None,
        }

    async def get_readiness_status(self) -> Tuple[bool, str]:
        """
        Check if service is ready to accept traffic.

        Returns:
            Tuple of (is_ready, reason)
        """
        try:
            # Quick readiness checks
            if self.event_queue and self.event_queue.size() >= self.event_queue.max_size:
                return False, "Event queue is full"

            if self.scheduler and not self.scheduler.running:
                return False, "Background scheduler is not running"

            # Check critical system resources
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return False, f"Memory usage critical: {memory.percent:.1f}%"

            disk = psutil.disk_usage("/")
            if disk.percent > 95:
                return False, f"Disk usage critical: {disk.percent:.1f}%"

            return True, "Service is ready"

        except Exception as e:
            return False, f"Readiness check failed: {e}"

    async def get_liveness_status(self) -> Tuple[bool, str]:
        """
        Check if service is alive and responding.

        Returns:
            Tuple of (is_alive, reason)
        """
        try:
            # Basic liveness check - can we respond?
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

            # Check if process is responding
            process = psutil.Process()
            if process.status() == psutil.STATUS_ZOMBIE:
                return False, "Process is zombie"

            return True, f"Service is alive (uptime: {uptime:.0f}s)"

        except Exception as e:
            return False, f"Liveness check failed: {e}"
