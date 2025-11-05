"""
Background scheduler for periodic evaluation runs and maintenance tasks.

Provides scheduled execution of windowed evaluations, report generation,
cleanup tasks, and system maintenance for online shadow evaluation.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ulei.core.evaluator import Evaluator
from ulei.http.alerts import AlertManager
from ulei.http.queue import EventQueue, QueuedEvent, WindowedProcessor
from ulei.reporters.prometheus import PrometheusExporter

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduled tasks."""

    INTERVAL = "interval"
    CRON = "cron"
    ONCE = "once"


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""

    task_id: str
    name: str
    schedule_type: ScheduleType
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    run_at: Optional[datetime] = None
    callback: Optional[Callable] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class BackgroundScheduler:
    """Background scheduler for periodic evaluation and maintenance tasks."""

    def __init__(
        self,
        evaluator: Evaluator,
        event_queue: EventQueue,
        alert_manager: Optional[AlertManager] = None,
        prometheus_exporter: Optional[PrometheusExporter] = None,
    ):
        """
        Initialize background scheduler.

        Args:
            evaluator: Core evaluator instance
            event_queue: Event queue for processing
            alert_manager: Alert manager for notifications
            prometheus_exporter: Metrics exporter
        """
        self.evaluator = evaluator
        self.event_queue = event_queue
        self.alert_manager = alert_manager
        self.prometheus_exporter = prometheus_exporter

        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None

        # Windowed processor for aggregated evaluations
        self.windowed_processor = WindowedProcessor(
            window_size_minutes=60, overlap_minutes=10, min_samples_per_window=5
        )

        # Setup default tasks
        self._setup_default_tasks()

    def _setup_default_tasks(self):
        """Setup default scheduled tasks."""

        # Windowed evaluation processing
        self.add_task(
            task_id="windowed_evaluation",
            name="Process Windowed Evaluations",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=300,  # Every 5 minutes
            callback=self._process_windowed_evaluations,
        )

        # Queue maintenance
        self.add_task(
            task_id="queue_maintenance",
            name="Queue Maintenance and Cleanup",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=900,  # Every 15 minutes
            callback=self._queue_maintenance,
        )

        # Metrics export
        if self.prometheus_exporter:
            self.add_task(
                task_id="metrics_export",
                name="Export System Metrics",
                schedule_type=ScheduleType.INTERVAL,
                interval_seconds=60,  # Every minute
                callback=self._export_metrics,
            )

        # Alert system cleanup
        if self.alert_manager:
            self.add_task(
                task_id="alert_cleanup",
                name="Alert System Cleanup",
                schedule_type=ScheduleType.INTERVAL,
                interval_seconds=3600,  # Every hour
                callback=self._alert_cleanup,
            )

        # Report cleanup (daily)
        self.add_task(
            task_id="report_cleanup",
            name="Clean Up Old Reports",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=86400,  # Daily
            callback=self._cleanup_old_reports,
        )

    def add_task(
        self,
        task_id: str,
        name: str,
        schedule_type: ScheduleType,
        callback: Callable,
        interval_seconds: Optional[int] = None,
        cron_expression: Optional[str] = None,
        run_at: Optional[datetime] = None,
        enabled: bool = True,
    ) -> ScheduledTask:
        """
        Add a scheduled task.

        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            schedule_type: Type of scheduling
            callback: Function to execute
            interval_seconds: Interval for INTERVAL type
            cron_expression: Cron expression for CRON type
            run_at: Specific time for ONCE type
            enabled: Whether task is enabled

        Returns:
            Created ScheduledTask
        """
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            schedule_type=schedule_type,
            interval_seconds=interval_seconds,
            cron_expression=cron_expression,
            run_at=run_at,
            callback=callback,
            enabled=enabled,
        )

        # Calculate next run time
        task.next_run = self._calculate_next_run(task)

        self.tasks[task_id] = task
        logger.info(f"Added scheduled task: {name} (next run: {task.next_run})")

        return task

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a scheduled task.

        Args:
            task_id: Task identifier to remove

        Returns:
            True if task was removed, False if not found
        """
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            logger.info(f"Removed scheduled task: {task.name}")
            return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """Enable a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.tasks[task_id].next_run = self._calculate_next_run(self.tasks[task_id])
            logger.info(f"Enabled task: {task_id}")
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            logger.info(f"Disabled task: {task_id}")
            return True
        return False

    async def start(self):
        """Start the background scheduler."""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Background scheduler started")

    async def stop(self):
        """Stop the background scheduler."""
        if not self.running:
            return

        self.running = False

        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Background scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self.running:
            try:
                now = datetime.utcnow()

                # Find tasks ready to run
                ready_tasks = [
                    task
                    for task in self.tasks.values()
                    if (task.enabled and task.next_run and task.next_run <= now)
                ]

                # Execute ready tasks
                for task in ready_tasks:
                    await self._execute_task(task)

                # Sleep for a short interval before checking again
                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(30)  # Longer sleep on error

    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        try:
            logger.info(f"Executing scheduled task: {task.name}")
            task.last_run = datetime.utcnow()

            # Execute the callback
            if asyncio.iscoroutinefunction(task.callback):
                await task.callback()
            else:
                task.callback()

            task.run_count += 1
            task.last_error = None

            # Calculate next run time
            if task.schedule_type != ScheduleType.ONCE:
                task.next_run = self._calculate_next_run(task)
            else:
                task.enabled = False  # Disable one-time tasks after execution

            logger.debug(f"Task {task.name} completed successfully")

        except Exception as e:
            task.error_count += 1
            task.last_error = str(e)
            logger.error(f"Task {task.name} failed: {e}")

            # Recalculate next run even after error
            if task.schedule_type != ScheduleType.ONCE:
                task.next_run = self._calculate_next_run(task)

            # Send alert for critical task failures
            if self.alert_manager and task.error_count >= 3:
                await self.alert_manager.send_system_alert(
                    title=f"Scheduled Task Failing: {task.name}",
                    message=f"Task has failed {task.error_count} times. Last error: {e}",
                    severity="high",
                    details={
                        "task_id": task.task_id,
                        "task_name": task.name,
                        "error_count": task.error_count,
                        "last_error": str(e),
                    },
                )

    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next run time for a task."""
        now = datetime.utcnow()

        if task.schedule_type == ScheduleType.INTERVAL:
            if task.interval_seconds:
                if task.last_run:
                    return task.last_run + timedelta(seconds=task.interval_seconds)
                else:
                    return now + timedelta(seconds=task.interval_seconds)

        elif task.schedule_type == ScheduleType.ONCE:
            return task.run_at

        elif task.schedule_type == ScheduleType.CRON:
            # Simplified cron parsing - in production would use croniter library
            # For now, return None (not implemented)
            logger.warning("CRON scheduling not implemented yet")
            return None

        return None

    async def _process_windowed_evaluations(self):
        """Process windowed evaluations from accumulated events."""
        try:
            # Get ready windows from processor
            ready_windows = self.windowed_processor.get_ready_windows()

            if not ready_windows:
                logger.debug("No ready windows for evaluation")
                return

            logger.info(f"Processing {len(ready_windows)} evaluation windows")

            for window_events in ready_windows:
                # Group events by suite
                events_by_suite = {}
                for event in window_events:
                    suite_name = event.suite_name
                    if suite_name not in events_by_suite:
                        events_by_suite[suite_name] = []
                    events_by_suite[suite_name].append(event)

                # Process each suite
                for suite_name, suite_events in events_by_suite.items():
                    await self._evaluate_windowed_batch(suite_name, suite_events)

        except Exception as e:
            logger.error(f"Error processing windowed evaluations: {e}")

    async def _evaluate_windowed_batch(self, suite_name: str, events: List[QueuedEvent]):
        """Evaluate a windowed batch of events."""
        try:
            # Create default suite configuration
            from ulei.core.schemas import EvaluationSuite, MetricSpec

            suite_config = EvaluationSuite(
                name=f"{suite_name}_windowed",
                metrics=[MetricSpec(name="faithfulness"), MetricSpec(name="answer_relevancy")],
                provider_priority=["ragas", "deepeval"],
                thresholds={"faithfulness": 0.7, "answer_relevancy": 0.8},
            )

            # Convert events to dataset items
            dataset_items = [event.dataset_item for event in events]

            # Run evaluation
            report = await self.evaluator.run_evaluation(suite=suite_config, dataset=dataset_items)

            logger.info(f"Completed windowed evaluation for {suite_name}: {len(events)} items")

            # Export metrics
            if self.prometheus_exporter:
                self.prometheus_exporter.export_evaluation_report(report)

            # Check for alerts
            if self.alert_manager:
                await self.alert_manager.check_report_for_alerts(report)

        except Exception as e:
            logger.error(f"Error evaluating windowed batch for {suite_name}: {e}")

    async def _queue_maintenance(self):
        """Perform queue maintenance and cleanup."""
        try:
            stats = self.event_queue.get_statistics()

            # Log queue statistics
            logger.info(
                f"Queue stats: size={stats['current_size']}, "
                f"failed={stats['failed_size']}, "
                f"utilization={stats['utilization']:.2%}"
            )

            # Retry failed events if failure rate is manageable
            if stats["failed_size"] > 0 and stats["failed_size"] < 100:
                retried = await self.event_queue.retry_failed_events()
                if retried > 0:
                    logger.info(f"Retried {retried} failed events")

            # Alert if queue is getting full
            if stats["utilization"] > 0.8 and self.alert_manager:
                await self.alert_manager.send_system_alert(
                    title="High Queue Utilization",
                    message=f"Event queue is {stats['utilization']:.1%} full",
                    severity="medium" if stats["utilization"] < 0.95 else "high",
                    details=stats,
                )

        except Exception as e:
            logger.error(f"Error in queue maintenance: {e}")

    async def _export_metrics(self):
        """Export system metrics to Prometheus."""
        try:
            if not self.prometheus_exporter:
                return

            # Export queue metrics
            queue_stats = self.event_queue.get_statistics()
            self.prometheus_exporter.export_queue_metrics(queue_stats)

            # Export scheduler metrics
            active_tasks = sum(1 for task in self.tasks.values() if task.enabled)
            self.prometheus_exporter.update_metric("scheduler_active_tasks", active_tasks)

            # Export windowed processor metrics
            buffer_stats = self.windowed_processor.get_buffer_statistics()
            self.prometheus_exporter.update_metric(
                "windowed_buffer_size", buffer_stats["buffer_size"]
            )

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

    async def _alert_cleanup(self):
        """Clean up old alert tracking data."""
        try:
            if self.alert_manager:
                await self.alert_manager.cleanup_old_tracking(days=7)

        except Exception as e:
            logger.error(f"Error in alert cleanup: {e}")

    async def _cleanup_old_reports(self):
        """Clean up old evaluation reports and data."""
        try:
            # This would clean up old report files, database records, etc.
            # For now, just log the activity
            logger.info("Performing report cleanup (placeholder implementation)")

            # Could implement:
            # - Delete old report files
            # - Clean up database records
            # - Archive old data
            # - Free up disk space

        except Exception as e:
            logger.error(f"Error in report cleanup: {e}")

    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all scheduled tasks."""
        now = datetime.utcnow()

        task_statuses = []
        for task in self.tasks.values():
            task_statuses.append(
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "enabled": task.enabled,
                    "schedule_type": task.schedule_type.value,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                    "run_count": task.run_count,
                    "error_count": task.error_count,
                    "last_error": task.last_error,
                    "overdue": (task.next_run and task.enabled and task.next_run < now),
                }
            )

        return {
            "scheduler_running": self.running,
            "total_tasks": len(self.tasks),
            "enabled_tasks": sum(1 for task in self.tasks.values() if task.enabled),
            "tasks": task_statuses,
        }

    async def run_task_now(self, task_id: str) -> bool:
        """Manually trigger a task to run immediately."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if not task.enabled:
            return False

        await self._execute_task(task)
        return True
