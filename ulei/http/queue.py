"""
Event queue system for batch processing of ingested evaluation samples.

Provides asynchronous queuing with priority support, batch dequeue,
and configurable processing strategies for online shadow evaluation.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from ulei.core.schemas import DatasetItem

logger = logging.getLogger(__name__)


@dataclass
class QueuedEvent:
    """Represents a queued evaluation event."""

    event_id: str
    dataset_item: DatasetItem
    suite_name: str
    queued_at: datetime
    priority: int = 1  # Higher values = higher priority
    attempts: int = 0
    last_error: Optional[str] = None


class EventQueue:
    """
    Asynchronous event queue for processing evaluation events in batches.

    Supports priority-based ordering and configurable batch processing.
    """

    def __init__(self, max_size: int = 10000, batch_size: int = 50, max_retries: int = 3):
        """
        Initialize event queue.

        Args:
            max_size: Maximum queue size
            batch_size: Default batch size for dequeue operations
            max_retries: Maximum retry attempts for failed events
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Use deque for O(1) append/popleft operations
        self._queue: deque[QueuedEvent] = deque()
        self._failed_events: deque[QueuedEvent] = deque()

        # Asyncio synchronization
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

        # Statistics
        self.enqueued_count = 0
        self.dequeued_count = 0
        self.failed_count = 0

    async def enqueue(self, event: QueuedEvent) -> bool:
        """
        Add event to queue.

        Args:
            event: Event to enqueue

        Returns:
            True if enqueued successfully, False if queue is full
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                logger.warning(f"Queue full, rejecting event {event.event_id}")
                return False

            # Insert based on priority (higher priority first)
            inserted = False
            for i, queued_event in enumerate(self._queue):
                if event.priority > queued_event.priority:
                    self._queue.insert(i, event)
                    inserted = True
                    break

            if not inserted:
                self._queue.append(event)

            self.enqueued_count += 1

            logger.debug(f"Enqueued event {event.event_id} (priority {event.priority})")

            # Notify waiting consumers
            self._not_empty.notify()

            return True

    async def dequeue_batch(
        self, batch_size: Optional[int] = None, timeout: Optional[float] = None
    ) -> List[QueuedEvent]:
        """
        Dequeue a batch of events for processing.

        Args:
            batch_size: Size of batch to dequeue (uses default if None)
            timeout: Maximum time to wait for events (waits indefinitely if None)

        Returns:
            List of events to process (may be empty if timeout)
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch = []

        async with self._lock:
            # Wait for events if queue is empty
            if not self._queue and timeout is not None:
                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    return batch  # Return empty batch on timeout
            elif not self._queue:
                await self._not_empty.wait()

            # Dequeue up to batch_size events
            while self._queue and len(batch) < batch_size:
                event = self._queue.popleft()
                batch.append(event)
                self.dequeued_count += 1

            if batch:
                logger.debug(f"Dequeued batch of {len(batch)} events")

        return batch

    async def requeue_failed_event(self, event: QueuedEvent, error: str):
        """
        Requeue a failed event for retry or move to failed queue.

        Args:
            event: Failed event
            error: Error message
        """
        async with self._lock:
            event.attempts += 1
            event.last_error = error

            if event.attempts < self.max_retries:
                # Retry with lower priority
                event.priority = max(1, event.priority - 1)

                # Insert back into queue for retry
                inserted = False
                for i, queued_event in enumerate(self._queue):
                    if event.priority > queued_event.priority:
                        self._queue.insert(i, event)
                        inserted = True
                        break

                if not inserted:
                    self._queue.append(event)

                logger.info(
                    f"Requeued event {event.event_id} for retry "
                    f"(attempt {event.attempts}/{self.max_retries})"
                )
            else:
                # Move to failed events queue
                self._failed_events.append(event)
                self.failed_count += 1

                logger.error(
                    f"Event {event.event_id} failed permanently after "
                    f"{event.attempts} attempts: {error}"
                )

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def failed_size(self) -> int:
        """Get failed events queue size."""
        return len(self._failed_events)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def get_statistics(self) -> dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        return {
            "current_size": len(self._queue),
            "failed_size": len(self._failed_events),
            "max_size": self.max_size,
            "enqueued_total": self.enqueued_count,
            "dequeued_total": self.dequeued_count,
            "failed_total": self.failed_count,
            "utilization": len(self._queue) / self.max_size,
            "is_full": len(self._queue) >= self.max_size,
        }

    async def clear(self):
        """Clear all events from queue."""
        async with self._lock:
            self._queue.clear()
            self._failed_events.clear()
            logger.info("Queue cleared")

    async def get_failed_events(self, limit: int = 100) -> List[QueuedEvent]:
        """
        Get failed events for inspection.

        Args:
            limit: Maximum number of failed events to return

        Returns:
            List of failed events
        """
        async with self._lock:
            return list(self._failed_events)[-limit:]

    async def retry_failed_events(self) -> int:
        """
        Retry all failed events by moving them back to main queue.

        Returns:
            Number of events moved back to queue
        """
        async with self._lock:
            retried_count = 0

            while self._failed_events:
                event = self._failed_events.popleft()

                # Reset attempts and increase priority for retry
                event.attempts = 0
                event.priority += 1
                event.last_error = None

                # Add back to main queue
                inserted = False
                for i, queued_event in enumerate(self._queue):
                    if event.priority > queued_event.priority:
                        self._queue.insert(i, event)
                        inserted = True
                        break

                if not inserted:
                    self._queue.append(event)

                retried_count += 1

            if retried_count > 0:
                logger.info(f"Retried {retried_count} failed events")
                self._not_empty.notify()

            return retried_count


class WindowedProcessor:
    """
    Processes events in time-based windows for aggregated evaluation.
    """

    def __init__(
        self,
        window_size_minutes: int = 60,
        overlap_minutes: int = 10,
        min_samples_per_window: int = 10,
    ):
        """
        Initialize windowed processor.

        Args:
            window_size_minutes: Size of each evaluation window
            overlap_minutes: Overlap between consecutive windows
            min_samples_per_window: Minimum samples required for evaluation
        """
        self.window_size = window_size_minutes * 60  # Convert to seconds
        self.overlap = overlap_minutes * 60
        self.min_samples = min_samples_per_window

        # Buffer for accumulating events
        self.event_buffer: List[QueuedEvent] = []
        self.last_window_time: Optional[datetime] = None

    def add_event(self, event: QueuedEvent):
        """Add event to the window buffer."""
        self.event_buffer.append(event)

        # Sort by timestamp to maintain order
        self.event_buffer.sort(key=lambda e: e.queued_at)

    def get_ready_windows(self) -> List[List[QueuedEvent]]:
        """
        Get windows that are ready for evaluation.

        Returns:
            List of event lists, each representing a complete window
        """
        if not self.event_buffer:
            return []

        now = datetime.utcnow()
        windows = []

        # Calculate window boundaries
        earliest_time = self.event_buffer[0].queued_at
        current_window_start = earliest_time

        while True:
            window_end = (
                current_window_start.replace(second=0, microsecond=0).timestamp() + self.window_size
            )

            # Check if window is complete (enough time has passed)
            if now.timestamp() < window_end:
                break

            # Collect events in this window
            window_events = [
                event
                for event in self.event_buffer
                if (current_window_start.timestamp() <= event.queued_at.timestamp() < window_end)
            ]

            # Only include windows with minimum sample size
            if len(window_events) >= self.min_samples:
                windows.append(window_events)

                # Remove processed events from buffer
                for event in window_events:
                    if event in self.event_buffer:
                        self.event_buffer.remove(event)

            # Move to next window (with overlap)
            advance_time = self.window_size - self.overlap
            current_window_start = datetime.fromtimestamp(
                current_window_start.timestamp() + advance_time
            )

        return windows

    def get_buffer_statistics(self) -> dict[str, Any]:
        """Get statistics about the event buffer."""
        if not self.event_buffer:
            return {
                "buffer_size": 0,
                "oldest_event": None,
                "newest_event": None,
                "time_span_minutes": 0,
            }

        oldest = min(self.event_buffer, key=lambda e: e.queued_at)
        newest = max(self.event_buffer, key=lambda e: e.queued_at)
        time_span = (newest.queued_at - oldest.queued_at).total_seconds() / 60

        return {
            "buffer_size": len(self.event_buffer),
            "oldest_event": oldest.queued_at.isoformat(),
            "newest_event": newest.queued_at.isoformat(),
            "time_span_minutes": time_span,
        }
