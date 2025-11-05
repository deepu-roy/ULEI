"""
Performance optimization system for custom metric execution.

Provides caching, parallel processing, resource management, and intelligent
scheduling to optimize evaluation performance and resource utilization.
"""

import asyncio
import logging
import multiprocessing
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil

from ulei.core.custom_metrics import (
    BaseMetric,
    MetricComplexity,
    MetricExecutionContext,
    MetricExecutionMode,
    MetricExecutionResult,
)

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies for metric processing."""

    SEQUENTIAL = "sequential"  # Execute one at a time
    PARALLEL = "parallel"  # Execute in parallel threads
    BATCH = "batch"  # Batch similar metrics
    PRIORITY = "priority"  # Priority-based scheduling
    ADAPTIVE = "adaptive"  # Adapt based on system load


class CacheStrategy(Enum):
    """Caching strategies."""

    NONE = "none"  # No caching
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage


@dataclass
class ExecutionPlan:
    """Plan for executing a set of metrics."""

    metrics: List[BaseMetric]
    contexts: List[MetricExecutionContext]
    strategy: ExecutionStrategy
    max_parallelism: int = 4
    timeout_seconds: float = 300.0
    priority: int = 0
    cache_results: bool = True

    def get_estimated_execution_time(self) -> float:
        """Estimate total execution time."""
        total_time = 0.0

        for metric in self.metrics:
            # Base estimate from complexity
            if metric.config.complexity == MetricComplexity.LOW:
                base_time = 1.0
            elif metric.config.complexity == MetricComplexity.MEDIUM:
                base_time = 5.0
            elif metric.config.complexity == MetricComplexity.HIGH:
                base_time = 30.0
            else:  # VERY_HIGH
                base_time = 120.0

            total_time += base_time * len(self.contexts)

        # Adjust for parallelism
        if self.strategy in [ExecutionStrategy.PARALLEL, ExecutionStrategy.BATCH]:
            total_time /= min(self.max_parallelism, len(self.metrics))

        return total_time


@dataclass
class CacheEntry:
    """Cache entry for metric results."""

    result: MetricExecutionResult
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False

        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def access(self):
        """Record access to this cache entry."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class MetricCache:
    """High-performance cache for metric results."""

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 3600.0,
        strategy: CacheStrategy = CacheStrategy.LRU,
    ):
        """
        Initialize metric cache.

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
            strategy: Cache eviction strategy
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _generate_key(
        self, metric_name: str, context: MetricExecutionContext, parameters: Dict[str, Any]
    ) -> str:
        """Generate cache key."""
        import hashlib

        key_data = {
            "metric": metric_name,
            "prediction": context.prediction,
            "reference": context.reference,
            "context": context.context,
            "params": sorted(parameters.items()) if parameters else None,
        }

        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self, metric_name: str, context: MetricExecutionContext, parameters: Dict[str, Any]
    ) -> Optional[MetricExecutionResult]:
        """Get cached result."""
        key = self._generate_key(metric_name, context, parameters)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self.misses += 1
                    return None

                # Record access and return result
                entry.access()
                self.hits += 1

                # Update result metadata to indicate cache hit
                cached_result = MetricExecutionResult(
                    metric_name=entry.result.metric_name,
                    value=entry.result.value,
                    confidence=entry.result.confidence,
                    explanation=entry.result.explanation,
                    execution_time_ms=0.1,  # Minimal cache lookup time
                    execution_mode=MetricExecutionMode.CACHED,
                    error=entry.result.error,
                    metadata={
                        **(entry.result.metadata or {}),
                        "cache_hit": True,
                        "cached_at": entry.created_at.isoformat(),
                        "access_count": entry.access_count,
                    },
                )

                return cached_result

            self.misses += 1
            return None

    def set(
        self,
        metric_name: str,
        context: MetricExecutionContext,
        parameters: Dict[str, Any],
        result: MetricExecutionResult,
        ttl: Optional[float] = None,
    ):
        """Cache result."""
        key = self._generate_key(metric_name, context, parameters)
        ttl = ttl or self.default_ttl

        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size:
                self._evict_entries()

            # Create cache entry
            entry = CacheEntry(
                result=result,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl_seconds=ttl,
            )

            self._cache[key] = entry

    def _evict_entries(self):
        """Evict entries based on strategy."""
        if not self._cache:
            return

        # Calculate how many entries to evict (25% of cache)
        evict_count = max(1, len(self._cache) // 4)

        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
        else:  # TTL or other strategies
            # Evict oldest entries
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].created_at)

        # Remove entries
        for i in range(evict_count):
            if i < len(sorted_entries):
                key_to_remove = sorted_entries[i][0]
                del self._cache[key_to_remove]
                self.evictions += 1

    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "strategy": self.strategy.value,
                "default_ttl": self.default_ttl,
            }


class ResourceMonitor:
    """Monitor system resources for adaptive execution."""

    def __init__(self):
        """Initialize resource monitor."""
        self.cpu_usage_history = deque(maxlen=60)  # Last 60 measurements
        self.memory_usage_history = deque(maxlen=60)
        self.last_update = time.time()

    def update_metrics(self):
        """Update resource metrics."""
        current_time = time.time()

        # Update every second
        if current_time - self.last_update >= 1.0:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent

                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_percent)

                self.last_update = current_time
            except Exception as e:
                logger.warning(f"Failed to update resource metrics: {e}")

    def get_cpu_usage(self) -> float:
        """Get average CPU usage."""
        self.update_metrics()
        return (
            sum(self.cpu_usage_history) / len(self.cpu_usage_history)
            if self.cpu_usage_history
            else 0
        )

    def get_memory_usage(self) -> float:
        """Get average memory usage."""
        self.update_metrics()
        return (
            sum(self.memory_usage_history) / len(self.memory_usage_history)
            if self.memory_usage_history
            else 0
        )

    def is_system_under_load(self) -> bool:
        """Check if system is under high load."""
        return self.get_cpu_usage() > 80 or self.get_memory_usage() > 85

    def get_recommended_parallelism(self, max_parallelism: int) -> int:
        """Get recommended level of parallelism based on system load."""
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()

        # Reduce parallelism under high load
        if cpu_usage > 80 or memory_usage > 85:
            return max(1, max_parallelism // 4)
        elif cpu_usage > 60 or memory_usage > 70:
            return max(1, max_parallelism // 2)
        else:
            return max_parallelism


class MetricExecutionEngine:
    """High-performance execution engine for metrics."""

    def __init__(
        self,
        max_workers: int = None,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        default_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    ):
        """
        Initialize execution engine.

        Args:
            max_workers: Maximum number of worker threads/processes
            cache_enabled: Enable result caching
            cache_size: Maximum cache size
            default_strategy: Default execution strategy
        """
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.default_strategy = default_strategy

        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(self.max_workers, multiprocessing.cpu_count() or 1)
        )

        # Caching
        self.cache_enabled = cache_enabled
        self.cache = MetricCache(max_size=cache_size) if cache_enabled else None

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()

        # Execution statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0

        # Batch processing
        self.pending_batches: Dict[str, List] = defaultdict(list)
        self.batch_timeout = 1.0  # seconds

        # Background tasks
        self._batch_processor_task = asyncio.create_task(self._process_batches())

    async def execute_metric(
        self,
        metric: BaseMetric,
        context: MetricExecutionContext,
        use_cache: bool = True,
        timeout: Optional[float] = None,
    ) -> MetricExecutionResult:
        """
        Execute a single metric with optimization.

        Args:
            metric: Metric to execute
            context: Execution context
            use_cache: Whether to use caching
            timeout: Execution timeout

        Returns:
            Execution result
        """
        start_time = time.time()
        self.total_executions += 1

        try:
            # Check cache first
            if use_cache and self.cache_enabled and self.cache:
                cached_result = self.cache.get(metric.config.name, context, metric.parameters)
                if cached_result:
                    return cached_result

            # Determine execution strategy
            strategy = self._choose_execution_strategy(metric)

            # Execute metric
            if strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential(metric, context, timeout)
            elif strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(metric, context, timeout)
            else:
                # Default to sequential for single metric
                result = await self._execute_sequential(metric, context, timeout)

            # Cache result if successful
            if use_cache and self.cache_enabled and self.cache and result.error is None:
                ttl = self._calculate_cache_ttl(metric, result)
                self.cache.set(metric.config.name, context, metric.parameters, result, ttl)

            self.successful_executions += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            return result

        except Exception as e:
            self.failed_executions += 1
            execution_time = (time.time() - start_time) * 1000

            return MetricExecutionResult(
                metric_name=metric.config.name,
                value=None,
                error=str(e),
                execution_time_ms=execution_time,
                execution_mode=context.execution_mode,
            )

    async def execute_plan(self, plan: ExecutionPlan) -> List[MetricExecutionResult]:
        """
        Execute an execution plan with optimization.

        Args:
            plan: Execution plan

        Returns:
            List of execution results
        """
        start_time = time.time()

        try:
            if plan.strategy == ExecutionStrategy.SEQUENTIAL:
                results = await self._execute_plan_sequential(plan)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_plan_parallel(plan)
            elif plan.strategy == ExecutionStrategy.BATCH:
                results = await self._execute_plan_batch(plan)
            elif plan.strategy == ExecutionStrategy.PRIORITY:
                results = await self._execute_plan_priority(plan)
            else:  # ADAPTIVE
                results = await self._execute_plan_adaptive(plan)

            execution_time = time.time() - start_time
            logger.info(f"Executed plan with {len(plan.metrics)} metrics in {execution_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Failed to execute plan: {e}")
            # Return error results for all metrics
            return [
                MetricExecutionResult(
                    metric_name=metric.config.name,
                    value=None,
                    error=str(e),
                    execution_time_ms=0,
                    execution_mode=MetricExecutionMode.LOCAL,
                )
                for metric in plan.metrics
            ]

    def _choose_execution_strategy(self, metric: BaseMetric) -> ExecutionStrategy:
        """Choose optimal execution strategy for a metric."""
        # Check system load
        if self.resource_monitor.is_system_under_load():
            return ExecutionStrategy.SEQUENTIAL

        # Choose based on complexity
        complexity = metric.config.complexity

        if complexity == MetricComplexity.LOW:
            return ExecutionStrategy.PARALLEL
        elif complexity == MetricComplexity.MEDIUM:
            return ExecutionStrategy.PARALLEL
        elif complexity == MetricComplexity.HIGH:
            return ExecutionStrategy.SEQUENTIAL
        else:  # VERY_HIGH
            return ExecutionStrategy.SEQUENTIAL

    async def _execute_sequential(
        self, metric: BaseMetric, context: MetricExecutionContext, timeout: Optional[float]
    ) -> MetricExecutionResult:
        """Execute metric sequentially."""
        timeout = timeout or metric.config.timeout_seconds

        try:
            result = await asyncio.wait_for(metric.compute(context), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return MetricExecutionResult(
                metric_name=metric.config.name,
                value=None,
                error="Execution timeout",
                execution_time_ms=timeout * 1000,
                execution_mode=context.execution_mode,
            )

    async def _execute_parallel(
        self, metric: BaseMetric, context: MetricExecutionContext, timeout: Optional[float]
    ) -> MetricExecutionResult:
        """Execute metric in parallel (for I/O bound operations)."""
        timeout = timeout or metric.config.timeout_seconds

        loop = asyncio.get_event_loop()

        try:
            # Run in thread pool for I/O bound operations
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool, lambda: asyncio.run(metric.compute(context))
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return MetricExecutionResult(
                metric_name=metric.config.name,
                value=None,
                error="Execution timeout",
                execution_time_ms=timeout * 1000,
                execution_mode=context.execution_mode,
            )

    async def _execute_plan_sequential(self, plan: ExecutionPlan) -> List[MetricExecutionResult]:
        """Execute plan sequentially."""
        results = []

        for metric in plan.metrics:
            for context in plan.contexts:
                result = await self.execute_metric(
                    metric, context, plan.cache_results, plan.timeout_seconds
                )
                results.append(result)

        return results

    async def _execute_plan_parallel(self, plan: ExecutionPlan) -> List[MetricExecutionResult]:
        """Execute plan in parallel."""
        # Adjust parallelism based on system load
        max_parallelism = self.resource_monitor.get_recommended_parallelism(plan.max_parallelism)

        # Create tasks
        tasks = []
        for metric in plan.metrics:
            for context in plan.contexts:
                task = self.execute_metric(
                    metric, context, plan.cache_results, plan.timeout_seconds
                )
                tasks.append(task)

        # Execute with limited concurrency
        semaphore = asyncio.Semaphore(max_parallelism)

        async def limited_execute(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_execute(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                metric_idx = i // len(plan.contexts)
                metric = (
                    plan.metrics[metric_idx] if metric_idx < len(plan.metrics) else plan.metrics[0]
                )

                processed_results.append(
                    MetricExecutionResult(
                        metric_name=metric.config.name,
                        value=None,
                        error=str(result),
                        execution_time_ms=0,
                        execution_mode=MetricExecutionMode.LOCAL,
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_plan_batch(self, plan: ExecutionPlan) -> List[MetricExecutionResult]:
        """Execute plan using batch processing."""
        # Group metrics by similarity (same type/complexity)
        batches = defaultdict(list)

        for metric in plan.metrics:
            batch_key = f"{metric.config.type.value}_{metric.config.complexity.value}"
            batches[batch_key].append(metric)

        # Execute batches
        all_results = []
        for batch_metrics in batches.values():
            batch_plan = ExecutionPlan(
                metrics=batch_metrics,
                contexts=plan.contexts,
                strategy=ExecutionStrategy.PARALLEL,
                max_parallelism=plan.max_parallelism,
                timeout_seconds=plan.timeout_seconds,
                cache_results=plan.cache_results,
            )

            batch_results = await self._execute_plan_parallel(batch_plan)
            all_results.extend(batch_results)

        return all_results

    async def _execute_plan_priority(self, plan: ExecutionPlan) -> List[MetricExecutionResult]:
        """Execute plan with priority ordering."""
        # Sort metrics by priority and complexity
        sorted_metrics = sorted(
            plan.metrics, key=lambda m: (plan.priority, m.config.complexity.value), reverse=True
        )

        priority_plan = ExecutionPlan(
            metrics=sorted_metrics,
            contexts=plan.contexts,
            strategy=ExecutionStrategy.PARALLEL,
            max_parallelism=plan.max_parallelism,
            timeout_seconds=plan.timeout_seconds,
            cache_results=plan.cache_results,
        )

        return await self._execute_plan_parallel(priority_plan)

    async def _execute_plan_adaptive(self, plan: ExecutionPlan) -> List[MetricExecutionResult]:
        """Execute plan with adaptive strategy selection."""
        # Choose strategy based on system state and plan characteristics
        system_load = self.resource_monitor.is_system_under_load()
        high_complexity_count = sum(
            1
            for m in plan.metrics
            if m.config.complexity in [MetricComplexity.HIGH, MetricComplexity.VERY_HIGH]
        )

        if system_load or high_complexity_count > len(plan.metrics) // 2:
            # Use sequential for high load or many complex metrics
            return await self._execute_plan_sequential(plan)
        elif len(plan.metrics) <= 5:
            # Use parallel for small number of metrics
            return await self._execute_plan_parallel(plan)
        else:
            # Use batch for larger sets
            return await self._execute_plan_batch(plan)

    def _calculate_cache_ttl(self, metric: BaseMetric, result: MetricExecutionResult) -> float:
        """Calculate appropriate cache TTL based on metric characteristics."""
        base_ttl = 3600.0  # 1 hour

        # Adjust based on complexity
        if metric.config.complexity == MetricComplexity.VERY_HIGH:
            base_ttl *= 4  # Cache expensive operations longer
        elif metric.config.complexity == MetricComplexity.HIGH:
            base_ttl *= 2
        elif metric.config.complexity == MetricComplexity.LOW:
            base_ttl *= 0.5  # Cache simple operations for shorter time

        # Adjust based on execution time
        if result.execution_time_ms and result.execution_time_ms > 10000:  # > 10 seconds
            base_ttl *= 2

        return base_ttl

    async def _process_batches(self):
        """Background task to process batched executions."""
        while True:
            try:
                await asyncio.sleep(self.batch_timeout)

                # Process pending batches
                if self.pending_batches:
                    for batch_key, batch_items in list(self.pending_batches.items()):
                        if (
                            len(batch_items) >= 5
                            or time.time() - batch_items[0]["timestamp"] > self.batch_timeout
                        ):
                            # Process batch
                            await self._execute_batch(batch_key, batch_items)
                            del self.pending_batches[batch_key]

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(1)

    async def _execute_batch(self, batch_key: str, batch_items: List[Any]):
        """Execute a batch of similar metrics."""
        # This is a placeholder for batch optimization logic
        logger.debug(f"Processing batch {batch_key} with {len(batch_items)} items")

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_execution_time = (
            self.total_execution_time / self.total_executions if self.total_executions > 0 else 0
        )

        success_rate = (
            self.successful_executions / self.total_executions * 100
            if self.total_executions > 0
            else 0
        )

        stats = {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "total_execution_time": self.total_execution_time,
            "max_workers": self.max_workers,
            "resource_usage": {
                "cpu_percent": self.resource_monitor.get_cpu_usage(),
                "memory_percent": self.resource_monitor.get_memory_usage(),
                "system_under_load": self.resource_monitor.is_system_under_load(),
            },
        }

        # Add cache statistics if available
        if self.cache:
            stats["cache"] = self.cache.get_statistics()

        return stats

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "_batch_processor_task"):
            self._batch_processor_task.cancel()

        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)


# Global execution engine instance
execution_engine: Optional[MetricExecutionEngine] = None


def get_execution_engine() -> MetricExecutionEngine:
    """Get global execution engine instance."""
    global execution_engine

    if execution_engine is None:
        execution_engine = MetricExecutionEngine()

    return execution_engine


def initialize_execution_engine(**kwargs) -> MetricExecutionEngine:
    """Initialize global execution engine with custom settings."""
    global execution_engine

    execution_engine = MetricExecutionEngine(**kwargs)
    return execution_engine
