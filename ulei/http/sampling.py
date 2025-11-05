"""
Production data sampling and traffic control for online shadow evaluation.

Provides intelligent sampling strategies, rate limiting, and traffic shaping
to manage evaluation load and costs while maintaining statistical significance.
"""

import hashlib
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Sampling strategy types."""

    RANDOM = "random"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    HASH_BASED = "hash_based"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


class TrafficSource(Enum):
    """Traffic source types for differentiated handling."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"


@dataclass
class SamplingConfig:
    """Configuration for sampling strategy."""

    strategy: SamplingStrategy
    rate: float  # 0.0 to 1.0
    max_samples_per_minute: Optional[int] = None
    max_samples_per_hour: Optional[int] = None
    max_samples_per_day: Optional[int] = None

    # Stratified sampling parameters
    strata_field: Optional[str] = None  # Field to stratify on
    strata_rates: Optional[Dict[str, float]] = None

    # Hash-based sampling parameters
    hash_field: Optional[str] = None  # Field to hash for consistent sampling
    hash_salt: Optional[str] = None

    # Adaptive sampling parameters
    min_rate: Optional[float] = None
    max_rate: Optional[float] = None
    adapt_interval_minutes: int = 5

    # Priority-based sampling parameters
    priority_field: Optional[str] = None
    priority_rates: Optional[Dict[str, float]] = None


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_second: Optional[float] = None
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None

    # Burst handling
    burst_size: Optional[int] = None
    burst_rate: Optional[float] = None

    # Traffic source differentiation
    source_limits: Optional[Dict[TrafficSource, Dict[str, Union[int, float]]]] = None


@dataclass
class SamplingDecision:
    """Result of sampling decision."""

    should_sample: bool
    reason: str
    sampling_rate: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrafficStats:
    """Traffic statistics for monitoring."""

    total_requests: int = 0
    sampled_requests: int = 0
    rejected_requests: int = 0
    rate_limited_requests: int = 0

    last_reset: datetime = field(default_factory=datetime.utcnow)
    window_start: datetime = field(default_factory=datetime.utcnow)

    # Per-source stats
    source_stats: Dict[TrafficSource, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second refill rate
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        now = time.time()
        elapsed = now - self.last_refill

        # Refill tokens
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def available_tokens(self) -> int:
        """Get number of available tokens."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        return min(self.capacity, self.tokens + new_tokens)


class DataSampler:
    """Production data sampling and traffic control."""

    def __init__(
        self, sampling_config: SamplingConfig, rate_limit_config: Optional[RateLimitConfig] = None
    ):
        """
        Initialize data sampler.

        Args:
            sampling_config: Sampling strategy configuration
            rate_limit_config: Rate limiting configuration
        """
        self.sampling_config = sampling_config
        self.rate_limit_config = rate_limit_config or RateLimitConfig()

        # Sampling state
        self.sample_count = 0
        self.total_count = 0
        self.last_sample_time = time.time()

        # Systematic sampling state
        self.systematic_counter = 0
        self.systematic_interval = (
            int(1.0 / sampling_config.rate) if sampling_config.rate > 0 else 1
        )

        # Stratified sampling state
        self.strata_counts: Dict[str, int] = defaultdict(int)
        self.strata_samples: Dict[str, int] = defaultdict(int)

        # Adaptive sampling state
        self.current_adaptive_rate = sampling_config.rate
        self.last_adaptation = time.time()
        self.recent_load_metrics = deque(maxlen=100)

        # Rate limiting
        self.rate_limiters: Dict[str, TokenBucket] = {}
        self._setup_rate_limiters()

        # Statistics
        self.stats = TrafficStats()
        self.hourly_samples = deque(maxlen=60)  # Track samples per minute for last hour
        self.daily_samples = deque(maxlen=24)  # Track samples per hour for last day

    def _setup_rate_limiters(self):
        """Setup rate limiting token buckets."""
        config = self.rate_limit_config

        if config.requests_per_second:
            self.rate_limiters["per_second"] = TokenBucket(
                capacity=int(config.requests_per_second * 2),  # 2x burst capacity
                refill_rate=config.requests_per_second,
            )

        if config.requests_per_minute:
            self.rate_limiters["per_minute"] = TokenBucket(
                capacity=config.requests_per_minute, refill_rate=config.requests_per_minute / 60.0
            )

        if config.requests_per_hour:
            self.rate_limiters["per_hour"] = TokenBucket(
                capacity=config.requests_per_hour, refill_rate=config.requests_per_hour / 3600.0
            )

        # Setup source-specific rate limiters
        if config.source_limits:
            for source, limits in config.source_limits.items():
                source_key = f"source_{source.value}"

                if "requests_per_second" in limits:
                    self.rate_limiters[f"{source_key}_per_second"] = TokenBucket(
                        capacity=int(limits["requests_per_second"] * 2),
                        refill_rate=limits["requests_per_second"],
                    )

                if "requests_per_minute" in limits:
                    self.rate_limiters[f"{source_key}_per_minute"] = TokenBucket(
                        capacity=limits["requests_per_minute"],
                        refill_rate=limits["requests_per_minute"] / 60.0,
                    )

    async def should_sample(
        self, data: Dict[str, Any], source: TrafficSource = TrafficSource.PRODUCTION
    ) -> SamplingDecision:
        """
        Determine if data should be sampled.

        Args:
            data: Data to potentially sample
            source: Traffic source

        Returns:
            Sampling decision
        """
        self.total_count += 1
        self.stats.total_requests += 1
        self.stats.source_stats[source]["total"] += 1

        # Check rate limits first
        rate_limit_result = self._check_rate_limits(source)
        if not rate_limit_result:
            self.stats.rate_limited_requests += 1
            self.stats.source_stats[source]["rate_limited"] += 1
            return SamplingDecision(should_sample=False, reason="Rate limited", sampling_rate=0.0)

        # Check sampling limits
        if not self._check_sampling_limits():
            self.stats.rejected_requests += 1
            self.stats.source_stats[source]["rejected"] += 1
            return SamplingDecision(
                should_sample=False, reason="Sampling quota exceeded", sampling_rate=0.0
            )

        # Apply sampling strategy
        decision = await self._apply_sampling_strategy(data, source)

        if decision.should_sample:
            self.sample_count += 1
            self.stats.sampled_requests += 1
            self.stats.source_stats[source]["sampled"] += 1
            self._update_sampling_tracking()
        else:
            self.stats.rejected_requests += 1
            self.stats.source_stats[source]["rejected"] += 1

        return decision

    def _check_rate_limits(self, source: TrafficSource) -> bool:
        """Check if request is within rate limits."""
        # Check global rate limits
        for limiter in self.rate_limiters.values():
            if not limiter.consume():
                return False

        # Check source-specific rate limits
        source_key = f"source_{source.value}"
        for name, limiter in self.rate_limiters.items():
            if name.startswith(source_key) and not limiter.consume():
                return False

        return True

    def _check_sampling_limits(self) -> bool:
        """Check if sampling is within configured limits."""
        config = self.sampling_config
        now = time.time()

        # Check per-minute limit
        if config.max_samples_per_minute:
            minute_samples = sum(1 for timestamp, _ in self.hourly_samples if now - timestamp < 60)
            if minute_samples >= config.max_samples_per_minute:
                return False

        # Check per-hour limit
        if config.max_samples_per_hour:
            hour_samples = sum(
                count for timestamp, count in self.hourly_samples if now - timestamp < 3600
            )
            if hour_samples >= config.max_samples_per_hour:
                return False

        # Check per-day limit
        if config.max_samples_per_day:
            day_samples = sum(
                count for timestamp, count in self.daily_samples if now - timestamp < 86400
            )
            if day_samples >= config.max_samples_per_day:
                return False

        return True

    async def _apply_sampling_strategy(
        self, data: Dict[str, Any], source: TrafficSource
    ) -> SamplingDecision:
        """Apply configured sampling strategy."""
        strategy = self.sampling_config.strategy

        if strategy == SamplingStrategy.RANDOM:
            return self._random_sampling()
        elif strategy == SamplingStrategy.SYSTEMATIC:
            return self._systematic_sampling()
        elif strategy == SamplingStrategy.STRATIFIED:
            return self._stratified_sampling(data)
        elif strategy == SamplingStrategy.HASH_BASED:
            return self._hash_based_sampling(data)
        elif strategy == SamplingStrategy.ADAPTIVE:
            return await self._adaptive_sampling(data, source)
        elif strategy == SamplingStrategy.PRIORITY_BASED:
            return self._priority_based_sampling(data)
        else:
            return SamplingDecision(
                should_sample=False,
                reason=f"Unknown sampling strategy: {strategy}",
                sampling_rate=0.0,
            )

    def _random_sampling(self) -> SamplingDecision:
        """Random sampling strategy."""
        should_sample = random.random() < self.sampling_config.rate
        return SamplingDecision(
            should_sample=should_sample,
            reason="Random sampling",
            sampling_rate=self.sampling_config.rate,
        )

    def _systematic_sampling(self) -> SamplingDecision:
        """Systematic sampling strategy."""
        self.systematic_counter += 1
        should_sample = (self.systematic_counter % self.systematic_interval) == 0

        return SamplingDecision(
            should_sample=should_sample,
            reason=f"Systematic sampling (interval: {self.systematic_interval})",
            sampling_rate=self.sampling_config.rate,
        )

    def _stratified_sampling(self, data: Dict[str, Any]) -> SamplingDecision:
        """Stratified sampling strategy."""
        strata_field = self.sampling_config.strata_field
        strata_rates = self.sampling_config.strata_rates or {}

        if not strata_field or strata_field not in data:
            # Fall back to random sampling
            return self._random_sampling()

        stratum = str(data[strata_field])
        self.strata_counts[stratum] += 1

        # Get rate for this stratum
        rate = strata_rates.get(stratum, self.sampling_config.rate)

        should_sample = random.random() < rate

        if should_sample:
            self.strata_samples[stratum] += 1

        return SamplingDecision(
            should_sample=should_sample,
            reason=f"Stratified sampling (stratum: {stratum}, rate: {rate})",
            sampling_rate=rate,
            metadata={"stratum": stratum},
        )

    def _hash_based_sampling(self, data: Dict[str, Any]) -> SamplingDecision:
        """Hash-based consistent sampling."""
        hash_field = self.sampling_config.hash_field
        hash_salt = self.sampling_config.hash_salt or ""

        if not hash_field or hash_field not in data:
            # Fall back to random sampling
            return self._random_sampling()

        # Create hash of field value + salt
        hash_input = f"{data[hash_field]}{hash_salt}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()

        # Convert to number between 0 and 1
        hash_number = int(hash_value[:8], 16) / (16**8)

        should_sample = hash_number < self.sampling_config.rate

        return SamplingDecision(
            should_sample=should_sample,
            reason=f"Hash-based sampling (field: {hash_field})",
            sampling_rate=self.sampling_config.rate,
            metadata={"hash_value": hash_number},
        )

    async def _adaptive_sampling(
        self, data: Dict[str, Any], source: TrafficSource
    ) -> SamplingDecision:
        """Adaptive sampling based on system load."""
        now = time.time()

        # Check if it's time to adapt
        if now - self.last_adaptation > (self.sampling_config.adapt_interval_minutes * 60):
            await self._adapt_sampling_rate()
            self.last_adaptation = now

        should_sample = random.random() < self.current_adaptive_rate

        return SamplingDecision(
            should_sample=should_sample,
            reason=f"Adaptive sampling (current rate: {self.current_adaptive_rate:.3f})",
            sampling_rate=self.current_adaptive_rate,
        )

    def _priority_based_sampling(self, data: Dict[str, Any]) -> SamplingDecision:
        """Priority-based sampling."""
        priority_field = self.sampling_config.priority_field
        priority_rates = self.sampling_config.priority_rates or {}

        if not priority_field or priority_field not in data:
            # Fall back to random sampling
            return self._random_sampling()

        priority = str(data[priority_field])
        rate = priority_rates.get(priority, self.sampling_config.rate)

        should_sample = random.random() < rate

        return SamplingDecision(
            should_sample=should_sample,
            reason=f"Priority-based sampling (priority: {priority}, rate: {rate})",
            sampling_rate=rate,
            metadata={"priority": priority},
        )

    async def _adapt_sampling_rate(self):
        """Adapt sampling rate based on system metrics."""
        # This could be enhanced to consider:
        # - System CPU/memory usage
        # - Queue depth
        # - Processing latency
        # - Error rates
        # - Cost metrics

        # Simple example: adjust based on recent load
        if len(self.recent_load_metrics) < 10:
            return

        avg_load = sum(self.recent_load_metrics) / len(self.recent_load_metrics)

        min_rate = self.sampling_config.min_rate or 0.01
        max_rate = self.sampling_config.max_rate or 1.0

        if avg_load > 0.8:  # High load
            self.current_adaptive_rate = max(min_rate, self.current_adaptive_rate * 0.8)
        elif avg_load < 0.3:  # Low load
            self.current_adaptive_rate = min(max_rate, self.current_adaptive_rate * 1.2)

        logger.info(
            f"Adapted sampling rate to {self.current_adaptive_rate:.3f} (load: {avg_load:.2f})"
        )

    def _update_sampling_tracking(self):
        """Update sampling tracking for limits."""
        now = time.time()

        # Update per-minute tracking
        self.hourly_samples.append((now, 1))

        # Update per-hour tracking (aggregate per-minute samples)
        if len(self.daily_samples) == 0 or now - self.daily_samples[-1][0] > 3600:
            hour_count = sum(
                count for timestamp, count in self.hourly_samples if now - timestamp < 3600
            )
            self.daily_samples.append((now, hour_count))

    def report_load_metric(self, load: float):
        """Report system load metric for adaptive sampling."""
        self.recent_load_metrics.append(load)

    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get current sampling statistics."""
        now = time.time()
        uptime = now - self.stats.window_start.timestamp()

        sampling_rate = self.sample_count / self.total_count if self.total_count > 0 else 0

        return {
            "total_requests": self.total_count,
            "sampled_requests": self.sample_count,
            "rejected_requests": self.stats.rejected_requests,
            "rate_limited_requests": self.stats.rate_limited_requests,
            "effective_sampling_rate": sampling_rate,
            "configured_sampling_rate": self.sampling_config.rate,
            "current_adaptive_rate": getattr(self, "current_adaptive_rate", None),
            "uptime_seconds": uptime,
            "requests_per_second": self.total_count / uptime if uptime > 0 else 0,
            "samples_per_second": self.sample_count / uptime if uptime > 0 else 0,
            "strategy": self.sampling_config.strategy.value,
            "rate_limiter_status": {
                name: {"available_tokens": limiter.available_tokens(), "capacity": limiter.capacity}
                for name, limiter in self.rate_limiters.items()
            },
            "source_stats": dict(self.stats.source_stats),
            "strata_stats": {
                "counts": dict(self.strata_counts),
                "samples": dict(self.strata_samples),
            }
            if hasattr(self, "strata_counts")
            else None,
        }

    def reset_statistics(self):
        """Reset sampling statistics."""
        self.sample_count = 0
        self.total_count = 0
        self.stats = TrafficStats()
        self.hourly_samples.clear()
        self.daily_samples.clear()

        # Reset strategy-specific state
        self.systematic_counter = 0
        self.strata_counts.clear()
        self.strata_samples.clear()

        logger.info("Sampling statistics reset")


class SamplingManager:
    """Manager for multiple sampling configurations and strategies."""

    def __init__(self):
        """Initialize sampling manager."""
        self.samplers: Dict[str, DataSampler] = {}
        self.default_sampler: Optional[DataSampler] = None

    def add_sampler(self, name: str, sampler: DataSampler, is_default: bool = False):
        """Add a data sampler."""
        self.samplers[name] = sampler
        if is_default:
            self.default_sampler = sampler

    def get_sampler(self, name: Optional[str] = None) -> Optional[DataSampler]:
        """Get sampler by name or default."""
        if name:
            return self.samplers.get(name)
        return self.default_sampler

    async def should_sample(
        self,
        data: Dict[str, Any],
        sampler_name: Optional[str] = None,
        source: TrafficSource = TrafficSource.PRODUCTION,
    ) -> Optional[SamplingDecision]:
        """Check if data should be sampled using specified or default sampler."""
        sampler = self.get_sampler(sampler_name)
        if sampler:
            return await sampler.should_sample(data, source)
        return None

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all samplers."""
        return {name: sampler.get_sampling_statistics() for name, sampler in self.samplers.items()}
