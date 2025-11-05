"""
Local metric fallbacks for cost-effective evaluation.

Provides local alternatives to expensive metrics (e.g., LLM-based scoring)
using lightweight models, rule-based approaches, and statistical methods.
"""

import asyncio
import logging
import re
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from ulei.core.custom_metrics import (
    BaseMetric,
    MetricExecutionContext,
    MetricExecutionMode,
    MetricExecutionResult,
)

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies for expensive metrics."""

    DISABLED = "disabled"  # No fallback, fail if primary unavailable
    LOCAL_ONLY = "local_only"  # Only use local alternatives
    FALLBACK_ON_ERROR = "fallback_on_error"  # Try primary, fallback on error
    FALLBACK_ON_TIMEOUT = "fallback_on_timeout"  # Try primary, fallback on timeout
    COST_THRESHOLD = "cost_threshold"  # Use local if cost exceeds threshold
    HYBRID = "hybrid"  # Combine primary and local results


@dataclass
class FallbackConfig:
    """Configuration for metric fallbacks."""

    strategy: FallbackStrategy = FallbackStrategy.FALLBACK_ON_ERROR
    timeout_seconds: float = 30.0
    cost_threshold: Optional[float] = None
    confidence_threshold: float = 0.7
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_retries: int = 2


class LocalMetricCache:
    """Cache for local metric results."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _generate_key(self, context: MetricExecutionContext, params: Dict[str, Any]) -> str:
        """Generate cache key from context and parameters."""
        import hashlib

        key_data = {
            "prediction": context.prediction,
            "reference": context.reference,
            "context": context.context,
            "params": params,
        }

        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, context: MetricExecutionContext, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result."""
        key = self._generate_key(context, params)

        if key in self.cache:
            entry = self.cache[key]

            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["result"]
            else:
                # Remove expired entry
                del self.cache[key]

        return None

    def set(self, context: MetricExecutionContext, params: Dict[str, Any], result: Any):
        """Cache result."""
        # Check cache size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        key = self._generate_key(context, params)
        self.cache[key] = {"result": result, "timestamp": time.time()}

    def clear(self):
        """Clear cache."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {"size": len(self.cache), "max_size": self.max_size, "ttl_seconds": self.ttl_seconds}


class BaseLocalMetric(BaseMetric):
    """Base class for local metric implementations."""

    def __init__(self, config, fallback_config: Optional[FallbackConfig] = None, **kwargs):
        """
        Initialize local metric.

        Args:
            config: Metric configuration
            fallback_config: Fallback configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, **kwargs)
        self.fallback_config = fallback_config or FallbackConfig()
        self.cache = LocalMetricCache() if self.fallback_config.enable_caching else None

    async def compute(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Compute metric with fallback support."""
        start_time = time.time()

        # Check cache first
        if self.cache:
            cached_result = self.cache.get(context, self.parameters)
            if cached_result is not None:
                return MetricExecutionResult(
                    metric_name=self.config.name,
                    value=cached_result,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    execution_mode=MetricExecutionMode.CACHED,
                )

        try:
            # Compute local metric
            result = await self._compute_local(context)

            # Cache result
            if self.cache:
                self.cache.set(context, self.parameters, result.value)

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return MetricExecutionResult(
                metric_name=self.config.name,
                value=None,
                error=str(e),
                execution_time_ms=execution_time,
                execution_mode=MetricExecutionMode.LOCAL,
            )

    @abstractmethod
    async def _compute_local(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Compute local metric implementation."""
        pass


class LocalSemanticSimilarity(BaseLocalMetric):
    """Local semantic similarity using sentence embeddings."""

    def __init__(self, config, **kwargs):
        """Initialize local semantic similarity."""
        super().__init__(config, **kwargs)

        # Initialize sentence transformer model
        model_name = self.parameters.get("model_name", "all-MiniLM-L6-v2")
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except ImportError:
            logger.warning("sentence-transformers not available, using TF-IDF fallback")
            self.model = None
            self.vectorizer = TfidfVectorizer()

    async def _compute_local(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Compute semantic similarity locally."""
        start_time = time.time()

        if not context.reference:
            raise ValueError("Reference text required for semantic similarity")

        if self.model:
            # Use sentence transformer
            embeddings = self.model.encode([context.prediction, context.reference])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        else:
            # Use TF-IDF fallback
            tfidf_matrix = self.vectorizer.fit_transform([context.prediction, context.reference])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        execution_time = (time.time() - start_time) * 1000

        return MetricExecutionResult(
            metric_name=self.config.name,
            value=float(similarity),
            execution_time_ms=execution_time,
            execution_mode=MetricExecutionMode.LOCAL,
            confidence=0.8,  # Local models have moderate confidence
        )


class LocalFactualAccuracy(BaseLocalMetric):
    """Local factual accuracy using rule-based and statistical methods."""

    def __init__(self, config, **kwargs):
        """Initialize local factual accuracy."""
        super().__init__(config, **kwargs)

        # Load NER model for entity extraction
        try:
            self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        except Exception:
            logger.warning("NER pipeline not available")
            self.ner_pipeline = None

    async def _compute_local(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Compute factual accuracy using local methods."""
        start_time = time.time()

        if not context.reference:
            raise ValueError("Reference text required for factual accuracy")

        # Extract facts using multiple methods
        pred_facts = self._extract_facts(context.prediction)
        ref_facts = self._extract_facts(context.reference)

        # Calculate accuracy
        if not ref_facts:
            accuracy = 1.0  # No facts to verify
        else:
            correct_facts = len(pred_facts.intersection(ref_facts))
            total_facts = len(ref_facts)
            accuracy = correct_facts / total_facts

        execution_time = (time.time() - start_time) * 1000

        return MetricExecutionResult(
            metric_name=self.config.name,
            value=accuracy,
            execution_time_ms=execution_time,
            execution_mode=MetricExecutionMode.LOCAL,
            confidence=0.6,  # Rule-based methods have lower confidence
            metadata={
                "pred_facts": list(pred_facts),
                "ref_facts": list(ref_facts),
                "correct_facts": len(pred_facts.intersection(ref_facts)),
            },
        )

    def _extract_facts(self, text: str) -> Set[str]:
        """Extract facts from text using multiple methods."""
        facts = set()

        # Named entity extraction
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                for entity in entities:
                    facts.add(f"{entity['entity_group']}:{entity['word']}")
            except Exception:
                pass

        # Number extraction
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        for num in numbers:
            facts.add(f"NUMBER:{num}")

        # Date extraction
        dates = re.findall(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", text)
        for date in dates:
            facts.add(f"DATE:{date}")

        # Proper noun extraction (simple heuristic)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for noun in proper_nouns:
            if len(noun.split()) <= 3:  # Avoid long phrases
                facts.add(f"PROPER_NOUN:{noun}")

        return facts


class LocalHallucinationDetection(BaseLocalMetric):
    """Local hallucination detection using statistical and rule-based methods."""

    def __init__(self, config, **kwargs):
        """Initialize local hallucination detection."""
        super().__init__(config, **kwargs)

        # Common hallucination patterns
        self.hallucination_patterns = [
            r"\b(?:I think|I believe|I guess|probably|maybe|perhaps)\b",
            r"\b(?:fictional|imaginary|made-up|invented)\b",
            r"\b(?:according to|source:|citation:)\b",  # Fabricated citations
            r"\b\d{4}\b.*(?:published|released|announced)",  # Specific dates
        ]

    async def _compute_local(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Detect hallucinations using local methods."""
        start_time = time.time()

        hallucination_score = 0.0
        detected_patterns = []

        # Pattern-based detection
        for pattern in self.hallucination_patterns:
            matches = re.findall(pattern, context.prediction, re.IGNORECASE)
            if matches:
                hallucination_score += 0.2
                detected_patterns.extend(matches)

        # Statistical anomaly detection
        if context.reference:
            # Check for information not in reference
            pred_words = set(context.prediction.lower().split())
            ref_words = set(context.reference.lower().split())

            # Calculate ratio of novel information
            novel_words = pred_words - ref_words
            if pred_words:
                novel_ratio = len(novel_words) / len(pred_words)
                if novel_ratio > 0.5:  # High novel content
                    hallucination_score += 0.3

        # Confidence indicators (inverse patterns)
        confidence_patterns = [
            r"\b(?:according to the document|as stated|as mentioned)\b",
            r"\b(?:the text says|the passage indicates)\b",
        ]

        confidence_boost = 0.0
        for pattern in confidence_patterns:
            if re.search(pattern, context.prediction, re.IGNORECASE):
                confidence_boost += 0.1

        final_score = max(0.0, min(1.0, hallucination_score - confidence_boost))

        execution_time = (time.time() - start_time) * 1000

        return MetricExecutionResult(
            metric_name=self.config.name,
            value=final_score,
            execution_time_ms=execution_time,
            execution_mode=MetricExecutionMode.LOCAL,
            confidence=0.7,
            metadata={
                "detected_patterns": detected_patterns,
                "novel_ratio": novel_ratio if context.reference else None,
                "confidence_boost": confidence_boost,
            },
        )


class LocalBiasDetection(BaseLocalMetric):
    """Local bias detection using lexical analysis."""

    def __init__(self, config, **kwargs):
        """Initialize local bias detection."""
        super().__init__(config, **kwargs)

        # Bias-indicating terms (simplified examples)
        self.bias_terms = {
            "gender": ["he", "she", "his", "her", "man", "woman", "male", "female"],
            "racial": ["white", "black", "asian", "latino", "hispanic", "native"],
            "age": ["young", "old", "elderly", "teenager", "millennial", "boomer"],
            "socioeconomic": ["rich", "poor", "wealthy", "homeless", "privileged"],
        }

        # Negative sentiment terms
        self.negative_terms = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "dangerous",
            "threatening",
            "suspicious",
            "lazy",
            "incompetent",
            "aggressive",
            "violent",
        ]

    async def _compute_local(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Detect potential bias using lexical analysis."""
        start_time = time.time()

        text_lower = context.prediction.lower()
        bias_scores = {}
        detected_biases = []

        # Check for bias in each category
        for category, terms in self.bias_terms.items():
            category_score = 0.0
            found_terms = []

            for term in terms:
                if term in text_lower:
                    found_terms.append(term)

                    # Check if bias term is associated with negative sentiment
                    term_index = text_lower.index(term)
                    surrounding_text = text_lower[max(0, term_index - 50) : term_index + 50]

                    for neg_term in self.negative_terms:
                        if neg_term in surrounding_text:
                            category_score += 0.3
                            detected_biases.append(
                                {
                                    "category": category,
                                    "term": term,
                                    "negative_association": neg_term,
                                    "context": surrounding_text.strip(),
                                }
                            )
                            break

            if found_terms:
                bias_scores[category] = category_score

        # Calculate overall bias score
        overall_score = min(1.0, sum(bias_scores.values()) / len(self.bias_terms))

        execution_time = (time.time() - start_time) * 1000

        return MetricExecutionResult(
            metric_name=self.config.name,
            value=overall_score,
            execution_time_ms=execution_time,
            execution_mode=MetricExecutionMode.LOCAL,
            confidence=0.5,  # Lexical analysis has limited confidence
            metadata={"category_scores": bias_scores, "detected_biases": detected_biases},
        )


class LocalRelevanceScore(BaseLocalMetric):
    """Local relevance scoring using keyword matching and similarity."""

    def __init__(self, config, **kwargs):
        """Initialize local relevance scoring."""
        super().__init__(config, **kwargs)
        self.vectorizer = TfidfVectorizer(stop_words="english")

    async def _compute_local(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Compute relevance using local methods."""
        start_time = time.time()

        if not context.context:
            raise ValueError("Context required for relevance scoring")

        # TF-IDF similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform([context.prediction, context.context])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception:
            tfidf_similarity = 0.0

        # Keyword overlap
        pred_words = set(context.prediction.lower().split())
        context_words = set(context.context.lower().split())

        if context_words:
            keyword_overlap = len(pred_words.intersection(context_words)) / len(context_words)
        else:
            keyword_overlap = 0.0

        # Combined relevance score
        relevance_score = (tfidf_similarity * 0.7) + (keyword_overlap * 0.3)

        execution_time = (time.time() - start_time) * 1000

        return MetricExecutionResult(
            metric_name=self.config.name,
            value=relevance_score,
            execution_time_ms=execution_time,
            execution_mode=MetricExecutionMode.LOCAL,
            confidence=0.8,
            metadata={
                "tfidf_similarity": tfidf_similarity,
                "keyword_overlap": keyword_overlap,
                "shared_keywords": len(pred_words.intersection(context_words)),
            },
        )


class FallbackMetricWrapper(BaseMetric):
    """Wrapper that provides fallback capabilities for any metric."""

    def __init__(
        self,
        primary_metric: BaseMetric,
        fallback_metric: BaseLocalMetric,
        fallback_config: FallbackConfig,
    ):
        """
        Initialize fallback wrapper.

        Args:
            primary_metric: Primary (possibly expensive) metric
            fallback_metric: Local fallback metric
            fallback_config: Fallback configuration
        """
        super().__init__(primary_metric.config)
        self.primary_metric = primary_metric
        self.fallback_metric = fallback_metric
        self.fallback_config = fallback_config

    async def compute(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """Compute metric with fallback strategy."""
        strategy = self.fallback_config.strategy

        if strategy == FallbackStrategy.LOCAL_ONLY:
            return await self.fallback_metric.compute(context)

        elif strategy == FallbackStrategy.DISABLED:
            return await self.primary_metric.compute(context)

        elif strategy == FallbackStrategy.FALLBACK_ON_ERROR:
            try:
                return await self.primary_metric.compute(context)
            except Exception as e:
                logger.warning(f"Primary metric failed, using fallback: {e}")
                fallback_result = await self.fallback_metric.compute(context)
                fallback_result.metadata = fallback_result.metadata or {}
                fallback_result.metadata["fallback_reason"] = "primary_error"
                fallback_result.metadata["primary_error"] = str(e)
                return fallback_result

        elif strategy == FallbackStrategy.FALLBACK_ON_TIMEOUT:
            try:
                return await asyncio.wait_for(
                    self.primary_metric.compute(context),
                    timeout=self.fallback_config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("Primary metric timed out, using fallback")
                fallback_result = await self.fallback_metric.compute(context)
                fallback_result.metadata = fallback_result.metadata or {}
                fallback_result.metadata["fallback_reason"] = "timeout"
                return fallback_result

        elif strategy == FallbackStrategy.HYBRID:
            # Run both metrics and combine results
            primary_task = asyncio.create_task(self.primary_metric.compute(context))
            fallback_task = asyncio.create_task(self.fallback_metric.compute(context))

            try:
                primary_result = await asyncio.wait_for(
                    primary_task, timeout=self.fallback_config.timeout_seconds
                )
                fallback_result = await fallback_task

                # Combine results (weighted average)
                if (
                    primary_result.value is not None
                    and fallback_result.value is not None
                    and isinstance(primary_result.value, (int, float))
                    and isinstance(fallback_result.value, (int, float))
                ):
                    combined_value = (primary_result.value * 0.7) + (fallback_result.value * 0.3)

                    return MetricExecutionResult(
                        metric_name=self.config.name,
                        value=combined_value,
                        execution_time_ms=max(
                            primary_result.execution_time_ms or 0,
                            fallback_result.execution_time_ms or 0,
                        ),
                        execution_mode=MetricExecutionMode.HYBRID,
                        confidence=(primary_result.confidence or 0.8)
                        * 0.9,  # Slightly lower confidence for hybrid
                        metadata={
                            "primary_result": primary_result.value,
                            "fallback_result": fallback_result.value,
                            "combination_strategy": "weighted_average",
                        },
                    )
                else:
                    # Return primary result if available
                    return primary_result if primary_result.value is not None else fallback_result

            except asyncio.TimeoutError:
                # Primary timed out, cancel and return fallback
                primary_task.cancel()
                fallback_result = await fallback_task
                fallback_result.metadata = fallback_result.metadata or {}
                fallback_result.metadata["fallback_reason"] = "primary_timeout"
                return fallback_result

        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")


# Registry of local metric implementations
LOCAL_METRIC_IMPLEMENTATIONS = {
    "semantic_similarity": LocalSemanticSimilarity,
    "factual_accuracy": LocalFactualAccuracy,
    "hallucination_detection": LocalHallucinationDetection,
    "bias_detection": LocalBiasDetection,
    "relevance": LocalRelevanceScore,
}


def create_fallback_metric(
    primary_metric: BaseMetric, fallback_type: str, fallback_config: Optional[FallbackConfig] = None
) -> FallbackMetricWrapper:
    """
    Create a fallback-enabled metric.

    Args:
        primary_metric: Primary metric implementation
        fallback_type: Type of local fallback metric
        fallback_config: Fallback configuration

    Returns:
        Fallback-enabled metric wrapper
    """
    if fallback_type not in LOCAL_METRIC_IMPLEMENTATIONS:
        raise ValueError(f"Unknown fallback type: {fallback_type}")

    fallback_class = LOCAL_METRIC_IMPLEMENTATIONS[fallback_type]
    fallback_metric = fallback_class(primary_metric.config)

    config = fallback_config or FallbackConfig()

    return FallbackMetricWrapper(primary_metric, fallback_metric, config)
