"""
AI Model Performance Optimizer for Project QuickNav

Provides comprehensive AI model optimization including:
- Model inference caching and optimization
- Batch processing for multiple requests
- Model loading and memory management
- Response streaming optimization
- Prediction result caching
- Context window optimization
- Model selection based on performance
"""

import asyncio
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict
import json
import hashlib
import statistics
from pathlib import Path
import queue
import functools

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    throughput_per_second: float
    cache_hit_rate: float
    avg_tokens_per_request: int
    memory_usage_mb: float
    cost_per_request: float
    last_updated: datetime


@dataclass
class InferenceRequest:
    """AI inference request with metadata."""
    request_id: str
    model_name: str
    prompt: str
    parameters: Dict[str, Any]
    context: Optional[str]
    priority: int  # 1=high, 2=medium, 3=low
    submitted_at: datetime
    cache_key: Optional[str] = None
    batch_compatible: bool = True


@dataclass
class InferenceResult:
    """AI inference result with performance data."""
    request_id: str
    response: str
    model_name: str
    latency_ms: float
    tokens_generated: int
    cache_hit: bool
    cost_estimate: float
    metadata: Dict[str, Any]
    completed_at: datetime


class ModelCache:
    """Intelligent caching for AI model responses."""

    def __init__(self, max_size_mb: int = 512):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()

        # Semantic similarity threshold for cache matching
        self.similarity_threshold = 0.85

    def get(self, key: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached response with semantic similarity matching."""
        with self.lock:
            # Exact match first
            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry):
                    self.cache.move_to_end(key)  # LRU update
                    entry['access_count'] += 1
                    entry['last_accessed'] = datetime.utcnow()
                    self.hits += 1
                    return entry['response']
                else:
                    self._remove_entry(key)

            # Semantic similarity search for related prompts
            if context:
                similar_entry = self._find_similar_entry(key, context)
                if similar_entry:
                    self.hits += 1
                    return similar_entry

            self.misses += 1
            return None

    def set(self, key: str, response: Dict[str, Any], ttl_seconds: int = 3600,
           context: Optional[str] = None):
        """Cache AI response with metadata."""
        with self.lock:
            # Calculate size
            entry_size = self._estimate_size(response)

            # Make space if needed
            while (self.current_size + entry_size > self.max_size_bytes and
                   len(self.cache) > 0):
                self._evict_oldest()

            # Create cache entry
            entry = {
                'response': response,
                'created_at': datetime.utcnow(),
                'last_accessed': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(seconds=ttl_seconds),
                'access_count': 0,
                'size_bytes': entry_size,
                'context': context
            }

            self.cache[key] = entry
            self.current_size += entry_size

    def _find_similar_entry(self, key: str, context: str) -> Optional[Dict[str, Any]]:
        """Find semantically similar cached entry."""
        # Simplified similarity matching
        # In production, use proper embedding-based similarity

        best_match = None
        best_score = 0

        for cached_key, entry in self.cache.items():
            if entry.get('context'):
                # Simple token overlap similarity
                similarity = self._calculate_similarity(context, entry['context'])
                if similarity > self.similarity_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = entry['response']

        return best_match

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simplified Jaccard similarity
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() > entry['expires_at']

    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry['size_bytes']

    def _evict_oldest(self):
        """Evict least recently used entry."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(json.dumps(obj, default=str).encode())
        except:
            return 1024  # Default estimate

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'hit_rate': hit_rate,
                'total_entries': len(self.cache),
                'current_size_mb': self.current_size / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'total_requests': total_requests
            }

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0


class BatchProcessor:
    """Batch processing for AI inference requests."""

    def __init__(self, max_batch_size: int = 8, max_wait_time_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.request_queue = queue.Queue()
        self.result_futures = {}
        self.processing_active = False
        self.processor_thread = None

    def start_processing(self, batch_handler: Callable):
        """Start batch processing thread."""
        if self.processing_active:
            return

        self.processing_active = True
        self.batch_handler = batch_handler
        self.processor_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processor_thread.start()
        logger.info("Batch processor started")

    def stop_processing(self):
        """Stop batch processing."""
        self.processing_active = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        logger.info("Batch processor stopped")

    async def submit_request(self, request: InferenceRequest) -> InferenceResult:
        """Submit request for batch processing."""
        if not request.batch_compatible:
            # Process immediately for non-batchable requests
            return await self._process_single_request(request)

        # Create future for result
        future = asyncio.Future()
        self.result_futures[request.request_id] = future

        # Add to queue
        self.request_queue.put(request)

        # Wait for result
        return await future

    def _process_batches(self):
        """Process requests in batches."""
        while self.processing_active:
            try:
                batch = self._collect_batch()
                if batch:
                    asyncio.run(self._process_batch(batch))
                else:
                    time.sleep(0.01)  # Short sleep if no requests

            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests for batching."""
        batch = []
        start_time = time.time()

        while (len(batch) < self.max_batch_size and
               (time.time() - start_time) * 1000 < self.max_wait_time_ms):

            try:
                request = self.request_queue.get(timeout=0.01)
                batch.append(request)
            except queue.Empty:
                if batch:  # Have some requests, don't wait longer
                    break
                continue

        return batch

    async def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests."""
        try:
            # Group by model for efficient processing
            model_groups = defaultdict(list)
            for request in batch:
                model_groups[request.model_name].append(request)

            # Process each model group
            for model_name, requests in model_groups.items():
                results = await self.batch_handler(model_name, requests)

                # Return results to futures
                for request, result in zip(requests, results):
                    if request.request_id in self.result_futures:
                        future = self.result_futures.pop(request.request_id)
                        if not future.cancelled():
                            future.set_result(result)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all futures
            for request in batch:
                if request.request_id in self.result_futures:
                    future = self.result_futures.pop(request.request_id)
                    if not future.cancelled():
                        future.set_exception(e)

    async def _process_single_request(self, request: InferenceRequest) -> InferenceResult:
        """Process single non-batchable request."""
        return await self.batch_handler(request.model_name, [request])[0]


class ContextOptimizer:
    """Optimizes context windows for AI models."""

    def __init__(self, max_context_length: int = 4096):
        self.max_context_length = max_context_length
        self.context_strategies = {
            'truncate_start': self._truncate_start,
            'truncate_end': self._truncate_end,
            'compress_middle': self._compress_middle,
            'extract_relevant': self._extract_relevant
        }

    def optimize_context(self, context: str, query: str, strategy: str = 'compress_middle') -> str:
        """Optimize context based on strategy."""
        if len(context) <= self.max_context_length:
            return context

        if strategy in self.context_strategies:
            return self.context_strategies[strategy](context, query)
        else:
            return self._truncate_end(context, query)

    def _truncate_start(self, context: str, query: str) -> str:
        """Truncate from the beginning."""
        max_chars = self.max_context_length - len(query) - 100  # Buffer
        if len(context) > max_chars:
            return context[-max_chars:]
        return context

    def _truncate_end(self, context: str, query: str) -> str:
        """Truncate from the end."""
        max_chars = self.max_context_length - len(query) - 100  # Buffer
        if len(context) > max_chars:
            return context[:max_chars]
        return context

    def _compress_middle(self, context: str, query: str) -> str:
        """Keep beginning and end, compress middle."""
        max_chars = self.max_context_length - len(query) - 100
        if len(context) <= max_chars:
            return context

        # Keep first and last portions
        keep_start = max_chars // 3
        keep_end = max_chars // 3

        start_part = context[:keep_start]
        end_part = context[-keep_end:]

        return f"{start_part}\n\n... [content compressed] ...\n\n{end_part}"

    def _extract_relevant(self, context: str, query: str) -> str:
        """Extract most relevant parts based on query."""
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        sentences = context.split('.')

        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            score = len(query_words.intersection(sentence_words))
            scored_sentences.append((score, sentence))

        # Sort by relevance and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        result = ""
        for score, sentence in scored_sentences:
            if len(result) + len(sentence) <= self.max_context_length - len(query) - 100:
                result += sentence + "."
            else:
                break

        return result


class ModelManager:
    """Manages multiple AI models and their performance."""

    def __init__(self):
        self.models = {}
        self.performance_stats = {}
        self.model_costs = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},  # per 1K tokens
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        }

    def register_model(self, name: str, handler: Callable, capabilities: Dict[str, Any]):
        """Register an AI model."""
        self.models[name] = {
            'handler': handler,
            'capabilities': capabilities,
            'active': True
        }
        self.performance_stats[name] = {
            'requests': 0,
            'total_latency': 0,
            'latencies': [],
            'errors': 0,
            'total_tokens': 0,
            'total_cost': 0
        }
        logger.info(f"Registered model: {name}")

    def select_best_model(self, requirements: Dict[str, Any]) -> str:
        """Select best model based on requirements and performance."""
        candidates = []

        for name, model_info in self.models.items():
            if not model_info['active']:
                continue

            # Check if model meets requirements
            capabilities = model_info['capabilities']
            score = 0

            # Performance scoring
            stats = self.performance_stats.get(name, {})
            if stats.get('requests', 0) > 0:
                avg_latency = stats['total_latency'] / stats['requests']
                error_rate = stats['errors'] / stats['requests']

                # Lower latency is better
                latency_score = max(0, 100 - avg_latency / 10)

                # Lower error rate is better
                error_score = max(0, 100 - error_rate * 100)

                score = (latency_score + error_score) / 2
            else:
                score = 50  # Default for untested models

            # Capability matching
            if requirements.get('max_tokens', 0) <= capabilities.get('max_tokens', float('inf')):
                score += 10

            if requirements.get('streaming', False) == capabilities.get('streaming', False):
                score += 5

            candidates.append((score, name))

        # Return best scoring model
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Fallback
        return next(iter(self.models.keys())) if self.models else None

    def update_performance_stats(self, model_name: str, latency_ms: float,
                                tokens: int, error: bool = False):
        """Update performance statistics for a model."""
        if model_name not in self.performance_stats:
            return

        stats = self.performance_stats[model_name]
        stats['requests'] += 1
        stats['total_latency'] += latency_ms
        stats['latencies'].append(latency_ms)
        stats['total_tokens'] += tokens

        # Keep only recent latencies for percentile calculations
        if len(stats['latencies']) > 1000:
            stats['latencies'] = stats['latencies'][-500:]

        if error:
            stats['errors'] += 1

        # Calculate cost
        if model_name in self.model_costs:
            cost = self._calculate_cost(model_name, tokens)
            stats['total_cost'] += cost

    def _calculate_cost(self, model_name: str, tokens: int) -> float:
        """Calculate cost for model usage."""
        if model_name not in self.model_costs:
            return 0.0

        costs = self.model_costs[model_name]
        # Simplified: assume 75% input, 25% output
        input_tokens = int(tokens * 0.75)
        output_tokens = int(tokens * 0.25)

        cost = (input_tokens * costs['input'] + output_tokens * costs['output']) / 1000
        return cost

    def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a model."""
        if model_name not in self.performance_stats:
            return None

        stats = self.performance_stats[model_name]
        if stats['requests'] == 0:
            return None

        latencies = stats['latencies']
        avg_latency = stats['total_latency'] / stats['requests']
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg_latency
        p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else avg_latency
        error_rate = stats['errors'] / stats['requests']

        return ModelPerformance(
            model_name=model_name,
            total_requests=stats['requests'],
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            error_rate=error_rate,
            throughput_per_second=1000 / avg_latency if avg_latency > 0 else 0,
            cache_hit_rate=0.0,  # Would need to integrate with cache
            avg_tokens_per_request=stats['total_tokens'] / stats['requests'],
            memory_usage_mb=0.0,  # Would need memory tracking
            cost_per_request=stats['total_cost'] / stats['requests'],
            last_updated=datetime.utcnow()
        )


class AIModelOptimizer:
    """Main AI model optimization manager."""

    def __init__(self, cache_size_mb: int = 512):
        self.cache = ModelCache(cache_size_mb)
        self.batch_processor = BatchProcessor()
        self.context_optimizer = ContextOptimizer()
        self.model_manager = ModelManager()

        # Performance tracking
        self.request_times = []
        self.active_requests = {}
        self.stats_lock = threading.Lock()

        # Configuration
        self.enable_caching = True
        self.enable_batching = True
        self.enable_context_optimization = True

    async def initialize(self):
        """Initialize the optimizer."""
        if self.enable_batching:
            self.batch_processor.start_processing(self._handle_batch_requests)
        logger.info("AI Model Optimizer initialized")

    async def inference(self,
                       prompt: str,
                       model_name: str = None,
                       parameters: Dict[str, Any] = None,
                       context: str = None,
                       cache_key: str = None,
                       priority: int = 2,
                       stream: bool = False) -> Union[InferenceResult, AsyncIterator[str]]:
        """Perform optimized AI inference."""

        parameters = parameters or {}
        request_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()

        # Select model if not specified
        if not model_name:
            requirements = {
                'max_tokens': parameters.get('max_tokens', 1000),
                'streaming': stream
            }
            model_name = self.model_manager.select_best_model(requirements)
            if not model_name:
                raise ValueError("No suitable model available")

        # Generate cache key if not provided
        if not cache_key and self.enable_caching:
            cache_input = f"{model_name}:{prompt}:{json.dumps(parameters, sort_keys=True)}"
            cache_key = hashlib.md5(cache_input.encode()).hexdigest()

        # Check cache
        if cache_key and self.enable_caching:
            cached_result = self.cache.get(cache_key, context)
            if cached_result:
                return InferenceResult(
                    request_id=request_id,
                    response=cached_result['response'],
                    model_name=model_name,
                    latency_ms=0,  # Cache hit
                    tokens_generated=cached_result.get('tokens', 0),
                    cache_hit=True,
                    cost_estimate=0,
                    metadata=cached_result.get('metadata', {}),
                    completed_at=datetime.utcnow()
                )

        # Optimize context if provided
        if context and self.enable_context_optimization:
            context = self.context_optimizer.optimize_context(context, prompt)

        # Create request
        request = InferenceRequest(
            request_id=request_id,
            model_name=model_name,
            prompt=prompt,
            parameters=parameters,
            context=context,
            priority=priority,
            submitted_at=datetime.utcnow(),
            cache_key=cache_key,
            batch_compatible=not stream  # Streaming requests can't be batched
        )

        # Process request
        if self.enable_batching and request.batch_compatible:
            result = await self.batch_processor.submit_request(request)
        else:
            result = await self._process_single_request(request, stream)

        # Cache result if successful
        if cache_key and result and not result.cache_hit:
            cache_data = {
                'response': result.response,
                'tokens': result.tokens_generated,
                'metadata': result.metadata
            }
            self.cache.set(cache_key, cache_data, ttl_seconds=3600, context=context)

        return result

    async def _handle_batch_requests(self, model_name: str,
                                   requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Handle batch of requests for the same model."""
        results = []
        start_time = time.perf_counter()

        try:
            # Get model handler
            if model_name not in self.model_manager.models:
                raise ValueError(f"Unknown model: {model_name}")

            handler = self.model_manager.models[model_name]['handler']

            # Process batch (implementation depends on specific model API)
            batch_responses = await handler(requests)

            # Create results
            for request, response in zip(requests, batch_responses):
                latency_ms = (time.perf_counter() - start_time) * 1000
                tokens = len(response.split()) if isinstance(response, str) else 0

                result = InferenceResult(
                    request_id=request.request_id,
                    response=response,
                    model_name=model_name,
                    latency_ms=latency_ms,
                    tokens_generated=tokens,
                    cache_hit=False,
                    cost_estimate=self.model_manager._calculate_cost(model_name, tokens),
                    metadata={'batch_size': len(requests)},
                    completed_at=datetime.utcnow()
                )
                results.append(result)

                # Update performance stats
                self.model_manager.update_performance_stats(
                    model_name, latency_ms, tokens
                )

        except Exception as e:
            # Create error results
            for request in requests:
                self.model_manager.update_performance_stats(
                    model_name, 0, 0, error=True
                )
                results.append(InferenceResult(
                    request_id=request.request_id,
                    response=f"Error: {str(e)}",
                    model_name=model_name,
                    latency_ms=0,
                    tokens_generated=0,
                    cache_hit=False,
                    cost_estimate=0,
                    metadata={'error': str(e)},
                    completed_at=datetime.utcnow()
                ))

        return results

    async def _process_single_request(self, request: InferenceRequest,
                                    stream: bool = False) -> Union[InferenceResult, AsyncIterator[str]]:
        """Process single request (non-batchable or streaming)."""
        start_time = time.perf_counter()

        try:
            # Get model handler
            if request.model_name not in self.model_manager.models:
                raise ValueError(f"Unknown model: {request.model_name}")

            handler = self.model_manager.models[request.model_name]['handler']

            if stream:
                # Return streaming iterator
                async def stream_generator():
                    async for chunk in handler([request], stream=True):
                        yield chunk

                return stream_generator()
            else:
                # Process single request
                responses = await handler([request])
                response = responses[0] if responses else ""

                latency_ms = (time.perf_counter() - start_time) * 1000
                tokens = len(response.split()) if isinstance(response, str) else 0

                result = InferenceResult(
                    request_id=request.request_id,
                    response=response,
                    model_name=request.model_name,
                    latency_ms=latency_ms,
                    tokens_generated=tokens,
                    cache_hit=False,
                    cost_estimate=self.model_manager._calculate_cost(request.model_name, tokens),
                    metadata={},
                    completed_at=datetime.utcnow()
                )

                # Update performance stats
                self.model_manager.update_performance_stats(
                    request.model_name, latency_ms, tokens
                )

                return result

        except Exception as e:
            self.model_manager.update_performance_stats(
                request.model_name, 0, 0, error=True
            )
            raise

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache.get_stats()

        model_performances = {}
        for model_name in self.model_manager.models:
            perf = self.model_manager.get_model_performance(model_name)
            if perf:
                model_performances[model_name] = asdict(perf)

        return {
            'cache_stats': cache_stats,
            'model_performances': model_performances,
            'batch_processor_active': self.batch_processor.processing_active,
            'optimization_features': {
                'caching_enabled': self.enable_caching,
                'batching_enabled': self.enable_batching,
                'context_optimization_enabled': self.enable_context_optimization
            }
        }

    def clear_cache(self):
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("AI model cache cleared")

    async def shutdown(self):
        """Shutdown the optimizer."""
        if self.enable_batching:
            self.batch_processor.stop_processing()
        logger.info("AI Model Optimizer shut down")


# Decorator for caching AI responses
def cached_inference(cache_ttl: int = 3600, optimizer: AIModelOptimizer = None):
    """Decorator for caching AI inference results."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = hashlib.md5(
                f"{func.__name__}:{args}:{kwargs}".encode()
            ).hexdigest()

            if optimizer:
                # Check cache first
                cached = optimizer.cache.get(cache_key)
                if cached:
                    return cached

                # Execute function and cache result
                result = await func(*args, **kwargs)
                optimizer.cache.set(cache_key, {'response': result}, cache_ttl)
                return result
            else:
                # Execute without caching
                return await func(*args, **kwargs)

        return wrapper
    return decorator


# Global optimizer instance
_global_ai_optimizer: Optional[AIModelOptimizer] = None


def get_ai_optimizer() -> AIModelOptimizer:
    """Get or create global AI optimizer."""
    global _global_ai_optimizer
    if _global_ai_optimizer is None:
        _global_ai_optimizer = AIModelOptimizer()
        asyncio.create_task(_global_ai_optimizer.initialize())
    return _global_ai_optimizer