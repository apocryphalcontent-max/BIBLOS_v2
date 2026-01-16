"""
BIBLOS v2 - ML Batch Processor

Provides true parallel batch processing for ML inference:
- GPU-optimized batch sizing
- Dynamic batch accumulation
- Memory-aware processing
- Automatic tensor cleanup
- OpenTelemetry tracing

Fixes the sequential batch processing issue in the original implementation.
"""

from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 32
    max_batch_size: int = 128
    min_batch_size: int = 4
    max_wait_ms: float = 50.0
    dynamic_batching: bool = True
    memory_limit_mb: int = 1024
    device: str = "cuda"
    num_workers: int = 2


@dataclass
class BatchStats:
    """Statistics for batch processing."""

    total_batches: int = 0
    total_items: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_batches": self.total_batches,
            "total_items": self.total_items,
            "avg_batch_size": self.avg_batch_size,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "gpu_utilization": self.gpu_utilization,
        }


class GPUMemoryManager:
    """
    Manages GPU memory for batch processing.

    Features:
    - Automatic memory tracking
    - OOM prevention
    - Tensor cleanup
    - Memory pressure alerts
    """

    def __init__(self, device: str = "cuda", limit_mb: int = 1024):
        self.device = device
        self.limit_bytes = limit_mb * 1024 * 1024
        self._peak_memory = 0
        self._torch_available = False

        try:
            import torch
            self._torch_available = torch.cuda.is_available() if device == "cuda" else False
            self._torch = torch
        except ImportError:
            pass

    def get_memory_usage(self) -> Tuple[int, int]:
        """Get current and peak GPU memory usage in bytes."""
        if not self._torch_available:
            return 0, 0

        allocated = self._torch.cuda.memory_allocated(self.device)
        peak = self._torch.cuda.max_memory_allocated(self.device)
        self._peak_memory = max(self._peak_memory, peak)
        return allocated, peak

    def get_available_memory(self) -> int:
        """Get available GPU memory in bytes."""
        if not self._torch_available:
            return self.limit_bytes

        allocated, _ = self.get_memory_usage()
        return max(0, self.limit_bytes - allocated)

    def can_fit_batch(self, estimated_size_bytes: int) -> bool:
        """Check if a batch can fit in available memory."""
        available = self.get_available_memory()
        # Keep 20% buffer for safety
        return estimated_size_bytes < available * 0.8

    def cleanup(self) -> None:
        """Force GPU memory cleanup."""
        if self._torch_available:
            self._torch.cuda.empty_cache()
            gc.collect()

    def reset_peak_stats(self) -> None:
        """Reset peak memory tracking."""
        if self._torch_available:
            self._torch.cuda.reset_peak_memory_stats()
            self._peak_memory = 0


class DynamicBatcher(Generic[T, R]):
    """
    Dynamic batch accumulator for optimal GPU utilization.

    Collects items and forms optimal batch sizes based on:
    - Current memory availability
    - Processing time targets
    - Queue depth
    """

    def __init__(
        self,
        process_fn: Callable[[List[T]], List[R]],
        config: Optional[BatchConfig] = None,
    ):
        self.process_fn = process_fn
        self.config = config or BatchConfig()
        self.memory_manager = GPUMemoryManager(
            self.config.device,
            self.config.memory_limit_mb,
        )
        self._queue: Queue[Tuple[T, asyncio.Future]] = Queue()
        self._running = False
        self._stats = BatchStats()
        self._total_processing_time = 0.0
        self._batch_count = 0
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self._worker_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _determine_batch_size(self) -> int:
        """Determine optimal batch size based on current conditions."""
        if not self.config.dynamic_batching:
            return self.config.batch_size

        # Start with configured batch size
        batch_size = self.config.batch_size

        # Adjust based on queue depth
        queue_size = self._queue.qsize()
        if queue_size > self.config.batch_size * 2:
            # High queue depth - increase batch size
            batch_size = min(self.config.max_batch_size, batch_size * 2)
        elif queue_size < self.config.min_batch_size:
            # Low queue depth - decrease batch size for responsiveness
            batch_size = max(self.config.min_batch_size, batch_size // 2)

        # Adjust based on available memory
        available_memory = self.memory_manager.get_available_memory()
        estimated_per_item = 4 * 768 * 4  # Approximate embedding size
        max_by_memory = available_memory // estimated_per_item

        return min(batch_size, max_by_memory)

    def _process_batch(self, batch: List[Tuple[T, asyncio.Future]]) -> None:
        """Process a batch and resolve futures."""
        items = [item for item, _ in batch]
        futures = [fut for _, fut in batch]

        try:
            with tracer.start_as_current_span("batch.process") as span:
                span.set_attribute("batch.size", len(items))

                start_time = time.perf_counter()
                results = self.process_fn(items)
                processing_time = (time.perf_counter() - start_time) * 1000

                span.set_attribute("batch.processing_time_ms", processing_time)

                # Update stats
                self._batch_count += 1
                self._total_processing_time += processing_time
                self._stats.total_batches = self._batch_count
                self._stats.total_items += len(items)
                self._stats.avg_batch_size = self._stats.total_items / self._batch_count
                self._stats.avg_processing_time_ms = self._total_processing_time / self._batch_count

                # Resolve futures
                for i, result in enumerate(results):
                    if not futures[i].done():
                        self._loop.call_soon_threadsafe(futures[i].set_result, result)

        except Exception as e:
            for fut in futures:
                if not fut.done():
                    self._loop.call_soon_threadsafe(fut.set_exception, e)

    def _worker_loop(self) -> None:
        """Background worker that processes batches."""
        while self._running:
            batch: List[Tuple[T, asyncio.Future]] = []
            batch_size = self._determine_batch_size()
            deadline = time.monotonic() + (self.config.max_wait_ms / 1000)

            # Collect items up to batch size or deadline
            while len(batch) < batch_size and time.monotonic() < deadline:
                try:
                    remaining = deadline - time.monotonic()
                    timeout = max(0.001, remaining)
                    item = self._queue.get(timeout=timeout)
                    batch.append(item)
                except Empty:
                    break

            if batch:
                self._process_batch(batch)

    async def submit(self, item: T) -> R:
        """Submit an item for batch processing."""
        if not self._running:
            self.start()

        future = asyncio.get_event_loop().create_future()
        self._queue.put((item, future))
        return await future

    async def submit_batch(self, items: List[T]) -> List[R]:
        """Submit multiple items for batch processing."""
        if not self._running:
            self.start()

        futures = []
        for item in items:
            future = asyncio.get_event_loop().create_future()
            self._queue.put((item, future))
            futures.append(future)

        return await asyncio.gather(*futures)

    def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_event_loop()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        self._executor.shutdown(wait=False)
        self.memory_manager.cleanup()

    def get_stats(self) -> BatchStats:
        """Get processing statistics."""
        return BatchStats(
            total_batches=self._stats.total_batches,
            total_items=self._stats.total_items,
            avg_batch_size=self._stats.avg_batch_size,
            avg_processing_time_ms=self._stats.avg_processing_time_ms,
            peak_memory_mb=self.memory_manager._peak_memory / (1024 * 1024),
        )


class EmbeddingBatcher:
    """
    Specialized batcher for embedding generation.

    Optimized for sentence transformer models with:
    - Automatic text length grouping
    - Padding optimization
    - Model-specific batch sizes
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        config: Optional[BatchConfig] = None,
    ):
        self.embed_fn = embed_fn
        self.config = config or BatchConfig()
        self._batcher = DynamicBatcher[str, np.ndarray](
            self._process_texts,
            self.config,
        )

    def _group_by_length(
        self,
        texts: List[str],
    ) -> List[Tuple[List[int], List[str]]]:
        """Group texts by similar lengths for efficient padding."""
        # Define length buckets
        buckets: Dict[int, Tuple[List[int], List[str]]] = {}

        for i, text in enumerate(texts):
            length = len(text)
            # Round to nearest 64 tokens for grouping
            bucket_key = (length // 64) * 64
            if bucket_key not in buckets:
                buckets[bucket_key] = ([], [])
            buckets[bucket_key][0].append(i)
            buckets[bucket_key][1].append(text)

        return list(buckets.values())

    def _process_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Process texts with length-based grouping."""
        results: List[Tuple[int, np.ndarray]] = []

        # Group by similar lengths
        groups = self._group_by_length(texts)

        for indices, group_texts in groups:
            with tracer.start_as_current_span("embedding.group") as span:
                span.set_attribute("group.size", len(group_texts))

                # Generate embeddings for this group
                embeddings = self.embed_fn(group_texts)

                # Store with original indices
                for i, idx in enumerate(indices):
                    results.append((idx, embeddings[i]))

        # Sort by original index and extract embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return await self._batcher.submit(text)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts with optimal batching."""
        if not texts:
            return []

        with tracer.start_as_current_span("embedding.batch") as span:
            span.set_attribute("batch.size", len(texts))
            results = await self._batcher.submit_batch(texts)
            span.set_attribute("batch.completed", len(results))
            return results

    def start(self) -> None:
        """Start the batcher."""
        self._batcher.start()

    def stop(self) -> None:
        """Stop the batcher."""
        self._batcher.stop()

    def get_stats(self) -> BatchStats:
        """Get processing statistics."""
        return self._batcher.get_stats()


def true_batch_embed(
    model,
    texts: List[str],
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Perform true parallel batch embedding.

    Unlike sequential processing, this uses the model's native
    batch processing capabilities for maximum GPU utilization.

    Args:
        model: Sentence transformer model
        texts: List of texts to embed
        batch_size: Batch size for processing
        device: Device to use (cuda/cpu)

    Returns:
        Array of embeddings [N, D]
    """
    with tracer.start_as_current_span("embedding.true_batch") as span:
        span.set_attribute("input.count", len(texts))
        span.set_attribute("batch_size", batch_size)
        span.set_attribute("device", device)

        # Use model's native encode_multi_process or encode with show_progress_bar=False
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )

        span.set_attribute("output.shape", str(embeddings.shape))
        return embeddings
