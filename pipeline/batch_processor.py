"""
Batch Processor

Process multiple verses efficiently with adaptive concurrency and backpressure.
"""
import asyncio
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Iterator, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4

from pipeline.unified_orchestrator import UnifiedOrchestrator
from pipeline.golden_record import GoldenRecord


logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Strategy for batch processing."""
    SEQUENTIAL = "sequential"      # One at a time (safest)
    CHUNKED = "chunked"            # Process in fixed chunks
    ADAPTIVE = "adaptive"          # Adjust concurrency based on performance
    PRIORITY = "priority"          # Process high-value verses first

    @property
    def base_concurrency(self) -> int:
        """Default concurrency level."""
        return {
            BatchStrategy.SEQUENTIAL: 1,
            BatchStrategy.CHUNKED: 10,
            BatchStrategy.ADAPTIVE: 5,
            BatchStrategy.PRIORITY: 8,
        }[self]


@dataclass
class BackpressureState:
    """Track backpressure metrics for adaptive processing."""
    current_concurrency: int
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    error_streak: int = 0
    latency_samples: List[float] = field(default_factory=list)

    MAX_LATENCY_SAMPLES = 100
    ERROR_THRESHOLD = 3
    LATENCY_INCREASE_THRESHOLD = 1.5

    def record_success(self, latency_ms: float) -> None:
        """Record a successful processing."""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.MAX_LATENCY_SAMPLES:
            self.latency_samples.pop(0)
        self.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)
        self.error_streak = 0

    def record_failure(self) -> None:
        """Record a failed processing."""
        self.error_streak += 1

    @property
    def should_reduce_concurrency(self) -> bool:
        """Check if we should reduce concurrency due to backpressure."""
        if self.error_streak >= self.ERROR_THRESHOLD:
            return True
        if len(self.latency_samples) >= 10:
            recent_avg = sum(self.latency_samples[-10:]) / 10
            if recent_avg > self.avg_latency_ms * self.LATENCY_INCREASE_THRESHOLD:
                return True
        return False

    @property
    def should_increase_concurrency(self) -> bool:
        """Check if we can safely increase concurrency."""
        return (
            self.error_streak == 0 and
            len(self.latency_samples) >= 20 and
            sum(self.latency_samples[-10:]) / 10 < self.avg_latency_ms * 0.8
        )


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    strategy: BatchStrategy = BatchStrategy.CHUNKED
    chunk_size: int = 50
    max_concurrency: int = 20
    enable_progress_events: bool = True


@dataclass
class BatchResult:
    """Result from batch processing."""
    batch_id: str
    book_id: str
    results: List[GoldenRecord]
    errors: List[Dict[str, Any]]
    duration_ms: float
    throughput_per_second: float


class BatchProcessor:
    """
    Process multiple verses efficiently with adaptive concurrency.
    """

    def __init__(self, orchestrator: UnifiedOrchestrator, config: Optional[BatchConfig] = None):
        self.orchestrator = orchestrator
        self.config = config or BatchConfig()
        self.strategy = self.config.strategy
        self._backpressure = BackpressureState(
            current_concurrency=self.strategy.base_concurrency
        )
        self._semaphore = asyncio.Semaphore(self._backpressure.current_concurrency)
        self._start_time: Optional[float] = None

    async def process_book(self, book_id: str) -> BatchResult:
        """
        Process all verses in a book.

        Args:
            book_id: Book identifier (e.g., "GEN")

        Returns:
            BatchResult with statistics
        """
        self._start_time = time.time()

        # Get verses for book (placeholder - would query database)
        verses = await self._get_book_verses(book_id)

        batch_id = str(uuid4())
        results = []
        errors = []

        logger.info(f"Starting batch {batch_id} for book {book_id} with {len(verses)} verses")

        # Emit batch started event
        if self.orchestrator.event_publisher and self.config.enable_progress_events:
            from db.events import BatchProcessingStarted
            await self.orchestrator.event_publisher.publish(
                BatchProcessingStarted(
                    aggregate_id=batch_id,
                    correlation_id=batch_id,
                    batch_id=batch_id,
                    total_verses=len(verses)
                )
            )

        # Optionally prioritize verses
        if self.strategy == BatchStrategy.PRIORITY:
            verses = await self._prioritize_verses(verses)

        # Process in chunks with adaptive sizing
        chunk_size = self._calculate_chunk_size()
        for chunk_idx, chunk in enumerate(self._chunk_verses(verses, chunk_size)):
            chunk_results = await self._process_chunk(chunk, batch_id)

            for verse_id, result in chunk_results:
                if isinstance(result, Exception):
                    errors.append({
                        "verse_id": verse_id,
                        "error": str(result),
                        "error_type": type(result).__name__
                    })
                else:
                    results.append(result)

            # Adaptive concurrency adjustment
            if self.strategy == BatchStrategy.ADAPTIVE:
                await self._adjust_concurrency()

            # Progress update with ETA
            processed = len(results) + len(errors)
            eta_seconds = self._calculate_eta(processed, len(verses))

            logger.info(f"Batch progress: {processed}/{len(verses)} verses ({processed/len(verses)*100:.1f}%), ETA: {eta_seconds:.0f}s")

            if self.orchestrator.event_publisher and self.config.enable_progress_events:
                from db.events import BatchProgressUpdated
                await self.orchestrator.event_publisher.publish(
                    BatchProgressUpdated(
                        aggregate_id=batch_id,
                        correlation_id=batch_id,
                        batch_id=batch_id,
                        processed_count=processed,
                        total_count=len(verses),
                        success_count=len(results),
                        error_count=len(errors)
                    )
                )

        # Calculate final statistics
        duration_ms = (time.time() - self._start_time) * 1000

        # Emit batch completed event
        if self.orchestrator.event_publisher and self.config.enable_progress_events:
            from db.events import BatchProcessingCompleted
            await self.orchestrator.event_publisher.publish(
                BatchProcessingCompleted(
                    aggregate_id=batch_id,
                    correlation_id=batch_id,
                    batch_id=batch_id,
                    total_verses=len(verses),
                    success_count=len(results),
                    error_count=len(errors)
                )
            )

        logger.info(f"Batch {batch_id} completed: {len(results)} success, {len(errors)} errors in {duration_ms:.0f}ms")

        return BatchResult(
            batch_id=batch_id,
            book_id=book_id,
            results=results,
            errors=errors,
            duration_ms=duration_ms,
            throughput_per_second=len(verses) / (duration_ms / 1000) if duration_ms > 0 else 0
        )

    async def _process_chunk(
        self,
        chunk: List[str],
        batch_id: str
    ) -> List[Tuple[str, Union[GoldenRecord, Exception]]]:
        """Process a chunk of verses with semaphore control."""
        tasks = [
            self._process_with_semaphore(verse_id, batch_id)
            for verse_id in chunk
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(zip(chunk, results))

    async def _process_with_semaphore(
        self,
        verse_id: str,
        batch_id: str
    ) -> GoldenRecord:
        """Process single verse with semaphore and backpressure tracking."""
        async with self._semaphore:
            start = time.time()
            try:
                result = await self.orchestrator.process_verse(
                    verse_id,
                    correlation_id=f"{batch_id}:{verse_id}"
                )
                latency = (time.time() - start) * 1000
                self._backpressure.record_success(latency)
                return result
            except Exception as e:
                self._backpressure.record_failure()
                raise

    async def _adjust_concurrency(self) -> None:
        """Adjust concurrency based on backpressure state."""
        if self._backpressure.should_reduce_concurrency:
            new_concurrency = max(1, self._backpressure.current_concurrency - 2)
            self._backpressure.current_concurrency = new_concurrency
            self._semaphore = asyncio.Semaphore(new_concurrency)
            logger.info(f"Reduced concurrency to {new_concurrency}")
        elif self._backpressure.should_increase_concurrency:
            new_concurrency = min(
                self.config.max_concurrency,
                self._backpressure.current_concurrency + 1
            )
            self._backpressure.current_concurrency = new_concurrency
            self._semaphore = asyncio.Semaphore(new_concurrency)
            logger.info(f"Increased concurrency to {new_concurrency}")

    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on strategy."""
        if self.strategy == BatchStrategy.SEQUENTIAL:
            return 1
        return min(self.config.chunk_size, self._backpressure.current_concurrency * 2)

    def _calculate_eta(self, processed: int, total: int) -> float:
        """Calculate estimated time to completion."""
        if processed == 0 or self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        rate = processed / elapsed
        remaining = total - processed
        return remaining / rate if rate > 0 else 0.0

    async def _prioritize_verses(self, verses: List[str]) -> List[str]:
        """Prioritize verses for processing (high-value first)."""
        # Placeholder - would get centrality scores from graph
        # For now, return as-is
        return verses

    def _chunk_verses(self, verses: List[str], chunk_size: int) -> Iterator[List[str]]:
        """Yield chunks of verses."""
        for i in range(0, len(verses), chunk_size):
            yield verses[i:i + chunk_size]

    async def _get_book_verses(self, book_id: str) -> List[str]:
        """Get all verse IDs for a book."""
        # Placeholder - would query database
        # For now, return mock data
        return [
            f"{book_id}.1.1",
            f"{book_id}.1.2",
            f"{book_id}.1.3",
        ]
