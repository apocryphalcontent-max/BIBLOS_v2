"""
BIBLOS v2 - Arrow Batch Processing

High-performance batch processing utilities using Apache Arrow for zero-copy
data transfer between components, processes, and systems.

Key Features:
1. Zero-copy IPC for inter-process communication
2. Memory-mapped file support for large datasets
3. Streaming batch processing with backpressure
4. Efficient serialization/deserialization
5. Integration with existing pipeline infrastructure

Performance Characteristics:
- Zero-copy reads from memory-mapped files
- Columnar compression (LZ4, ZSTD, Snappy)
- Predicate pushdown for filtered reads
- Parallel chunk processing
"""
from __future__ import annotations

import asyncio
import logging
import mmap
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather
import pyarrow.parquet as pq

from data.schemas import (
    BaseSchema,
    VerseSchema,
    CrossReferenceSchema,
    GoldenRecordSchema,
    ExtractionResultSchema,
    InferenceCandidateSchema,
)
from data.polars_schemas import (
    SchemaConverter,
    ArrowSchemas,
    DataFrameBuilder,
    PolarsSchemas,
)

logger = logging.getLogger("biblos.data.arrow_batch")

T = TypeVar("T", bound=BaseSchema)


# =============================================================================
# BATCH CONFIGURATION
# =============================================================================

@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""

    # Batch sizes
    batch_size: int = 1000              # Default batch size
    max_batch_bytes: int = 64 * 1024 * 1024  # 64MB max batch size

    # Memory settings
    memory_map: bool = True             # Use memory-mapped I/O
    use_mmap_threshold: int = 10 * 1024 * 1024  # Files > 10MB use mmap

    # Compression settings
    compression: str = "zstd"           # lz4, zstd, snappy, none
    compression_level: int = 3          # 1-22 for zstd

    # Parallelism
    num_threads: int = 4                # Thread pool size
    use_threads: bool = True            # Enable parallel processing

    # Streaming
    stream_buffer_size: int = 10        # Number of batches to buffer
    prefetch_batches: int = 2           # Batches to prefetch

    # IPC settings
    ipc_write_options: Optional[Dict[str, Any]] = None
    ipc_read_options: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set default IPC options based on compression settings."""
        if self.ipc_write_options is None:
            self.ipc_write_options = {
                "compression": self.compression if self.compression != "none" else None,
            }


@dataclass
class BatchMetrics:
    """Metrics for batch processing operations."""

    total_rows: int = 0
    total_bytes: int = 0
    batches_processed: int = 0
    processing_time_seconds: float = 0.0
    serialization_time_seconds: float = 0.0
    deserialization_time_seconds: float = 0.0
    io_time_seconds: float = 0.0
    compression_ratio: float = 1.0

    @property
    def rows_per_second(self) -> float:
        """Calculate processing throughput."""
        if self.processing_time_seconds > 0:
            return self.total_rows / self.processing_time_seconds
        return 0.0

    @property
    def bytes_per_second(self) -> float:
        """Calculate I/O throughput."""
        if self.io_time_seconds > 0:
            return self.total_bytes / self.io_time_seconds
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "total_rows": self.total_rows,
            "total_bytes": self.total_bytes,
            "batches_processed": self.batches_processed,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "rows_per_second": round(self.rows_per_second, 1),
            "bytes_per_second": round(self.bytes_per_second, 1),
            "compression_ratio": round(self.compression_ratio, 2),
        }


# =============================================================================
# ARROW IPC READER/WRITER
# =============================================================================

class ArrowIPCWriter:
    """
    High-performance Arrow IPC writer with streaming support.

    Supports writing to files or memory buffers with optional compression.
    Enables zero-copy reading by consumers.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        schema: Optional[pa.Schema] = None,
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize IPC writer.

        Args:
            path: Output file path (None for in-memory buffer)
            schema: Arrow schema for the data
            config: Batch processing configuration
        """
        self.path = Path(path) if path else None
        self.schema = schema
        self.config = config or BatchConfig()
        self._writer: Optional[ipc.RecordBatchFileWriter] = None
        self._buffer: Optional[pa.BufferOutputStream] = None
        self._sink: Optional[Any] = None
        self._metrics = BatchMetrics()
        self._start_time: float = 0.0

    def __enter__(self) -> "ArrowIPCWriter":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open the writer for writing."""
        self._start_time = time.time()

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._sink = pa.OSFile(str(self.path), "wb")
        else:
            self._buffer = pa.BufferOutputStream()
            self._sink = self._buffer

        options = ipc.IpcWriteOptions(
            compression=pa.Codec(self.config.compression)
            if self.config.compression != "none"
            else None
        )

        if self.schema:
            self._writer = ipc.new_file(self._sink, self.schema, options=options)

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """
        Write a single record batch.

        Args:
            batch: Arrow RecordBatch to write
        """
        if self._writer is None:
            if self.schema is None:
                self.schema = batch.schema
            options = ipc.IpcWriteOptions(
                compression=pa.Codec(self.config.compression)
                if self.config.compression != "none"
                else None
            )
            self._writer = ipc.new_file(self._sink, self.schema, options=options)

        ser_start = time.time()
        self._writer.write_batch(batch)
        self._metrics.serialization_time_seconds += time.time() - ser_start

        self._metrics.total_rows += batch.num_rows
        self._metrics.batches_processed += 1

    def write_table(self, table: pa.Table) -> None:
        """
        Write an entire Arrow table.

        Args:
            table: Arrow Table to write
        """
        for batch in table.to_batches(max_chunksize=self.config.batch_size):
            self.write_batch(batch)

    def write_dataframe(self, df: pl.DataFrame) -> None:
        """
        Write a Polars DataFrame.

        Args:
            df: Polars DataFrame to write
        """
        table = df.to_arrow()
        self.write_table(table)

    def write_dataclasses(
        self,
        items: List[T],
        schema_type: Optional[Type[T]] = None
    ) -> None:
        """
        Write a list of dataclass instances.

        Args:
            items: List of dataclass instances
            schema_type: Optional schema type hint
        """
        df = SchemaConverter.dataclass_to_polars(items, schema_type)
        self.write_dataframe(df)

    def close(self) -> None:
        """Close the writer and finalize output."""
        if self._writer:
            self._writer.close()
            self._writer = None

        if self._sink and hasattr(self._sink, "close"):
            self._sink.close()

        self._metrics.processing_time_seconds = time.time() - self._start_time

        if self.path and self.path.exists():
            self._metrics.total_bytes = self.path.stat().st_size

        logger.debug(f"IPC write complete: {self._metrics.to_dict()}")

    def get_buffer(self) -> Optional[pa.Buffer]:
        """Get the in-memory buffer (if writing to memory)."""
        if self._buffer:
            return self._buffer.getvalue()
        return None

    @property
    def metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self._metrics


class ArrowIPCReader:
    """
    High-performance Arrow IPC reader with memory-mapping support.

    Supports zero-copy reads from files or memory buffers.
    """

    def __init__(
        self,
        source: Union[str, Path, pa.Buffer, bytes],
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize IPC reader.

        Args:
            source: File path, Buffer, or bytes to read from
            config: Batch processing configuration
        """
        self.source = source
        self.config = config or BatchConfig()
        self._reader: Optional[ipc.RecordBatchFileReader] = None
        self._metrics = BatchMetrics()

    def __enter__(self) -> "ArrowIPCReader":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open the reader."""
        start_time = time.time()

        if isinstance(self.source, (str, Path)):
            path = Path(self.source)
            file_size = path.stat().st_size
            self._metrics.total_bytes = file_size

            # Use memory mapping for large files
            if self.config.memory_map and file_size >= self.config.use_mmap_threshold:
                self._reader = ipc.open_file(pa.memory_map(str(path), "r"))
                logger.debug(f"Opened {path} with memory mapping ({file_size:,} bytes)")
            else:
                self._reader = ipc.open_file(str(path))
                logger.debug(f"Opened {path} ({file_size:,} bytes)")

        elif isinstance(self.source, pa.Buffer):
            self._reader = ipc.open_file(pa.BufferReader(self.source))
            self._metrics.total_bytes = self.source.size

        elif isinstance(self.source, bytes):
            self._reader = ipc.open_file(pa.BufferReader(pa.py_buffer(self.source)))
            self._metrics.total_bytes = len(self.source)

        self._metrics.io_time_seconds = time.time() - start_time

    @property
    def schema(self) -> pa.Schema:
        """Get the schema of the data."""
        return self._reader.schema if self._reader else None

    @property
    def num_batches(self) -> int:
        """Get the number of record batches."""
        return self._reader.num_record_batches if self._reader else 0

    def read_batch(self, index: int) -> pa.RecordBatch:
        """
        Read a specific record batch by index.

        Args:
            index: Batch index (0-based)

        Returns:
            Arrow RecordBatch
        """
        if not self._reader:
            raise RuntimeError("Reader not opened")

        deser_start = time.time()
        batch = self._reader.get_batch(index)
        self._metrics.deserialization_time_seconds += time.time() - deser_start
        self._metrics.total_rows += batch.num_rows
        self._metrics.batches_processed += 1
        return batch

    def read_all(self) -> pa.Table:
        """
        Read all batches as a single Arrow Table.

        Returns:
            Arrow Table containing all data
        """
        if not self._reader:
            raise RuntimeError("Reader not opened")

        start_time = time.time()
        table = self._reader.read_all()
        self._metrics.deserialization_time_seconds = time.time() - start_time
        self._metrics.total_rows = table.num_rows
        self._metrics.batches_processed = self._reader.num_record_batches
        return table

    def read_dataframe(self) -> pl.DataFrame:
        """
        Read all data as a Polars DataFrame.

        Returns:
            Polars DataFrame
        """
        table = self.read_all()
        return pl.from_arrow(table)

    def read_dataclasses(self, schema_type: Type[T]) -> List[T]:
        """
        Read all data as a list of dataclass instances.

        Args:
            schema_type: Target dataclass type

        Returns:
            List of dataclass instances
        """
        df = self.read_dataframe()
        return SchemaConverter.polars_to_dataclass(df, schema_type)

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        """
        Iterate over record batches.

        Yields:
            Arrow RecordBatch for each batch in the file
        """
        for i in range(self.num_batches):
            yield self.read_batch(i)

    def iter_dataframes(
        self,
        chunk_size: Optional[int] = None
    ) -> Iterator[pl.DataFrame]:
        """
        Iterate over data as Polars DataFrames.

        Args:
            chunk_size: Optional rows per chunk (None = one batch per chunk)

        Yields:
            Polars DataFrame for each chunk
        """
        for batch in self.iter_batches():
            df = pl.from_arrow(pa.Table.from_batches([batch]))

            if chunk_size and len(df) > chunk_size:
                # Split large batches
                for start in range(0, len(df), chunk_size):
                    yield df.slice(start, chunk_size)
            else:
                yield df

    def close(self) -> None:
        """Close the reader."""
        self._reader = None

    @property
    def metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self._metrics


# =============================================================================
# STREAMING BATCH PROCESSOR
# =============================================================================

class StreamingBatchProcessor:
    """
    Streaming batch processor for large datasets.

    Processes data in batches with backpressure control and efficient
    memory usage. Supports async processing and parallel execution.
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize streaming processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self._metrics = BatchMetrics()

    async def process_stream(
        self,
        source: Union[str, Path, pl.LazyFrame],
        processor: Callable[[pl.DataFrame], pl.DataFrame],
        sink: Optional[Union[str, Path]] = None,
    ) -> BatchMetrics:
        """
        Process a data stream with batched execution.

        Args:
            source: Input source (file path or LazyFrame)
            processor: Function to process each batch
            sink: Optional output destination

        Returns:
            Processing metrics
        """
        start_time = time.time()

        # Create output writer if sink specified
        writer = None
        if sink:
            writer = ArrowIPCWriter(sink, config=self.config)
            writer.open()

        try:
            async for batch_df in self._iter_source_async(source):
                # Process batch
                processed = processor(batch_df)

                # Write output
                if writer:
                    writer.write_dataframe(processed)

                self._metrics.total_rows += len(processed)
                self._metrics.batches_processed += 1

        finally:
            if writer:
                writer.close()
                self._metrics.total_bytes = writer.metrics.total_bytes

        self._metrics.processing_time_seconds = time.time() - start_time
        return self._metrics

    async def _iter_source_async(
        self,
        source: Union[str, Path, pl.LazyFrame],
    ) -> AsyncIterator[pl.DataFrame]:
        """
        Async iterator over source data.

        Args:
            source: Input source

        Yields:
            Polars DataFrame for each batch
        """
        if isinstance(source, pl.LazyFrame):
            # Collect in batches using streaming
            for batch in source.collect(streaming=True).iter_slices(
                n_rows=self.config.batch_size
            ):
                yield batch
                await asyncio.sleep(0)  # Yield control

        elif isinstance(source, (str, Path)):
            path = Path(source)

            if path.suffix == ".parquet":
                # Stream from parquet
                lf = pl.scan_parquet(path)
                async for batch in self._iter_source_async(lf):
                    yield batch

            elif path.suffix in (".ipc", ".arrow", ".feather"):
                # Stream from IPC
                with ArrowIPCReader(path, config=self.config) as reader:
                    for df in reader.iter_dataframes(self.config.batch_size):
                        yield df
                        await asyncio.sleep(0)

            else:
                # Try JSON
                df = pl.read_json(path)
                for batch in df.iter_slices(n_rows=self.config.batch_size):
                    yield batch
                    await asyncio.sleep(0)

    def process_parallel(
        self,
        batches: List[pl.DataFrame],
        processor: Callable[[pl.DataFrame], pl.DataFrame],
        max_workers: Optional[int] = None,
    ) -> List[pl.DataFrame]:
        """
        Process batches in parallel using thread pool.

        Args:
            batches: List of DataFrames to process
            processor: Processing function
            max_workers: Maximum parallel workers

        Returns:
            List of processed DataFrames
        """
        from concurrent.futures import ThreadPoolExecutor

        workers = max_workers or self.config.num_threads

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(processor, batches))

        return results

    @property
    def metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self._metrics


# =============================================================================
# BATCH VERSE PROCESSOR
# =============================================================================

class BatchVerseProcessor:
    """
    Specialized batch processor for verse data.

    Optimized for processing biblical verses through the SDES pipeline
    with efficient memory usage and parallel execution.
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize verse batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self._metrics = BatchMetrics()

    def load_verses(
        self,
        source: Union[str, Path, List[Dict], List[VerseSchema]],
    ) -> pl.DataFrame:
        """
        Load verses from various sources into DataFrame.

        Args:
            source: File path, list of dicts, or list of VerseSchema

        Returns:
            Polars DataFrame with verse data
        """
        start_time = time.time()

        if isinstance(source, (str, Path)):
            path = Path(source)

            if path.suffix == ".parquet":
                df = pl.read_parquet(path)
            elif path.suffix in (".ipc", ".arrow", ".feather"):
                df = pl.read_ipc(path, memory_map=self.config.memory_map)
            elif path.suffix == ".json":
                df = pl.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        elif isinstance(source, list):
            if not source:
                df = DataFrameBuilder.empty(VerseSchema)
            elif isinstance(source[0], VerseSchema):
                df = SchemaConverter.dataclass_to_polars(source, VerseSchema)
            else:
                df = DataFrameBuilder.verses(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        self._metrics.total_rows = len(df)
        self._metrics.io_time_seconds = time.time() - start_time

        logger.info(f"Loaded {len(df):,} verses in {self._metrics.io_time_seconds:.2f}s")
        return df

    def iter_batches(
        self,
        df: pl.DataFrame,
        batch_size: Optional[int] = None,
    ) -> Iterator[Tuple[int, pl.DataFrame]]:
        """
        Iterate over verses in batches.

        Args:
            df: Verses DataFrame
            batch_size: Batch size (defaults to config)

        Yields:
            Tuple of (batch_index, batch_dataframe)
        """
        size = batch_size or self.config.batch_size

        for i, batch in enumerate(df.iter_slices(n_rows=size)):
            yield i, batch

    def process_batch(
        self,
        batch: pl.DataFrame,
        processor: Callable[[pl.DataFrame], pl.DataFrame],
    ) -> pl.DataFrame:
        """
        Process a single batch of verses.

        Args:
            batch: Batch DataFrame
            processor: Processing function

        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        result = processor(batch)
        self._metrics.processing_time_seconds += time.time() - start_time
        self._metrics.batches_processed += 1
        return result

    def filter_by_books(
        self,
        df: pl.DataFrame,
        books: List[str],
    ) -> pl.DataFrame:
        """
        Filter verses by book codes.

        Args:
            df: Verses DataFrame
            books: List of book codes (e.g., ["GEN", "EXO"])

        Returns:
            Filtered DataFrame
        """
        return df.filter(pl.col("book").is_in(books))

    def filter_by_range(
        self,
        df: pl.DataFrame,
        start_ref: str,
        end_ref: str,
    ) -> pl.DataFrame:
        """
        Filter verses by reference range.

        Args:
            df: Verses DataFrame
            start_ref: Starting verse reference
            end_ref: Ending verse reference

        Returns:
            Filtered DataFrame
        """
        # Parse references
        start_parts = start_ref.upper().replace(":", ".").split(".")
        end_parts = end_ref.upper().replace(":", ".").split(".")

        return df.filter(
            (pl.col("book") >= start_parts[0]) &
            (pl.col("book") <= end_parts[0]) &
            (pl.col("chapter") >= int(start_parts[1])) &
            (pl.col("chapter") <= int(end_parts[1]))
        )

    def save_results(
        self,
        df: pl.DataFrame,
        path: Union[str, Path],
        format: str = "parquet",
    ) -> None:
        """
        Save processed results to file.

        Args:
            df: DataFrame to save
            path: Output path
            format: Output format (parquet, ipc, json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        if format == "parquet":
            df.write_parquet(
                path,
                compression=self.config.compression,
            )
        elif format in ("ipc", "arrow", "feather"):
            df.write_ipc(
                path,
                compression=self.config.compression,
            )
        elif format == "json":
            df.write_json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self._metrics.io_time_seconds += time.time() - start_time
        self._metrics.total_bytes = path.stat().st_size

        logger.info(f"Saved {len(df):,} rows to {path} ({self._metrics.total_bytes:,} bytes)")

    @property
    def metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self._metrics


# =============================================================================
# CROSS-REFERENCE BATCH PROCESSOR
# =============================================================================

class BatchCrossRefProcessor:
    """
    Specialized batch processor for cross-reference data.

    Optimized for processing and discovering cross-references with
    efficient graph operations and similarity computations.
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize cross-reference processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self._metrics = BatchMetrics()

    def load_crossrefs(
        self,
        source: Union[str, Path, List[Dict], List[CrossReferenceSchema]],
    ) -> pl.DataFrame:
        """
        Load cross-references from various sources.

        Args:
            source: File path, list of dicts, or list of CrossReferenceSchema

        Returns:
            Polars DataFrame with cross-reference data
        """
        start_time = time.time()

        if isinstance(source, (str, Path)):
            path = Path(source)

            if path.suffix == ".parquet":
                df = pl.read_parquet(path)
            elif path.suffix in (".ipc", ".arrow", ".feather"):
                df = pl.read_ipc(path, memory_map=self.config.memory_map)
            elif path.suffix == ".json":
                df = pl.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        elif isinstance(source, list):
            if not source:
                df = DataFrameBuilder.empty(CrossReferenceSchema)
            elif isinstance(source[0], CrossReferenceSchema):
                df = SchemaConverter.dataclass_to_polars(source, CrossReferenceSchema)
            else:
                df = DataFrameBuilder.cross_references(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        self._metrics.total_rows = len(df)
        self._metrics.io_time_seconds = time.time() - start_time

        logger.info(f"Loaded {len(df):,} cross-refs in {self._metrics.io_time_seconds:.2f}s")
        return df

    def filter_by_type(
        self,
        df: pl.DataFrame,
        connection_types: List[str],
    ) -> pl.DataFrame:
        """
        Filter cross-references by connection type.

        Args:
            df: Cross-references DataFrame
            connection_types: List of connection types to include

        Returns:
            Filtered DataFrame
        """
        return df.filter(pl.col("connection_type").is_in(connection_types))

    def filter_by_confidence(
        self,
        df: pl.DataFrame,
        min_confidence: float,
    ) -> pl.DataFrame:
        """
        Filter cross-references by minimum confidence.

        Args:
            df: Cross-references DataFrame
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered DataFrame
        """
        return df.filter(pl.col("confidence") >= min_confidence)

    def get_source_refs(
        self,
        df: pl.DataFrame,
        verse_id: str,
    ) -> pl.DataFrame:
        """
        Get all cross-references from a source verse.

        Args:
            df: Cross-references DataFrame
            verse_id: Source verse ID

        Returns:
            DataFrame of cross-references from this verse
        """
        return df.filter(pl.col("source_ref") == verse_id.upper())

    def get_target_refs(
        self,
        df: pl.DataFrame,
        verse_id: str,
    ) -> pl.DataFrame:
        """
        Get all cross-references to a target verse.

        Args:
            df: Cross-references DataFrame
            verse_id: Target verse ID

        Returns:
            DataFrame of cross-references to this verse
        """
        return df.filter(pl.col("target_ref") == verse_id.upper())

    def compute_stats(
        self,
        df: pl.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compute statistics for cross-reference dataset.

        Args:
            df: Cross-references DataFrame

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_count": len(df),
            "unique_sources": df["source_ref"].n_unique(),
            "unique_targets": df["target_ref"].n_unique(),
            "avg_confidence": df["confidence"].mean(),
            "verified_count": df.filter(pl.col("verified") == True).height,
        }

        # Count by connection type
        type_counts = df.group_by("connection_type").count().to_dict()
        stats["by_type"] = dict(zip(
            type_counts.get("connection_type", []),
            type_counts.get("count", [])
        ))

        # Count by strength
        if "strength" in df.columns:
            strength_counts = df.group_by("strength").count().to_dict()
            stats["by_strength"] = dict(zip(
                strength_counts.get("strength", []),
                strength_counts.get("count", [])
            ))

        return stats

    def deduplicate(
        self,
        df: pl.DataFrame,
        keep: str = "first",
    ) -> pl.DataFrame:
        """
        Remove duplicate cross-references.

        Args:
            df: Cross-references DataFrame
            keep: Which duplicate to keep ("first", "last", "none")

        Returns:
            Deduplicated DataFrame
        """
        return df.unique(
            subset=["source_ref", "target_ref", "connection_type"],
            keep=keep,
        )

    @property
    def metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self._metrics


# =============================================================================
# MEMORY-MAPPED FILE UTILITIES
# =============================================================================

class MemoryMappedStore:
    """
    Memory-mapped file store for large datasets.

    Provides efficient random access to large files without loading
    the entire dataset into memory.
    """

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "r",
    ):
        """
        Initialize memory-mapped store.

        Args:
            path: File path
            mode: Open mode ("r" for read, "w" for write, "rw" for read-write)
        """
        self.path = Path(path)
        self.mode = mode
        self._mmap: Optional[mmap.mmap] = None
        self._file: Optional[Any] = None

    def __enter__(self) -> "MemoryMappedStore":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open the memory-mapped file."""
        if self.mode == "w":
            self._file = open(self.path, "wb")
        elif self.mode == "rw":
            self._file = open(self.path, "r+b")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                0,
                access=mmap.ACCESS_WRITE
            )
        else:  # read mode
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                0,
                access=mmap.ACCESS_READ
            )

    def read_table(self) -> pa.Table:
        """
        Read entire file as Arrow Table.

        Returns:
            Arrow Table
        """
        if self._mmap:
            reader = pa.ipc.open_file(pa.py_buffer(self._mmap[:]))
            return reader.read_all()
        raise RuntimeError("File not opened in read mode")

    def read_dataframe(self) -> pl.DataFrame:
        """
        Read entire file as Polars DataFrame.

        Returns:
            Polars DataFrame
        """
        table = self.read_table()
        return pl.from_arrow(table)

    def write_table(self, table: pa.Table, compression: str = "zstd") -> None:
        """
        Write Arrow Table to file.

        Args:
            table: Arrow Table to write
            compression: Compression codec
        """
        if self.mode not in ("w", "rw"):
            raise RuntimeError("File not opened in write mode")

        options = ipc.IpcWriteOptions(
            compression=pa.Codec(compression) if compression else None
        )

        with ipc.new_file(self._file, table.schema, options=options) as writer:
            writer.write_table(table)

    def write_dataframe(self, df: pl.DataFrame, compression: str = "zstd") -> None:
        """
        Write Polars DataFrame to file.

        Args:
            df: DataFrame to write
            compression: Compression codec
        """
        table = df.to_arrow()
        self.write_table(table, compression)

    def close(self) -> None:
        """Close the memory-mapped file."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None


# =============================================================================
# IPC CHANNEL FOR INTER-PROCESS COMMUNICATION
# =============================================================================

class IPCChannel:
    """
    Inter-process communication channel using Arrow IPC.

    Enables zero-copy data sharing between processes using shared memory
    or file-based IPC.
    """

    def __init__(
        self,
        name: str,
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize IPC channel.

        Args:
            name: Channel name
            temp_dir: Directory for temporary files
        """
        self.name = name
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._path = self.temp_dir / f"biblos_ipc_{name}.arrow"

    def send(self, df: pl.DataFrame) -> None:
        """
        Send DataFrame through the channel.

        Args:
            df: DataFrame to send
        """
        with ArrowIPCWriter(self._path) as writer:
            writer.write_dataframe(df)

        logger.debug(f"Sent {len(df):,} rows through channel {self.name}")

    def receive(self) -> pl.DataFrame:
        """
        Receive DataFrame from the channel.

        Returns:
            Received DataFrame
        """
        if not self._path.exists():
            return pl.DataFrame()

        with ArrowIPCReader(self._path) as reader:
            df = reader.read_dataframe()

        logger.debug(f"Received {len(df):,} rows from channel {self.name}")
        return df

    def send_schema(
        self,
        items: List[T],
        schema_type: Optional[Type[T]] = None,
    ) -> None:
        """
        Send dataclass instances through the channel.

        Args:
            items: List of dataclass instances
            schema_type: Optional schema type hint
        """
        df = SchemaConverter.dataclass_to_polars(items, schema_type)
        self.send(df)

    def receive_schema(self, schema_type: Type[T]) -> List[T]:
        """
        Receive dataclass instances from the channel.

        Args:
            schema_type: Target dataclass type

        Returns:
            List of dataclass instances
        """
        df = self.receive()
        return SchemaConverter.polars_to_dataclass(df, schema_type)

    def clear(self) -> None:
        """Clear the channel (delete temporary file)."""
        if self._path.exists():
            self._path.unlink()

    def exists(self) -> bool:
        """Check if channel has data."""
        return self._path.exists()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def read_verses_batch(
    path: Union[str, Path],
    batch_size: int = 1000,
    **kwargs
) -> Iterator[pl.DataFrame]:
    """
    Read verses file in batches.

    Args:
        path: File path
        batch_size: Rows per batch
        **kwargs: Additional options

    Yields:
        Polars DataFrame for each batch
    """
    processor = BatchVerseProcessor(BatchConfig(batch_size=batch_size))
    df = processor.load_verses(path)

    for _, batch in processor.iter_batches(df):
        yield batch


def write_verses_batch(
    verses: Union[List[Dict], List[VerseSchema], pl.DataFrame],
    path: Union[str, Path],
    format: str = "parquet",
    **kwargs
) -> BatchMetrics:
    """
    Write verses to file.

    Args:
        verses: Verse data
        path: Output path
        format: Output format
        **kwargs: Additional options

    Returns:
        Processing metrics
    """
    processor = BatchVerseProcessor()
    df = processor.load_verses(verses)
    processor.save_results(df, path, format)
    return processor.metrics


def read_crossrefs_batch(
    path: Union[str, Path],
    batch_size: int = 1000,
    **kwargs
) -> Iterator[pl.DataFrame]:
    """
    Read cross-references file in batches.

    Args:
        path: File path
        batch_size: Rows per batch
        **kwargs: Additional options

    Yields:
        Polars DataFrame for each batch
    """
    config = BatchConfig(batch_size=batch_size)

    with ArrowIPCReader(path, config) as reader:
        for df in reader.iter_dataframes(batch_size):
            yield df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "BatchConfig",
    "BatchMetrics",
    # IPC Reader/Writer
    "ArrowIPCWriter",
    "ArrowIPCReader",
    # Streaming Processor
    "StreamingBatchProcessor",
    # Specialized Processors
    "BatchVerseProcessor",
    "BatchCrossRefProcessor",
    # Memory-mapped Store
    "MemoryMappedStore",
    # IPC Channel
    "IPCChannel",
    # Convenience Functions
    "read_verses_batch",
    "write_verses_batch",
    "read_crossrefs_batch",
]
