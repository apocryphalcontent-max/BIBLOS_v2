#!/usr/bin/env python3
"""
BIBLOS v2 - Data Plane Performance Benchmarks

Comprehensive benchmarks comparing the performance of:
1. Original Python dict/dataclass-based data handling
2. New Polars/Arrow-based data plane

Metrics measured:
- Serialization/deserialization throughput
- Memory usage
- Batch processing speed
- Zero-copy IPC overhead
- File I/O performance (JSON vs Parquet vs Arrow IPC)

Usage:
    python scripts/benchmark_data_plane.py --rows 100000 --iterations 5
    python scripts/benchmark_data_plane.py --full  # Run full benchmark suite
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import polars as pl
    import pyarrow as pa
    import pyarrow.ipc as ipc
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("WARNING: Polars/PyArrow not installed. Install with: pip install polars pyarrow")

from data.schemas import (
    VerseSchema,
    CrossReferenceSchema,
    ConnectionType,
    ConnectionStrength,
)

if POLARS_AVAILABLE:
    from data.polars_schemas import (
        SchemaConverter,
        DataFrameBuilder,
        PolarsSchemas,
    )
    from data.arrow_batch import (
        BatchConfig,
        ArrowIPCWriter,
        ArrowIPCReader,
        BatchVerseProcessor,
        BatchCrossRefProcessor,
        IPCChannel,
    )


# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def generate_verse_dataclasses(n: int) -> List[VerseSchema]:
    """Generate n VerseSchema instances for testing."""
    books = ["GEN", "EXO", "LEV", "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "PSA"]
    testaments = ["OT", "NT"]
    languages = ["hebrew", "aramaic", "greek"]

    verses = []
    for i in range(n):
        book = books[i % len(books)]
        testament = "OT" if book in ["GEN", "EXO", "LEV", "PSA"] else "NT"
        language = "hebrew" if testament == "OT" else "greek"

        verse = VerseSchema(
            verse_id=f"{book}.{(i // 30) + 1}.{(i % 30) + 1}",
            book=book,
            book_name=book.title(),
            chapter=(i // 30) + 1,
            verse=(i % 30) + 1,
            text=f"This is the text of verse {i + 1}. Lorem ipsum dolor sit amet.",
            original_text=f"Original text {i + 1} in {language}.",
            testament=testament,
            language=language,
        )
        verses.append(verse)

    return verses


def generate_crossref_dataclasses(n: int) -> List[CrossReferenceSchema]:
    """Generate n CrossReferenceSchema instances for testing."""
    connection_types = [e.value for e in ConnectionType]
    strengths = [e.value for e in ConnectionStrength]
    books = ["GEN", "EXO", "PSA", "ISA", "MAT", "JHN", "ROM", "HEB", "REV"]

    refs = []
    for i in range(n):
        source_book = books[i % len(books)]
        target_book = books[(i + 3) % len(books)]

        ref = CrossReferenceSchema(
            source_ref=f"{source_book}.{(i % 50) + 1}.{(i % 30) + 1}",
            target_ref=f"{target_book}.{((i + 7) % 50) + 1}.{((i + 5) % 30) + 1}",
            connection_type=connection_types[i % len(connection_types)],
            strength=strengths[i % len(strengths)],
            confidence=0.5 + (i % 50) / 100.0,
            bidirectional=(i % 5 == 0),
            verified=(i % 10 == 0),
            patristic_support=(i % 7 == 0),
            notes=[f"Note {i + 1}"] if i % 3 == 0 else [],
            sources=["TSK"] if i % 4 == 0 else [],
        )
        refs.append(ref)

    return refs


def generate_verse_dicts(n: int) -> List[Dict[str, Any]]:
    """Generate n verse dicts for testing."""
    return [asdict(v) for v in generate_verse_dataclasses(n)]


def generate_crossref_dicts(n: int) -> List[Dict[str, Any]]:
    """Generate n cross-reference dicts for testing."""
    return [asdict(r) for r in generate_crossref_dataclasses(n)]


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

class BenchmarkTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        gc.collect()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def format_size(size_bytes: int) -> str:
    """Format byte size as human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def format_rate(rows: int, seconds: float) -> str:
    """Format processing rate as rows/second."""
    if seconds == 0:
        return "inf rows/s"
    rate = rows / seconds
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f}M rows/s"
    elif rate >= 1_000:
        return f"{rate / 1_000:.2f}K rows/s"
    return f"{rate:.2f} rows/s"


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

class DataPlaneBenchmarks:
    """Benchmark suite for data plane operations."""

    def __init__(self, num_rows: int = 10000, iterations: int = 3):
        self.num_rows = num_rows
        self.iterations = iterations
        self.results: Dict[str, Dict[str, Any]] = {}
        self.temp_dir = tempfile.mkdtemp(prefix="biblos_bench_")

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all benchmarks."""
        print(f"\n{'=' * 70}")
        print(f"BIBLOS v2 Data Plane Benchmarks")
        print(f"{'=' * 70}")
        print(f"Rows: {self.num_rows:,} | Iterations: {self.iterations}")
        print(f"Temp directory: {self.temp_dir}")
        print(f"{'=' * 70}\n")

        # Run benchmarks
        self.benchmark_dataclass_creation()
        self.benchmark_dict_vs_polars_conversion()
        self.benchmark_serialization()
        self.benchmark_file_io()
        self.benchmark_batch_processing()
        self.benchmark_ipc_transfer()
        self.benchmark_filtering()

        # Print summary
        self.print_summary()

        return self.results

    def benchmark_dataclass_creation(self):
        """Benchmark dataclass instantiation."""
        print("1. Dataclass Creation Benchmark")
        print("-" * 40)

        times = []
        for i in range(self.iterations):
            with BenchmarkTimer() as timer:
                verses = generate_verse_dataclasses(self.num_rows)
            times.append(timer.elapsed)
            del verses

        avg_time = sum(times) / len(times)
        rate = format_rate(self.num_rows, avg_time)

        self.results["dataclass_creation"] = {
            "operation": "Create VerseSchema instances",
            "rows": self.num_rows,
            "avg_time_sec": avg_time,
            "rate": rate,
        }

        print(f"  Create {self.num_rows:,} VerseSchema: {avg_time:.4f}s ({rate})")
        print()

    def benchmark_dict_vs_polars_conversion(self):
        """Benchmark dict to Polars DataFrame conversion."""
        print("2. Dict vs Polars Conversion Benchmark")
        print("-" * 40)

        verses = generate_verse_dataclasses(self.num_rows)

        # Dataclass to dict
        dict_times = []
        for i in range(self.iterations):
            with BenchmarkTimer() as timer:
                dicts = [asdict(v) for v in verses]
            dict_times.append(timer.elapsed)
            del dicts

        avg_dict_time = sum(dict_times) / len(dict_times)
        dict_rate = format_rate(self.num_rows, avg_dict_time)

        self.results["dataclass_to_dict"] = {
            "operation": "Dataclass to dict conversion",
            "rows": self.num_rows,
            "avg_time_sec": avg_dict_time,
            "rate": dict_rate,
        }
        print(f"  Dataclass -> dict: {avg_dict_time:.4f}s ({dict_rate})")

        if POLARS_AVAILABLE:
            # Dataclass to Polars DataFrame
            polars_times = []
            for i in range(self.iterations):
                with BenchmarkTimer() as timer:
                    df = SchemaConverter.dataclass_to_polars(verses, VerseSchema)
                polars_times.append(timer.elapsed)
                del df

            avg_polars_time = sum(polars_times) / len(polars_times)
            polars_rate = format_rate(self.num_rows, avg_polars_time)
            speedup = avg_dict_time / avg_polars_time if avg_polars_time > 0 else float('inf')

            self.results["dataclass_to_polars"] = {
                "operation": "Dataclass to Polars DataFrame",
                "rows": self.num_rows,
                "avg_time_sec": avg_polars_time,
                "rate": polars_rate,
                "speedup_vs_dict": speedup,
            }
            print(f"  Dataclass -> Polars: {avg_polars_time:.4f}s ({polars_rate})")
            print(f"  Speedup: {speedup:.2f}x")

        del verses
        print()

    def benchmark_serialization(self):
        """Benchmark JSON vs Arrow serialization."""
        print("3. Serialization Benchmark")
        print("-" * 40)

        verses = generate_verse_dataclasses(self.num_rows)
        dicts = [asdict(v) for v in verses]

        # JSON serialization
        json_ser_times = []
        json_sizes = []
        for i in range(self.iterations):
            with BenchmarkTimer() as timer:
                json_data = json.dumps(dicts)
            json_ser_times.append(timer.elapsed)
            json_sizes.append(len(json_data.encode('utf-8')))
            del json_data

        avg_json_ser = sum(json_ser_times) / len(json_ser_times)
        avg_json_size = sum(json_sizes) / len(json_sizes)

        self.results["json_serialization"] = {
            "operation": "JSON serialization",
            "rows": self.num_rows,
            "avg_time_sec": avg_json_ser,
            "avg_size_bytes": avg_json_size,
            "rate": format_rate(self.num_rows, avg_json_ser),
        }
        print(f"  JSON serialize: {avg_json_ser:.4f}s ({format_size(int(avg_json_size))})")

        if POLARS_AVAILABLE:
            # Arrow serialization
            df = SchemaConverter.dataclass_to_polars(verses, VerseSchema)

            arrow_ser_times = []
            arrow_sizes = []
            for i in range(self.iterations):
                with BenchmarkTimer() as timer:
                    table = df.to_arrow()
                    sink = pa.BufferOutputStream()
                    with ipc.new_file(sink, table.schema) as writer:
                        writer.write_table(table)
                    buffer = sink.getvalue()
                arrow_ser_times.append(timer.elapsed)
                arrow_sizes.append(buffer.size)
                del buffer

            avg_arrow_ser = sum(arrow_ser_times) / len(arrow_ser_times)
            avg_arrow_size = sum(arrow_sizes) / len(arrow_sizes)
            speedup = avg_json_ser / avg_arrow_ser if avg_arrow_ser > 0 else float('inf')
            compression = avg_json_size / avg_arrow_size if avg_arrow_size > 0 else 1.0

            self.results["arrow_serialization"] = {
                "operation": "Arrow IPC serialization",
                "rows": self.num_rows,
                "avg_time_sec": avg_arrow_ser,
                "avg_size_bytes": avg_arrow_size,
                "rate": format_rate(self.num_rows, avg_arrow_ser),
                "speedup_vs_json": speedup,
                "compression_vs_json": compression,
            }
            print(f"  Arrow serialize: {avg_arrow_ser:.4f}s ({format_size(int(avg_arrow_size))})")
            print(f"  Speedup: {speedup:.2f}x | Size ratio: {compression:.2f}x")

            del df

        del verses, dicts
        print()

    def benchmark_file_io(self):
        """Benchmark file I/O operations."""
        print("4. File I/O Benchmark")
        print("-" * 40)

        verses = generate_verse_dataclasses(self.num_rows)
        dicts = [asdict(v) for v in verses]

        json_path = Path(self.temp_dir) / "benchmark.json"
        parquet_path = Path(self.temp_dir) / "benchmark.parquet"
        arrow_path = Path(self.temp_dir) / "benchmark.arrow"

        # JSON write
        json_write_times = []
        for i in range(self.iterations):
            with BenchmarkTimer() as timer:
                with open(json_path, "w") as f:
                    json.dump(dicts, f)
            json_write_times.append(timer.elapsed)

        avg_json_write = sum(json_write_times) / len(json_write_times)
        json_size = json_path.stat().st_size

        # JSON read
        json_read_times = []
        for i in range(self.iterations):
            with BenchmarkTimer() as timer:
                with open(json_path, "r") as f:
                    loaded = json.load(f)
            json_read_times.append(timer.elapsed)
            del loaded

        avg_json_read = sum(json_read_times) / len(json_read_times)

        self.results["json_file_io"] = {
            "operation": "JSON file I/O",
            "rows": self.num_rows,
            "write_time_sec": avg_json_write,
            "read_time_sec": avg_json_read,
            "file_size_bytes": json_size,
        }
        print(f"  JSON write: {avg_json_write:.4f}s | read: {avg_json_read:.4f}s | size: {format_size(json_size)}")

        if POLARS_AVAILABLE:
            df = SchemaConverter.dataclass_to_polars(verses, VerseSchema)

            # Parquet write/read
            parquet_write_times = []
            for i in range(self.iterations):
                with BenchmarkTimer() as timer:
                    df.write_parquet(parquet_path, compression="zstd")
                parquet_write_times.append(timer.elapsed)

            avg_parquet_write = sum(parquet_write_times) / len(parquet_write_times)
            parquet_size = parquet_path.stat().st_size

            parquet_read_times = []
            for i in range(self.iterations):
                with BenchmarkTimer() as timer:
                    loaded_df = pl.read_parquet(parquet_path)
                parquet_read_times.append(timer.elapsed)
                del loaded_df

            avg_parquet_read = sum(parquet_read_times) / len(parquet_read_times)

            self.results["parquet_file_io"] = {
                "operation": "Parquet file I/O",
                "rows": self.num_rows,
                "write_time_sec": avg_parquet_write,
                "read_time_sec": avg_parquet_read,
                "file_size_bytes": parquet_size,
                "write_speedup": avg_json_write / avg_parquet_write,
                "read_speedup": avg_json_read / avg_parquet_read,
                "compression": json_size / parquet_size,
            }
            print(f"  Parquet write: {avg_parquet_write:.4f}s | read: {avg_parquet_read:.4f}s | size: {format_size(parquet_size)}")

            # Arrow IPC write/read
            arrow_write_times = []
            for i in range(self.iterations):
                with BenchmarkTimer() as timer:
                    df.write_ipc(arrow_path, compression="zstd")
                arrow_write_times.append(timer.elapsed)

            avg_arrow_write = sum(arrow_write_times) / len(arrow_write_times)
            arrow_size = arrow_path.stat().st_size

            arrow_read_times = []
            for i in range(self.iterations):
                with BenchmarkTimer() as timer:
                    loaded_df = pl.read_ipc(arrow_path, memory_map=True)
                arrow_read_times.append(timer.elapsed)
                del loaded_df

            avg_arrow_read = sum(arrow_read_times) / len(arrow_read_times)

            self.results["arrow_file_io"] = {
                "operation": "Arrow IPC file I/O",
                "rows": self.num_rows,
                "write_time_sec": avg_arrow_write,
                "read_time_sec": avg_arrow_read,
                "file_size_bytes": arrow_size,
                "write_speedup": avg_json_write / avg_arrow_write,
                "read_speedup": avg_json_read / avg_arrow_read,
                "compression": json_size / arrow_size,
            }
            print(f"  Arrow write: {avg_arrow_write:.4f}s | read: {avg_arrow_read:.4f}s | size: {format_size(arrow_size)}")

            del df

        del verses, dicts
        print()

    def benchmark_batch_processing(self):
        """Benchmark batch processing operations."""
        print("5. Batch Processing Benchmark")
        print("-" * 40)

        verses = generate_verse_dataclasses(self.num_rows)

        # Dict batch iteration
        dicts = [asdict(v) for v in verses]
        batch_size = 1000

        dict_batch_times = []
        for _ in range(self.iterations):
            with BenchmarkTimer() as timer:
                batches = []
                for i in range(0, len(dicts), batch_size):
                    batch = dicts[i:i + batch_size]
                    # Simulate processing
                    processed = [d.copy() for d in batch]
                    batches.append(processed)
            dict_batch_times.append(timer.elapsed)
            del batches

        avg_dict_batch = sum(dict_batch_times) / len(dict_batch_times)

        self.results["dict_batch_processing"] = {
            "operation": "Dict batch processing",
            "rows": self.num_rows,
            "batch_size": batch_size,
            "avg_time_sec": avg_dict_batch,
            "rate": format_rate(self.num_rows, avg_dict_batch),
        }
        print(f"  Dict batching: {avg_dict_batch:.4f}s ({format_rate(self.num_rows, avg_dict_batch)})")

        if POLARS_AVAILABLE:
            df = SchemaConverter.dataclass_to_polars(verses, VerseSchema)

            polars_batch_times = []
            for _ in range(self.iterations):
                with BenchmarkTimer() as timer:
                    batches = []
                    for batch in df.iter_slices(n_rows=batch_size):
                        # Simulate processing (zero-copy operation)
                        processed = batch.clone()
                        batches.append(processed)
                polars_batch_times.append(timer.elapsed)
                del batches

            avg_polars_batch = sum(polars_batch_times) / len(polars_batch_times)
            speedup = avg_dict_batch / avg_polars_batch if avg_polars_batch > 0 else float('inf')

            self.results["polars_batch_processing"] = {
                "operation": "Polars batch processing",
                "rows": self.num_rows,
                "batch_size": batch_size,
                "avg_time_sec": avg_polars_batch,
                "rate": format_rate(self.num_rows, avg_polars_batch),
                "speedup_vs_dict": speedup,
            }
            print(f"  Polars batching: {avg_polars_batch:.4f}s ({format_rate(self.num_rows, avg_polars_batch)})")
            print(f"  Speedup: {speedup:.2f}x")

            del df

        del verses, dicts
        print()

    def benchmark_ipc_transfer(self):
        """Benchmark inter-process communication."""
        print("6. IPC Transfer Benchmark")
        print("-" * 40)

        if not POLARS_AVAILABLE:
            print("  Skipped (Polars not available)")
            print()
            return

        verses = generate_verse_dataclasses(self.num_rows)
        df = SchemaConverter.dataclass_to_polars(verses, VerseSchema)

        # Simulate IPC via file
        ipc_path = Path(self.temp_dir) / "benchmark_ipc.arrow"

        ipc_times = []
        for _ in range(self.iterations):
            with BenchmarkTimer() as timer:
                # Write (sender)
                df.write_ipc(ipc_path, compression="lz4")
                # Read (receiver) with memory mapping
                received_df = pl.read_ipc(ipc_path, memory_map=True)
            ipc_times.append(timer.elapsed)
            del received_df

        avg_ipc_time = sum(ipc_times) / len(ipc_times)
        ipc_size = ipc_path.stat().st_size

        self.results["arrow_ipc_transfer"] = {
            "operation": "Arrow IPC transfer (file-based)",
            "rows": self.num_rows,
            "avg_time_sec": avg_ipc_time,
            "transfer_size_bytes": ipc_size,
            "rate": format_rate(self.num_rows, avg_ipc_time),
            "throughput_MB_sec": (ipc_size / (1024 * 1024)) / avg_ipc_time if avg_ipc_time > 0 else 0,
        }
        print(f"  IPC transfer: {avg_ipc_time:.4f}s ({format_size(ipc_size)})")
        print(f"  Throughput: {(ipc_size / (1024 * 1024)) / avg_ipc_time:.2f} MB/s")

        del df, verses
        print()

    def benchmark_filtering(self):
        """Benchmark filtering operations."""
        print("7. Filtering Benchmark")
        print("-" * 40)

        crossrefs = generate_crossref_dataclasses(self.num_rows)
        dicts = [asdict(r) for r in crossrefs]

        # Dict filtering
        dict_filter_times = []
        for _ in range(self.iterations):
            with BenchmarkTimer() as timer:
                filtered = [d for d in dicts if d["confidence"] >= 0.8]
            dict_filter_times.append(timer.elapsed)
            del filtered

        avg_dict_filter = sum(dict_filter_times) / len(dict_filter_times)

        self.results["dict_filtering"] = {
            "operation": "Dict list comprehension filter",
            "rows": self.num_rows,
            "avg_time_sec": avg_dict_filter,
            "rate": format_rate(self.num_rows, avg_dict_filter),
        }
        print(f"  Dict filtering: {avg_dict_filter:.4f}s")

        if POLARS_AVAILABLE:
            df = SchemaConverter.dataclass_to_polars(crossrefs, CrossReferenceSchema)

            polars_filter_times = []
            for _ in range(self.iterations):
                with BenchmarkTimer() as timer:
                    filtered = df.filter(pl.col("confidence") >= 0.8)
                polars_filter_times.append(timer.elapsed)
                del filtered

            avg_polars_filter = sum(polars_filter_times) / len(polars_filter_times)
            speedup = avg_dict_filter / avg_polars_filter if avg_polars_filter > 0 else float('inf')

            self.results["polars_filtering"] = {
                "operation": "Polars DataFrame filter",
                "rows": self.num_rows,
                "avg_time_sec": avg_polars_filter,
                "rate": format_rate(self.num_rows, avg_polars_filter),
                "speedup_vs_dict": speedup,
            }
            print(f"  Polars filtering: {avg_polars_filter:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")

            # Lazy evaluation benchmark
            lazy_filter_times = []
            for _ in range(self.iterations):
                with BenchmarkTimer() as timer:
                    filtered = (
                        df.lazy()
                        .filter(pl.col("confidence") >= 0.8)
                        .filter(pl.col("connection_type") == "typological")
                        .collect()
                    )
                lazy_filter_times.append(timer.elapsed)
                del filtered

            avg_lazy_filter = sum(lazy_filter_times) / len(lazy_filter_times)

            self.results["polars_lazy_filtering"] = {
                "operation": "Polars LazyFrame chained filters",
                "rows": self.num_rows,
                "avg_time_sec": avg_lazy_filter,
                "rate": format_rate(self.num_rows, avg_lazy_filter),
            }
            print(f"  Lazy chained filter: {avg_lazy_filter:.4f}s")

            del df

        del crossrefs, dicts
        print()

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'=' * 70}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 70}")

        print(f"\nDataset size: {self.num_rows:,} rows")
        print(f"Iterations: {self.iterations}")

        if POLARS_AVAILABLE:
            print("\nKey Performance Improvements (Polars vs Dict):")
            print("-" * 50)

            metrics = [
                ("Serialization", "arrow_serialization", "speedup_vs_json"),
                ("Batch Processing", "polars_batch_processing", "speedup_vs_dict"),
                ("Filtering", "polars_filtering", "speedup_vs_dict"),
            ]

            for name, key, speedup_key in metrics:
                if key in self.results and speedup_key in self.results[key]:
                    speedup = self.results[key][speedup_key]
                    print(f"  {name}: {speedup:.2f}x faster")

            if "parquet_file_io" in self.results:
                print(f"\nFile I/O Improvements (Parquet vs JSON):")
                print("-" * 50)
                r = self.results["parquet_file_io"]
                print(f"  Write: {r.get('write_speedup', 0):.2f}x faster")
                print(f"  Read: {r.get('read_speedup', 0):.2f}x faster")
                print(f"  Size: {r.get('compression', 0):.2f}x smaller")

        print(f"\n{'=' * 70}")
        print("Benchmark complete.")
        print(f"{'=' * 70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BIBLOS v2 Data Plane Performance Benchmarks"
    )
    parser.add_argument(
        "--rows", "-r",
        type=int,
        default=10000,
        help="Number of rows to benchmark (default: 10000)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Number of iterations per test (default: 3)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full benchmark suite with 100K rows"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for JSON results"
    )

    args = parser.parse_args()

    if args.full:
        rows = 100000
        iterations = 5
    else:
        rows = args.rows
        iterations = args.iterations

    benchmarks = DataPlaneBenchmarks(num_rows=rows, iterations=iterations)
    results = benchmarks.run_all()

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
